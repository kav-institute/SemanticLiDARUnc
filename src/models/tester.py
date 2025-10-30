# models/tester.py
import os
import warnings
import math
import json
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from utils.inputs import set_model_inputs
from models.temp_scaling import cache_calib_logits, calibrate_temperature_from_cache
from utils.reliability import reliability_diagram_from_probs
from utils.mc_dropout import predictive_entropy_mc

# aggregator classes for evaluation
from models.evaluator import (
    #SemanticSegmentationEvaluator,
    IoUEvaluator,
    UncertaintyPerClassAggregator,
    UncertaintyAccuracyAggregator,
    plot_iou_sorted_by_uncertainty
)

from models.probability_helper import (
    to_alpha_concentrations_from_shape_and_scale,
    get_predictive_entropy_norm,
    get_eps_value
)

# >>> use the shared classifier you already defined in models/losses.py
from models.losses import classify_output_kind

from utils.mc_dropout import (
    mc_forward,
)

from metrics.ece import ECEAggregator
from metrics.auroc import AUROCAggregator

from dataset.definitions import class_names, color_map

import json

# @@@ Visualization imports
from utils.vis_cv2 import (
    visualize_semantic_segmentation_cv2,
)
from utils.viz_panel import (
    create_ia_plots,
    register_optional_names
)
from models.probability_helper import (
    build_uncertainty_layers
)
import cv2

# ------------------------------
# Small helpers
# ------------------------------
@torch.no_grad()
def _classify_output_kind(outputs: torch.Tensor, class_dim: int = 1, sample_fraction: float = 0.1):
    """Classify model output as 'logits' | 'probs' | 'log_probs'.
    Copied locally to avoid import-cycle
    
    Heuristic:
        - probs:   values in [0,1] and sum over class_dim approx. 1 per pixel
        - log_probs: values <= 0 typically, and exp(outputs) behaves like probs
        - else: logits
    """
    x = outputs
    if sample_fraction and sample_fraction < 1.0 and x.ndim > 2:
        x_perm = x.movedim(class_dim, 1).contiguous()
        x_flat = x_perm.flatten(start_dim=2)
        S = x_flat.size(-1)
        k = max(1, int(S * sample_fraction))
        idx = torch.randperm(S, device=x.device)[:k]
        x = x_flat[..., idx]
    else:
        x = x.movedim(class_dim, 1).contiguous()

    in_range = (x.min() >= -1e-6) and (x.max() <= 1 + 1e-6)
    sums = x.sum(dim=1)
    if in_range and torch.allclose(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3):
        return 'probs'
    if x.max() <= 1e-6:
        ex = x.exp(); ex_sums = ex.sum(dim=1)
        if torch.allclose(ex_sums, torch.ones_like(ex_sums), atol=1e-3, rtol=1e-3):
            return 'log_probs'
    return 'logits'


class Tester:
    """
    Standalone tester:
    - loads checkpoint (if provided)
    - optional temperature scaling on a given calibration loader
    - runs a test pass with (optional) MC-dropout and ECE / RC plots saved

    Handles models whose forward() returns logits, probs, or log-probs:
      logits     -> (divide by T if set) -> softmax -> probs
      probs      -> log() -> (divide by T) -> softmax -> probs
      log_probs  -> (divide by T) -> softmax -> probs
    """

    EPS = 1e-12

    def __init__(self, model, cfg, visualize=False, logging=False, checkpoint: str | None = None):
        self.model = model
        self.cfg = cfg
        self.visualize = visualize
        
        self.use_mc_sampling = bool(self.cfg["model_settings"].get("use_mc_sampling", 0))
        
        # loss selection
        self.loss_name = cfg["model_settings"]["loss_function"]
        #declare which optional layer names exist for active visu
        if self.visualize:
            if self.loss_name == "Dirichlet":
                self.viz_optional_names = [
                    "H_norm", "AU_norm", "EU_norm",
                    "alpha0", "AU_frac", "EU_frac", "EU_minus_AU_frac",
                ]
            elif self.use_mc_sampling:
                self.viz_optional_names = [
                    "H_norm"
                ]
            else:
                self.viz_optional_names = []  # keep modular; add others here if needed
            
            # make all optional names visible but unticked in the panel
            if self.viz_optional_names:
                register_optional_names(self.viz_optional_names, default_enabled=False)
            
        self.logging = logging

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.num_classes = int(cfg["extras"]["num_classes"])-1 if self.loss_name=="Dirichlet" else int(cfg["extras"]["num_classes"])
        
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        
        # get baseline name
        self.baseline = cfg["model_settings"]["baseline"]
        
        # timers
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        
        # timers (CUDA events if available)
        self._use_cuda_events = torch.cuda.is_available()
        
        #self.save_path = cfg["extras"].get("save_path", "")
        # SemanticKitti test dataset has 95_937_758 valid lidar points not including class unlabeled
        self.iou_evaluator = IoUEvaluator(self.num_classes)
        
        if cfg["extras"].get("test_mask", 0):
            self.test_mask = list(cfg["extras"]["test_mask"].values())
        else:
            self.test_mask = [0] + [1] * (cfg["extras"]["num_classes"] - 1)
        
        # ignore index
        self.ignore_idx = 0
        
        self.max_samples_reservoir_sampling = 1_000_000
        self.unc_agg = UncertaintyPerClassAggregator(num_classes=self.num_classes, 
                                                     max_per_class=self.max_samples_reservoir_sampling)  # cap is optional
        self.ua_agg = UncertaintyAccuracyAggregator(max_samples=self.max_samples_reservoir_sampling)  # cap optional
        
        # ece_mode in {"alpha", "logits", "probs"}
        if self.loss_name=="Dirichlet": eval_on_outputkind = "alpha"
        elif self.use_mc_sampling: eval_on_outputkind = "probs"
        else: eval_on_outputkind = "logits"
        self.ece_eval = ECEAggregator(
                            n_bins=15,
                            mode=eval_on_outputkind,          # "alpha" | "logits" | "probs" depending on what you feed
                            ignore_index=self.ignore_idx,        
                            max_samples=self.max_samples_reservoir_sampling       # None or an int cap like to bound memory
                        )
        self.auroc_eval = AUROCAggregator(mode=eval_on_outputkind, 
                                          score="entropy_norm", 
                                          ignore_index=self.ignore_idx,
                                          max_samples=self.max_samples_reservoir_sampling)
        # Additional AUROC aggregator for epistemic MI (mutual information)
        mi_mode = "alpha" if self.loss_name == "Dirichlet" else ("probs" if self.use_mc_sampling else eval_on_outputkind)
        self.auroc_eval_mi = AUROCAggregator(
            mode=mi_mode,
            score="mi_norm",
            ignore_index=self.ignore_idx,
            max_samples=self.max_samples_reservoir_sampling
        )

        self.checkpoint = checkpoint
        # Load checkpoint if provided
        if checkpoint:
            try:
                sd = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(sd)
                print(f"[Tester] Loaded checkpoint: {checkpoint}")
            except Exception as e:
                print(f"[Tester] WARNING: failed to load checkpoint: {checkpoint}\n{e}")

        # Detected once on first batch (kept consistent with Trainer)
        self._model_act_kind: str | None = None  # 'logits' | '' | 'log_probs'

    # ---------- helpers for activations / temperature ----------

    def save_results(self, result_dict, out_dir):
        per_class_keys = [k for k in result_dict.keys() if k not in ("mIoU")]
        out = {
            "iou": {k: result_dict[k] for k in per_class_keys},  # all class-wise IoUs
            "mIoU": result_dict["mIoU"],
            "checkpoint": self.checkpoint,             # separate top-level line
        }

        def clean_nans(obj):
            """Recursively replace float('nan') with None (=> null in JSON)."""
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_nans(v) for v in obj]
            if isinstance(obj, float):
                return None if math.isnan(obj) else obj
            return obj
        
        # replace NaN -> null
        out = clean_nans(out)

        # Save result_dict to JSON in the same folder
        result_path = os.path.join(out_dir, "result_dict.json")
        with open(result_path, "w") as f:
            json.dump(out, f, indent=4)

        print(f"✅ Saved results to {result_path}")

    @torch.no_grad()
    def _to_logits(self, out: torch.Tensor, kind: str) -> torch.Tensor:
        """Convert model forward() output to logits."""
        if kind == 'logits':
            return out
        elif kind == 'probs':
            return out.clamp_min(self.EPS).log()
        elif kind == 'log_probs':
            return out  # already log-softmax
        else:
            raise ValueError(f"Unknown output kind: {kind}")

    # ------------------------------
    # utilities
    # ------------------------------
    def _start_timer(self):
        if self.logging:   # gate timing with logging
            if self._use_cuda_events:
                self._start.record()
            else:
                self._t0 = self._time.perf_counter()

    def _stop_timer_ms(self) -> float:
        if not self.logging:
            return 0.0
        if self._use_cuda_events:
            self._end.record(); torch.cuda.synchronize()
            return float(self._start.elapsed_time(self._end))
        else:
            return float((self._time.perf_counter() - self._t0) * 1000.0)
    
    
    # ---------- Testing ----------   
    @torch.no_grad()
    def test_epoch(self, loader, epoch: int = 0):
        from models.losses import classify_output_kind  # use your canonical detector
        import torch.nn.functional as F
        EPS = 1e-12
        def _to_logits(out: torch.Tensor, kind: str) -> torch.Tensor:
            if kind == 'logits':
                return out
            elif kind == 'probs':
                return out.clamp_min(EPS).log()
            elif kind == 'log_probs':
                return out
            else:
                raise ValueError(f"Unknown output kind: {kind}")

        inference_times = []
        self.model.eval()

        # ----- Determine epoch name early and set output directories -----
        try:
            path, ext = os.path.splitext(self.checkpoint)
            _, epoch_name = os.path.basename(path).split("_")
            try:
                epoch_name = f"{int(epoch_name):03d}"
            except ValueError:
                epoch_name = "final"
        except Exception:
            epoch_name = "final"

        # standard plots/metrics dir
        out_dir = os.path.join(os.path.dirname(self.checkpoint), "test"); os.makedirs(out_dir, exist_ok=True)
        # new cached summary dir (to avoid re-running model inference)
        summary_dir = os.path.join(os.path.dirname(self.checkpoint), "outputs_summary"); os.makedirs(summary_dir, exist_ok=True)
        summary_file = os.path.join(summary_dir, f"summary_epoch_{epoch_name}.pt")

        # If a cached summary exists, load it and skip the heavy inference loop
        loaded_summary = False
        if os.path.isfile(summary_file):
            try:
                cache = torch.load(summary_file, map_location="cpu")
                if len(loader) * loader.batch_size != cache["meta"].get("num_frames", None):
                    raise ValueError("Cached summary num_frames does not match current loader size.")

                # Validate required keys before restoring anything; fall back to recompute if missing
                need_uncert_things = bool(self.use_mc_sampling or self.loss_name == "Dirichlet")
                required_keys = ["iou_confmat"]
                if need_uncert_things:
                    required_keys += [
                        "ece_conf", "ece_correct",
                        "auroc_scores", "auroc_is_error",
                        "auroc_mi_scores", "auroc_mi_is_error",
                        "ua_uncert", "ua_correct",
                        "unc_values", "unc_seen_counts",
                    ]
                missing = [k for k in required_keys if cache.get(k) is None]
                if missing:
                    raise KeyError(f"Summary file missing keys: {missing}")

                # Restore IoU confusion matrix
                self.iou_evaluator.confmat = cache["iou_confmat"].to(self.iou_evaluator.device).clone().long()
                
                if need_uncert_things:
                    # Restore ECE buffers
                    if hasattr(self.ece_eval, "_conf"):
                        self.ece_eval._conf = cache["ece_conf"].clone()
                    if hasattr(self.ece_eval, "_correct"):
                        self.ece_eval._correct = cache["ece_correct"].clone().to(torch.uint8)
                    # Restore AUROC buffers
                    if hasattr(self.auroc_eval, "_scores"):
                        self.auroc_eval._scores = cache["auroc_scores"].clone()
                    if hasattr(self.auroc_eval, "_is_error"):
                        self.auroc_eval._is_error = cache["auroc_is_error"].clone().to(torch.uint8)
                    # Restore AUROC-MI buffers
                    if hasattr(self.auroc_eval_mi, "_scores"):
                        self.auroc_eval_mi._scores = cache["auroc_mi_scores"].clone()
                    if hasattr(self.auroc_eval_mi, "_is_error"):
                        self.auroc_eval_mi._is_error = cache["auroc_mi_is_error"].clone().to(torch.uint8)
                    # Restore UncertaintyAccuracy buffers
                    if hasattr(self.ua_agg, "_uncert"):
                        self.ua_agg._uncert = cache["ua_uncert"].clone()
                    if hasattr(self.ua_agg, "_correct"):
                        self.ua_agg._correct = cache["ua_correct"].clone().to(torch.uint8)
                    # Restore per-class uncertainty values (ridgeline etc.)
                    vals = cache["unc_values"]
                    if isinstance(vals, list):
                        self.unc_agg._values = [v.clone() for v in vals]
                    self.unc_agg._seen_counts = list(cache["unc_seen_counts"])  # keep as list[int]

                self.total_train_steps = len(loader) * self.cfg["train_params"]["batch_size"]
                loaded_summary = True
                print(f"[Tester] Loaded cached outputs summary: {summary_file}")
            except Exception as e:
                warnings.warn(
                    f"[Tester] WARNING: Cached summary exists but could not be fully loaded (will recompute).\n{e}"
                )

        # reset aggregators (only if we computed them anew)
        if not loaded_summary:
            self.ece_eval.reset()
            self.ua_agg.reset()
            self.unc_agg.reset()
            self.auroc_eval.reset()
            self.auroc_eval_mi.reset()
            self.iou_evaluator.reset()

        # toggles
        mc_T = int(self.cfg["model_settings"].get("mc_samples", 30))

        # prepare aggregators
        n_bins = 10
        thresholds = np.linspace(0.0, 1.0, n_bins, endpoint=False)

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"eval {epoch+1}")
        ):
            if loaded_summary:
                # Skip heavy per-batch inference if cache already available
                break
            # --- inputs/labels prep (unchanged) ---
            range_img = range_img.to(self.device)
            reflectivity = reflectivity.to(self.device)
            xyz = xyz.to(self.device)
            normals = normals.to(self.device)
            labels = labels.to(self.device)
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()
            inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)
            
            # statistics inference time start
            self._start_timer()
            
            # ---- If MC-dropout is enabled, do ONLY the MC path (single evaluator update) ----
            if self.use_mc_sampling:
                mc_outputs = mc_forward(self.model, inputs, T=mc_T)     # [T,B,C,H,W]
                
                # stop timer and log
                inference_times.append(self._stop_timer_ms())
                
                # debugging sanity check: print(mc_outputs.std().item(), mc_outputs.std(dim=0).mean().item())
                probs = F.softmax(mc_outputs, dim=2)    # softmax probs, [T,B,C,H,W]
                
                # Predictive distribution
                p_bar = probs.mean(dim=0)   # [B,C,H,W]
                # Predicted label (argmax)
                preds = p_bar.argmax(dim=1) # [B,1,H,W]
                
                @torch.no_grad()
                def mc_predictive_entropy_norm(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
                    # probs: [T,B,C,H,W]
                    # Predictive entropy: H[p_bar] = -Sum_c {p_bar}_c * log( {p_bar}_c )
                    p_bar = probs.mean(dim=0)                                # [B,C,H,W]
                    H = -(p_bar.clamp_min(eps) * p_bar.clamp_min(eps).log()).sum(dim=1)
                    return H / math.log(p_bar.size(1))                       # [B,H,W]

                @torch.no_grad()
                def mc_mutual_information_norm(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
                    """
                    probs: [T,B,C,H,W] where probs[t] is softmax(logits_t)
                    returns: normalized epistemic MI in [B,H,W], scaled to [0,1]
                    """
                    # predictive mean
                    p_bar = probs.mean(dim=0)  # [B,C,H,W]

                    # total predictive entropy H[p_bar]
                    H_bar = -(p_bar.clamp_min(eps) * p_bar.clamp_min(eps).log()).sum(dim=1)  # [B,H,W]

                    # per-sample entropies H[p_t]
                    H_t = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=2)    # [T,B,H,W]

                    # expected entropy E_t[H[p_t]]
                    EH = H_t.mean(dim=0)  # [B,H,W]

                    # epistemic = mutual information
                    C = p_bar.size(1)
                    MI = H_bar - EH        # [B,H,W]

                    # normalize epistemic by log(C) so it's ~= [0,1]
                    MI_norm = (MI / math.log(C)).clamp_min(0.0)
                    return MI_norm  # [B,H,W]
                
                # Predictive entropy norm
                H_norm = mc_predictive_entropy_norm(probs)
                
                # Metric aggregator accumulation
                ## iou
                self.iou_evaluator.update(preds, labels)
                ## H_norm vs accuracy
                self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,))
                ## calibration
                self.ece_eval.update(p_bar, labels)
                ## auroc (predictive entropy)
                self.auroc_eval.update(p_bar, labels)
                ## auroc (epistemic MI via override)
                MI_norm = mc_mutual_information_norm(probs)                  # [B,H,W]
                self.auroc_eval_mi.update(p_bar, labels, score_override=MI_norm)
                
                # log inference times
                inference_times.append(self._start.elapsed_time(self._end))


            ###############################################
            ### ---- Non-MC path ---- ###
            ###############################################
            else: # single pass (no MC)
                outputs = self.model(*inputs)
                inference_times.append(self._stop_timer_ms())

                if self.loss_name == "Dirichlet":
                    # outputs are expected to be split: first K channels -> shape logits,
                    # next 1 channel -> scale logits (same convention as Trainer)
                    shape_logits = outputs[:, :self.num_classes, ...]
                    scale_logits = outputs[:, self.num_classes:self.num_classes+1, ...] 
                    logits = shape_logits  # use shape logits only for prediction
                else:
                    logits = outputs

                probs = F.softmax(logits, dim=1)    # [B,C,H,W]

                preds = probs.argmax(dim=1) # [B,1,H,W]
                
                self.iou_evaluator.update(preds, labels)
                
                if self.loss_name=="Dirichlet":
                    alpha = to_alpha_concentrations_from_shape_and_scale(shape_logits, scale_logits)
                    H_norm = get_predictive_entropy_norm(alpha)
                    
                    self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,)     # ignore unlabeled if ignore_ids==0
                    )
                    self.ece_eval.update(alpha, labels)
                    self.auroc_eval.update(alpha, labels)
                    # AUROC using Dirichlet mutual information
                    self.auroc_eval_mi.update(alpha, labels)
            
            # END of branches MC or standard
            if self.use_mc_sampling or self.loss_name=="Dirichlet":
                self.unc_agg.update(labels=labels, uncertainty=H_norm)

            if self.visualize and step%20:
                idx0 = 0
                want_cuda_viz_calc = True
                
                # -> GT class
                semantics_gt = labels[idx0].detach().cpu().numpy()  # [H, W]
                if self.ignore_idx is not None:
                    mask = np.argwhere(semantics_gt==self.ignore_idx)
                else:
                    mask = None
                
                # -> Predicted Semantic Class
                # outputs_cpu = outputs[idx0].detach().cpu().numpy()  # [B, C, H, W] -> [C, H, W]
                # semantics_pred = np.argmax(outputs_cpu, axis=0)     # [H, W]
                semantics_pred = preds[idx0].detach().cpu().numpy() 
                    # apply mask for ignored index
                if self.ignore_idx is not None:
                    semantics_pred[mask[:, 0], mask[:, 1]] = self.ignore_idx
                    
                # -> Reflectivity
                reflectivity_img = reflectivity[idx0].permute(1, 2, 0).detach().cpu().numpy()
                reflectivity_img = cv2.applyColorMap((reflectivity_img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

                # -> XYZ
                xyz_img = xyz[idx0].permute(1, 2, 0).detach().cpu().numpy()
                
                # -> Normals
                normal_img = np.uint8(
                    255.0 * (normals[idx0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
                )
                
                # -> Error Mask (GT vs Prediction)
                err_img = np.uint8(
                    np.where(semantics_pred[..., None] != semantics_gt[..., None], (0, 0, 255), (0, 0, 0))
                )

                # Pred & GT to 3ch RGB image
                prev_sem_pred = visualize_semantic_segmentation_cv2(semantics_pred, class_colors=self.class_colors)
                prev_sem_gt = visualize_semantic_segmentation_cv2(semantics_gt, class_colors=self.class_colors)

                # Define main images dict to be displayed
                base_img_sources = {
                    "Reflectivity": reflectivity_img,
                    "Normals": normal_img,
                    "Pred": prev_sem_pred,
                    "GT": prev_sem_gt,
                    "ErrorMask": err_img,
                }
                
                # Define optional visus
                optional_builders = {}
                
                if self.loss_name == "Dirichlet":
                    alpha_src = alpha.detach()
                    
                    alpha_dev = alpha_src if ((want_cuda_viz_calc and alpha_src.is_cuda) or ((not want_cuda_viz_calc) and (not alpha_src.is_cuda))) \
                        else alpha_src.to("cuda" if want_cuda_viz_calc else "cpu", non_blocking=True)
                        
                    # build dict only for enabled names; lambdas are lazy and called only when drawn
                    optional_builders = {
                        n: (lambda name=n: build_uncertainty_layers(alpha_dev, [name], idx=idx0, mask=mask)[name])
                        for n in self.viz_optional_names
                    }
                elif self.use_mc_sampling:
                    H_norm_map = (H_norm[idx0].detach().cpu().numpy() * 255).astype(np.uint8)
                    H_norm_colormap = cv2.applyColorMap(H_norm_map, cv2.COLORMAP_TURBO)
                    if self.ignore_idx is not None:
                        H_norm_colormap[mask[:, 0], mask[:, 1]]=[0,0,0]
                    
                    images = [H_norm_colormap]
                    optional_builders = {
                        n: (lambda name=n: image)
                        for n, image in zip(self.viz_optional_names, images)
                    }
                    
                create_ia_plots(
                    base_images_dict=base_img_sources,
                    optional_builders=optional_builders,
                    args_o3d=(xyz_img, prev_sem_pred),
                    save_dir="",
                    enable=self.visualize
                )
        # @@@ END of Epoch
        
        # --> METRICS (epoch_name already computed above)
        
        # per class mIoU and overall mIoU
        mIoU, result_dict = self.iou_evaluator.compute(
            class_names=self.class_names,
            test_mask=self.test_mask,
            ignore_gt=[self.ignore_idx],
            reduce="mean",
            ignore_th=None
        )
        print(f"[eval] epoch {epoch + 1},  mIoU={mIoU:.4f}")

        # Save per-epoch summary (only if we actually ran the loop and built fresh stats)
        if not loaded_summary:
            try:
                summary = {
                    "meta": {
                        "epoch_name": epoch_name,
                        "checkpoint": self.checkpoint,
                        "loss_name": self.loss_name,
                        "use_mc": bool(self.use_mc_sampling),
                        "num_frames": len(loader) * loader.batch_size,
                    },
                    # IoU
                    "iou_confmat": self.iou_evaluator.confmat.detach().cpu(),
                    # ECE
                    "ece_conf": getattr(self.ece_eval, "_conf", torch.empty(0)).detach().cpu()
                                  if hasattr(self.ece_eval, "_conf") else None,
                    "ece_correct": getattr(self.ece_eval, "_correct", torch.empty(0, dtype=torch.uint8)).detach().cpu()
                                     if hasattr(self.ece_eval, "_correct") else None,
                    # AUROC
                    "auroc_scores": getattr(self.auroc_eval, "_scores", torch.empty(0)).detach().cpu()
                                       if hasattr(self.auroc_eval, "_scores") else None,
                    "auroc_is_error": getattr(self.auroc_eval, "_is_error", torch.empty(0, dtype=torch.uint8)).detach().cpu()
                                        if hasattr(self.auroc_eval, "_is_error") else None,
                    # AUROC (MI)
                    "auroc_mi_scores": getattr(self.auroc_eval_mi, "_scores", torch.empty(0)).detach().cpu()
                                           if hasattr(self.auroc_eval_mi, "_scores") else None,
                    "auroc_mi_is_error": getattr(self.auroc_eval_mi, "_is_error", torch.empty(0, dtype=torch.uint8)).detach().cpu()
                                            if hasattr(self.auroc_eval_mi, "_is_error") else None,
                    # UA aggregator
                    "ua_uncert": getattr(self.ua_agg, "_uncert", torch.empty(0)).detach().cpu()
                                   if hasattr(self.ua_agg, "_uncert") else None,
                    "ua_correct": getattr(self.ua_agg, "_correct", torch.empty(0, dtype=torch.uint8)).detach().cpu()
                                    if hasattr(self.ua_agg, "_correct") else None,
                    # Per-class uncertainty for ridgeline plots
                    "unc_values": [v.detach().cpu() for v in getattr(self.unc_agg, "_values", [])],
                    "unc_seen_counts": list(getattr(self.unc_agg, "_seen_counts", [])),
                }
                torch.save(summary, summary_file)
                print(f"✅ Saved outputs summary to {summary_file}")
            except Exception as e:
                print(f"[Tester] WARNING: Failed to save summary file.\n{e}")

        self.save_results(result_dict, out_dir)
        
        if self.use_mc_sampling or self.loss_name=="Dirichlet":
            # Entropy norm binned vs accuracy
            self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                bin_width=0.05, 
                show_percent_on_bars=True,
                title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch_name}.png"),
            )
            
            # ECE plot
            (ece, mce), ece_stats, fig_ece = self.ece_eval.compute(
                save_plot_path=os.path.join(out_dir, f"ece_epoch_{epoch_name}.png"),
                title=f"Reliability (epoch {epoch_name})"
            )
            print(f"Epoch {epoch} ECE: {ece:.4f}, MCE: {mce:.4f}")
            
            # AUROC
                # sanity check, after auroc_eval.compute(), before reset
            scores = self.auroc_eval._scores.numpy()
            errs   = self.auroc_eval._is_error.numpy().astype(bool)

            print("err_rate:", errs.mean())
            print("mean(score|error):   ", scores[errs].mean(),  "±", scores[errs].std())
            print("mean(score|correct): ", scores[~errs].mean(), "±", scores[~errs].std())
            
            auroc, _, fig_auroc = self.auroc_eval.compute(
                save_plot_path=os.path.join(out_dir, f"roc_epoch_{epoch_name}.png"),
                title=f"Error detection ROC (epoch {epoch_name})"
            )
            print(f"Epoch {epoch} AUROC: {auroc:.4f}")
            # AUROC (Epistemic MI)
            auroc_mi, _, fig_auroc_mi = self.auroc_eval_mi.compute(
                save_plot_path=os.path.join(out_dir, f"roc_mi_epoch_{epoch_name}.png"),
                title=f"Error detection ROC (MI, epoch {epoch_name})"
            )
            print(f"Epoch {epoch} AUROC (MI): {auroc_mi:.4f}")

        #--> IoU vs thresholded predictive uncertainty
        # self.unc_agg.plot_boxplot(
        #     class_names=list(class_names.values()),
        #     color_map=color_map,
        #     ignore_ids=(0,),  # ignore unlabeled
        #     title="Normalized Uncertainty per Class (Boxplot)",
        #     save_path="uncertainty_boxplot.png"
        # )
        
        # 1: car, 6: person, 9: road, 11: sidewalk, 13: building, 15: vegetation, 17: terrain, 18: pole
        # self.unc_agg.plot_ridgeline(
        #     class_names=list(class_names.values()),
        #     color_map=color_map,
        #     ignore_ids=(0,2,3,4,5,7,8,10,12,14,16,19),  # ignore unlabeled
        #     title="Normalized Uncertainty per Class (Ridged Plot)",
        #     save_path="uncertainty_ridgedplot.png"
        # )
        self.unc_agg.plot_ridgeline_fast(
            class_names=list(class_names.values()),
            color_map=color_map,
            ignore_ids=(0,2,3,4,5,7,8,10,12,14,16,19),
            bins=100_000,                 
            bandwidth="scott",     # or "scott" or a float (e.g., 0.02)
            title="Normalized Uncertainty per Class (Ridged Plot)",
            save_path=os.path.join(out_dir, f"uncertainty_ridgedplot_{epoch_name}.png"),
        )

        # plot_iou_sorted_by_uncertainty(
        #     self.unc_agg,
        #     result_dict=result_dict,
        #     class_names=list(class_names.values()),
        #     color_map=color_map,
        #     ignore_ids=self.test_mask,  # ignore 'unlabeled'
        #     save_path="iou_sorted_by_uncertainty.png"
        # )
        

    # convenience wrapper
    def run(self, dataloader_test, mc_samples: int = 30):
        """
        If do_calibration=True and calib_loader is given, fit T first.
        Then test one epoch and return (mIoU, class_metrics).
        """
        return self.test_epoch(dataloader_test, epoch=0)
