# models/tester.py
import os
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
    SemanticSegmentationEvaluator,
    UncertaintyPerClassAggregator,
    UncertaintyAccuracyAggregator,
    plot_iou_sorted_by_uncertainty
)

from models.probability_helper import (
    to_alpha_concentrations, get_predictive_entropy,
    compute_mc_reliability_bins, save_reliability_diagram,
    compute_entropy_error_iou, plot_mIOU_errorEntropy,
    compute_entropy_reliability
)

# >>> use the shared classifier you already defined in models/losses.py
from models.losses import classify_output_kind

from utils.mc_dropout import (
    mc_dropout_probs
)

from dataset.definitions import class_names, color_map

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
        self.logging = logging

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_classes = int(cfg["extras"]["num_classes"])
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        
        # loss selection
        self.loss_name = cfg["model_settings"]["loss_function"]
        
        # get baseline name
        self.baseline = cfg["model_settings"]["baseline"]
        
        # timers
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        
        self.save_path = cfg["extras"].get("save_path", "")
        # SemanticKitti test dataset has 95_937_758 valid lidar points not including class unlabeled
        self.evaluator = SemanticSegmentationEvaluator(self.num_classes)
        self.unc_agg = UncertaintyPerClassAggregator(num_classes=self.num_classes, max_per_class=100_000_000)  # cap is optional
        self.ua_agg = UncertaintyAccuracyAggregator(max_samples=100_000_000)  # cap optional


        # Load checkpoint if provided
        if checkpoint:
            try:
                sd = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(sd)
                print(f"[Tester] Loaded checkpoint: {checkpoint}")
            except Exception as e:
                print(f"[Tester] WARNING: failed to load checkpoint: {checkpoint}\n{e}")

        # Temperature scalar (optional; set by calibrate_temperature)
        self.T_value: float | None = None

        # Detected once on first batch (kept consistent with Trainer)
        self._model_act_kind: str | None = None  # 'logits' | 'probs' | 'log_probs'

    # ---------- helpers for activations / temperature ----------

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

    @torch.no_grad()
    def _forward_probs(self, inputs):
        """
        Returns probabilities [B,C,H,W].
        - Calls model.forward once.
        - Detects output kind on first batch and caches it in self._model_act_kind.
        - Applies temperature on logits, then softmax.
        """
        out = self.model(*inputs)  # logits/probs/log_probs
        if self._model_act_kind is None:
            self._model_act_kind = classify_output_kind(out, class_dim=1)

        logits = self._to_logits(out, self._model_act_kind)
        if self.T_value is not None:
            logits = logits / max(1e-3, float(self.T_value))
        probs = torch.softmax(logits, dim=1)
        return probs

    def _enable_dropout_only(self):
        """Enable only Dropout-like layers for MC sampling, keep everything else in eval."""
        import torch.nn as nn
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                              nn.AlphaDropout, nn.FeatureAlphaDropout)):
                m.train()

    # ---------- Temperature scaling API ----------

    @torch.no_grad()
    def cache_logits(self, calib_loader, ignore_index=255, mode="default", mc_samples=30):
        """Wrapper to use the refactored cache helper."""
        return cache_calib_logits(
            model=self.model,
            val_loader=calib_loader,
            device=self.device,
            cfg=self.cfg,
            ignore_index=ignore_index,
            mode=mode,
            mc_samples=mc_samples
        )

    def calibrate_temperature(
        self,
        calib_loader,
        ignore_index: int = 255,
        mode: str = "default",          # {"default","mc"}
        mc_samples: int = 30,
        optimizer_type: str = "lbfgs",  # {"lbfgs","adam"}
        epochs: int = 2,                # Adam only
        chunk_size: int = 1_000_000,
        max_iter_lbfgs: int = 100,
        init_T: float | str = "auto",
        save_json: str | None = None
    ):
        """Fit temperature on cached logits/labels and store in self.T_value."""
        logits_cpu, labels_cpu = self.cache_logits(
            calib_loader, ignore_index=ignore_index, mode=mode, mc_samples=mc_samples
        )
        T = calibrate_temperature_from_cache(
            logits_cpu, labels_cpu,
            device=self.device,
            init_T=init_T,
            optimizer_type=optimizer_type,
            lr=0.05,
            epochs=epochs,
            chunk_size=chunk_size,
            max_iter_lbfgs=max_iter_lbfgs,
            prev_T=self.T_value,
            save_path=save_json
        )
        self.T_value = float(T)
        print(f"[Tester] Calibrated temperature T={self.T_value:.4f}")

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
        self.evaluator.reset()

        # toggles
        use_dropout = bool(self.cfg["model_settings"].get("use_dropout", 0))
        want_dirichlet_metrics = (self.loss_name == "Dirichlet")

        ts_cache_mode = "mc" if use_dropout else "default"
                # MC sample count (re-use calibration setting if present)
        mc_T = int(self.cfg.get("calibration", {}).get("mc_samples", 30))
        
        # optional uncertainty plots config
        metrics_cfg = self.cfg.get("logging_settings", {}).get("metrics", {})
        do_mc_conf_acc = bool(metrics_cfg.get("McConfEmpAcc", 0))
        do_entropy_iou = bool(metrics_cfg.get("IouPlt", 0))
        do_entropy_rel = bool(metrics_cfg.get("EntErrRel", 0))

        if (do_mc_conf_acc or do_entropy_iou or do_entropy_rel) and not want_dirichlet_metrics and not use_dropout:
            # nothing to compute; silently skip
            do_mc_conf_acc = do_entropy_iou = do_entropy_rel = False

        # prepare aggregators
        n_bins = 10
        thresholds = np.linspace(0.0, 1.0, n_bins, endpoint=False)

        if do_mc_conf_acc:
            mc_hits = np.zeros(n_bins, dtype=np.float64)
            mc_tot = np.zeros(n_bins, dtype=np.float64)

        if do_entropy_iou:
            all_ious = []

        if do_entropy_rel:
            ent_err_tot = np.zeros(n_bins, dtype=np.float64)
            ent_err_err = np.zeros(n_bins, dtype=np.float64)

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"eval {epoch+1}")
        ):
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
            self._start.record()
            
            # ---- If MC-dropout is enabled, do ONLY the MC path (single evaluator update) ----
            if use_dropout:
                # (Optional) cache model output kind once for telemetry/consistency
                if not hasattr(self, "_model_act_kind") or self._model_act_kind is None:
                    outputs = self.model(*inputs)
                    
                    assert not (isinstance(outputs, tuple) and len(outputs) >2), "Model returned/generated unexpectedly too many outputs"
                    if isinstance(outputs, tuple) and len(outputs) == 2 and self.baseline=="SalsaNextAdf":
                        logits_mean, logits_var = outputs
                        outputs = logits_mean
                    
                    self._model_act_kind = classify_output_kind(outputs, class_dim=1)
                    del outputs

                # >>> MC sanity checks after mc_dropout_probs(...) is called
                    # - global std: mc_probs.std(dim=0).mean().item() ---> >~1e-6 
                    # - fraction of pixels that changed class: (mc_probs.argmax(dim=2) != mc_probs.argmax(dim=2)[0:1]).any(dim=0).float().mean().item() ---> >0
                
                # Decide where to apply T based on how T was calibrated
                if self.T_value is not None and ts_cache_mode == "default":
                    # Per-sample scaling: softmax(logits_like/T) then average
                    T_val = max(1e-3, float(self.T_value))
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=T_val)  # [T,B,C,H,W]
                    probs = mc_probs.mean(dim=0)  # [B,C,H,W]
                else:
                    # No per-sample TS; average first. If T exists & was MC-calibrated, apply post-mean.
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=None)  # [T,B,C,H,W]
                    mean_p = mc_probs.mean(dim=0).clamp_min(EPS)  # [B,C,H,W]
                    if self.T_value is not None and ts_cache_mode == "mc":
                        T_val = max(1e-3, float(self.T_value))
                        probs = torch.softmax(torch.log(mean_p) / T_val, dim=1)
                    else:
                        probs = mean_p

                preds = probs.argmax(dim=1)
                self.evaluator.update(preds, labels)
                
                # statistics inference time end
                self._end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                # log inference times
                inference_times.append(self._start.elapsed_time(self._end))

                # --- optional uncertainty metrics (unchanged logic) ---
                if (do_entropy_iou or do_entropy_rel) and self.logging:
                    ent_mc = predictive_entropy_mc(mc_probs, normalize=True)  # [B,H,W]
                    err_mask = (preds != labels).to(torch.int32)
                    if do_entropy_iou:
                        ious = compute_entropy_error_iou(ent_mc, err_mask, thresholds=thresholds)
                        all_ious.append(ious.cpu().numpy())
                    if do_entropy_rel:
                        tot, err, err_rate, ece = compute_entropy_reliability(ent_mc, err_mask, n_bins=n_bins)
                        ent_err_tot += tot
                        ent_err_err += err
            ###############################################
            ### ---- Non-MC path ---- ###
            ###############################################
            else:
                outputs = self.model(*inputs)  # logits/probs/log_probs
                assert not (isinstance(outputs, tuple) and len(outputs) >2), "Model returned/generated unexpectedly too many outputs"
                if isinstance(outputs, tuple) and len(outputs) == 2 and self.baseline=="SalsaNextAdf":
                    logits_mean, logits_var = outputs
                    outputs = logits_mean
                        
                if not hasattr(self, "_model_act_kind") or self._model_act_kind is None:
                    self._model_act_kind = classify_output_kind(outputs, class_dim=1)

                logits_raw = _to_logits(outputs, self._model_act_kind)  # un-tempered logits-like
                logits = logits_raw
                if self.T_value is not None:
                    logits = logits / max(1e-3, float(self.T_value))

                probs = F.softmax(logits, dim=1)
                assert probs.shape[1] == self.num_classes, \
                    f"Channel mismatch: probs C={probs.shape[1]} vs cfg num_classes={self.num_classes}"

                preds = probs.argmax(dim=1)
                self.evaluator.update(preds, labels)
                
                # statistics inference time end
                self._end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                # log inference times
                inference_times.append(self._start.elapsed_time(self._end))
                
                # ---- optional uncertainty metrics (Dirichlet branch uses UN-TEMPERED logits) ----
                if want_dirichlet_metrics:
                    alpha = to_alpha_concentrations(logits_raw)  # use raw logits for alpha
                    pred_entropy = get_predictive_entropy(alpha)  # [B,H,W]
                    pred_entropy_norm = pred_entropy / math.log(self.num_classes)

                    self.unc_agg.update(labels=labels, uncertainty=pred_entropy_norm)
                    self.ua_agg.update(
                        labels=labels,
                        preds=preds,
                        uncertainty=pred_entropy_norm,
                        ignore_ids=(0,),                       # ignore unlabeled if you use 0
                    )

                    if do_mc_conf_acc:
                        hits, tot = compute_mc_reliability_bins(alpha, labels, n_bins=n_bins, n_samples=120)
                        mc_hits += hits
                        mc_tot += tot

                    if do_entropy_iou:
                        err_mask = (preds != labels).to(torch.int32)
                        ious = compute_entropy_error_iou(pred_entropy_norm, err_mask, thresholds=thresholds)
                        all_ious.append(ious.cpu().numpy())

                    if do_entropy_rel:
                        err_mask = (preds != labels).to(torch.int32)
                        tot, err, err_rate, ece = compute_entropy_reliability(pred_entropy_norm, err_mask, n_bins=n_bins)
                        ent_err_tot += tot
                        ent_err_err += err
        
        # per class mIoU and overall mIoU
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        print(f"[eval] epoch {epoch + 1},  mIoU={mIoU:.4f}")

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
            bins=100_000,                 # try 4096 if you like
            bandwidth="scott",     # or "scott" or a float (e.g., 0.02)
            title="Normalized Uncertainty per Class (Ridged Plot)",
            save_path="uncertainty_ridgedplot.png",
        )
        
        self.ua_agg.plot_accuracy_vs_uncertainty_bins(
            bin_edges=np.linspace(0.0, 1.0, 11),  # 10 bins
            title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
            save_path="acc_vs_unc_bins.png",
        )

        plot_iou_sorted_by_uncertainty(
            self.unc_agg,
            result_dict=result_dict,
            class_names=list(class_names.values()),
            color_map=color_map,
            ignore_ids=(0,),  # ignore 'unlabeled'
            save_path="iou_sorted_by_uncertainty.png"
        )
        
        # --- plots / logs ---
        if self.logging and self.save_path:
            out_dir = os.path.join(self.save_path, "eval")
            os.makedirs(out_dir, exist_ok=True)

            # log IoU to tensorboard
            for cls in range(self.num_classes):
                self.writer.add_scalar('IoU_{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            
            self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
            self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)
            
            
            if do_mc_conf_acc and mc_tot.sum() > 0:
                emp_acc = np.divide(mc_hits, mc_tot, out=np.zeros_like(mc_hits), where=mc_tot > 0)
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                save_reliability_diagram(
                    empirical_acc=emp_acc,
                    bin_centers=centers,
                    tot_counts=mc_tot,
                    output_path=os.path.join(out_dir, f"reliability_epoch_{epoch:06d}.png"),
                    title='Reliability diagram\n(dot area ∝ #pixels per bin — sharpness)',
                    xlabel='Predicted confidence (MC estimate)',
                    ylabel='Empirical accuracy',
                )

            if do_entropy_iou and len(all_ious) > 0:
                all_ious_np = np.stack(all_ious, axis=0)
                mean_ious = all_ious_np.mean(axis=0)
                plot_mIOU_errorEntropy(
                    mean_ious,
                    thresholds,
                    output_path=os.path.join(out_dir, f"entropy_iou_epoch_{epoch:06d}.png"),
                )

            if do_entropy_rel and ent_err_tot.sum() > 0:
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                emp_err_rate = np.divide(ent_err_err, ent_err_tot, out=np.zeros_like(ent_err_err), where=ent_err_tot > 0)
                save_reliability_diagram(
                    empirical_acc=emp_err_rate,
                    bin_centers=centers,
                    tot_counts=ent_err_tot,
                    output_path=os.path.join(out_dir, f"entropy_error_rel_epoch_{epoch:06d}.png"),
                    title='Reliability of Entropy as Error Predictor\n(dot area ∝ #pixels per entropy bin — sharpness)',
                    xlabel='Normalized entropy',
                    ylabel='Observed error rate',
                )

        return mIoU, result_dict

    # convenience wrapper
    def run(self, dataloader_test, calib_loader=None, do_calibration: bool = False,
            ts_mode: str = "default", mc_samples: int = 30):
        """
        If do_calibration=True and calib_loader is given, fit T first.
        Then test one epoch and return (mIoU, class_metrics).
        """
        if do_calibration and calib_loader is not None:
            json_out = None
            if self.save_path:
                json_out = os.path.join(self.save_path, "eval", "temperature.json")
            self.calibrate_temperature(
                calib_loader,
                mode=ts_mode,
                mc_samples=mc_samples,
                optimizer_type="lbfgs",
                max_iter_lbfgs=100,
                epochs=2,
                chunk_size=1_000_000,
                init_T="auto",
                save_json=json_out
            )
        return self.test_epoch(dataloader_test, epoch=0)
