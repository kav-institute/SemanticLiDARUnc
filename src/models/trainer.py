from __future__ import annotations

import os
import math
import numpy as np
import tqdm
import torch
import cv2
cv2.setNumThreads(1)    # avoid oversubscribing CPU during training
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# --- project modules ---
from models.evaluator import (
    SemanticSegmentationEvaluator,
    UncertaintyAccuracyAggregator
)

from models.temp_scaling import (
    cache_calib_logits,
    calibrate_temperature_from_cache,
)
from utils.mc_dropout import (
    set_dropout_mode,
    mc_dropout_probs,
    predictive_entropy_mc,
)
from utils.inputs import set_model_inputs
from utils.loss_balancer import LossBalancer
from utils.vis_cv2 import (
    visualize_semantic_segmentation_cv2,
)

from models.probability_helper import (
    to_alpha_concentrations,
    get_predictive_entropy_norm,
    build_uncertainty_layers,
    smoothing_schedule,
    # Global parameter
    get_eps_value,
    get_alpha_temperature,
    # Metrics
    compute_entropy_error_iou,
    plot_mIOU_errorEntropy,
    compute_entropy_reliability,
    compute_mc_reliability_bins,
    save_reliability_diagram
)

from typing import Dict, Callable, Optional, Tuple
import numpy as np
from utils.viz_panel import (
    create_ia_plots,
    register_optional_names
)
# from utils.viz_panel import VizPanel
# from utils.viz_panel import register_optional_names

# _PANEL: Optional[VizPanel] = None

# def create_ia_plots(
#     base_images_dict: Dict[str, np.ndarray],
#     optional_builders: Dict[str, Callable[[], np.ndarray]],
#     args_o3d: Tuple[np.ndarray, np.ndarray],
#     save_dir: str = "",
#     enable: bool = True,
# ):
#     """
#     base_images_dict: name -> HxWx3 uint8 (already computed)
#     optional_builders: name -> () -> HxWx3 uint8 (lazy; only when enabled)
#     args_o3d: (xyz, color_bgr) for 'q' shortcut (unchanged)
#     enable: if False, close window and return
#     """
#     if not enable:
#         destroy_panel()
#         return

#     panel = get_panel()
#     panel.render_with_builders(base_sources=base_images_dict,
#                                optional_builders=optional_builders,
#                                scale=1.5)

#     key = cv2.waitKey(1) & 0xFF
#     if key != 0xFF:
#         panel.handle_key(key)

#     if key == ord("q"):
#         # 3D viewer hook (unchanged)
#         import open3d as o3d
#         from utils.o3d_env import ensure_o3d_runtime, has_display
#         if not has_display():
#             return
#         ensure_o3d_runtime()
#         xyz, color_bgr = args_o3d
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
#         rgb = (color_bgr[..., ::-1].reshape(-1, 3).astype(np.float32)) / 255.0
#         pcd.colors = o3d.utility.Vector3dVector(rgb)
#         mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         o3d.visualization.draw_geometries([mesh, pcd])
        
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


def _to_logits(out: torch.Tensor, kind: str) -> torch.Tensor:
    eps = get_eps_value()
    if kind == 'logits':
        return out
    if kind == 'probs':
        return out.clamp_min(eps).log()
    if kind == 'log_probs':
        return out
    raise ValueError(f"Unknown output kind: {kind}")


def grad_norm_of(weighted_loss, logits):
    g = torch.autograd.grad(weighted_loss, logits, retain_graph=True, create_graph=False, allow_unused=False)[0]
    return float(g.norm().detach().cpu())
                    

#@torch.no_grad()    # TODO: for debugging only!
class Trainer:
    """
    Clean trainer with:
        - standard train / eval loops
        - optional MC-dropout path
        - Dirichlet-aware caching so alpha is computed ONCE per batch when needed
        - optional post-hoc Temperature Scaling (TS)

    Expectation:
      - model.forward(*inputs) returns either logits, probs, or log_probs
      - for ADF baseline returns (logits_mean, logits_var)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        visualize: bool = False,
        logging: bool = False,
        test_mask=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.visualize = visualize
        self.logging = logging

        if test_mask is None:
            test_mask = [1] * cfg["extras"]["num_classes"]
            test_mask[0] = 0

        # data / task meta
        self.num_classes = int(cfg["extras"]["num_classes"])
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        self.global_step = 0

        # loss selection & baseline
        self.loss_name = cfg["model_settings"]["loss_function"]
        self.baseline = cfg["model_settings"]["baseline"]

        # evaluator
        self.evaluator = SemanticSegmentationEvaluator(self.num_classes, test_mask=test_mask)

        # device & writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_path = cfg["extras"].get("save_path", "")
        self.writer = SummaryWriter(log_dir=self.save_path) if self.logging and self.save_path else None

        # output kind cache (set on first batch)
        self._model_act_kind: str | None = None

        # temperature value (optional, filled once calibrated)
        self.T_value: float | None = None

        # timers (CUDA events if available)
        self._use_cuda_events = torch.cuda.is_available()
        if self._use_cuda_events:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
        else:
            import time
            self._t0 = 0.0
            self._time = time

        # losses
        self._init_losses()
        
        #declare which optional layer names exist for the active loss
        if self.visualize:
            if self.loss_name == "Dirichlet":
                self.viz_optional_names = [
                    "H_norm", "AU_norm", "EU_norm",
                    "alpha0", "AU_frac", "EU_frac", "EU_minus_AU_frac",
                ]
            elif self.loss_name == "SalsaNextAdf":
                self.viz_optional_names = ["unc_norm"]
            else:
                self.viz_optional_names = []  # keep modular; add others here if needed
            
            # make all optional names visible but unticked in the panel
            if self.viz_optional_names:
                register_optional_names(self.viz_optional_names, default_enabled=False)
                
        if self.loss_name=="Dirichlet":
            self._alpha_cache: torch.Tensor | None = None
            
        # EVALUATION inits
        self.ua_agg = UncertaintyAccuracyAggregator(max_samples=100_000_000)  # cap optional


    # ------------------------------
    # loss setup
    # ------------------------------
    def _init_losses(self):
        from models.losses import (
            TverskyLoss,
            CrossEntropyLoss,
            LovaszSoftmaxStable,
            DirichletLoss,
            SoftmaxHeteroscedasticLoss,
        )

        if self.loss_name == "Tversky":
            self.criterion_ce = CrossEntropyLoss()
            self.criterion_tversky = TverskyLoss()
        elif self.loss_name == "CE":
            self.criterion_ce = CrossEntropyLoss()
        elif self.loss_name == "Lovasz":
            self.criterion_lovasz = LovaszSoftmaxStable()
        elif self.loss_name == "Dirichlet":
            nll_args = dict(
                num_classes=self.num_classes,   # number of classes
                objective="nll",                # "nll" | "dce" | "imax"
                smoothing=0.25,                 # only used when objective="nll"
                temperature=get_alpha_temperature(),                # logits -> alpha temperature
                prior_concentration=3.0,        # evidence strength s (typ 1..5)
                kl_weight=0.0,                  # keep 0.0 here; apply KL weight in trainer
                eps=get_eps_value(),                       # numerical epsilon
                kl_prior_mode="evidence",       # "evidence" keeps mean p_hat; "symmetric" pulls toward uniform
                ignore_index=0,              # set to 0 to ignore unlabeled class, None to include all
            )
            
            imax_args = dict(
                num_classes=self.num_classes,   # number of classes
                objective="imax",               # "nll" | "dce" | "imax"
                temperature=get_alpha_temperature(),                # logits -> alpha temperature
                prior_concentration=3.0,        # evidence strength s (typ 1..5)
                kl_weight=0.0,                  # keep 0.0 here; apply KL weight in trainer
                eps=get_eps_value(),                       # numerical epsilon
                kl_prior_mode="evidence",       # "evidence" keeps mean p_hat; "symmetric" pulls toward uniform
                ignore_index=0,                 # set to 0 to ignore unlabeled class, None to include all
                p_moment=4.0,                   # p for iMAX Lp bound, 4.0
            )

            self.criterion_dirichlet_nll_loss = DirichletLoss(**nll_args)
            self.criterion_dirichlet_imax_loss = DirichletLoss(**imax_args)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=0)

            self.w_nll    = 0.15         # with 0.25smoothing, w_nll =0.1, (semantic kitti)
            self.w_imax   = 3.0         # with 0.25smoothing, w_imax=6.0, (semantic kitti)
            self.w_ls     = 2.5         # with 0.25smoothing, w_ls  =2.5, (semantic kitti)
            
            # KL term
            self.w_kl     = 1e-3
            
            
            # # Adaptive 2-head balancer (Dirichlet data + Lovasz)
            # self.balancer = LossBalancer(
            #     targets={'dir': 0.7, 'ls': 0.3},
            #     warmup_epochs=0,
            #     every_n_epochs=1,
            #     bootstrap_batches=128,
            #     ema=0.6,
            #     clamp=(0.2, 5.0),
            #     start_weights=None,
            # )
        elif self.loss_name == "SalsaNext":
            self.criterion_ce = CrossEntropyLoss()
            self.criterion_lovasz = LovaszSoftmaxStable()
        elif self.loss_name == "SalsaNextAdf":
            self.criterion_ce = CrossEntropyLoss()
            self.criterion_lovasz = LovaszSoftmaxStable()
            self.het = SoftmaxHeteroscedasticLoss(num_classes=self.num_classes)
        else:
            raise NotImplementedError(f"Unknown loss function: {self.loss_name}")

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

    # ------------------------------
    # training
    # ------------------------------
    def train_one_epoch(self, loader, epoch: int):
        self.model.train()
        total_loss = 0.0

        # Dirichlet-specific schedules
        if self.loss_name == "Dirichlet":
            get_predictive_entropy_norm.reset()
            self.criterion_dirichlet_nll_loss.smoothing = smoothing_schedule(epoch, self.num_epochs)
            # self.balancer.begin_epoch(epoch)

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"train {epoch+1}")
        ):
            self.global_step = epoch * len(loader) + step

            range_img    = range_img.to(self.device, non_blocking=True)
            reflectivity = reflectivity.to(self.device, non_blocking=True)
            xyz          = xyz.to(self.device, non_blocking=True)
            normals      = normals.to(self.device, non_blocking=True)
            labels       = labels.to(self.device, non_blocking=True)
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()

            inputs = set_model_inputs(range_img, reflectivity, xyz, normals, self.cfg)

            self._start_timer()
            outputs = self.model(*inputs)
            logits_var = None
            if isinstance(outputs, tuple) and len(outputs) == 2 and self.baseline == "SalsaNextAdf":
                logits_mean, logits_var = outputs
                outputs = logits_mean
            assert not (isinstance(outputs, tuple) and len(outputs) > 2), "Unexpected model outputs"

            # Decide output kind once (first batch)
            if self._model_act_kind is None:
                self._model_act_kind = _classify_output_kind(outputs, class_dim=1)

            # ---- compute loss (single-pass α caching for Dirichlet) ----
            if self.loss_name == "Tversky":
                loss_sem = self.criterion_ce(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss_t = self.criterion_tversky(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss = loss_sem + loss_t
                
            elif self.loss_name == "CE":
                loss = self.criterion_ce(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                
            elif self.loss_name == "Lovasz":
                loss = self.criterion_lovasz(outputs, labels, model_act=self._model_act_kind)
                
            elif self.loss_name == "SalsaNext":
                loss_ce = self.criterion_ce(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss_ls = self.criterion_lovasz(outputs, labels, model_act=self._model_act_kind)
                loss = loss_ce + loss_ls
                
            elif self.loss_name == "SalsaNextAdf":
                # CE on mean logits
                loss_ce = self.criterion_ce(logits_mean, labels, num_classes=self.num_classes, model_act="logits")
                # Lovasz on mean probs
                mean_probs = torch.softmax(logits_mean, dim=1)
                loss_ls = self.criterion_lovasz(mean_probs, labels, model_act="probs")
                # Heteroscedastic
                loss_het = self.het([logits_mean, logits_var], labels)
                loss = loss_ce + loss_ls + 0.1 * loss_het
                
            elif self.loss_name == "Dirichlet":
                # alpa computed ONCE here and reused everywhere below
                T = self.criterion_dirichlet_nll_loss.temperature
                eps = self.criterion_dirichlet_nll_loss.eps
                # For stability, convert to *logits-like* first and then α
                logits_like = _to_logits(outputs, self._model_act_kind)
                alpha = to_alpha_concentrations(logits_like, T=T, eps=eps)
                
                # Dirichlet data-term
                # data terms
                loss_nll  = self.criterion_dirichlet_nll_loss .data_from_alpha(alpha, labels)   # scalar
                loss_imax = self.criterion_dirichlet_imax_loss.data_from_alpha(alpha, labels)   # scalar

                # Lovasz on Dirichlet mean (reuse alpha)
                p_hat = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
                loss_ls = self.criterion_lovasz(p_hat, labels.squeeze(1).long() if labels.ndim == 4 else labels.long(), model_act="probs")

                # Combine with balancer
                #loss_main = self.balancer.combine(dir=data_dir, ls=loss_ls)
                loss =  self.w_nll  * loss_nll + \
                        self.w_imax * loss_imax + \
                        self.w_ls   * loss_ls
                
                # Add evidence prior
                loss += self.w_kl * self.criterion_dirichlet_nll_loss.kl_from_alpha(alpha, labels)
                
                # after computing losses and before optimizer.step():
                gn_nll  = grad_norm_of(self.w_nll  * loss_nll,  outputs)
                gn_imax = grad_norm_of(self.w_imax * loss_imax, outputs)
                gn_ls   = grad_norm_of(self.w_ls   * loss_ls,   outputs)

                # gentle evidence prior
                    # per-epoch schedule
                # if epoch < 3:   kl_w = 0.0
                # elif epoch < 8: kl_w = 5e-4
                # else:           kl_w = 1e-3
                
                # kl_term = self.criterion_dirichlet_loss.kl_from_alpha(alpha, labels)
                # loss += kl_w * kl_term

                self._alpha_cache = alpha.detach()  # keep a clean, detached copy for viz (AFTER loss is computed!)
                
            else:
                raise RuntimeError("unreachable")

            # Backprop
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

            # --- Post-step bookkeeping
            elapsed_ms = self._stop_timer_ms()

            # Dirichlet running uncertainty stat (reuse cached alpha)
            if self.loss_name == "Dirichlet" and self._alpha_cache is not None:
                H_norm = get_predictive_entropy_norm.accumulate(self._alpha_cache.cpu())

            # Logging (every 10 steps)
            if self.logging and self.writer and step % 10 == 0:
                self.writer.add_scalar("train/loss/iter", loss.item(), self.global_step)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.global_step)
                if self.loss_name == "Dirichlet":
                    #w = self.balancer.current_weights()
                    self.writer.add_scalar('train/loss_nll', loss_nll.item() * self.w_nll, self.global_step)
                    self.writer.add_scalar('train/loss_imax', loss_imax.item() * self.w_imax, self.global_step)
                    self.writer.add_scalar('train/loss_ls', loss_ls.item() * self.w_ls, self.global_step)
                    with torch.no_grad(): 
                        alpha0 = alpha.sum(dim=1, keepdim=False) # [B,H,W] 
                        med_alpha0 = alpha0.median().item() # per-class evidence (scale-invariant across C): 
                        med_alpha0_per_cls = (alpha0 / float(alpha.shape[1])).median().item()
                    self.writer.add_scalar('train/alpha0_median', med_alpha0, self.global_step)
                    self.writer.add_scalar('train/alpha0_per_class_median', med_alpha0_per_cls, self.global_step)
                    
                    # entropy masses less than threshold
                    self.writer.add_scalar("H_norm/pct_lt_0.1", (H_norm < 0.1).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_lt_0.25", (H_norm < 0.25).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.5", (H_norm > 0.5).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.75", (H_norm > 0.75).float().mean().item(), self.global_step)
                    
                    # who actually dominates the update
                    self.writer.add_scalar('train/grad_norm_logits/nll', gn_nll, self.global_step)
                    self.writer.add_scalar('train/grad_norm_logits/imax', gn_imax, self.global_step)
                    self.writer.add_scalar('train/grad_norm_logits/ls', gn_ls, self.global_step)

            # Interactive visualization (cheap, reusing computed items)
            if self.visualize:
                idx0 = 0
                want_cuda_viz_calc = True
                
                # -> Predicted Semantic Class
                outputs_cpu = outputs[idx0].detach().cpu().numpy()  # [B, C, H, W] -> [C, H, W]
                semantics_pred = np.argmax(outputs_cpu, axis=0)     # [H, W]
                
                # -> GT class
                semantics_gt = labels[idx0].detach().cpu().numpy()  # [H, W]
                
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

                # Define optional
                optional_builders = {}
                
                if self.loss_name == "Dirichlet":
                    alpha_src = self._alpha_cache
                    alpha_dev = alpha_src if ((want_cuda_viz_calc and alpha_src.is_cuda) or ((not want_cuda_viz_calc) and (not alpha_src.is_cuda))) \
                        else alpha_src.to("cuda" if want_cuda_viz_calc else "cpu", non_blocking=True)

                    # build dict only for enabled names; lambdas are lazy and called only when drawn
                    optional_builders = {
                        n: (lambda name=n: build_uncertainty_layers(alpha_dev, [name], idx=idx0)[name])
                        for n in self.viz_optional_names
                    }
                elif self.loss_name=="SalsaNextAdf":
                    # Extract variance of predicted class
                    logits_var = logits_var[idx0].detach().cpu()    # [B, 1, H, W] -> [1, H, W]
                    pred_var = logits_var.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B, H, W]    # TODO: check if working

                    # Option: convert variance -> std
                    pred_std = torch.sqrt(pred_var + 1e-8)
                    # Normalize per image (0..255 for heatmap)
                    unc_map = pred_std.numpy()
                    unc_map = (unc_map - unc_map.min()) / (unc_map.max() - unc_map.min() + 1e-8)
                    unc_map = (unc_map * 255).astype(np.uint8)

                    # Apply colormap (e.g. JET or TURBO)
                    unc_colormap = cv2.applyColorMap(unc_map, cv2.COLORMAP_TURBO)
                
                    optional_builders = {
                        n: (lambda name=n: unc_colormap)
                        for n in self.viz_optional_names
                    }
    
                create_ia_plots(
                    base_images_dict=base_img_sources,
                    optional_builders=optional_builders,
                    args_o3d=(xyz_img, prev_sem_pred),
                    save_dir="",
                    enable=self.visualize
                )
                
                #print(H_norm.min(), H_norm.max(), H_norm.median())
        avg_loss = total_loss / max(1, len(loader))
        print(f"[train] epoch {epoch+1}/{self.num_epochs}, loss={avg_loss:.4f}")
        if self.logging and self.writer:
            self.writer.add_scalar("train/loss/epoch", avg_loss, epoch)
            if self.loss_name == "Dirichlet":
                H_norm_epoch = get_predictive_entropy_norm.mean(reset=True)
                self.writer.add_scalar('train/unc_tot/epoch', H_norm_epoch, self.global_step)
                #self.balancer.end_epoch(epoch)

    # ------------------------------
    # evaluation
    # ------------------------------
    @torch.no_grad()
    def test_one_epoch(self, loader, epoch: int):
        inference_times = []
        self.model.eval()
        self.evaluator.reset()

        use_dropout = bool(self.cfg["model_settings"].get("use_dropout", 0))
        want_dirichlet_metrics = (self.loss_name == "Dirichlet")

        ts_cache_mode = "mc" if use_dropout else "default"
        mc_T = int(self.cfg.get("calibration", {}).get("mc_samples", 30))

        metrics_cfg = self.cfg.get("logging_settings", {}).get("metrics", {})
        do_mc_conf_acc = bool(metrics_cfg.get("McConfEmpAcc", 0))
        do_entropy_iou = bool(metrics_cfg.get("IouPlt", 0))
        do_entropy_rel = bool(metrics_cfg.get("EntErrRel", 0))
        if (do_mc_conf_acc or do_entropy_iou or do_entropy_rel) and not want_dirichlet_metrics and not use_dropout:
            do_mc_conf_acc = do_entropy_iou = do_entropy_rel = False

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

            self._start_timer()

            if use_dropout:
                if self._model_act_kind is None:
                    outputs = self.model(*inputs)
                    if isinstance(outputs, tuple) and len(outputs) == 2 and self.baseline == "SalsaNextAdf":
                        outputs = outputs[0]
                    self._model_act_kind = _classify_output_kind(outputs, class_dim=1)
                    del outputs

                if self.T_value is not None and ts_cache_mode == "default":
                    T_val = max(1e-3, float(self.T_value))
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=T_val)
                    probs = mc_probs.mean(dim=0)
                else:
                    mc_probs = mc_dropout_probs(self.model, inputs, T=mc_T, temperature=None)
                    mean_p = mc_probs.mean(dim=0).clamp_min(1e-12)
                    if self.T_value is not None and ts_cache_mode == "mc":
                        T_val = max(1e-3, float(self.T_value))
                        probs = torch.softmax(torch.log(mean_p) / T_val, dim=1)
                    else:
                        probs = mean_p

                preds = probs.argmax(dim=1)
                self.evaluator.update(preds, labels)
                inference_times.append(self._stop_timer_ms())

                if (do_entropy_iou or do_entropy_rel) and self.logging:
                    ent_mc = predictive_entropy_mc(mc_probs, normalize=True)
                    err_mask = (preds != labels).to(torch.int32)
                    if do_entropy_iou:
                        ious = compute_entropy_error_iou(ent_mc, err_mask, thresholds=thresholds)
                        all_ious.append(ious.cpu().numpy())
                    if do_entropy_rel:
                        tot, err, err_rate, ece = compute_entropy_reliability(ent_mc, err_mask, n_bins=n_bins)
                        ent_err_tot += tot; ent_err_err += err
            else:
                outputs = self.model(*inputs)
                if isinstance(outputs, tuple) and len(outputs) == 2 and self.baseline == "SalsaNextAdf":
                    outputs = outputs[0]
                if self._model_act_kind is None:
                    self._model_act_kind = _classify_output_kind(outputs, class_dim=1)

                logits_raw = _to_logits(outputs, self._model_act_kind)
                logits = logits_raw if self.T_value is None else (logits_raw / max(1e-3, float(self.T_value)))
                probs = F.softmax(logits, dim=1)

                preds = probs.argmax(dim=1)
                self.evaluator.update(preds, labels)
                inference_times.append(self._stop_timer_ms())

                if want_dirichlet_metrics:
                    # alpha computed ONCE for all metrics
                    alpha = to_alpha_concentrations(logits_raw)
                    pred_entropy_norm = get_predictive_entropy_norm(alpha)
                    
                    self.ua_agg.update(
                        labels=labels,
                        preds=preds,
                        uncertainty=pred_entropy_norm,
                        ignore_ids=(0,),                       # ignore unlabeled if you use 0
                    )
                    # TODO: check test metrics
                    # if do_mc_conf_acc:
                    #     hits, tot = compute_mc_reliability_bins(alpha, labels, n_bins=n_bins, n_samples=120)
                    #     mc_hits += hits; mc_tot += tot
                    # if do_entropy_iou:
                    #     err_mask = (preds != labels).to(torch.int32)
                    #     ious = compute_entropy_error_iou(pred_entropy_norm, err_mask, thresholds=thresholds)
                    #     all_ious.append(ious.cpu().numpy())
                    # if do_entropy_rel:
                    #     err_mask = (preds != labels).to(torch.int32)
                    #     tot, err, err_rate, ece = compute_entropy_reliability(pred_entropy_norm, err_mask, n_bins=n_bins)
                    #     ent_err_tot += tot; ent_err_err += err

        # metrics
        mIoU, result_dict = self.evaluator.compute_final_metrics(class_names=self.class_names)
        print(f"[eval] epoch {epoch + 1}/{self.num_epochs},  mIoU={mIoU:.4f}")

        # logs/plots
        if self.logging and self.save_path:
            out_dir = os.path.join(self.save_path, "eval"); os.makedirs(out_dir, exist_ok=True)
            for cls in range(self.num_classes):
                self.writer.add_scalar('IoU_{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
            self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)

            if use_dropout and (do_entropy_iou or do_entropy_rel):
                pass  # already saved below as needed

            if (not use_dropout) and want_dirichlet_metrics:
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_edges=np.linspace(0.0, 1.0, 11),  # 10 bins
                    title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:06d}.png")
                )
                self.ua_agg.reset()
                
                # TODO: check test metrics
                # if do_mc_conf_acc and mc_tot.sum() > 0:
                #     emp_acc = np.divide(mc_hits, mc_tot, out=np.zeros_like(mc_hits), where=mc_tot > 0)
                #     edges = np.linspace(0.0, 1.0, n_bins + 1)
                #     centers = 0.5 * (edges[:-1] + edges[1:])
                #     save_reliability_diagram(
                #         empirical_acc=emp_acc,
                #         bin_centers=centers,
                #         tot_counts=mc_tot,
                #         output_path=os.path.join(out_dir, f"reliability_epoch_{epoch:06d}.png"),
                #         title='Reliability diagram\n(dot area ∝ #pixels per bin — sharpness)',
                #         xlabel='Predicted confidence (MC estimate)',
                #         ylabel='Empirical accuracy',
                #     )
                # if do_entropy_iou and len(all_ious) > 0:
                #     all_ious_np = np.stack(all_ious, axis=0)
                #     mean_ious = all_ious_np.mean(axis=0)
                #     plot_mIOU_errorEntropy(
                #         mean_ious,
                #         thresholds,
                #         output_path=os.path.join(out_dir, f"entropy_iou_epoch_{epoch:06d}.png"),
                #     )
                # if do_entropy_rel and ent_err_tot.sum() > 0:
                #     edges = np.linspace(0.0, 1.0, n_bins + 1)
                #     centers = 0.5 * (edges[:-1] + edges[1:])
                #     emp_err_rate = np.divide(ent_err_err, ent_err_tot, out=np.zeros_like(ent_err_err), where=ent_err_tot > 0)
                #     save_reliability_diagram(
                #         empirical_acc=emp_err_rate,
                #         bin_centers=centers,
                #         tot_counts=ent_err_tot,
                #         output_path=os.path.join(out_dir, f"entropy_error_rel_epoch_{epoch:06d}.png"),
                #         title='Reliability of Entropy as Error Predictor\n(dot area ∝ #pixels per entropy bin — sharpness)',
                #         xlabel='Normalized entropy',
                #         ylabel='Observed error rate',
                #     )

        return mIoU

    # ------------------------------
    # main loop
    # ------------------------------
    def __call__(self, train_loader, val_loader):
        self.num_epochs = int(self.cfg["train_params"]["num_epochs"]) + int(self.cfg["train_params"].get("num_warmup_epochs", 0))
        test_every = int(self.cfg["logging_settings"]["test_every_nth_epoch"])

        # TS config
        cal_cfg = self.cfg.get("calibration", {})
        ts_enable = bool(cal_cfg.get("enable", False))
        ts_run_each_eval = bool(cal_cfg.get("run_each_eval", False))
        ts_cache_mode = "mc" if self.cfg['model_settings'].get('use_dropout') else "default"
        ts_optimizer = cal_cfg.get("optimizer", "lbfgs")
        ts_lr = float(cal_cfg.get("lr", 0.05))
        ts_epochs = int(cal_cfg.get("epochs", 2))
        ts_chunk = int(cal_cfg.get("chunk_size", 1_000_000))
        ts_max_iter = int(cal_cfg.get("max_iter", 100))
        ts_save_json = cal_cfg.get("save_path", None)

        best_mIoU = -1.0
        for epoch in range(self.num_epochs):
            self.train_one_epoch(train_loader, epoch)

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            if epoch % max(1, test_every) == 0:
                mIoU = self.test_one_epoch(val_loader, epoch)

                if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(mIoU)

                if self.logging and self.save_path and mIoU > best_mIoU:
                    best_mIoU = mIoU
                    ckpt_path = os.path.join(self.save_path, f"best_epoch_{epoch:03d}.pt")
                    torch.save(self.model.state_dict(), ckpt_path)

                if ts_enable and (self.T_value is None or ts_run_each_eval):
                    logits_cpu, labels_cpu = cache_calib_logits(
                        model=self.model,
                        val_loader=val_loader,
                        device=self.device,
                        cfg=self.cfg,
                        ignore_index=255,
                        mode=ts_cache_mode,
                        mc_samples=int(cal_cfg.get("mc_samples", 30)),
                    )
                    self.T_value = calibrate_temperature_from_cache(
                        logits_cpu=logits_cpu,
                        labels_cpu=labels_cpu,
                        device=self.device,
                        init_T="auto",
                        optimizer_type=ts_optimizer,
                        lr=ts_lr,
                        epochs=ts_epochs,
                        chunk_size=ts_chunk,
                        max_iter_lbfgs=ts_max_iter,
                        prev_T=self.T_value,
                        save_path=ts_save_json,
                    )
                    print(f"[TS] calibrated T = {self.T_value:.4f}")
                    self.test_one_epoch(val_loader, epoch)

        # final eval & save
        self.test_one_epoch(val_loader, self.num_epochs - 1)
        if self.logging and self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))

