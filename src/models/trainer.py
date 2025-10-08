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
    IoUEvaluator,
    UncertaintyAccuracyAggregator
)

from models.temp_scaling import (
    cache_calib_logits,
    calibrate_temperature_from_cache,
)
from utils.mc_dropout import (
    set_dropout_mode,
    mc_forward,
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
    predictive_entropy_from_logistic_normal,
    # Global parameter getter
    get_eps_value,
    get_alpha_temperature,
    # Metrics
    compute_entropy_error_iou,
    plot_mIOU_errorEntropy,
    compute_entropy_reliability,
    compute_mc_reliability_bins,
    save_reliability_diagram
)

from baselines.SalsaNext import adf as adf # TODO: testing original version
from baselines.SalsaNext.SalsaNextAdf_utils import (
    adf_mc_passes, 
    salsanext_uncertainties_from_mc
)

from typing import Dict, Callable, Optional, Tuple, Iterable, Union
import numpy as np
from utils.viz_panel import (
    create_ia_plots,
    register_optional_names
)


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


@torch.no_grad()
def _safe_l2(x: torch.Tensor) -> float:
    if x is None or x.numel() == 0:
        return 0.0
    v = torch.linalg.vector_norm(x.float(), ord=2)  # scalar tensor
    if not torch.isfinite(v).item():
        return 0.0
    return v.item()

def grad_norm_of(
    loss: torch.Tensor,
    wrt: Union[torch.Tensor, Iterable[torch.Tensor]],
    *,
    retain_graph: bool = True,
) -> float:
    """
    L2 norm of d(loss)/d(wrt). Returns 0.0 if grads are unavailable.
    - loss can be non-scalar; it will be mean-reduced.
    - wrt can be a single tensor or an iterable of tensors.
    """
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return 0.0
    if loss.ndim > 0:
        loss = loss.mean()

    if isinstance(wrt, (list, tuple)):
        wrt_list = [t for t in wrt if isinstance(t, torch.Tensor) and t.requires_grad]
    else:
        wrt_list = [wrt] if isinstance(wrt, torch.Tensor) and wrt.requires_grad else []

    if not wrt_list:
        return 0.0

    grads = torch.autograd.grad(
        loss, wrt_list,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    # global L2 across all grads: sqrt(sum_i ||g_i||^2)
    sq_sum = 0.0
    for g in grads:
        sq_sum += _safe_l2(g.detach()) ** 2
    return math.sqrt(sq_sum)


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

        # data / task meta
        self.num_classes = int(cfg["extras"]["num_classes"])
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        self.global_step = 0

        # loss selection & baseline
        self.loss_name = cfg["model_settings"]["loss_function"]
        self.baseline = cfg["model_settings"]["baseline"]

        # evaluator
        self.iou_evaluator = IoUEvaluator(self.num_classes)
        if test_mask is None:
            self.test_mask = [1] * cfg["extras"]["num_classes"]
            self.test_mask[0] = 0

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

        # ignore index for training, e.g. unlabeled class; else None
        self.ignore_idx: int| None = 0
        
        # losses
        self._init_losses()
        
        #declare which optional layer names exist for the active loss
        if self.visualize:
            if self.loss_name == "Dirichlet":
                self.viz_optional_names = [
                    "H_norm", "AU_norm", "EU_norm",
                    "alpha0", "AU_frac", "EU_frac", "EU_minus_AU_frac",
                ]
            elif self.loss_name == "SalsaNextAdf" or self.baseline == "SalsaNextAdf":
                self.viz_optional_names = ["std_hat", "H_norm"]
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
            DirichletCriterion,
            SoftmaxHeteroscedasticLoss
            #DirichletLoss
        )
        
        def load_loss_weights(cfg: dict, loss_name: str, defaults: dict) -> dict:
            """
            Minimal: defaults -> model_weights.default -> model_weights[loss_name]
            """
            w = dict(defaults)
            mw = (cfg.get("model_weights") or {})

            def apply(section):
                if not isinstance(section, dict): return
                for k, v in section.items():
                    if k in w:
                        try: w[k] = max(0.0, float(v))
                        except: pass  # ignore bad values

            apply(mw.get("default"))
            apply(mw.get(loss_name))  # e.g. "Dirichlet"
            return w

        if self.loss_name == "Tversky":
            self.criterion_ce = CrossEntropyLoss(ignore_index=self.ignore_idx)
            self.criterion_tversky = TverskyLoss(ignore_index=self.ignore_idx)
            
            # loss weights
            defaults = dict(w_ce=1.0, w_tversky=1.0)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)
            self.w_ce, self.w_tversky = w["w_ce"], w["w_tversky"]
        elif self.loss_name == "CE":
            self.criterion_ce = CrossEntropyLoss(ignore_index=self.ignore_idx)
        elif self.loss_name == "Lovasz":
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
        elif self.loss_name == "Dirichlet":
            # def s_for_entropy_floor(C, h_norm, tol=1e-8):
            #     import math
            #     def H_norm(s):
            #         if s<=0: return 0.0
            #         return (-(1-s)*math.log(1-s) - s*math.log(s/(C-1))) / math.log(C)
            #     # binary search on s in [0, 1 - 1/C)
            #     lo, hi = 0.0, 1.0 - 1.0/C
            #     for _ in range(80):
            #         mid = (lo+hi)/2
            #         if H_norm(mid) < h_norm: lo = mid
            #         else: hi = mid
            #     return (lo+hi)/2
            # set target smoothing value for minimum desired enryopy [0,1]
            self.nll_smoothing_start = 0.25 # s_for_entropy_floor(self.num_classes, 0.05)
            self. criterion_dirichlet = DirichletCriterion(
                num_classes=self.num_classes,
                ignore_index=self.ignore_idx,
                eps=get_eps_value(),
                prior_concentration=1.5*self.num_classes,
                p_moment=2.0,
                smoothing=self.nll_smoothing_start,
                kl_mode="evidence", # "evidence" keeps mean p_hat, pins alpha0 to your target so certainty stays calibrated; "symmetric" pulls toward uniform
                nll_mode="dircat",   # "density" | "dircat" (stabilizes class ranking, scale-invariant)
                comp_gamma = 2.0,
                comp_tau   = 0.75,    # if you still see confident mistakes, try 0.75 later
                comp_sigma = 0.10,      # bounded to [0.06, 0.12]
                comp_normalize = True
            )

            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
            
            self.criterion_nll_temp = torch.nn.NLLLoss()
            
            # loss weights
            defaults = dict(w_nll=0.0, 
                            w_imax=3.0, 
                            w_dce=1.0, 
                            w_ls=2.5, 
                            w_kl=0.5, 
                            w_ir=0.3,
                            w_comp=0.2)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)

            self.w_nll, self.w_imax, self.w_dce = w["w_nll"], w["w_imax"], w["w_dce"]
            self.w_ls,  self.w_kl,   self.w_ir  = w["w_ls"],  w["w_kl"],  w["w_ir"]
            self.w_comp = w["w_comp"]
            # lovasz baseline gn_fractions DCE ≈ 0.6–1.0×, iMAX ≈ 0.3–0.7×, KL ≈ 0.3–0.6× (more if alpha0 grows), IR ≈ 0.2–0.4×, nll≈0.9x not above 1.0x
        elif self.loss_name == "SalsaNext":
            self.criterion_nll = torch.nn.NLLLoss()#CrossEntropyLoss(ignore_index=self.ignore_idx)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
            
            # loss weights
            defaults = dict(w_nll=1.0, w_ls=1.0)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)
            self.w_nll, self.w_ls = w["w_nll"], w["w_ls"]
        elif self.loss_name == "SalsaNextAdf":
            self.criterion_nll = torch.nn.NLLLoss()#(ignore_index=self.ignore_idx)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
            self.het = SoftmaxHeteroscedasticLoss(num_classes=self.num_classes, ignore_index=self.ignore_idx)
            self.w_het = 1
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
        self.iou_evaluator.reset()

        # Dirichlet-specific schedules
        if self.loss_name == "Dirichlet":
            get_predictive_entropy_norm.reset()
            # gentle late late decay
            #self.criterion_dirichlet.smoothing = smoothing_schedule(epoch, self.num_epochs, s0=self.nll_smoothing_start, s_min=0.15, start_frac=0.25, end_frac=0.8)
            # self.balancer.begin_epoch(epoch)

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"train {epoch+1}")
        ):
            self.global_step = epoch * len(loader) + step
            # define logging step
            will_log = bool(self.logging and self.writer and (step % 10 == 0))

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
                output_SalsaNextAdf = outputs
                sm = adf.Softmax(dim=1, keep_variance_fn=adf.keep_variance_fn)
                logits_mean, logits_var = sm(*outputs)                       # probs + their var
                logits_mean = logits_mean.clamp(min=1e-7, max=1.0 - 1e-7)
                logits_var  = torch.clamp(logits_var, min=1e-6, max=10.0)    # just in case

                #logits_mean, logits_var = adf.Softmax(dim=1, keep_variance_fn=lambda x: x+1e-3)(*outputs)   # TODO: was adf.keep_variance_fn
                outputs = logits_mean   # TODO: maybe we must clone here
            assert not (isinstance(outputs, tuple) and len(outputs) > 2), "Unexpected model outputs"

            # Decide output kind once (first batch)
            if self._model_act_kind is None:
                self._model_act_kind = _classify_output_kind(outputs, class_dim=1)

            # ---- compute loss (single-pass α caching for Dirichlet) ----
            if self.loss_name == "Tversky":
                loss_sem = self.criterion_ce(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss_t = self.criterion_tversky(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                loss = self.w_ce * loss_sem + self.w_tversky * loss_t
                
            elif self.loss_name == "CE":
                loss = self.criterion_ce(outputs, labels, num_classes=self.num_classes, model_act=self._model_act_kind)
                
            elif self.loss_name == "Lovasz":
                loss = self.criterion_lovasz(outputs, labels, model_act=self._model_act_kind)
                
            elif self.loss_name == "SalsaNext":
                # add softmax layer in line with SalsaNext code, as we removed it inside the class (baselines/SalsaNext/SalsaNext.py):
                    # https://github.com/TiagoCortinhal/SalsaNext/blob/7548c124b48f0259cdc40e98dfc3aeeadca6070c/train/tasks/semantic/modules/SalsaNext.py#L213
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                # SalsaNext loss function as in:
                    # https://github.com/TiagoCortinhal/SalsaNext/blob/7548c124b48f0259cdc40e98dfc3aeeadca6070c/train/tasks/semantic/modules/trainer.py#L379
                loss_nll = self.criterion_nll(torch.log(outputs.clamp(min=1e-8)), labels)#, num_classes=self.num_classes, model_act="log_probs")
                loss_ls = self.criterion_lovasz(outputs, labels, model_act="probs")
                loss = self.w_nll * loss_nll + self.w_ls * loss_ls
                
            elif self.loss_name == "SalsaNextAdf":
                # CE on mean logits
                loss_nll = self.criterion_nll(torch.log(logits_mean.clamp(min=1e-8)), labels)#, num_classes=self.num_classes, model_act="probs")
                # Lovasz on mean logits
                loss_ls = self.criterion_lovasz(logits_mean, labels, model_act="probs") # TODO: check if it's probs or logits 
                # Heteroscedastic
                loss_het = self.het(output_SalsaNextAdf, labels)
                loss = loss_nll + loss_ls + self.w_het * loss_het
                
            elif self.loss_name == "Dirichlet":
                # alpa computed ONCE here and reused everywhere below
                # For stability, convert to *logits-like* first and then alpha
                #logits_like = _to_logits(outputs, self._model_act_kind)
                alpha = to_alpha_concentrations(outputs)
                
                # get dirichlet losses
                L_dir_dict={}
                if step % 2 ==0:    # every 2nd iteration update ema class weight accumulator
                    self.criterion_dirichlet.update_class_weights(labels, method="effective_num", beta=0.999)   # access with self.criterion_dirichlet.class_weights  
                if True: #epoch >=2:   # TODO: hard coded warm-start num epochs
                    loss_dirichlet, L_dir_dict = self.criterion_dirichlet(
                        alpha, labels,
                        w_nll=self.w_nll,
                        w_dce=self.w_dce,
                        w_imax=self.w_imax,
                        w_ir=self.w_ir,
                        w_kl=self.w_kl,        # tune to keep alpha0 in range
                        w_comp=self.w_comp
                    )
            
                # Lovasz on either Dirichlet mean 
                    # alpha0 = alpha.sum(dim=1, keepdim=True) + get_eps_value()
                    # p_hat = alpha / alpha0
                # or softmax
                    # alpha0 treated as constant -> It stops Lovasz gradients from shrinking as alpha0 inflates (without the detach they scale like ~1/alpha0
                    # should shape class proportions only; letting it backprop through alpha0 makes it push the evidence scale (can collapse or inflate alpha0)
                #.detach() # NOTE: detach here is wanted. # debugging: print(p_hat[0,:,32,32],"\n", alpha[0,:,32,32],"\n", alpha0[0,:,32,32]) 
                loss_ls = self.criterion_lovasz(F.softmax(outputs, dim=1), labels.long(), model_act="probs")
                
                if False: #epoch < 2: # first two epoch geometry warm-start # TODO: hard coded warm-start num epochs
                    from models.losses import nll_dirichlet_categorical, dce_from_alpha
                    #loss_dircat = nll_dirichlet_categorical(alpha, labels, self.ignore_idx)
                    loss_dce = dce_from_alpha(alpha, labels, self.ignore_idx)
                    loss = self.w_ls * loss_ls + loss_dce #self.criterion_nll_temp(torch.log(p_hat), labels))
                else:
                    loss = loss_dirichlet + (self.w_ls * loss_ls)
                
                # ----------------- PRE-BACKWARD: compute grad norms -----------------
                def _to_float(x):
                    if isinstance(x, (float, int)): return float(x)
                    if torch.is_tensor(x): return float(x.detach().cpu().item())
                    return float(x)
                
                def _safe_ratio(num: float, denom: float, eps: float = 1e-12) -> float:
                    return float(num) / (float(denom) + eps)

                gn_terms_f = {}   # name -> float grad norm
                w_map = {"dce": self.w_dce, "nll": self.w_nll, "imax": self.w_imax, "ir": self.w_ir, "kl": self.w_kl, "comp": self.w_comp}

                if will_log:
                    # Compute per-term grad norms w.r.t. logits (outputs). Use retain_graph=True since we will still call backward().
                    for name, val in L_dir_dict.items():
                        w = w_map.get(name, 0.0)
                        if w != 0.0 and torch.is_tensor(val) and val.requires_grad:
                            gn = grad_norm_of(w * val, outputs, retain_graph=True)
                            gn_terms_f[name] = _to_float(gn)
                        else:
                            gn_terms_f[name] = 0.0

                    gn_ls_f = _to_float(grad_norm_of(self.w_ls * loss_ls, outputs, retain_graph=True))
                    
                    # Ratios of each term's grad pressure vs Lovasz
                    rat_vs_ls={}
                    # rat_vs_ls = {
                    #     "dce": _safe_ratio(gn_terms_f.get("dce", 0.0), gn_ls_f),
                    #     "imax": _safe_ratio(gn_terms_f.get("imax", 0.0), gn_ls_f),
                    #     "kl":  _safe_ratio(gn_terms_f.get("kl",  0.0), gn_ls_f),
                    #     "ir":  _safe_ratio(gn_terms_f.get("ir",  0.0), gn_ls_f),
                    #     "nll": _safe_ratio(gn_terms_f.get("nll", 0.0), gn_ls_f),
                    # }
                else:
                    gn_ls_f = 0.0

                self._alpha_cache = alpha.detach()  # keep a clean, detached copy for viz (AFTER loss is computed!)
                
            else:
                raise RuntimeError("unreachable")

            # @@@ After loss calculaltion @@@
            
            # Guard 1: skip non-finite loss (don't backprop zeros or 1e6!)
            # if not torch.isfinite(loss):
            #     print("Non-finite loss - skipping step")
            #     self.optimizer.zero_grad(set_to_none=True)
            #     continue
            
            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Guard 2: Clip + sanitize grads; skip step if any grad is non-finite. Used for SalsaNextAdf
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # bad = False
            # for p in self.model.parameters():
            #     if p.grad is not None and not torch.isfinite(p.grad).all():
            #         bad = True
            #         break
            # if bad:
            #     print("Non-finite grads — skipping step")
            #     self.optimizer.zero_grad(set_to_none=True)
            #     continue
    
            self.optimizer.step()
            total_loss += float(loss.item())

            # ----------------- POST-STEP: logging -----------------
            elapsed_ms = self._stop_timer_ms()
            
            # IoU handler accumulate
            #log_probs = torch.log_softmax(outputs, dim=1)
            #probs = log_probs.exp() # [B,C,H,W]
            preds = outputs.argmax(dim=1)
            self.iou_evaluator.update(preds, labels)
            
            # Dirichlet running uncertainty stat (reuse cached alpha)
            if self.loss_name == "Dirichlet" and self._alpha_cache is not None:
                H_norm = get_predictive_entropy_norm.accumulate(self._alpha_cache.cpu())
            elif self.loss_name=="SalsaNextAdf" and self.baseline=="SalsaNextAdf":
                H_pred, H_norm = predictive_entropy_from_logistic_normal(*output_SalsaNextAdf)
                
            # Logging (every 10 steps)
            if will_log:
                self.writer.add_scalar("loss/iter", loss.item(), self.global_step)
                self.writer.add_scalar("LR/iter", self.optimizer.param_groups[0]['lr'], self.global_step)
                if self.loss_name == "Dirichlet":
                    # weighted loss terms that exist
                    for name, val in L_dir_dict.items():
                        w = w_map.get(name, 0.0)
                        if w != 0.0:
                            self.writer.add_scalar(f"loss/{name}", _to_float(val) * w, self.global_step)

                    # Lovasz
                    self.writer.add_scalar("loss/ls", _to_float(loss_ls) * self.w_ls, self.global_step)

                    # grad norms (we cached floats pre-backward)
                    for name, g in gn_terms_f.items():
                        self.writer.add_scalar(f"grad_norm/logits/{name}", g, self.global_step)
                    self.writer.add_scalar("grad_norm/logits/ls", gn_ls_f, self.global_step)
                    
                    if rat_vs_ls:
                        for k, v in rat_vs_ls.items():
                            self.writer.add_scalar(f"grad_norm/ratio_vs_ls/{k}", v, self.global_step)

                    # effective pressure — prefer RSS over sum
                    rss = (sum(g*g for g in gn_terms_f.values()) + gn_ls_f*gn_ls_f) ** 0.5
                    self.writer.add_scalar("grad_norm/logits/rss_total", rss, self.global_step)
                    
                    # alpha0 stats (use cached alpha)
                    with torch.no_grad():
                        a0_hw = self._alpha_cache.sum(dim=1)              # [B,H,W]
                        med_alpha0 = a0_hw.median().item()
                        med_alpha0_per_cls = (a0_hw / float(alpha.shape[1])).median().item()
                    self.writer.add_scalar("alpha0/median", med_alpha0, self.global_step)
                    self.writer.add_scalar("alpha0/median_per_class", med_alpha0_per_cls, self.global_step)

                    # H_norm coverage
                    self.writer.add_scalar("H_norm/pct_lt_0.1",  (H_norm < 0.10).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_lt_0.25", (H_norm < 0.25).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.5",  (H_norm > 0.50).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.75", (H_norm > 0.75).float().mean().item(), self.global_step)
                elif self.loss_name=="SalsaNext" and self.baseline=="SalsaNext":
                    self.writer.add_scalar('loss/loss_nll', loss_nll.item(), self.global_step)
                    self.writer.add_scalar('loss/loss_ls', loss_ls.item(), self.global_step)
                elif self.loss_name=="SalsaNextAdf" and self.baseline=="SalsaNextAdf":
                    self.writer.add_scalar('loss/loss_nll', loss_nll.item(), self.global_step)
                    self.writer.add_scalar('loss/loss_ls', loss_ls.item(), self.global_step)
                    self.writer.add_scalar('loss/loss_hetero', loss_het.item() * self.w_het, self.global_step)
                    
                    # entropy masses less than threshold
                    self.writer.add_scalar("H_norm/pct_lt_0.1", (H_norm < 0.1).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_lt_0.25", (H_norm < 0.25).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.5", (H_norm > 0.5).float().mean().item(), self.global_step)
                    self.writer.add_scalar("H_norm/pct_gt_0.75", (H_norm > 0.75).float().mean().item(), self.global_step)
            # Interactive visualization (cheap, reusing computed items)
            if self.visualize:
                idx0 = 0
                want_cuda_viz_calc = True
                
                # -> GT class
                semantics_gt = labels[idx0].detach().cpu().numpy()  # [H, W]
                if self.ignore_idx is not None:
                    mask = np.argwhere(semantics_gt==self.ignore_idx)
                else:
                    mask = None
                
                # -> Predicted Semantic Class
                outputs_cpu = outputs[idx0].detach().cpu().numpy()  # [B, C, H, W] -> [C, H, W]
                semantics_pred = np.argmax(outputs_cpu, axis=0)     # [H, W]
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

                # Define optional
                optional_builders = {}
                
                if self.loss_name == "Dirichlet":
                    alpha_src = self._alpha_cache
                    
                    alpha_dev = alpha_src if ((want_cuda_viz_calc and alpha_src.is_cuda) or ((not want_cuda_viz_calc) and (not alpha_src.is_cuda))) \
                        else alpha_src.to("cuda" if want_cuda_viz_calc else "cpu", non_blocking=True)
                        
                    # build dict only for enabled names; lambdas are lazy and called only when drawn
                    optional_builders = {
                        n: (lambda name=n: build_uncertainty_layers(alpha_dev, [name], idx=idx0, mask=mask)[name])
                        for n in self.viz_optional_names
                    }
                elif self.loss_name=="SalsaNextAdf" and self.baseline=="SalsaNextAdf":
                    # Extract variance of predicted class
                    logits_var = logits_var[idx0].detach().cpu()    # [B, C, H, W] -> [C, H, W]
                    pred_var = logits_var.gather(0, torch.tensor(semantics_gt, dtype=int).unsqueeze(0)).squeeze(0)  # [H,W]

                    # convert variance -> std
                    std_hat = torch.sqrt(pred_var + get_eps_value()).numpy()
                    # Normalize per image (0..255 for heatmap)
                    std_hat_map = (std_hat - std_hat.min()) / (std_hat.max() - std_hat.min() + 1e-8)
                    std_hat_map = (std_hat_map * 255).astype(np.uint8)

                    # Apply colormap (e.g. JET or TURBO)
                    std_hat_map_colormap = cv2.applyColorMap(std_hat_map, cv2.COLORMAP_TURBO)
                    if self.ignore_idx is not None:
                        std_hat_map_colormap[mask[:, 0], mask[:, 1]]=[0,0,0]
                    
                    # get H_norm from logits_mean + logits_var + MC Sampling
                    H_norm_map = (H_norm[idx0].detach().cpu().numpy() * 255).astype(np.uint8)
                    H_norm_colormap = cv2.applyColorMap(H_norm_map, cv2.COLORMAP_TURBO)
                    if self.ignore_idx is not None:
                        H_norm_colormap[mask[:, 0], mask[:, 1]]=[0,0,0]
                    
                    images = [std_hat_map_colormap, H_norm_colormap]
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
                
                #print(H_norm.min(), H_norm.max(), H_norm.median())
        
        # @@@ END of Epoch
        
        mIoU, result_dict = self.iou_evaluator.compute(
            class_names=self.class_names,
            test_mask=self.test_mask,
            ignore_gt=[self.ignore_idx],
            reduce="mean",
            ignore_th=None
        )
        print(f"[train] epoch {epoch + 1}/{self.num_epochs}, mIoU={mIoU:.4f}")
        avg_loss = total_loss / max(1, len(loader))
        print(f"[train] epoch {epoch+1}/{self.num_epochs}, loss={avg_loss:.4f}")
        if self.logging and self.writer:
            # LR epoch
            self.writer.add_scalar("LR/epoch", self.optimizer.param_groups[0]['lr'], epoch)
            # loss epoch
            self.writer.add_scalar("loss/epoch", avg_loss, epoch)
            # IoU logging
            for cls in range(self.num_classes):
                self.writer.add_scalar('IoU_Train/{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            self.writer.add_scalar('mIoU_Train', mIoU*100, epoch)
            if self.loss_name == "Dirichlet":
                H_norm_epoch = get_predictive_entropy_norm.mean(reset=True)
                self.writer.add_scalar('unc_tot/epoch', H_norm_epoch, self.global_step)
                #self.balancer.end_epoch(epoch)

    # ------------------------------
    # evaluation
    # ------------------------------
    @torch.no_grad()
    def test_one_epoch(self, loader, epoch: int):
        self.model.eval()
        self.iou_evaluator.reset()
        inference_times = []

        use_mc_sampling = bool(self.cfg["model_settings"].get("use_mc_sampling", 0))
        mc_T = int(self.cfg["model_settings"].get("mc_samples", 30))

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"test {epoch+1}")
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

            # --- inference ---
            self._start_timer()
            # --- inside test_one_epoch loop, MC path ---
            if use_mc_sampling:
                mc_outputs = mc_forward(self.model, inputs, T=mc_T)     # [T,B,C,H,W]
                # debugging sanity check: print(mc_outputs.std().item(), mc_outputs.std(dim=0).mean().item())
                log_probs = F.log_softmax(mc_outputs, dim=2)    # Log-softmax for numerical stability
                probs = log_probs.exp()                             # get probs

                # Predictive distribution
                p_bar = probs.mean(dim=0)   # [B,C,H,W]

                # Predictive entropy: H[p_bar] = -Sum_c {p_bar}_c * log( {p_bar}_c )
                entropy = -(p_bar * (p_bar + get_eps_value()).log()).sum(dim=1)   # [B,(H,W)]
                H_norm = entropy/ math.log(self.num_classes)    # [B,(H,W)]
                
                preds = p_bar.argmax(dim=1) # [B,1,H,W]

                inference_times.append(self._stop_timer_ms())
                
                self.iou_evaluator.update(preds, labels)
                self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,))
    
            else: # single pass (no MC)
                outputs = self.model(*inputs)
                if isinstance(outputs, tuple) and len(outputs)==2 and self.baseline=="SalsaNextAdf":
                    output_SalsaNextAdf = outputs
                    sm = adf.Softmax(dim=1, keep_variance_fn=adf.keep_variance_fn)
                    logits_mean, logits_var = sm(*outputs)                       # probs + their var
                    logits_mean = logits_mean.clamp(min=1e-7, max=1.0 - 1e-7)
                    logits_var  = torch.clamp(logits_var, min=1e-6, max=10.0)    # just in case

                    #logits_mean, logits_var = adf.Softmax(dim=1, keep_variance_fn=lambda x: x+1e-3)(*outputs)   # TODO: was adf.keep_variance_fn
                    outputs = logits_mean 
                    probs = outputs
                if self._model_act_kind is None:
                    self._model_act_kind = _classify_output_kind(outputs, class_dim=1)
                    
                if not (self.baseline=="SalsaNextAdf" and self.loss_name=="SalsaNextAdf"):
                    logits = _to_logits(outputs, self._model_act_kind)  # [B,C,H,W]
                    log_probs = F.log_softmax(logits, dim=1)
                    probs = log_probs.exp() # [B,C,H,W]

                preds = probs.argmax(dim=1) # [B,1,H,W]
                inference_times.append(self._stop_timer_ms())
                
                self.iou_evaluator.update(preds, labels)
                
                if self.baseline=="SalsaNextAdf" and self.loss_name=="SalsaNextAdf":
                    H_pred, H_norm = predictive_entropy_from_logistic_normal(*output_SalsaNextAdf)
                if self.loss_name=="Dirichlet":
                    # alpha computed ONCE for all metrics
                    alpha = to_alpha_concentrations(logits)
                    H_norm = get_predictive_entropy_norm(alpha)
                
                if (self.baseline=="SalsaNextAdf" and self.loss_name=="SalsaNextAdf") or \
                        self.loss_name=="Dirichlet":
                    self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,)     # ignore unlabeled if ignore_ids==0
                    )
        
        # @@@ END of Epoch
        
        # metrics
        mIoU, result_dict = self.iou_evaluator.compute(
            class_names=self.class_names,
            test_mask=self.test_mask,
            ignore_gt=[self.ignore_idx],
            reduce="mean",
            ignore_th=None
        )
        print(f"[eval] epoch {epoch + 1}/{self.num_epochs},  mIoU={mIoU:.4f}")

        # logs/plots
        if self.logging and self.save_path:
            out_dir = os.path.join(self.save_path, "eval"); os.makedirs(out_dir, exist_ok=True)
            for cls in range(self.num_classes):
                self.writer.add_scalar('IoU_Test/{}'.format(self.class_names[cls]), result_dict[self.class_names[cls]]*100, epoch)
            self.writer.add_scalar('mIoU_Test', mIoU*100, epoch)
            self.writer.add_scalar('Inference Time', np.median(inference_times), epoch)
        
            if (not use_mc_sampling) and \
                    (self.loss_name=="Dirichlet" or (self.baseline=="SalsaNextAdf" and self.loss_name=="SalsaNextAdf")):
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_edges=np.linspace(0.0, 1.0, 11),  # 10 bins
                    title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:06d}.png")
                )
            elif use_mc_sampling:
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_edges=np.linspace(0.0, 1.0, 11),  # 10 bins
                    title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:06d}.png")
                )
        self.ua_agg.reset()

        return mIoU

    # ------------------------------
    # main loop
    # ------------------------------
    def __call__(self, train_loader, val_loader):
        self.num_epochs = int(self.cfg["train_params"]["num_epochs"]) + int(self.cfg["train_params"].get("num_warmup_epochs", 0))
        test_every = int(self.cfg["logging_settings"]["test_every_nth_epoch"])

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

        # final eval & save
        self.test_one_epoch(val_loader, self.num_epochs - 1)
        if self.logging and self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))

