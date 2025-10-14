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
    IoUEvaluator,
    UncertaintyAccuracyAggregator
)
from utils.mc_dropout import (
    mc_forward,
)
from utils.inputs import set_model_inputs
#from utils.loss_balancer import LossBalancer
from utils.vis_cv2 import (
    visualize_semantic_segmentation_cv2,
)

from models.probability_helper import (
    to_alpha_concentrations,
    get_predictive_entropy_norm,
    build_uncertainty_layers,
    # Global parameter getter
    get_eps_value,
    get_alpha_temperature,
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

def _to_float(x):
    if isinstance(x, (float, int)): return float(x)
    if torch.is_tensor(x): return float(x.detach().cpu().item())
    return float(x)

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

import math
import torch
from typing import Iterable, Union

def grad_norm_wrt(
    loss: torch.Tensor,
    wrt: Union[torch.Tensor, Iterable[torch.Tensor]],
    *,
    retain_graph: bool = False,
) -> float:
    # mean-reduce if needed
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return 0.0
    if loss.ndim > 0:
        loss = loss.mean()

    if isinstance(wrt, (list, tuple)):
        wrt_list = [p for p in wrt if isinstance(p, torch.Tensor) and p.requires_grad]
    else:
        wrt_list = [wrt] if isinstance(wrt, torch.Tensor) and wrt.requires_grad else []
    if not wrt_list:
        return 0.0

    grads = torch.autograd.grad(loss, wrt_list, retain_graph=retain_graph, allow_unused=True)
    sq = 0.0
    for g in grads:
        if g is None:
            continue
        v = torch.linalg.vector_norm(g.float(), ord=2)
        if torch.isfinite(v):
            sq += float(v.item()) ** 2
    return math.sqrt(sq)

# ---- helpers you already have ----
# grad_norm_wrt(loss, ref_params, retain_graph=True)

class ProportionalGradNorm:
    """
    Anchored GradNorm with target ratios.
    Keeps weights near your base priors but nudges them so that per-loss raw grad norms
    approach the desired proportions.
    """
    def __init__(self, targets: dict[str, float], base: dict[str, float],
                 ema_beta: float = 0.9, power: float = 0.5,
                 min_mult: float = 0.2, max_mult: float = 5.0):
        """
        targets: desired relative strengths, e.g. {'nll':1.0, 'ls':1.0, 'kl':0.3, ...}
        base:    your prior weights (what you set in cfg), same keys; base[k]=0 disables
        ema_beta: log-EMA smoothing for measured grad norms. incrase to 0.95-0.98 to reduce jitter.
        power:   update aggressiveness (0<k<=1). 0.5 is gentle, 1.0 aggressive. redice power to 0.3-0.4 for less jitter.
        min_mult/max_mult: clamp on multiplicative adjustments a_i around base
        """
        self.targets = targets.copy()
        self.base = base.copy()
        self.beta = ema_beta
        self.k = power
        self.min_mult = min_mult
        self.max_mult = max_mult

        self.ema = {k: 0.0 for k in targets}    # EMA of RAW grad norms
        self.mult = {k: 1.0 for k in targets}   # learned multiplicative adjustments

    def weights(self) -> dict[str, float]:
        return {k: (self.base[k] * self.mult.get(k, 1.0) if self.base.get(k, 0.0) > 0.0 else 0.0)
                for k in self.base}

    def step(self, raw_g: dict[str, float]):
        active = [k for k, v in self.base.items() if v > 0.0 and k in raw_g]
        if not active:
            return self.weights()

        # --- update EMA of RAW norms ---
        for k in active:
            g = max(0.0, float(raw_g[k]))
            self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * g

        # --- compute current EFFECTIVE norms E_i = w_i * EMA(raw G_i) ---
        w = {k: self.base[k] * self.mult[k] for k in active}
        eff = {k: w[k] * self.ema[k] for k in active}

        # normalize target ratios over ACTIVE keys
        mean_t = sum(self.targets[k] for k in active) / (len(active) + 1e-12)
        r = {k: self.targets[k] / (mean_t + 1e-12) for k in active}

        # mean effective norm
        mean_eff = sum(eff[k] for k in active) / (len(active) + 1e-12)

        # --- update multipliers to match EFFECTIVE targets ---
        # target E*_i = mean_eff * r_i
        for k in active:
            Ei = eff[k] + 1e-12
            Ei_star = mean_eff * r[k]
            scale = (Ei_star / Ei) ** self.k          # proportional correction
            self.mult[k] = float(self.mult[k] * scale)
            self.mult[k] = float(min(max(self.mult[k], self.min_mult), self.max_mult))

        # keep avg multiplier ≈ 1 to avoid drift
        avg_mult = sum(self.mult[k] for k in active) / (len(active) + 1e-12)
        for k in active:
            self.mult[k] /= (avg_mult + 1e-12)

        return self.weights()

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
        self.model_ref_params = list(self.model.parameters()) # or: list(model.head.parameters())

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
        if cfg["extras"].get("test_mask", 0):
            self.test_mask = list(cfg["extras"]["test_mask"].values())
        else:
            self.test_mask = [0] + [1] * (cfg["extras"]["num_classes"] - 1)

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
            DirichletCriterion
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
            self.nll_smoothing_start = 0.25
            self.criterion_dirichlet = DirichletCriterion(
                num_classes=self.num_classes,
                ignore_index=self.ignore_idx,
                eps=get_eps_value(),
                prior_concentration=2*self.num_classes,
                p_moment=1.0,
                smoothing=self.nll_smoothing_start,
                kl_mode="evidence", # "evidence" keeps mean p_hat, pins alpha0 to your target so certainty stays calibrated; "symmetric" pulls toward uniform
                nll_mode="dircat",   # "density" | "dircat" (stabilizes class ranking, acts on p_hat)
                comp_gamma = 2.0,
                comp_tau   = 0.75,    # if you still see confident mistakes, try 0.75 later
                comp_sigma = 0.10,      # bounded to [0.06, 0.12]
                comp_normalize = True
            )

            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
            
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
            # define prior/base weights (from cfg)
            self.base_weights = {
                "nll": self.w_nll, "ls": self.w_ls, "dce": self.w_dce,
                "imax": self.w_imax, "comp": self.w_comp,
                "ir": self.w_ir, "kl": self.w_kl    # should not be part of the loss weight balancer
            }
            
            # Which losses should be *balanced* by GradNorm (supervised/shape only)
            # Anything *not* in BALANCE_KEYS should be added to the loss with a fixed weight outside GradNorm.
            BALANCE_KEYS = ("nll", "ls", "imax", "comp", "dce")
            self.activeKeys_weightBalancer = [k for k in BALANCE_KEYS if self.base_weights.get(k, 0.0) > 0.0]
            
            # target proportions don't necessarily need to be softmaxed
            self.targets = {"nll": 1.0, "ls": 0.0, "dce": 0.0, "imax": 0.0, "comp": 0.2} # "ir": 0.0, "kl": 0.0
            
            base_for_balancer    = {k: self.base_weights[k] for k in self.activeKeys_weightBalancer}
            targets_for_balancer = {k: self.targets.get(k, 0.0) for k in self.activeKeys_weightBalancer}

            self.loss_w_eq = ProportionalGradNorm(      #ema_beta=0.9, power=0.5, # 0.9, min_mult=0.25, max_mult=4.0
                    targets=targets_for_balancer, 
                    base=base_for_balancer, 
                    ema_beta=0.99,
                    power=0.3, 
                    min_mult=0.5, 
                    max_mult=2.0)
            # set target kl lambda to the one in config
            def kl_lambda_schedule(step, max_val=self.base_weights.get("kl", 1e-2), warmup=1000):
                x = (step - warmup) / max(1, warmup // 3)
                # smooth 0 -> max_val
                return float(max_val) * (0.5 * (1.0 + torch.tanh(torch.tensor(x))).item())
            self.kl_lambda_schedule = kl_lambda_schedule

        elif self.loss_name == "SalsaNext":
            self.criterion_nll = torch.nn.NLLLoss()#CrossEntropyLoss(ignore_index=self.ignore_idx)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_idx)
            
            # loss weights
            defaults = dict(w_nll=1.0, w_ls=1.0)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)
            self.w_nll, self.w_ls = w["w_nll"], w["w_ls"]
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

        for step, (range_img, reflectivity, xyz, normals, labels) in enumerate(
            tqdm.tqdm(loader, desc=f"train {epoch+1}")
        ):
            self.global_step = epoch * len(loader) + step
            
            # Equalizer scheduling, decoupled from logging
            do_log = bool(self.logging and self.writer and (step % 20 == 0))
            if not hasattr(self, "update_step"):             # increments after each optimizer.step()
                self.update_step = 0
            eq_interval = int(getattr(self, "loss_w_eq_interval", 50))
            
            do_eq   = (self.update_step % eq_interval == 0)   # equalizer update cadence
            need_raw = (do_eq or do_log)  

            # input image prepare
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
            elapsed_ms = self._stop_timer_ms()
            
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

                raw_g = {}
                if need_raw:
                    # lovasz grad_norm
                    if self.w_nll > 0.0 and loss_ls.requires_grad:
                        raw_g["ls"] = grad_norm_wrt(loss_ls, self.model_ref_params, retain_graph=True)
                    # nll grad_norm
                    if self.w_nll > 0.0 and loss_ls.requires_grad:
                        raw_g["nll"] = grad_norm_wrt(loss_nll, self.model_ref_params, retain_graph=True)

                    # cache latest norms for later logging steps that don't measure
                    self._last_raw_g = raw_g
                else:
                    raw_g = getattr(self, "_last_raw_g", {})
                
            elif self.loss_name == "Dirichlet":
                alpha = to_alpha_concentrations(outputs)
                
                # get dirichlet losses
                L_dir_dict={}
                if step % 10 ==0:    # every 10 iteration update ema class weight accumulator, TODO: currently hard coded
                    self.criterion_dirichlet.update_class_weights(labels, method="effective_num", beta=0.999)   # access with self.criterion_dirichlet.class_weights  
                
                loss_dirichlet, L_dir_dict = self.criterion_dirichlet(
                    alpha, labels,
                    w_nll=self.w_nll,
                    w_dce=self.w_dce,
                    w_imax=self.w_imax,
                    w_ir=self.w_ir,
                    w_kl=self.w_kl,        # tune to keep alpha0 in range
                    w_comp=self.w_comp
                )
            
                # Lovasz on either Dirichlet mean (p_hat) or softmaxed model logits F.softmax(outputs, dim=1). We found on Dirichlet mean yielding better results.
                alpha0 = alpha.sum(dim=1, keepdim=True) + get_eps_value()
                p_hat = alpha / alpha0  # debugging: print(p_hat[0,:,32,32],"\n", alpha[0,:,32,32],"\n", alpha0[0,:,32,32]) 
                
                loss_ls = self.criterion_lovasz(p_hat, labels.long(), model_act="probs")
                
                # ----------------- PRE-BACKWARD: compute grad norms -----------------
                raw_g = {}
                if need_raw:
                    # Measure raw grad norms (for ALL losses you want to see)
                    for name, val in L_dir_dict.items():
                        if self.base_weights.get(name, 0.0) > 0.0 and isinstance(val, torch.Tensor) and val.requires_grad:
                            raw_g[name] = grad_norm_wrt(val, self.model_ref_params, retain_graph=True)
                    # Lovasz is separate
                    if self.base_weights.get("ls", 0.0) > 0.0 and isinstance(loss_ls, torch.Tensor) and loss_ls.requires_grad:
                        raw_g["ls"] = grad_norm_wrt(loss_ls, self.model_ref_params, retain_graph=True)
                        
                    # update equalizer only on schedule; else keep current weights
                    if do_eq and raw_g:
                        active_g = {k: raw_g[k] for k in self.activeKeys_weightBalancer if k in raw_g}
                        new_w = self.loss_w_eq.step(active_g)       # uses EFFECTIVE-norm control internally
                    else:
                        new_w = self.loss_w_eq.weights()

                    # Effective grad norms: use balanced weight if present, else fall back to fixed base weight
                    def _eff_weight(name: str) -> float:
                        return float(new_w.get(name, self.base_weights.get(name, 0.0)))
                    eff_g = {k: _eff_weight(k) * float(raw_g[k]) for k in raw_g}

                    self._last_raw_g = raw_g
                    self._last_eff_g = eff_g
                else:
                    new_w = self.loss_w_eq.weights()
                    raw_g = getattr(self, "_last_raw_g", {})
                    eff_g = getattr(self, "_last_eff_g", {})

                # ----------------- BUILD TOTAL LOSS -----------------
                loss = 0.0
                # balanced terms
                if new_w.get("dce", 0.0)  > 0: loss = loss + new_w["dce"]  * L_dir_dict.get("dce",  0.0)
                if new_w.get("nll", 0.0)  > 0: loss = loss + new_w["nll"]  * L_dir_dict.get("nll",  0.0)
                if new_w.get("imax", 0.0) > 0: loss = loss + new_w["imax"] * L_dir_dict.get("imax", 0.0)
                if new_w.get("comp", 0.0) > 0: loss = loss + new_w["comp"] * L_dir_dict.get("comp", 0.0)
                if new_w.get("ls", 0.0)   > 0: loss = loss + new_w["ls"]   * loss_ls
                
                # fixed (NOT balanced) regularizers - still included in total and logs
                if self.base_weights.get("ir", 0.0) > 0.0:
                    loss = loss + self.base_weights["ir"] * L_dir_dict.get("ir", 0.0)

                if self.base_weights.get("kl", 0.0) > 0.0:
                    lam_kl = self.kl_lambda_schedule(self.global_step)
                    self.base_weights["kl"] = lam_kl
                    loss = loss + self.base_weights["kl"] * L_dir_dict.get("kl", 0.0)

                # keep a clean, detached copy for viz AFTER losses computed
                self._alpha_cache = alpha.detach()
                
            else:
                raise RuntimeError("unreachable")

            # @@@ After loss calculaltion @@@
            total_loss += float(loss.item())
            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.update_step += 1
            # ----------------- POST-STEP: logging -----------------
            
            # get predicted class argmax
            preds = outputs.argmax(dim=1)
            
            # IoU handler accumulate
            self.iou_evaluator.update(preds, labels)
            
            # Dirichlet running uncertainty stat (reuse cached alpha)
            if self.loss_name == "Dirichlet" and self._alpha_cache is not None:
                H_norm = get_predictive_entropy_norm.accumulate(self._alpha_cache.cpu())
                
            # Logging
            if do_log:
                self.writer.add_scalar("loss/iter", loss.item(), self.global_step)
                self.writer.add_scalar("LR/iter", self.optimizer.param_groups[0]['lr'], self.global_step)
                if self.loss_name == "Dirichlet":
                    # weighted loss terms that exist (log with CURRENT weights)
                    for name, val in L_dir_dict.items():
                        w_cur = new_w.get(name, 0.0)
                        if w_cur != 0.0:
                            self.writer.add_scalar(f"loss/{name}", _to_float(val) * w_cur, self.global_step)

                    # Lovasz (same effective weight logic; new_w should have it, but fallback is fine)
                    w_ls_eff = new_w.get("ls", self.base_weights.get("ls", 0.0))
                    if w_ls_eff != 0.0:
                        self.writer.add_scalar("loss/ls", _to_float(loss_ls) * float(w_ls_eff), self.global_step)
                    
                    # KL evidence
                    if self.base_weights.get("kl", 0.0) != 0.0:
                        self.writer.add_scalar("loss/kl", _to_float(L_dir_dict.get("kl", 0.0)) * self.base_weights["kl"], self.global_step)
                    
                    # Information regularization term
                    if self.base_weights.get("ir", 0.0) != 0.0:
                        self.writer.add_scalar("loss/ir", _to_float(L_dir_dict.get("ir", 0.0)) * self.base_weights["ir"], self.global_step)

                    # grad norms raw / eff already prepared (eff_g uses effective weights for all keys)
                    if raw_g:
                        for name, g in raw_g.items():
                            self.writer.add_scalar(f"grad_norm/params/raw/{name}", float(g), self.global_step)
                        rss_raw = (sum(float(g)**2 for g in raw_g.values())) ** 0.5
                        self.writer.add_scalar("grad_norm/params/rss_raw", rss_raw, self.global_step)

                    if eff_g:
                        for name, g in eff_g.items():
                            self.writer.add_scalar(f"grad_norm/params/eff/{name}", float(g), self.global_step)
                        rss_eff = (sum(float(g)**2 for g in eff_g.values())) ** 0.5
                        self.writer.add_scalar("grad_norm/params/rss_eff", rss_eff, self.global_step)
                    
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

                    if raw_g:
                        for name, g in raw_g.items():
                            self.writer.add_scalar(f"grad_norm/params/raw/{name}", float(g), self.global_step)
                        # rss raw
                        rss_raw = (sum(float(g)**2 for g in raw_g.values())) ** 0.5 if raw_g else 0.0
                        self.writer.add_scalar("grad_norm/params/rss_raw", rss_raw, self.global_step)
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
            test_mask=[0] + [1] * (self.num_classes - 1),
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
                
                inference_times.append(self._stop_timer_ms())
                
                # debugging sanity check: print(mc_outputs.std().item(), mc_outputs.std(dim=0).mean().item())
                log_probs = F.log_softmax(mc_outputs, dim=2)    # Log-softmax for numerical stability
                probs = log_probs.exp()                             # get probs

                # Predictive distribution
                p_bar = probs.mean(dim=0)   # [B,C,H,W]

                # Predictive entropy: H[p_bar] = -Sum_c {p_bar}_c * log( {p_bar}_c )
                entropy = -(p_bar * (p_bar + get_eps_value()).log()).sum(dim=1)   # [B,(H,W)]
                H_norm = entropy/ math.log(self.num_classes)    # [B,(H,W)]
                
                preds = p_bar.argmax(dim=1) # [B,1,H,W]
                
                self.iou_evaluator.update(preds, labels)
                self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,))
    
            else: # single pass (no MC)
                outputs = self.model(*inputs)
                
                inference_times.append(self._stop_timer_ms())
                
                if self._model_act_kind is None:
                    self._model_act_kind = _classify_output_kind(outputs, class_dim=1)
                    
                logits = _to_logits(outputs, self._model_act_kind)  # [B,C,H,W]
                log_probs = F.log_softmax(logits, dim=1)
                probs = log_probs.exp() # [B,C,H,W]

                preds = probs.argmax(dim=1) # [B,1,H,W]
                
                self.iou_evaluator.update(preds, labels)
                
                if self.loss_name=="Dirichlet":
                    # alpha computed ONCE for all metrics
                    alpha = to_alpha_concentrations(logits)
                    H_norm = get_predictive_entropy_norm(alpha)
                    
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
                    (self.loss_name=="Dirichlet"):
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_width=0.05, 
                    show_percent_on_bars=True,
                    title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:03d}.png"),
                )
            elif use_mc_sampling:
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_width=0.05, 
                    show_percent_on_bars=True,
                    title="Pixel Accuracy vs Predictive-Uncertainty (binned)",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:03d}.png"),
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

                if self.logging and self.save_path: 
                    if mIoU > best_mIoU: # save best weight
                        best_mIoU = mIoU
                        ckpt_path = os.path.join(self.save_path, f"best_epoch_{epoch:03d}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)
                    else:   # save weights regardless but not labeled as "best"                       
                        ckpt_path = os.path.join(self.save_path, f"epoch_{epoch:03d}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)

        # final eval & save
        self.test_one_epoch(val_loader, self.num_epochs - 1)
        if self.logging and self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))

