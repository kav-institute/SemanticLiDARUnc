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
    to_alpha_concentrations_from_shape_and_scale,
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

from metrics.ece import ECEAggregator

from losses.dirichlet_losses import _valid_mask
from utils.grad_norm import grad_norm_wrt, AdaptiveLossBalancer, select_ref_params, discover_shared_params_from_losses

#from models.losses import brier_dirichlet, evidence_logspring
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


#@torch.no_grad()    # TODO: for debugging only!
class Trainer:
    """
    Clean trainer with:
        - standard train / eval loops
        - optional MC-dropout path
        - Dirichlet-aware caching so alpha is computed ONCE per batch when needed

    Expectation:
      - model.forward(*inputs) returns either logits, probs, or log_probs
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
        #self.model_ref_params = list(self.model.parameters()) # or: list(model.head.parameters())
        self.model_ref_params = None    # will be set later

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.visualize = visualize
        
        self.use_mc_sampling = bool(self.cfg["model_settings"].get("use_mc_sampling", 0))
        
        self.logging = logging
        self.num_epochs = int(self.cfg["train_params"]["num_epochs"]) + int(self.cfg["train_params"].get("num_warmup_epochs", 0))

        # loss selection & baseline
        self.loss_name = cfg["model_settings"]["loss_function"]
        self.baseline = cfg["model_settings"]["baseline"]
        
        # data / task meta
        self.num_classes = int(cfg["extras"]["num_classes"])-1 if self.loss_name=="Dirichlet" else int(cfg["extras"]["num_classes"])
        self.class_names = cfg["extras"]["class_names"]
        self.class_colors = cfg["extras"]["class_colors"]
        self.global_step = 0

        # Evaluator inits
        ## ignore index for training, e.g. unlabeled class; else None
        self.ignore_index: int| None = 0
        
        self.iou_evaluator = IoUEvaluator(self.num_classes)
        if cfg["extras"].get("test_mask", 0):
            self.test_mask = list(cfg["extras"]["test_mask"].values())
        else:
            self.test_mask = [0] + [1] * (cfg["extras"]["num_classes"] - 1)
        # eval_on_outputkind in {"alpha", "logits", "probs"}
        if self.loss_name=="Dirichlet": eval_on_outputkind = "alpha"
        elif self.use_mc_sampling: eval_on_outputkind = "probs"
        else: eval_on_outputkind = "logits"
        self.ece_eval = ECEAggregator(
                            n_bins=15,
                            mode=eval_on_outputkind,          # "alpha" | "logits" | "probs" depending on what you feed
                            ignore_index=self.ignore_index,        
                            max_samples=100_000_000,       # None or an int cap like 2_000_000 to bound memory
                            plot_style="classic+hist"
                        )
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
            #LovaszSoftmaxStable,   # TODO: remove
            #DirichletCriterion     # TODO: remove
        )
        from losses.lovasz import LovaszSoftmaxStable
        
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
            self.criterion_ce = CrossEntropyLoss(ignore_index=self.ignore_index)
            self.criterion_tversky = TverskyLoss(ignore_index=self.ignore_index)
            
            # loss weights
            defaults = dict(w_ce=1.0, w_tversky=1.0)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)
            self.w_ce, self.w_tversky = w["w_ce"], w["w_tversky"]
        elif self.loss_name == "CE":
            self.criterion_ce = CrossEntropyLoss(ignore_index=self.ignore_index)
        elif self.loss_name == "Lovasz":
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_index)
        elif self.loss_name == "Dirichlet":
            # build prior concentration from desired credible interval coverage   
            from utils.alpha_evid_prior import solve_alpha0_for_coverage
            # h=0.05 -> (95% central mass in [0.85, 0.95]), # delta=0.025 -> 95% central mass (1 - 2*delta)
            self.prior_concentration, self.prior_concentration_per_class = solve_alpha0_for_coverage(p_star=0.90, h=0.05, delta=0.025, K=self.num_classes)     # alpha0, alpha0/C target
            print(f"Dirichlet prior concentration set to {self.prior_concentration:.3f} (uniform base)")
            
            from utils.alpha_evid_prior import logit_threshold_for_alpha_cap
            # margin: how much over the target s_total you will tolerate before the hinge wakes up, m: assume up to m "active" classes per pixel
            z_thr, a_thr = logit_threshold_for_alpha_cap(self.prior_concentration, K=self.num_classes, m=3, margin=0.05, T=get_alpha_temperature())
            from losses.dirichlet_losses import (
                NLLDirichletCategorical,
                BrierDirichlet,
                ComplementKLUniform
            )
            self.crit_nll_dircat = NLLDirichletCategorical(ignore_index=self.ignore_index)
            self.crit_brier = BrierDirichlet(ignore_index=self.ignore_index, s_ref=None)  # set s_ref=None if you want the standard version or self.prior_concentration
            self.crit_comp = ComplementKLUniform(ignore_index=self.ignore_index, gamma=2.0, tau=0.55, sigma=0.12,
                                    s_target=None, normalize=True)

            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_index)

            # loss weights
            defaults = dict(w_nll=0.0, 
                            w_imax=3.0, 
                            w_dce=1.0, 
                            w_ls=2.5, 
                            w_kl=0.5, 
                            w_ir=0.3,
                            w_comp=0.2,
                            w_brier=0.05,
                            w_evid_reg=2e-3,
                            w_logit_reg=1e-4)
            w = load_loss_weights(self.cfg, self.loss_name, defaults)

            self.w_nll, self.w_imax, self.w_dce = w["w_nll"], w["w_imax"], w["w_dce"]
            self.w_ls,  self.w_kl,   self.w_ir  = w["w_ls"],  w["w_kl"],  w["w_ir"]
            self.w_comp = w["w_comp"]
            self.w_brier = w["w_brier"]
            self.w_evid_reg = w["w_evid_reg"]
            self.w_logit_reg = w["w_logit_reg"]
            # define prior/base weights (from cfg)
            self.base_weights = {
                "nll": self.w_nll, "ls": self.w_ls, "dce": self.w_dce,
                "imax": self.w_imax, "comp": self.w_comp, "brier": self.w_brier,
                "ir": self.w_ir, "kl": self.w_kl, 
                "evid_reg": self.w_evid_reg, "logit_reg": self.w_logit_reg
            }
            
            from losses.regularizers import EvidenceReg, LogitRegularizer, EvidenceRegBand
            # EvidenceReg: scale controller. penalizes mismatch between a0 and target s, while leaving class proportions alone.
                # choose mode: "one_sided" (penalize only over-confidence) or "log_squared" (penalize both under- and over-confidence)
            self.evidence_reg = EvidenceReg(s_target=self.prior_concentration, mode="log_squared", ignore_index=self.ignore_index, scale_correct=True, margin=0.1)
            # LogitRegularizer: hard guardrail on the pre-activation z. Stopping runaway growth early and everywhere.
            self.logit_reg = LogitRegularizer(threshold=z_thr, ignore_index=None)   


            # Which losses should be *balanced* by GradNorm (supervised/shape only)
            # Anything *not* in BALANCE_KEYS should be added to the loss with a fixed weight outside GradNorm.
            BALANCE_KEYS = ("nll", "ls", "comp", "brier")
            self.activeKeys_weightBalancer = [k for k in BALANCE_KEYS if self.base_weights.get(k, 0.0) > 0.0]
            
            # target proportions don't necessarily need to be softmaxed
            self.targets = {"nll": 0.75, "ls": 0.15, "dce": 0.0, "imax": 0.0, "comp": 0.00, "brier": 0.10}
            try:
                if self.cfg["model_weights"]["Dirichlet"].get("target_shares", 0) !=0 and \
                    isinstance(self.cfg["model_weights"]["Dirichlet"].get("target_shares", 0), dict):
                        if all([True if (k in self.cfg["model_weights"]["Dirichlet"]["target_shares"]) else False for k in BALANCE_KEYS] ):
                            self.targets = self.cfg["model_weights"]["Dirichlet"]["target_shares"]
                        else:
                            print(f"ERROR in getting target weight shares. Using default {self.targets}")
                else:
                    self.targets = {"nll": 0.8, "ls": 0.2, "dce": 0.0, "imax": 0.0, "comp": 0.0} # "ir": 0.0, "kl": 0.0
            except:
                print(f"ERROR in getting target weight shares. Using default {self.targets}")
            
            print(f"Using base weights: {self.base_weights},\ntarget weights: {self.targets}")
            base_for_balancer    = {k: self.base_weights[k] for k in self.activeKeys_weightBalancer}
            targets_for_balancer = {k: self.targets.get(k, 0.0) for k in self.activeKeys_weightBalancer}

            # self.loss_w_eq = ProportionalGradNorm(
            #     targets=targets_for_balancer,
            #     base=base_for_balancer,
            #     ema_beta=0.92,   # was 0.99, 0.97
            #     power=0.7,       # was 0.3, 0.5
            #     min_mult=0.05,
            #     max_mult=10.0,
            #     use_log_ema=True,
            #     step_cap=2.0     # gentle per-update cap, 1.25
            # )
            self.loss_w_eq_interval = 10
            self.loss_w_eq = AdaptiveLossBalancer(
                names=list(targets_for_balancer.keys()),
                mode="share",                   # warmup on shares, then GradNorm. options: "gradnorm", "share", "hybrid"
                target_share=targets_for_balancer,
                start_step_gradnorm=5000,        # e.g., after warmup
                alpha=0.5,                       # GradNorm exponent
                lr_mult=1.0,                     # multiplicative strength
                ema_beta_g=0.97, ema_beta_L=0.95,   # if testing every step use ema_beta_g=0.95, ema_beta_L=0.90
                step_cap=1.5, min_w=0.05, max_w=10.0, inactive_frac_of_median=0.05
            )
        elif self.loss_name == "SalsaNext":
            self.criterion_nll = torch.nn.NLLLoss()#CrossEntropyLoss(ignore_index=self.ignore_index)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_index)
            
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
            
            do_eq   = (self.global_step % int(getattr(self, "loss_w_eq_interval", 10)) == 0)   # equalizer update cadence
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
                shape_logits = outputs[:, :self.num_classes, ...]
                scale_logits = outputs[:, self.num_classes:self.num_classes+1, ...] 
                alpha = to_alpha_concentrations_from_shape_and_scale(shape_logits, scale_logits)
                #alpha = to_alpha_concentrations(scale_logits)
                #self._alpha_logits = outputs.detach()  # optional; used only for the precond metric

                alpha0 = alpha.sum(dim=1, keepdim=True) + get_eps_value()
                p_hat = alpha / alpha0
                
                # Terms computed with your new split loss classes
                L_terms = {}

                # nll (dirichlet-categorical)
                if self.base_weights.get("nll", 0.0) > 0.0:
                    loss_nll_dircat = self.crit_nll_dircat(alpha, labels)
                    L_terms["nll"] = loss_nll_dircat

                # Lovasz on Dirichlet mean
                if self.base_weights.get("ls", 0.0) > 0.0:
                    loss_ls = self.criterion_lovasz(p_hat, labels.long(), model_act="probs")
                    L_terms["ls"] = loss_ls

                # Complement KL to uniform on off-classes (if enabled)
                if self.base_weights.get("comp", 0.0) > 0.0:
                    loss_comp = self.crit_comp(alpha, labels)
                    L_terms["comp"] = loss_comp

                # Dirichlet Brier (expected)
                if self.base_weights.get("brier", 0.0) > 0.0:
                    loss_brier = self.crit_brier(alpha, labels)
                    L_terms["brier"] = loss_brier

                # Add regularization terms
                if self.base_weights.get("evid_reg", 0.0) > 0.0:
                    loss_evid_reg = self.evidence_reg(alpha)
                    L_terms["evid_reg"] = loss_evid_reg
                    
                if self.base_weights.get("logit_reg", 0.0) > 0.0:
                    loss_logit_reg = self.logit_reg(outputs)
                    L_terms["logit_reg"] = loss_logit_reg

                # Discover shared params for all losses, overwrite previous if any
                if getattr(self, "model_ref_params", None) is None:
                    if len(self.activeKeys_weightBalancer) >= 2:
                        self.model_ref_params = discover_shared_params_from_losses(L_terms, self.model, min_losses=2)
                        #self.model_ref_params = select_ref_params(self.model, strategy="all", exclude_bias_norm=True)

                # ----------------- PRE-BACKWARD: compute grad norms -----------------
                # keys the balancer should control this step (intersection of: active list, base>0, and present in L_terms)
                keys = [k for k in self.activeKeys_weightBalancer
                        if self.base_weights.get(k, 0.0) > 0.0 and (k in L_terms)]

                # --- get/update weights ---
                if do_eq and keys:
                    # Update balancer state and get fresh weights (balancer computes grads internally)
                    # Pass only the relevant losses to avoid touching unrelated ones
                    new_w = self.loss_w_eq.step({k: L_terms[k] for k in keys}, self.model_ref_params, global_step=self.global_step)
                else:
                    # Don't update; just use current normalized weights (avg=1) for the selected keys
                    new_w = self.loss_w_eq.get_weights(keys, global_step=self.global_step)

                # Fallback to base weight if a key somehow lacks a balancer weight (defensive)
                for k in keys:
                    if k not in new_w:
                        new_w[k] = float(self.base_weights.get(k, 0.0))

                # --- logging raw / effective gradient norms ---
                if need_raw and keys:
                    # Prefer what the balancer already measured this step
                    raw_g = getattr(self.loss_w_eq, "last_g_raw", None)
                    eff_g = getattr(self.loss_w_eq, "last_eff_g", None)

                    # If weights were NOT updated this step (do_eq=False), the balancer didn't measure grads.
                    # Compute once for logging only (retain_graph=True keeps backprop possible later).
                    if (raw_g is None or not raw_g) or (eff_g is None or not eff_g):
                        raw_g = {k: grad_norm_wrt(L_terms[k], self.model_ref_params, retain_graph=True) for k in keys}
                        eff_g = {k: float(new_w[k]) * float(raw_g[k]) for k in raw_g}

                    self._last_raw_g = raw_g
                    self._last_eff_g = eff_g
                else:
                    raw_g = getattr(self, "_last_raw_g", {})
                    eff_g = getattr(self, "_last_eff_g", {})

                # --- build total loss (balanced terms only) ---
                loss = 0.0
                for k in keys:
                    wk = float(new_w.get(k, 0.0))
                    if wk > 0.0:
                        loss = loss + wk * L_terms.get(k, 0.0)

                # (optional) if you also have fixed-weight terms not managed by the balancer:
                # for name, base_w in self.base_weights.items():
                #     if name not in keys and base_w > 0.0:
                #         loss = loss + float(base_w) * L_terms.get(name, 0.0)

                # fixed-weight terms
                if self.base_weights.get("evid_reg", 0.0) > 0.0 or self.base_weights.get("logit_reg", 0.0) > 0.0:
                    with torch.no_grad():
                        s    = float(self.prior_concentration)
                        band = 0.05                       # tolerance for a0 overshoot (5% is a good starting point)
                        a0   = alpha.sum(dim=1) + get_eps_value()   # [B,H,W]

                        valid = _valid_mask(labels, self.ignore_index)
                            
                if self.base_weights.get("evid_reg", 0.0) > 0.0:
                    with torch.no_grad():  
                        # absolute log error from target, per pixel
                        log_err_target = torch.abs(torch.log(a0 / s))
                        # band in log space; ~0.095 for band=0.10
                        band_log = float(torch.log(torch.tensor(1.0 + band)))
                        log_err_target = log_err_target[valid] if valid.any() else log_err_target.reshape(-1)

                        if log_err_target.numel() == 0:
                            gate_evid = 0.0
                        else:
                            # in [0,1]: near 0 inside band, rises smoothly outside
                            gate_evid = float((log_err_target / (log_err_target + band_log)).clamp(0, 1).median())

                    w_evid_eff = float(self.base_weights["evid_reg"]) * gate_evid
                    new_w["evid_reg"] = w_evid_eff
                    loss = loss + new_w["evid_reg"] * L_terms.get("evid_reg", 0.0)
                if self.base_weights.get("logit_reg", 0.0) > 0.0:
                    with torch.no_grad():
                        # overshoot ratio per pixel:
                        # r = 0 when a0 <= s; r > 0 measures how far above target we are (in relative terms)
                        r = ((a0 / s) - 1.0).clamp_min(0.0)
                        r = r[valid] if valid.any() else r.reshape(-1)

                        if r.numel() == 0:
                            frac = torch.tensor(0.0, device=a0.device)
                            over_p50 = over_p90 = over_p99 = torch.tensor(0.0, device=a0.device)
                        else:
                            # how widespread is the overshoot? (0..1)
                            frac = (r > 0).float().mean()
                            
                            over_p50 = r.quantile(0.50)                       # median overshoot of the tail
                            over_p90 = r.quantile(0.90)                       # tail overshoot
                            over_p99 = r.quantile(0.99)                       # extreme overshoot
                            
                        # tail magnitude: use a high quantile so single outliers do not jerk the gate
                            # typical choices: 0.90 (robust), 0.95 (stricter)
                        # magnitude gate in [0,1]: near 0 if over_p90 << band, -> 1 as over_p90 >> band
                        gate_mag = over_p90 / (over_p90 + band + 1e-8)

                        # spread amplifier (tunable)
                            # gamma = 0.0 => spread = 1.0 (no spread effect; recommended default)
                            # gamma = 1.0 => spread = frac (linear effect)
                            # gamma < 1.0 => concave boost: amplifies small frac (e.g., sqrt when gamma=0.5)
                            # gamma > 1.0 => suppresses small frac: needs large coverage to matter
                        gamma_spread = getattr(self, "logit_spread_gamma", 0.0)  # start at 0.0 (disabled)
                        spread = (frac.clamp_min(1e-6)) ** gamma_spread

                        # small always-on floor so we keep a tiny guardrail even when below s
                        gate_floor = 0.05

                        # final gate in [0,1]
                        gate_raw = float(gate_floor + (1.0 - gate_floor) * (gate_mag * spread))
                        gate_raw = max(0.0, min(1.0, gate_raw))
                        
                        # smooth with EMA to avoid sudden jumps
                        beta = getattr(self, "logit_gate_beta", 0.90)
                        if not hasattr(self, "_gate_logit_ema"):
                            # on the first step, start EMA at the observed value
                            self._gate_logit_ema = float(gate_raw)

                        gate = beta * float(self._gate_logit_ema) + (1.0 - beta) * float(gate_raw)
                        gate = max(0.0, min(1.0, gate))
                        self._gate_logit_ema = gate  # persist state

                    # effective weights: weight scales with gate; also cap the amplification
                    w_logit_eff = float(self.base_weights["logit_reg"]) * gate       
                    new_w["logit_reg"] = w_logit_eff
                    loss = loss + new_w["logit_reg"] * L_terms.get("logit_reg", 0.0)
                # keep a clean, detached copy for viz AFTER losses computed
                self._alpha_cache = alpha.detach()


            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item())

            # ----------------- POST-STEP: logging -----------------
            
            if self.loss_name == "Dirichlet":
                outputs = shape_logits  # for prediction purposes
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
                    for name, val in L_terms.items():
                        w_cur = new_w.get(name, 0.0)
                        if w_cur != 0.0:
                            self.writer.add_scalar(f"loss/{name}", _to_float(val) * w_cur, self.global_step)

                # Lovasz
                w_ls_eff = new_w.get("ls", self.base_weights.get("ls", 0.0))
                if w_ls_eff != 0.0:
                    self.writer.add_scalar("loss/ls", _to_float(loss_ls) * float(w_ls_eff), self.global_step)

                # grad norms raw / eff
                if raw_g:
                    for name, g in raw_g.items():
                        self.writer.add_scalar(f"grad_norm/params/raw/{name}", float(g), self.global_step)
                rss_raw = (sum(float(g)**2 for g in raw_g.values())) ** 0.5
                self.writer.add_scalar("grad_norm/params/rss_raw", rss_raw, self.global_step)

                if eff_g:
                    eff_sum = sum(eff_g.values()) + 1e-12
                    for name, g in eff_g.items():
                        self.writer.add_scalar(f"grad_norm/params/eff/{name}", float(g), self.global_step)
                        self.writer.add_scalar(f"weight_share/all/{name}", eff_g[name]/eff_sum, self.global_step)
                    
                    active = [k for k in self.activeKeys_weightBalancer if k in raw_g and self.base_weights.get(k,0)>0]
                    # reconstruct g_ema, eff, shares as in step()
                    # then log:
                    eff_sum = sum(eff_g[k] for k in active) + 1e-12
                    eff_share = {k: eff_g[k]/eff_sum for k in active}
                    for k,v in eff_share.items():
                        self.writer.add_scalar(f"weight_share/balancedKeys/{k}", v, self.global_step)
                    
                    eff_sum = sum(eff_g.values()) + 1e-12
                    for name, g in eff_g.items():
                        self.writer.add_scalar(f"weight_share/all/{name}", eff_g[name]/eff_sum, self.global_step)

                rss_eff = (sum(float(g)**2 for g in eff_g.values())) ** 0.5
                self.writer.add_scalar("grad_norm/params/rss_eff", rss_eff, self.global_step)
                
                if self.base_weights.get("logit_reg", 0.0) > 0.0: 
                    self.writer.add_scalar("logit_gate/frac_overshoot", float(frac.item()), self.global_step)
                    self.writer.add_scalar("logit_gate/over_p50", float(over_p50.item()), self.global_step)
                    self.writer.add_scalar("logit_gate/over_p90", float(over_p90.item()), self.global_step)
                    self.writer.add_scalar("logit_gate/over_p99", float(over_p99.item()), self.global_step)
                    self.writer.add_scalar("logit_gate/gate_mag", float(gate_mag.item()), self.global_step)
                    self.writer.add_scalar("logit_gate/gate", float(gate), self.global_step)
                    # fraction of logits over threshold (if threshold exists)
                    if getattr(self.logit_reg, "threshold", None) is not None:
                        thr = float(self.logit_reg.threshold)
                        frac_z_over = (outputs.detach() > thr).float().mean().item()
                        self.writer.add_scalar("logit_gate/frac_logits_over_thr", float(frac_z_over), self.global_step)

                    # Optional histogram to see the distribution of overshoot r
                    # if r.numel() > 0:
                    #     self.writer.add_histogram("logit_gate/r_hist", r.detach().cpu(), self.global_step)
        
                # alpha0 stats (use cached alpha)
                with torch.no_grad():
                    a0_hw = self._alpha_cache.sum(dim=1)              # [B,H,W]
                    med_alpha0 = a0_hw.median().item()
                    med_alpha0_per_cls = (a0_hw / float(alpha.shape[1])).median().item()
                self.writer.add_scalar("alpha0/median", med_alpha0, self.global_step)
                self.writer.add_scalar("alpha0/median_per_class", med_alpha0_per_cls, self.global_step)

                # comp gate diagnostics (if comp enabled)
                if self.base_weights.get("comp", 0.0) != 0.0:
                    with torch.no_grad():
                        py = (alpha/(alpha.sum(1, keepdim=True)+1e-8)).gather(1, labels.unsqueeze(1)).squeeze(1)
                        mean_gate = ((1 - py).pow(self.crit_comp.gamma) *
                                     torch.sigmoid((self.crit_comp.tau - py)/self.crit_comp.sigma)).mean().item()
                        frac_below_tau = (py < self.crit_comp.tau).float().mean().item()
                        self.writer.add_scalar("comp/mean_gate", mean_gate, self.global_step)
                        self.writer.add_scalar("comp/frac_below_tau", frac_below_tau, self.global_step)

                # H_norm coverage
                self.writer.add_scalar("H_norm/pct_lt_0.1",  (H_norm < 0.10).float().mean().item(), self.global_step)
                self.writer.add_scalar("H_norm/pct_lt_0.25", (H_norm < 0.25).float().mean().item(), self.global_step)
                self.writer.add_scalar("H_norm/pct_gt_0.5",  (H_norm > 0.50).float().mean().item(), self.global_step)
                self.writer.add_scalar("H_norm/pct_gt_0.75", (H_norm > 0.75).float().mean().item(), self.global_step)
            
            # Interactive visualization (cheap, reusing computed items)
            if self.visualize:
                idx0 = 0
                want_cuda_viz_calc = True
                
                # -> GT class
                semantics_gt = labels[idx0].detach().cpu().numpy()  # [H, W]
                if self.ignore_index is not None:
                    mask = np.argwhere(semantics_gt==self.ignore_index)
                else:
                    mask = None
                
                # -> Predicted Semantic Class
                outputs_cpu = outputs[idx0].detach().cpu().numpy()  # [B, C, H, W] -> [C, H, W]
                semantics_pred = np.argmax(outputs_cpu, axis=0)     # [H, W]
                    # apply mask for ignored index
                if self.ignore_index is not None:
                    semantics_pred[mask[:, 0], mask[:, 1]] = self.ignore_index
                    
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
            ignore_gt=[self.ignore_index],
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
        # Reset Eval Accumulators
        self.iou_evaluator.reset()
        self.ua_agg.reset()
        self.ece_eval.reset()
        
        self.model.eval()
        inference_times = []

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
            if self.use_mc_sampling:
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
    
            else: # single pass (no MC)
                outputs = self.model(*inputs)
                if self.loss_name == "Dirichlet":
                    shape_logits = outputs[:, :self.num_classes, ...]
                    scale_logits = outputs[:, self.num_classes:self.num_classes+1, ...] 
                    logits = shape_logits  # use shape logits only for prediction
                inference_times.append(self._stop_timer_ms())
                
                if self._model_act_kind is None:
                    self._model_act_kind = _classify_output_kind(outputs, class_dim=1)
                
                # if self.loss_name == "Dirichlet":
                #     outputs = outputs[:, :self.num_classes, ...]  # use shape logits only for prediction
                #logits = _to_logits(outputs, self._model_act_kind)  # [B,C,H,W]
                
                log_probs = F.log_softmax(logits, dim=1)
                probs = log_probs.exp() # [B,C,H,W]
                
                # get predicted class argmax
                preds = probs.argmax(dim=1) # [B,1,H,W]
                
                # Metric aggregator accumulation
                ## iou
                self.iou_evaluator.update(preds, labels)
                
                if self.loss_name=="Dirichlet":
                    # alpha computed ONCE for all metrics
                    #alpha = to_alpha_concentrations(logits)
                    alpha = to_alpha_concentrations_from_shape_and_scale(shape_logits, scale_logits)
                    H_norm = get_predictive_entropy_norm(alpha)
                    
                    ## H_norm vs accuracy
                    self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,)     # ignore unlabeled if ignore_ids==0
                    )
                    ## calibration
                    self.ece_eval.update(alpha, labels)
                else:
                    ## calibration
                    self.ece_eval.update(probs, labels)
        
        # @@@ END of Epoch
        
        # metrics
        mIoU, result_dict = self.iou_evaluator.compute(
            class_names=self.class_names,
            test_mask=self.test_mask,
            ignore_gt=[self.ignore_index],
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
        
            if self.use_mc_sampling or self.loss_name=="Dirichlet":
                self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                    bin_width=0.05, 
                    show_percent_on_bars=True,
                    title=f"Pixel Accuracy vs Predictive-Uncertainty (epoch {epoch:03d})",
                    save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:03d}.png"),
                )
                # ECE plot
                (ece, mce), ece_stats = self.ece_eval.compute(
                    save_plot_path=os.path.join(out_dir, f"ece_epoch_{epoch:03d}.png"),
                    title=f"Reliability (epoch {epoch:03d})"
                )
                print(f"Epoch {epoch} ECE: {ece:.4f}, Max_ECE: {mce:.4f}")

        return mIoU

    # ------------------------------
    # main loop
    # ------------------------------
    def __call__(self, train_loader, val_loader):
        test_every = int(self.cfg["logging_settings"]["test_every_nth_epoch"])
        
        self.total_train_steps = len(train_loader) * self.num_epochs
        if getattr(self, "loss_w_eq", None):
            if hasattr(self.loss_w_eq, "switch_step"):
                self.loss_w_eq.switch_step = self.total_train_steps // 10  # first 10% steps warmup

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
                    os.makedirs(os.path.join(self.save_path, "weights"), exist_ok=True)
                    if mIoU > best_mIoU: # save best weight
                        best_mIoU = mIoU
                        ckpt_path = os.path.join(self.save_path, "weights", f"best_epoch_{epoch:03d}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)
                    else:   # save weights regardless but not labeled as "best"                       
                        ckpt_path = os.path.join(self.save_path, "weights", f"epoch_{epoch:03d}.pt")
                        torch.save(self.model.state_dict(), ckpt_path)

        # final eval & save
        self.test_one_epoch(val_loader, self.num_epochs - 1)
        if self.logging and self.save_path:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_final.pt"))

