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
from metrics.auroc import AUROCAggregator

from losses.dirichlet_losses import _valid_mask
from utils.grad_norm import (
    grad_norm_wrt, 
    AdaptiveLossBalancer, 
    select_ref_params, 
    discover_shared_params_from_losses,
    _apply_share_cap_vs_reference
)

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

import math


# --- Unified weight ramp schedule (used for both comp and wle) ---
def _cosine_weight_ramp(step: int, total: int,
                       w0: float, w_peak: float, w_end: float,
                       warm_frac: float, hold_frac: float) -> float:
    """
    Generic cosine ramp: warmup -> hold -> cosine decay.
    Used for both comp and wle with different parameter sets.
    """
    s = step / max(1, total)
    if s <= warm_frac:
        return w0 + (w_peak - w0) * (s / warm_frac)
    if s <= hold_frac:
        return w_peak
    t = (s - hold_frac) / (1.0 - hold_frac)
    return w_end + 0.5 * (w_peak - w_end) * (1.0 + math.cos(math.pi * min(t, 1.0)))

# --- Unified share cap schedule (used for both comp and wle) ---
def _cosine_share_cap(step: int, total: int,
                     cap_start: float, cap_end: float, hold_frac: float) -> float:
    """
    Generic cosine cap decay: hold -> cosine decay.
    Represents % of reference loss's effective gradient.
    """
    s = step / max(1, total)
    if s <= hold_frac:
        return cap_start
    t = (s - hold_frac) / (1.0 - hold_frac)
    return cap_end + 0.5 * (cap_start - cap_end) * (1.0 + math.cos(math.pi * min(t, 1.0)))

# --- update balancer target shares at milestones (works for mode="share" and "hybrid" warmup)
def _set_target_share_for(bal, new_share: dict[str, float]):
    # normalize over the balancer's names
    s = sum(max(0.0, float(new_share.get(k, 0.0))) for k in bal.names) + 1e-12
    bal.share = {k: float(new_share.get(k, 0.0)) / s for k in bal.names}

# piecewise share schedule for {nll, brier}
def nb_share_schedule(step: int, total: int) -> dict[str, float]:
    r = step / max(1, total)
    if r < 0.15:
        return {"nll": 0.75, "brier": 0.25}
    elif r < 0.40:
        return {"nll": 0.60, "brier": 0.40}
    else:
        return {"nll": 0.55, "brier": 0.45}


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
        else: eval_on_outputkind = "probs"  # or logits if not used probs as argument
        self.ece_eval = ECEAggregator(
                            n_bins=15,
                            mode=eval_on_outputkind,          # "alpha" | "logits" | "probs" depending on what you feed
                            ignore_index=self.ignore_index,        
                            max_samples=100_000_000,       # None or an int cap like 2_000_000 to bound memory
                            plot_style="classic+hist"
                        )
        self.auroc_eval = AUROCAggregator(mode=eval_on_outputkind, score="entropy_norm", ignore_index=self.ignore_index)
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
                ComplementKLUniform,
                DirichletMSELoss,
                DigammaDirichletCE
            )
            self.crit_nll_dircat = NLLDirichletCategorical(ignore_index=self.ignore_index)
            self.crit_brier = BrierDirichlet(ignore_index=self.ignore_index, s_ref=float(self.num_classes+20))  # set s_ref=None if you want the standard version or self.prior_concentration
            self.crit_mse_dir = DirichletMSELoss(ignore_index=self.ignore_index)
            self.crit_digamma_ce = DigammaDirichletCE(ignore_index=self.ignore_index)
            self.criterion_lovasz = LovaszSoftmaxStable(ignore_index=self.ignore_index)
            
            # Regualarizers
            from losses.regularizers import KL_offClasses_to_uniform
            self.crit_kl = KL_offClasses_to_uniform(ignore_index=self.ignore_index)
            self.crit_comp = ComplementKLUniform(ignore_index=self.ignore_index, gamma=1.25, tau=0.65, sigma=0.15,   # gamma=2.0, tau=0.55, sigma=0.12
                                    s_target=None, normalize=True)
            
            
            # loss weights
            defaults = dict(w_nll=1.0, 
                            w_ls=2.5, 
                            w_kl=0.5, 
                            w_comp=0.2,
                            w_brier=0.05,
                            w_wle=0.05,          # Add default weight for WLE
                            w_mse=1.0,
                            w_digamma_ce=1.0,   
            )
            w = load_loss_weights(self.cfg, self.loss_name, defaults)

            self.w_nll = w["w_nll"]
            self.w_ls = w["w_ls"]
            self.w_kl = w["w_kl"]
            self.w_comp = w["w_comp"]
            self.w_brier = w["w_brier"]
            self.w_wle = w["w_wle"]            # Add this line to extract the wle weight
            self.w_mse = w["w_mse"]
            self.w_digamma_ce = w["w_digamma_ce"]
            # define prior/base weights (from cfg)
            self.base_weights = {
                "nll": self.w_nll, 
                "ls": self.w_ls, 
                "comp": self.w_comp, 
                "brier": self.w_brier,
                "kl": self.w_kl, 
                "wle": self.w_wle,      
                "mse": self.w_mse,
                "digamma_ce": self.w_digamma_ce
            }
            
            from losses.regularizers import WrongLowEvidence

            self.crit_wle = WrongLowEvidence(
                ignore_index=self.ignore_index,
                s_low=0.0,            # pull wrongs toward a0 = C
                margin=0.05,          # require some confidence gap to trigger
                soft_margin_k=0.08    # soft transition
            )

            # Which losses should be *balanced* by GradNorm (supervised/shape only)
            # Anything *not* in BALANCE_KEYS should be added to the loss with a fixed weight outside GradNorm.
            BALANCE_KEYS = ("nll", "ls", "brier", 'mse', 'digamma_ce')  # removed "comp"
            self.activeKeys_weightBalancer = [k for k in BALANCE_KEYS if self.base_weights.get(k, 0.0) > 0.0]
            
            self.reference_loss_term="mse"  # use selected reference loss for GradNorm
            assert hasattr(self, "reference_loss_term") and self.reference_loss_term in self.activeKeys_weightBalancer, \
                f"Reference loss term '{self.reference_loss_term}' not found in active keys."
            print(f"Using reference loss term for GradNorm: {self.reference_loss_term}")
            
            # target proportions for balanced losses only
            self.targets = {"nll": 0.75, "ls": 0.20, "brier": 0.05}
            try:
                if self.cfg["model_weights"]["Dirichlet"].get("target_shares", 0) != 0 and \
                    isinstance(self.cfg["model_weights"]["Dirichlet"].get("target_shares", 0), dict):
                        ts = self.cfg["model_weights"]["Dirichlet"]["target_shares"]
                        # only use keys that are in BALANCE_KEYS
                        if all(k in ts for k in BALANCE_KEYS):
                            self.targets = {k: ts[k] for k in BALANCE_KEYS}
                        else:
                            print(f"ERROR in target_shares; using default {self.targets}")
            except:
                print(f"ERROR in getting target weight shares. Using default {self.targets}")

            print(f"Using base weights: {self.base_weights},\ntarget weights: {self.targets}")
            #base_for_balancer    = {k: self.base_weights[k] for k in self.activeKeys_weightBalancer}
            targets_for_balancer = {k: self.targets.get(k, 0.0) for k in self.activeKeys_weightBalancer}

            self.loss_w_eq_interval = 10
            self.loss_w_eq = AdaptiveLossBalancer(
                names=list(targets_for_balancer.keys()),
                mode="gradnorm",   # in "gradnorm" | "share" | "hybrid"
                target_share=targets_for_balancer,
                start_step_gradnorm=5000,
                alpha=0.5,
                lr_mult=1.0,
                ema_beta_g=0.97, ema_beta_L=0.95,
                step_cap=2.0, min_w=0.05, max_w=10.0, inactive_frac_of_median=0.05
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
                    if self.w_nll > 0.0 and loss_nll.requires_grad:
                        raw_g["nll"] = grad_norm_wrt(loss_nll, self.model_ref_params, retain_graph=True)

                    # cache latest norms for later logging steps that don't measure
                    self._last_raw_g = raw_g
                else:
                    raw_g = getattr(self, "_last_raw_g", {})
                
            elif self.loss_name == "Dirichlet":
                shape_logits = outputs[:, :self.num_classes, ...]
                scale_logits = outputs[:, self.num_classes:self.num_classes+1, ...] 
                alpha = to_alpha_concentrations_from_shape_and_scale(shape_logits, scale_logits)

                alpha0 = alpha.sum(dim=1, keepdim=True) + get_eps_value()
                p_hat = alpha / alpha0
                
                # Terms computed with your new split loss classes
                L_terms = {}

                # nll (dirichlet-categorical)
                if self.base_weights.get("nll", 0.0) > 0.0:
                    loss_nll_dircat = self.crit_nll_dircat(alpha, labels)
                    L_terms["nll"] = loss_nll_dircat

                # Dirichlet MSE (accuracy-focused + add. variance term)
                if self.base_weights.get("mse", 0.0) > 0.0:
                    loss_mse_dir = self.crit_mse_dir(alpha, labels)
                    L_terms["mse"] = loss_mse_dir
                
                if self.base_weights.get("digamma_ce", 0.0) > 0.0:
                    loss_digamma_ce = self.crit_digamma_ce(alpha, labels)
                    L_terms["digamma_ce"] = loss_digamma_ce

                # Lovasz on Dirichlet mean
                if self.base_weights.get("ls", 0.0) > 0.0:
                    loss_ls = self.criterion_lovasz(p_hat, labels.long(), model_act="probs")
                    L_terms["ls"] = loss_ls

                # Complement KL to uniform on off-classes (FIXED WEIGHT with scheduling)
                if self.base_weights.get("comp", 0.0) > 0.0:
                    loss_comp = self.crit_comp(alpha, labels)
                    L_terms["comp"] = loss_comp

                # Dirichlet Brier (expected)
                if self.base_weights.get("brier", 0.0) > 0.0:
                    loss_brier = self.crit_brier(alpha, labels)
                    L_terms["brier"] = loss_brier

                # Incorrect low-evidence penalty
                if self.base_weights.get("wle", 0.0) > 0.0:
                    L_terms["wle"] = self.crit_wle(alpha, labels)
                    
                if self.base_weights.get("kl", 0.0) > 0.0:
                    loss_kl = self.crit_kl(alpha, labels)
                    L_terms["kl"] = loss_kl 


                assert all(k in L_terms for k in self.base_weights if self.base_weights[k] > 0.0), f"Missing loss terms for base weights"
                
                # Discover shared params for all losses (once)
                if getattr(self, "model_ref_params", None) is None:
                    if len(self.activeKeys_weightBalancer) >= 2:
                        self.model_ref_params = discover_shared_params_from_losses(L_terms, self.model, min_losses=2)
                    else:
                        self.model_ref_params = discover_shared_params_from_losses(L_terms, self.model, min_losses=1)

                # ----------------- PRE-BACKWARD: compute grad norms ONCE -----------------
                # Keys the balancer controls
                balanced_keys = [k for k in self.activeKeys_weightBalancer
                                 if self.base_weights.get(k, 0.0) > 0.0 and (k in L_terms)]

                # Optional: dynamically adjust target shares for balanced_keys during training
                if do_eq and balanced_keys and self.loss_w_eq.mode in ("share", "hybrid"):
                    new_share = nb_share_schedule(self.global_step, self.total_train_steps)
                    # Only update if the schedule changed
                    if new_share != getattr(self, "_last_share_sched", None):
                        _set_target_share_for(self.loss_w_eq, new_share)
                        self._last_share_sched = new_share

                # --- Update balancer and retrieve measured grads ---
                if do_eq and balanced_keys:
                    # Balancer measures grads internally
                    new_w = self.loss_w_eq.step(
                        {k: L_terms[k] for k in balanced_keys}, 
                        self.model_ref_params, 
                        global_step=self.global_step
                    )
                    
                    # REUSE the grads the balancer just computed
                    raw_g = dict(getattr(self.loss_w_eq, "last_g_raw", {}))
                    
                    # Compute grads for NON-balanced terms (comp, regularizers) ONCE per update
                    for name in L_terms:
                        if name not in raw_g:   # raw_g already has balanced keys
                            g = grad_norm_wrt(L_terms[name], self.model_ref_params, retain_graph=True)
                            raw_g[name] = g
                    
                    # Cache ALL grads for non-update steps
                    self._last_raw_g = raw_g
                    self._last_new_w = new_w
                    
                    self._last_eff_g = dict()   # effective grads are built later


                    assert all(k in self._last_new_w for k in balanced_keys), "Balancer failed to return new weights for all balanced keys."
                    assert all(k in self._last_raw_g for k in L_terms), "Failed to compute raw grads for all loss terms."
                else:
                    # Not an update step: reuse cached 
                        # weights self._last_new_w
                        # grads self._last_raw_g    
                    pass
                
                # --- Get effective gradient of reference loss ---
                g_ref_raw = 0.0
                if self.reference_loss_term in balanced_keys:    # with asserts above raw_g and new_w must exist
                    g_ref_raw = float(self._last_raw_g[self.reference_loss_term])
                    w_ref_eff = float(self._last_new_w[self.reference_loss_term])
                    self._last_eff_g[self.reference_loss_term] = g_ref_raw * w_ref_eff   # cache

                # --- Schedule comp weight with cap relative to reference loss ---
                if "comp" in L_terms:
                    base_w_comp = float(self.base_weights["comp"])
                    
                    # Get scheduled weight using unified helper
                    w_comp_scheduled = _cosine_weight_ramp(
                        self.global_step, 
                        self.total_train_steps,
                        w0=0.001 * base_w_comp,
                        w_peak=base_w_comp * 0.5,
                        w_end=base_w_comp * 0.2,
                        warm_frac=0.12,
                        hold_frac=0.35
                    )
                    
                    # Apply cap relative to NLL using unified helper
                    if g_ref_raw > 0.0:
                        comp_cap_ratio = _cosine_share_cap(
                            self.global_step,
                            self.total_train_steps,
                            cap_start=0.05,
                            cap_end=0.03,
                            hold_frac=0.3
                        )

                        w_comp_final = _apply_share_cap_vs_reference(
                            w_scheduled=w_comp_scheduled,
                            g_current_raw=float(self._last_raw_g["comp"]),
                            g_reference_raw=g_ref_raw,
                            w_ref=w_ref_eff,  # Use effective balanced weight of reference
                            cap_ratio=comp_cap_ratio,
                            name="comp"
                        )
                    else:
                        w_comp_final = w_comp_scheduled
                    
                    self._last_new_w["comp"] = w_comp_final
                
                # --- Schedule wle weight with cap relative to reference loss ---
                if "wle" in L_terms:
                    base_w_wle = float(self.base_weights["wle"])
                    
                    # Get scheduled weight using unified helper
                    w_wle_scheduled = _cosine_weight_ramp(
                        self.global_step,
                        self.total_train_steps,
                        w0=0.5*base_w_wle,   # 0.5 * base_w_wle
                        w_peak=base_w_wle,
                        w_end=base_w_wle * 0.25,
                        warm_frac=0.1,
                        hold_frac=0.3
                    )
                    
                    if g_ref_raw > 0.0:
                        wle_cap_ratio = _cosine_share_cap(
                            self.global_step,
                            self.total_train_steps,
                            cap_start=0.2,
                            cap_end=0.15,
                            hold_frac=0.3
                        )

                        w_wle_final = _apply_share_cap_vs_reference(
                            w_scheduled=w_wle_scheduled,
                            g_current_raw=float(self._last_raw_g["wle"]),
                            g_reference_raw=g_ref_raw,
                            w_ref=w_ref_eff,  # Use NLL's effective balanced weight
                            cap_ratio=wle_cap_ratio,
                            name="wle"
                        )
                    else:
                        w_wle_final = w_wle_scheduled
                    
                    self._last_new_w["wle"] = w_wle_final

                # --- Schedule kl weight with cap relative to reference loss ---
                if "kl" in L_terms:
                    base_w_kl = float(self.base_weights["kl"])

                    # Get scheduled weight using unified helper
                    w_kl_scheduled = _cosine_weight_ramp(
                        self.global_step, 
                        self.total_train_steps,
                        w0=0.01 * base_w_kl,
                        w_peak=base_w_kl,
                        w_end=base_w_kl,
                        warm_frac=0.1,  # 10% of total train steps is warmup
                        hold_frac=1.00   # hold at peak until end
                    )
                    
                    # Apply cap relative to NLL using unified helper
                    if g_ref_raw > 0.0:
                        # make permenant cap at 15% of reference
                        kl_cap_ratio = _cosine_share_cap(
                            self.global_step,
                            self.total_train_steps,
                            cap_start=0.15,
                            cap_end=0.15,
                            hold_frac=0.15
                        )

                        w_kl_final = _apply_share_cap_vs_reference(
                            w_scheduled=w_kl_scheduled,
                            g_current_raw=float(self._last_raw_g["kl"]),
                            g_reference_raw=g_ref_raw,
                            w_ref=w_ref_eff,  # Use effective balanced weight of reference
                            cap_ratio=kl_cap_ratio,
                            name="kl"
                        )
                    else:
                        w_kl_final = w_kl_scheduled

                    self._last_new_w["kl"] = w_kl_final

                # --- Build last effective gradients for all terms ---
                for k in L_terms:
                    if k not in self._last_new_w:
                        self._last_new_w[k] = float(self.base_weights.get(k, 0.0))
                    
                    if k not in self._last_eff_g:
                        self._last_eff_g[k] = float(self._last_new_w[k]) * float(self._last_raw_g.get(k, 0.0))

                assert len(L_terms) == len(self._last_new_w) == len(self._last_raw_g) == len(self._last_eff_g), "Mismatch in tracked loss term counts."

                # --- Build total loss ---
                loss = 0.0
                
                # Add balanced terms to loss
                L_terms_eff = dict()
                for k in L_terms:
                    wk = float(self._last_new_w.get(k, 0.0))
                    if wk > 0.0:
                        L_terms_eff[k] = wk * L_terms.get(k, 0.0)
                        loss = loss + L_terms_eff[k]
                                
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
                    # Log weighted loss terms
                    for name, val in L_terms_eff.items():
                        self.writer.add_scalar(f"loss/{name}", _to_float(val), self.global_step)

                    # Log scheduled weights
                    for name, w in self._last_new_w.items():
                        self.writer.add_scalar(f"scheduled_weights/{name}", float(w), self.global_step)

                    # Log raw gradient norms for ALL losses
                    rss_raw = 0.0
                    for name, g in self._last_raw_g.items():
                        self.writer.add_scalar(f"grad_norm/params/raw/{name}", float(g), self.global_step)
                        rss_raw += float(g)**2
                    
                    #rss_raw = (sum(float(g)**2 for g in self._last_raw_g.values())) ** 0.5
                    self.writer.add_scalar("grad_norm/params/rss_raw", rss_raw**0.5, self.global_step)

                    # Log effective gradient norms for ALL losses and their relative shares
                    eff_sum_all = sum(self._last_eff_g.values()) + 1e-12
                    for name, g in self._last_eff_g.items():
                        self.writer.add_scalar(f"grad_norm/params/eff/{name}", float(g), self.global_step)
                        self.writer.add_scalar(f"eff_grad_shares/all/{name}", float(g)/eff_sum_all, self.global_step)

                    # Separate view: shares among balanced keys only
                    if balanced_keys:
                        balanced_eff_sum = sum(float(self._last_eff_g[k]) for k in balanced_keys if k in self._last_eff_g) + 1e-12
                        for k in balanced_keys:
                            self.writer.add_scalar(
                                f"eff_grad_shares/balancedKeys/{k}", 
                                float(self._last_eff_g[k])/balanced_eff_sum, 
                                self.global_step
                            )

                    rss_eff = (sum(float(g)**2 for g in self._last_eff_g.values())) ** 0.5
                    self.writer.add_scalar("grad_norm/params/rss_eff", rss_eff, self.global_step)

                    # ---------------- alpha0 + concentration split stats ----------------
                    with torch.no_grad():
                        eps = 1e-12
                        alpha_cached = self._alpha_cache                     # [B,C,H,W]
                        B, C, H, W = alpha_cached.shape

                        a0_hw = alpha_cached.sum(dim=1)                      # [B,H,W]  (alpha0 per pixel)
                        s_hw  = a0_hw - float(C)                             # evidence scale s when alpha = 1 + s p

                        # --- percentiles helper (torch >= 1.7 has torch.quantile) ---
                        def _quantiles(x_flat: torch.Tensor, qs):
                            x_flat = x_flat.to(dtype=torch.float32)
                            q = torch.tensor(qs, device=x_flat.device, dtype=torch.float32)
                            try:
                                return torch.quantile(x_flat, q, interpolation="linear")
                            except TypeError:  # older PyTorch uses 'method'
                                return torch.quantile(x_flat, q, method="linear")

                        # flatten over spatial & batch
                        a0_vec = a0_hw.reshape(-1)

                        # Outer percentiles you asked for
                        qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
                        a0_q = _quantiles(a0_vec, qs)

                        qnames = ["p01","p05","p25","p50","p75","p95","p99"]
                        for name, val in zip(qnames, a0_q.tolist()):
                            self.writer.add_scalar(f"alpha0/percentile_{name}", val, self.global_step)

                        # --- how much proportion of alpha0 sits on the most likely class ---
                        # top-1 alpha share = alpha_max / alpha0  (equals p_hat_max)
                        alpha_max_hw, _ = alpha_cached.max(dim=1)            # [B,H,W]
                        top1_share_hw = alpha_max_hw / (a0_hw + eps)         # in [0,1]

                        share_vec = top1_share_hw.reshape(-1)
                        share_q   = _quantiles(share_vec, qs)
                        for name, val in zip(qnames, share_q.tolist()):
                            self.writer.add_scalar(f"alpha0/top1_share_percentile_{name}", val, self.global_step)

                        # helpful summary scalars
                        self.writer.add_scalar("alpha0/mean",  float(a0_vec.mean().item()),  self.global_step)
                        self.writer.add_scalar("alpha0/median",float(a0_q[3].item()),         self.global_step)  # p50
                        self.writer.add_scalar("alpha0/median_per_class", float((a0_vec/float(C)).median().item()), self.global_step)

                        self.writer.add_scalar("alpha0/top1_share_mean",   float(share_vec.mean().item()), self.global_step)
                        self.writer.add_scalar("alpha0/top1_share_p95",    float(share_q[5].item()),       self.global_step)
                        self.writer.add_scalar("alpha0/top1_share_p99",    float(share_q[6].item()),       self.global_step)

                        # thresholds: how concentrated is top-1?
                        for th in (0.5, 0.7, 0.9, 0.95, 0.99):
                            frac = float((share_vec >= th).float().mean().item())
                            self.writer.add_scalar(f"alpha0/top1_share_frac_ge_{th:.2f}", frac, self.global_step)
                    # ---------------- end of alpha0 stats ----------------
                    
                    # Log wle-specific scheduling info
                    if "wle" in L_terms:
                        # Add WLE activation rate logging
                        with torch.no_grad():
                            # Calculate the wrong prediction mask
                            wrong_mask = (preds != labels) & (labels != self.ignore_index)
                            
                            # Calculate the activation based on margin
                            alpha0 = self._alpha_cache.sum(dim=1, keepdim=True)
                            C = alpha.shape[1]  # Number of classes
                            target_log = math.log(C + self.crit_wle.s_low + 1e-8)
                            a0_log = alpha0.log().squeeze(1)
                            
                            # Calculate delta between log(alpha0) and target log value
                            delta = (a0_log - target_log).detach()
                            
                            # Get actual margin after adjustment for confidently wrong predictions
                            pred_class = preds.unsqueeze(1)
                            correct_class = labels.unsqueeze(1)
                            pred_alpha = alpha.gather(1, pred_class).squeeze(1)
                            correct_alpha = alpha.gather(1, correct_class).squeeze(1)
                            margin_factor = ((pred_alpha / (correct_alpha + 1e-8)) - 1.0).clamp_min(0.0)
                            effective_margin = self.crit_wle.margin * (1.0 + self.crit_wle.k * margin_factor)
                            
                            # Calculate activation rate - what percentage of wrong predictions activated the loss
                            active_wrong = (delta > effective_margin) & wrong_mask
                            
                            # Log activation stats
                            if wrong_mask.any():
                                activation_rate = active_wrong.float().sum() / wrong_mask.float().sum()
                                self.writer.add_scalar("wle/activation_rate", activation_rate.item(), self.global_step)
                                self.writer.add_scalar("wle/wrong_pixel_rate", wrong_mask.float().mean().item(), self.global_step)
                    
                    # comp gate diagnostics (if comp enabled)
                    if "comp" in L_terms:
                        with torch.no_grad():
                            py = (self._alpha_cache/(self._alpha_cache.sum(1, keepdim=True)+1e-8)).gather(1, labels.unsqueeze(1)).squeeze(1)
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
                # End of Dirichlet logging
            # End of logging
            
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
                self.writer.add_scalar('Calibration/entropy_tot/epoch', H_norm_epoch, self.global_step)
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
                @torch.no_grad()
                def mc_predictive_entropy_norm(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
                    # probs: [T,B,C,H,W]
                    # Predictive entropy: H[p_bar] = -Sum_c {p_bar}_c * log( {p_bar}_c )
                    p_bar = probs.mean(dim=0)                                # [B,C,H,W]
                    H = -(p_bar.clamp_min(eps) * p_bar.clamp_min(eps).log()).sum(dim=1)
                    return H / math.log(p_bar.size(1))    
            
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
                
                mc_outputs = mc_forward(self.model, inputs, T=mc_T)     # [T,B,C,H,W]
                
                inference_times.append(self._stop_timer_ms())
                
                # debugging sanity check: print(mc_outputs.std().item(), mc_outputs.std(dim=0).mean().item())
                log_probs = F.log_softmax(mc_outputs, dim=2)    # Log-softmax for numerical stability, [T,B,C,H,W]
                probs = log_probs.exp()                         # get probs, [T,B,C,H,W]

                # Predictive distribution
                p_bar = probs.mean(dim=0)   # [B,C,H,W]

                # Predictive entropy
                H_norm = mc_predictive_entropy_norm(probs)
                # MI/ Epistemic Uncertainty
                MI_norm = mc_mutual_information_norm(probs)

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
                inference_times.append(self._stop_timer_ms())
                if self.loss_name == "Dirichlet":
                    shape_logits = outputs[:, :self.num_classes, ...]
                    scale_logits = outputs[:, self.num_classes:self.num_classes+1, ...] 
                    logits = shape_logits  # use shape logits only for prediction
                else:
                    logits = outputs
                
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
                    ## auroc
                    self.auroc_eval.update(alpha, labels)
                else:
                    ## calibration
                    self.ece_eval.update(probs, labels)

                    # Predictive entropy: H[p_bar] = -Sum_c {p_bar}_c * log( {p_bar}_c )
                    p_bar = probs
                    entropy = -(p_bar * torch.clamp(p_bar, min=get_eps_value()).log()).sum(dim=1)   # [B,(H,W)]
                    H_norm = entropy/ math.log(self.num_classes)    # [B,(H,W)]

                    ## H_norm vs accuracy
                    self.ua_agg.update(
                        labels=labels, 
                        preds=preds, 
                        uncertainty=H_norm, 
                        ignore_ids=(0,)     # ignore unlabeled if ignore_ids==0
                    )

                    ## auroc
                    self.auroc_eval.update(p_bar, labels)
        
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
        
            fig_ua_agg, ax_ua_agg = self.ua_agg.plot_accuracy_vs_uncertainty_bins(
                bin_width=0.05, 
                show_percent_on_bars=True,
                title=f"Pixel Accuracy vs Predictive-Uncertainty (epoch {epoch:03d})",
                save_path=os.path.join(out_dir, f"acc_vs_unc_bins_{epoch:03d}.png"),
            )
            # ECE plot
            (ece, mce), ece_stats, fig_ece = self.ece_eval.compute(
                save_plot_path=os.path.join(out_dir, f"ece_epoch_{epoch:03d}.png"),
                title=f"Reliability (epoch {epoch:03d})"
            )
            self.writer.add_scalar('Calibration/ECE', ece, epoch)
            self.writer.add_scalar('Calibration/Max_ECE', mce, epoch)
            
            # AUROC
            auroc, _, fig_auroc = self.auroc_eval.compute(
                save_plot_path=os.path.join(out_dir, f"auroc_epoch_{epoch:03d}.png"),
            )
            self.writer.add_scalar('Calibration/AUROC', auroc, epoch)

            # now mutate fig for TB
            fig_ece.set_size_inches(4, 3)   # shrink physical size
            fig_ece.set_dpi(100)            # lower dpi so final pixel dims ~600x450

            fig_ua_agg.set_size_inches(4, 3)   # shrink physical size
            fig_ua_agg.set_dpi(100)            # lower dpi so final pixel dims ~600x450

            fig_auroc.set_size_inches(4, 3)   # shrink physical size
            fig_auroc.set_dpi(100)            # lower dpi so final pixel dims ~600x450

            self.writer.add_figure('Calibration/ECE_Plot', fig_ece, epoch, close=True)
            self.writer.add_figure('Calibration/Accuracy_vs_Entropy', fig_ua_agg, epoch, close=True)
            self.writer.add_figure('Calibration/AUROC_Plot', fig_auroc, epoch, close=True)

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
                self.loss_w_eq.switch_step = self.total_train_steps // 25  # first 25% steps warmup

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

