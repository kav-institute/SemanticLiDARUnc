import math
import torch
from typing import Dict, Iterable, Union, Optional

# ---------- your function (kept) ----------
# def grad_norm_wrt(
#     loss: torch.Tensor,
#     wrt: Union[torch.Tensor, Iterable[torch.Tensor]],
#     *,
#     retain_graph: bool = False,
# ) -> float:
#     if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
#         return 0.0
#     if loss.ndim > 0:
#         loss = loss.mean()

#     if isinstance(wrt, (list, tuple)):
#         wrt_list = [p for p in wrt if isinstance(p, torch.Tensor) and p.requires_grad]
#     else:
#         wrt_list = [wrt] if isinstance(wrt, torch.Tensor) and wrt.requires_grad else []
#     if not wrt_list:
#         return 0.0

#     grads = torch.autograd.grad(loss, wrt_list, retain_graph=retain_graph, allow_unused=True)
#     sq = 0.0
#     for g in grads:
#         if g is None:
#             continue
#         v = torch.linalg.vector_norm(g.float(), ord=2)
#         if torch.isfinite(v):
#             sq += float(v.item()) ** 2
#     return math.sqrt(sq)

def grad_norm_wrt(
    loss: torch.Tensor,
    wrt: Union[torch.Tensor, Iterable[torch.Tensor]],
    *,
    retain_graph: bool = False,
) -> float:
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

    # Accumulate on device — NO per-parameter .item()
    sq = None
    for g in grads:
        if g is None:
            continue
        gf = g.detach()
        if gf.dtype != torch.float32:
            gf = gf.float()
        val = (gf * gf).sum()
        sq = val if sq is None else (sq + val)

    if sq is None:
        return 0.0
    return float(sq.sqrt().item())  # single sync

class AdaptiveLossBalancer:
    """
    Drop-in scalar reweighter with three modes:
      - 'gradnorm' : GradNorm (Chen et al., ICML'18) with stable multiplicative update.
      - 'share'    : your ProportionalGradNorm (target gradient shares), improved & gated.
      - 'hybrid'   : warm up with 'share' then switch to 'gradnorm' after a given step.

    Call .step(loss_dict, ref_params) each iteration to get a {name: weight} dict.

    Key conveniences:
      * log-EMA smoothing on gradients and losses
      * inactivity gating (ignores near-zero-grad losses in normalization)
      * per-step cap and [min_w, max_w] clamps
      * average weight = 1.0 after each update (stable mixing with other terms)
    """
    def __init__(
        self,
        names: Iterable[str],
        mode: str = "gradnorm",
        # --- GradNorm options ---
        alpha: float = 0.5,                  # training-rate exponent
        lr_mult: float = 1.0,                # multiplicative step strength rho
        # --- Share mode options ---
        target_share: Optional[Dict[str, float]] = None,
        power: float = 0.7,                  # smoothing exponent (your k)
        # --- Common stabilizers ---
        ema_beta_g: float = 0.95,            # EMA for gradient norms
        ema_beta_L: float = 0.90,            # EMA for loss values (for r_i)
        ema_floor: float = 1e-8,
        inactive_frac_of_median: float = 0.05, # gating threshold as % of median grad
        min_w: float = 0.05,
        max_w: float = 10.0,
        step_cap: float = 1.5,               # per-step multiplier cap
        start_step_gradnorm: int = 0,        # for 'hybrid': switch step
    ):
        self.names = list(names)
        self.mode = mode.lower()
        assert self.mode in {"gradnorm", "share", "hybrid"}

        self.alpha = float(alpha)
        self.lr_mult = float(lr_mult)
        self.power = float(power)

        self.beta_g = float(ema_beta_g)
        self.beta_L = float(ema_beta_L)
        self.ema_floor = float(ema_floor)
        self.inactive_frac = float(inactive_frac_of_median)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self.step_cap = float(step_cap)
        self.switch_step = int(start_step_gradnorm)

        # weights live in positive space; start at 1.0
        self.w = {k: 1.0 for k in self.names}
        # EMAs
        self.g_ema = {k: 0.0 for k in self.names}
        self.L0 = {}          # initial losses for GradNorm (filled on first seen)
        self.L_ema = {k: None for k in self.names}
        # target shares (normalized over provided keys)
        if target_share is None:
            target_share = {k: 1.0 for k in self.names}
        s = sum(max(0.0, float(target_share.get(k, 0.0))) for k in self.names) + 1e-12
        self.share = {k: float(target_share.get(k, 0.0))/s for k in self.names}

        self._step_idx = 0
    
    def _avg1(self, keys):
        avg = sum(self.w[k] for k in keys) / (len(keys) + 1e-12)
        for k in keys:
            self.w[k] /= (avg + 1e-12)

    def _inactive_filter(self, keys):
        vals = [self.g_ema[k] for k in keys]
        med = sorted(vals)[len(vals)//2] if vals else 0.0
        thr = max(self.ema_floor, self.inactive_frac * max(med, self.ema_floor))
        act = [k for k in keys if self.g_ema[k] >= thr]
        return act if act else keys  # fallback: keep all

    def get_weights(self, keys=None, global_step: int | None = None) -> Dict[str, float]:
        """Return current weights (avg=1 normalized) without updating."""
        self._step_idx = int(global_step)
        keys = list(self.w.keys()) if keys is None else list(keys)
        avg = sum(self.w[k] for k in keys) / (len(keys) + 1e-12)
        return {k: float(self.w[k] / (avg + 1e-12)) for k in keys}
    
    @torch.no_grad()
    def step(self, losses: Dict[str, torch.Tensor], ref_params: Union[Iterable[torch.Tensor], torch.Tensor], global_step: int | None = None) -> Dict[str, float]:
        self._step_idx = int(global_step)
        keys = [k for k in self.names if k in losses]

        # 1) measure grad norms (unweighted, retain_graph so caller can still backprop total)
        g_raw = {}
        for k in keys:
            Lk = losses[k]
            g = grad_norm_wrt(Lk, ref_params, retain_graph=True)
            # log-EMA smoothing
            self.g_ema[k] = self.beta_g * self.g_ema[k] + (1 - self.beta_g) * math.log(max(g, 1e-12))
            g_raw[k] = g
        g_sm = {k: max(math.exp(self.g_ema[k]), self.ema_floor) for k in keys}

        # stash to get current grads without needing to call step() again
        self.last_g_raw = {k: float(v) for k, v in g_raw.items()}              # raw norms (unweighted)
        self.last_eff_g = {k: float(self.w[k] * g_sm[k]) for k in keys}         # weighted (effective) norms

        # 2) update EMA losses and record L0 if needed
        for k in keys:
            Lk = float(losses[k].mean().detach().item())
            if k not in self.L0:
                self.L0[k] = max(Lk, 1e-12)
            prev = self.L_ema[k]
            self.L_ema[k] = (self.beta_L * prev + (1 - self.beta_L) * Lk) if prev is not None else Lk

        # 3) choose mode
        use_gradnorm = (self.mode == "gradnorm") or (self.mode == "hybrid" and self._step_idx >= self.switch_step)
        if use_gradnorm:
            self._update_gradnorm(keys, g_sm)
        else:
            self._update_share(keys, g_sm)

        return {k: float(self.w[k]) for k in keys}

    # ---- GradNorm (stable multiplicative variant) ----
    def _update_gradnorm(self, keys, g_sm):
        # Active keys by gradient activity
        active = self._inactive_filter(keys)

        # training rates r_i (EMA losses / initial)
        r = {k: max(self.L_ema[k] / self.L0[k], 1e-12) for k in active}
        # target normalized rates  r_i^alpha / mean(r^alpha)
        rpow = {k: r[k] ** self.alpha for k in active}
        mean_rpow = sum(rpow.values()) / (len(active) + 1e-12)
        rstar = {k: rpow[k] / (mean_rpow + 1e-12) for k in active}

        # current weighted gradient norms G_i
        G = {k: self.w[k] * g_sm[k] for k in active}
        Gbar = sum(G.values()) / (len(active) + 1e-12)

        # desired ratio: G_i -> Gbar * rstar_i
        # multiplicative update: w_i <- w_i * ((Gbar * r*) / (G_i + eps))^rho, with caps
        rho = self.lr_mult
        for k in active:
            target = Gbar * rstar[k]
            ratio = (target / (G[k] + 1e-12)) ** rho
            # per-step cap
            ratio = float(min(max(ratio, 1.0 / self.step_cap), self.step_cap))
            self.w[k] = float(self.w[k] * ratio)
            self.w[k] = float(min(max(self.w[k], self.min_w), self.max_w))

        # softly relax inactive keys toward 1.0
        for k in keys:
            if k not in active:
                self.w[k] = 0.9 * self.w[k] + 0.1 * 1.0

        # keep average weight = 1
        self._avg1(keys)

    # ---- Target-share (your method, activity-aware) ----
    def _update_share(self, keys, g_sm):
        active = [k for k in keys if self.share.get(k, 0.0) > 0.0]
        if not active:
            return

        active = self._inactive_filter(active)

        # normalize target shares on active
        tot = sum(max(0.0, self.share.get(k, 0.0)) for k in active) + 1e-12
        sh = {k: self.share.get(k, 0.0) / tot for k in active}

        # desired multipliers inversely proportional to g * (current w baseline=1)
        raw = {k: sh[k] / (g_sm[k] + 1e-12) for k in active}
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in raw.values()) / len(active))
        m_des = {k: raw[k] / gm for k in active}

        # smooth move in multiplicative space with cap
        kpow = self.power
        for k in active:
            ratio = (m_des[k] / (self.w[k] + 1e-12)) ** kpow
            ratio = float(min(max(ratio, 1.0 / self.step_cap), self.step_cap))
            self.w[k] = float(self.w[k] * ratio)
            self.w[k] = float(min(max(self.w[k], self.min_w), self.max_w))

        for k in keys:
            if k not in active:
                self.w[k] = 0.9 * self.w[k] + 0.1 * 1.0

        self._avg1(keys)


def select_ref_params(model, strategy="backbone", *, exclude_bias_norm=True, name_filter=None):
    """
    strategy: "backbone" | "shared" | "all"
    name_filter: optional callable(name) -> bool to pick layers (e.g., last stage)
    """
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_bias_norm and (name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower()):
            continue
        if strategy == "backbone" and "backbone" not in name:
            continue
        if strategy == "shared" and ("head" in name and "shared" not in name):
            # keep only shared head parts if you have them; adjust to your naming
            continue
        if name_filter is not None and not name_filter(name):
            continue
        params.append(p)
    if not params:
        # fallback: all trainable (still respecting exclude_bias_norm)
        params = [p for p in model.parameters() if p.requires_grad]
    return params


def discover_shared_params_from_losses(losses: dict[str, torch.Tensor],
                                       model,
                                       min_losses: int = 2,
                                       exclude_bias_norm: bool = True):
    """
    Returns params that actually influence >= min_losses of the given losses.
    Run once (retain_graph=True) and cache the returned list.
    """
    # candidate params
    params = []
    names = []
    for n,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_bias_norm and (n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower()):
            continue
        params.append(p)
        names.append(n)

    # build a mask: which params affect which loss
    affect = {k: [False]*len(params) for k in losses}
    for k, L in losses.items():
        if not isinstance(L, torch.Tensor) or not L.requires_grad:
            continue
        g = torch.autograd.grad(L.mean(), params, retain_graph=True, allow_unused=True)
        for i, gi in enumerate(g):
            if gi is not None and torch.isfinite(gi.detach()).all():
                affect[k][i] = True

    # pick params touched by at least min_losses different losses
    shared_idx = [i for i in range(len(params))
                  if sum(affect[k][i] for k in losses) >= min_losses]

    # fallback if nothing qualifies
    if not shared_idx:
        shared_idx = list(range(len(params)))

    ref_params = [params[i] for i in shared_idx]
    return ref_params


class ProportionalGradNorm:
    def __init__(self, targets, base, ema_beta=0.99, power=0.3,
                 min_mult=0.5, max_mult=2.0,
                 use_log_ema=True, ema_floor=1e-8, step_cap=1.25):
        self.targets   = targets.copy()
        self.base      = base.copy()
        self.beta      = float(ema_beta)
        self.k         = float(power)
        self.min_mult  = float(min_mult)
        self.max_mult  = float(max_mult)
        self.use_log_ema = bool(use_log_ema)
        self.ema_floor = float(ema_floor)
        self.step_cap  = float(step_cap)
        self.ema  = {k: 0.0 for k in targets}   # EMA of g or log g
        self.mult = {k: 1.0 for k in targets}

    def weights(self):
        return {k: (self.base[k] * self.mult.get(k, 1.0) if self.base.get(k, 0.0) > 0.0 else 0.0)
                for k in self.base}
    
    def step(self, raw_g: dict[str, float]):
        active = [k for k, v in self.base.items() if v > 0.0 and k in raw_g]
        if not active:
            return self.weights()

        # --- log-EMA grads (same as you do) ---
        for k in active:
            g = max(0.0, float(raw_g[k]))
            if self.use_log_ema:
                lg = math.log(g + 1e-12)
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * lg
            else:
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * g

        g_ema = {k: (math.exp(self.ema[k]) if self.use_log_ema else self.ema[k]) for k in active}
        for k in active:
            g_ema[k] = max(g_ema[k], self.ema_floor)

        # --- normalize target shares over *active* keys (sum=1) ---
        tot_t = sum(max(0.0, float(self.targets.get(k, 0.0))) for k in active) + 1e-12
        share = {k: float(self.targets.get(k, 0.0)) / tot_t for k in active}

        # --- closed-form desired multipliers (inverse-grad, base-aware) ---
        raw = {k: share[k] / (g_ema[k] * self.base[k] + 1e-12) for k in active}
        # geometric mean normalization so avg multiplier stays ~1
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in raw.values()) / len(active))
        m_des = {k: raw[k] / gm for k in active}

        # --- smooth move toward m_des with per-step cap ---
        for k in active:
            ratio = (m_des[k] / (self.mult[k] + 1e-12)) ** self.k
            ratio = float(min(max(ratio, 1.0 / self.step_cap), self.step_cap))
            self.mult[k] = float(self.mult[k] * ratio)
            self.mult[k] = float(min(max(self.mult[k], self.min_mult), self.max_mult))

        # re-center to keep average ~1
        avg_mult = sum(self.mult[k] for k in active) / (len(active) + 1e-12)
        for k in active:
            self.mult[k] /= (avg_mult + 1e-12)

        return self.weights()


class ProportionalGradNorm:
    def __init__(self, targets, base, ema_beta=0.99, power=0.3,
                 min_mult=0.5, max_mult=2.0,
                 use_log_ema=True, ema_floor=1e-8, step_cap=1.25):
        self.targets   = targets.copy()
        self.base      = base.copy()
        self.beta      = float(ema_beta)
        self.k         = float(power)
        self.min_mult  = float(min_mult)
        self.max_mult  = float(max_mult)
        self.use_log_ema = bool(use_log_ema)
        self.ema_floor = float(ema_floor)
        self.step_cap  = float(step_cap)
        self.ema  = {k: 0.0 for k in targets}   # EMA of g or log g
        self.mult = {k: 1.0 for k in targets}

    def weights(self):
        return {k: (self.base[k] * self.mult.get(k, 1.0) if self.base.get(k, 0.0) > 0.0 else 0.0)
                for k in self.base}
    
    def step(self, raw_g: dict[str, float]):
        active = [k for k, v in self.base.items() if v > 0.0 and k in raw_g]
        if not active:
            return self.weights()

        # --- log-EMA grads (same as you do) ---
        for k in active:
            g = max(0.0, float(raw_g[k]))
            if self.use_log_ema:
                lg = math.log(g + 1e-12)
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * lg
            else:
                self.ema[k] = self.beta * self.ema[k] + (1 - self.beta) * g

        g_ema = {k: (math.exp(self.ema[k]) if self.use_log_ema else self.ema[k]) for k in active}
        for k in active:
            g_ema[k] = max(g_ema[k], self.ema_floor)

        # --- normalize target shares over *active* keys (sum=1) ---
        tot_t = sum(max(0.0, float(self.targets.get(k, 0.0))) for k in active) + 1e-12
        share = {k: float(self.targets.get(k, 0.0)) / tot_t for k in active}

        # --- closed-form desired multipliers (inverse-grad, base-aware) ---
        raw = {k: share[k] / (g_ema[k] * self.base[k] + 1e-12) for k in active}
        # geometric mean normalization so avg multiplier stays ~1
        gm = math.exp(sum(math.log(max(v, 1e-12)) for v in raw.values()) / len(active))
        m_des = {k: raw[k] / gm for k in active}

        # --- smooth move toward m_des with per-step cap ---
        for k in active:
            ratio = (m_des[k] / (self.mult[k] + 1e-12)) ** self.k
            ratio = float(min(max(ratio, 1.0 / self.step_cap), self.step_cap))
            self.mult[k] = float(self.mult[k] * ratio)
            self.mult[k] = float(min(max(self.mult[k], self.min_mult), self.max_mult))

        # re-center to keep average ~1
        avg_mult = sum(self.mult[k] for k in active) / (len(active) + 1e-12)
        for k in active:
            self.mult[k] /= (avg_mult + 1e-12)

        return self.weights()
    
    
class _CapState:
    # holds persistent per-loss state
    def __init__(self):
        self.ema_g_ref = None    # EMA of reference grad-norm
        self.ema_g_cur = None    # EMA of current loss grad-norm
        self.w_prev    = None    # last applied weight for this loss
        self.bind_ctr  = 0       # consecutive steps with applied > cap

_CAP_STATES = {}  # keyed by `name`

def _apply_share_cap_vs_reference(
    w_scheduled: float,        # schedule weight for this step
    g_current_raw: float,      # raw grad-norm of this loss
    g_reference_raw: float,    # raw grad-norm of reference loss
    w_ref: float,              # current effective weight on reference loss
    cap_ratio: float,          # max allowed eff_current <= cap_ratio * eff_ref
    name: str = "loss",
    *,
    # ema smoothing for grad norms
    ema_beta: float = 0.95, # smoothing for grad norms -> higher = smoother
    grad_floor: float = 1e-9,
    # per-step multiplicative limits
    ratio_cap_up: float = 1.12,    # at most +12% per step
    ratio_cap_dn: float = 0.92,    # at most -8%  per step
    # adaptive down after sustained binding
    adaptive_tighten_after: int = 5,
    adaptive_ratio_cap_dn: float = 0.85,  # at most -15% per step
    # emergency brake when applied >> limit and persistent
    emergency_patience: int = 2,    # need >= this many bound steps
    emergency_violation: float = 1.5,  # applied_eff > 1.5 * limit
    emergency_factor: float = 0.75, # multiply target by 0.75 (25% cut)
    # let emergency also loosen the per-step down cap (optional)
    emergency_loosen_down_cap: bool = True
) -> float:
    # -------------------------------------------------------------------------
    # 0) get per-loss state
    # -------------------------------------------------------------------------
    st = _CAP_STATES.setdefault(name, _CapState())

    # -------------------------------------------------------------------------
    # 1) EMA grad norms
    # -------------------------------------------------------------------------
    if st.ema_g_ref is None:
        st.ema_g_ref = float(g_reference_raw)
        st.ema_g_cur = float(g_current_raw)
    else:
        st.ema_g_ref = float(ema_beta * st.ema_g_ref + (1.0 - ema_beta) * g_reference_raw)
        st.ema_g_cur = float(ema_beta * st.ema_g_cur + (1.0 - ema_beta) * g_current_raw)

    g_ref = max(st.ema_g_ref, grad_floor)
    g_cur = max(st.ema_g_cur, grad_floor)

    # -------------------------------------------------------------------------
    # 2) effective gradients and cap limit
    # -------------------------------------------------------------------------
    eff_ref = float(w_ref) * g_ref
    limit   = cap_ratio * max(eff_ref, grad_floor)

    if st.w_prev is None:
        st.w_prev = float(w_scheduled)  # initialize on first call

    eff_applied = float(st.w_prev)   * g_cur   # last step actually used
    eff_sched   = float(w_scheduled) * g_cur   # what schedule wants now

    # -------------------------------------------------------------------------
    # 3) detect violations
    # -------------------------------------------------------------------------
    applied_viol   = (eff_applied > limit)     # only this advances bind_ctr
    scheduled_viol = (eff_sched   > limit)

    # -------------------------------------------------------------------------
    # 4) compute raw target weight for this step
    #     - if schedule exceeds cap: solve w_target so eff_current == limit
    #     - else: follow schedule
    # -------------------------------------------------------------------------
    if scheduled_viol and limit > 0.0:
        w_target = float(limit / g_cur)
        w_target = min(w_target, float(w_scheduled))
    else:
        w_target = float(w_scheduled)

    # -------------------------------------------------------------------------
    # 5) emergency: only when applied is big over the limit and persistent
    #     - shrink target weight multiplicatively
    #     - optionally loosen down cap so we can realize the cut in one step
    # -------------------------------------------------------------------------
    local_ratio_cap_dn = ratio_cap_dn
    if applied_viol:
        st.bind_ctr += 1
        if st.bind_ctr >= emergency_patience and eff_applied > emergency_violation * limit:
            w_target = max(grad_floor, w_target * emergency_factor)
            if emergency_loosen_down_cap:
                # allow the per-step drop to be at least as large as emergency_factor
                local_ratio_cap_dn = min(local_ratio_cap_dn, float(emergency_factor))
    else:
        st.bind_ctr = 0

    # -------------------------------------------------------------------------
    # 6) adaptive tightening after sustained binding (faster descent)
    # -------------------------------------------------------------------------
    if st.bind_ctr >= adaptive_tighten_after:
        local_ratio_cap_dn = min(local_ratio_cap_dn, adaptive_ratio_cap_dn)

    # -------------------------------------------------------------------------
    # 7) multiplicative move from previous weight with asymmetric caps
    # -------------------------------------------------------------------------
    ratio = float(w_target / max(st.w_prev, grad_floor))

    # NaN/Inf guard
    if not (0.0 < ratio < float("inf")):
        ratio = 1.0

    if ratio >= 1.0:
        ratio = min(ratio, ratio_cap_up)      # slow up moves
    else:
        ratio = max(ratio, local_ratio_cap_dn)  # limit down moves

    w_new = float(st.w_prev * ratio)

    # -------------------------------------------------------------------------
    # 8) final safety bounds
    # -------------------------------------------------------------------------
    if w_scheduled > 0.0:
        w_new = min(w_new, 2.0 * float(w_scheduled))  # never exceed 2x schedule
    w_new = max(w_new, grad_floor)

    # -------------------------------------------------------------------------
    # 9) persist and return
    # -------------------------------------------------------------------------
    st.w_prev = w_new
    return w_new
