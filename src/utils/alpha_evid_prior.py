import mpmath as mp
mp.mp.dps = 80

def coverage(alpha0, p_star, h):
    a = p_star * alpha0
    b = (1 - p_star) * alpha0
    # guard bounds to [0,1]
    lo = max(0.0, p_star - h)
    hi = min(1.0, p_star + h)
    cdf_hi = mp.betainc(a, b, 0, hi, regularized=True)
    cdf_lo = mp.betainc(a, b, 0, lo, regularized=True)
    return cdf_hi - cdf_lo

def solve_alpha0_for_coverage(p_star, h, delta, K=20):
    target = 1 - 2*delta
    lo, hi = 5.0, 1000.0
    cov_lo = coverage(lo, p_star, h)
    cov_hi = coverage(hi, p_star, h)
    # expand upper bound if needed
    while cov_hi < target and hi < 1e6:
        hi *= 2.0
        cov_hi = coverage(hi, p_star, h)
    # reduce lower if oddly too high
    while cov_lo > target and lo > 1.0:
        lo /= 2.0
        cov_lo = coverage(lo, p_star, h)
    # bisection
    for _ in range(200):
        mid = 0.5*(lo + hi)
        cov_mid = coverage(mid, p_star, h)
        if cov_mid >= target:
            hi = mid
        else:
            lo = mid
        if abs(cov_mid - target) < 1e-8 and (hi - lo) < 1e-6:
            break
    alpha0 = 0.5*(lo + hi)
    return alpha0, alpha0 / K

def alpha0_from_variance(p_star, v_star, K=20):
    a0 = (p_star*(1-p_star))/v_star - 1.0
    return a0, a0 / K

import math
def logit_threshold_for_alpha_cap(s_total, K, m=3, margin=0.10, T=1.0):
    """
    Compute a hinge threshold on logits (z_thr) so that, if at most m classes are
    "active" (carry most of the evidence) and all remaining K-m classes stay at
    the base level alpha ≈ 1, then the total evidence alpha0 does not exceed
    s_total*(1 + margin).
    
    Assumptions:            
      - Mapping to Dirichlet: alpha_k = 1 + softplus(z_k / T).  Base level ~1.
      - At a pixel, at most m classes are allowed to take on a large alpha.
      - The other (K-m) classes sit near alpha ≈ 1 (i.e., logits negative/low).
      - You want an upper cap on total evidence:
            alpha0 = sum_k alpha_k  <=  s_total * (1 + margin)  := s_hi
    Derivation:
      With m active classes at per-class level a_thr and K-m inactive at ~1,
      we require:  m * a_thr + (K - m) * 1  <=  s_hi
      Solve for a_thr:  a_thr = (s_hi - (K - m)) / m

      Convert that per-class alpha bound into a logit threshold by inverting
      the softplus:
          y = a_thr - 1
          softplus^{-1}(y) = log( exp(y) - 1 )
      Then scale by T:
          z_thr = T * log( exp(a_thr - 1) - 1 )

    Notes:
      - a_thr must be > 1 because alpha_k = 1 + softplus(...) >= 1.  We clamp to
        at least 1.001 to avoid numerical issues with expm1(0).
      - If z > z_thr, the hinge penalty should kick in (if you use it that way).
      - Large T -> larger z needed to reach the same alpha; hence we multiply by T.

    Returns
    -------
    z_thr : float
        Logit threshold (pre-softplus) to keep alpha0 under the cap under the
        m-active-classes approximation.
    a_thr : float
        The per-class alpha value implied by that threshold.
    """
    s_hi = s_total * (1.0 + margin)
    a_thr = (s_hi - (K - m)) / m  # per-class alpha cap at the threshold
    a_thr = max(a_thr, 1.001)     # must be >1 because alpha_i = 1 + softplus(...)
    z_thr = T * math.log(math.expm1(a_thr - 1.0))  # softplus^{-1}(a_thr-1)
    return z_thr, a_thr


if __name__ ==  "__main__":
    K = 20
    scenarios = [
        ("A", 0.90, 0.05, 0.025),
        ("B", 0.80, 0.06, 0.025),
    ]

    quantile_res = []
    for tag, p_star, h, delta in scenarios:
        a0, sper = solve_alpha0_for_coverage(p_star, h, delta, K=K)
        quantile_res.append((tag, p_star, h, a0, sper))

    variance_res = []
    for tag, p_star, h, delta in scenarios:
        v_star = 0.05**2  # std=0.05
        a0, sper = alpha0_from_variance(p_star, v_star, K=K)
        variance_res.append((tag, p_star, v_star, a0, sper))

    print("Quantile-matching (exact coverage), K=20:")
    for tag, p_star, h, a0, sper in quantile_res:
        print(f"[{tag}] p*={p_star:.2f}, half-width={h:.3f} -> alpha0≈{a0:.3f}, s_per_class≈{sper:.3f}")

    print("\nVariance-matching (exact Var), K=20, std=0.05:")
    for tag, p_star, v_star, a0, sper in variance_res:
        print(f"[{tag}] p*={p_star:.2f}, var={v_star:.4f} -> alpha0≈{a0:.3f}, s_per_class≈{sper:.3f}")
    
    a0, a0_per_class = solve_alpha0_for_coverage(p_star=0.90, h=0.05, delta=0.025, K=20)
    logit_threshold_for_alpha_cap(a0, K=20, m=3, margin=0.10, T=1.0)