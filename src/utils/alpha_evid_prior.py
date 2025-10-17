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