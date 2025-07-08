import scipy.special
import torch
from torch.special import digamma
import matplotlib.pyplot as plt
import scipy
import numpy as np

def smooth_one_hot(targets, num_classes, smoothing=0.1):
    """
    Smooths one-hot encoded target labels for semantic segmentation.

    Args:
        targets: LongTensor of shape [B, 1, H, W], class indices
        num_classes: int, number of classes
        smoothing: float, amount of label smoothing (0.0 = no smoothing)

    Returns:
        FloatTensor of shape [B, C, H, W], smoothed one-hot labels
    """
    confidence = 1.0 - smoothing
    low_conf = smoothing / (num_classes - 1)

    B, _, H, W = targets.shape

    # Initialize full tensor with low confidence
    one_hot = torch.full(
        (B, num_classes, H, W),
        fill_value=low_conf,
        device=targets.device,
        dtype=torch.float
    )

    # Squeeze channel dimension to get [B, H, W]
    targets = targets.squeeze(1)

    # Scatter high confidence to correct class
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    #one_hot = torch.nn.functional.softmax(one_hot, dim=1)
    return one_hot

def to_alpha_concentrations(predicted_logits: torch.Tensor):
    """Converts model's output Tensor to alpha concentrations (alpha > 0)

    Args:
        predicted_logits (torch.Tensor): output Tensor of the model with shape [B, C, H, W]

    Returns:
        torch.Tensor: strictly positive Tensor with Dirchlet + 1 adjustment
    """
    return torch.nn.functional.softplus(predicted_logits)+1

def alphas_to_Dirichlet(alpha: torch.Tensor):
    """Build Dirichlet distribution from alpha concentration parameters

    Args:
        alpha (torch.Tensor): strictly positive Dirichlet parameter Tensor of shape [B, C, H, W]
    
    Returns:
        torch.distributions.Dirichlet
    """
    
    # permute shape from [B, C, H, W] -> [B, H, W, C] which is expected by torch.distributions
    alpha = alpha.permute(0, 2, 3, 1)
    return torch.distributions.Dirichlet(alpha)

def get_predictive_entropy(alpha, eps=1e-10):
    """
    Computes predictive entropy H(E[p]) from Dirichlet parameters.
    H(E[p]) = -∑_j (α_j/α₀) · log(α_j/α₀)
    
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], entropy of expected class probabilities
    """
    
    # α₀ = ∑_j α_j
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True) + eps               # Total concentration alpha_0
    
    # E[p_j] = α_j / α₀
    Exp_prob = alpha / alpha_0                                          # Expected class probabilities
    
    # H(E[p]) = -∑_j E[p_j] · log(E[p_j])
    entropy = -torch.sum(Exp_prob * torch.log(Exp_prob + eps), dim=1)   # Entropy across classes
    return entropy

def get_aleatoric_uncertainty(alpha, eps=1e-10):
    """
    Computes aleatoric (data) uncertainty:
    E_{p∼Dir(α)} [ H(P(y|p)) ]
        = -∑_j (α_j/α₀) · [ ψ(α_j + 1) − ψ(α₀ + 1) ]
    
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], aleatoric uncertainty of expected class probabilities
    """
    
    # α₀ = ∑_j α_j
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True) + eps               # Total concentration alpha_0
    
    # ψ-term: ψ(α_j + 1) − ψ(α₀ + 1)
    term = digamma(alpha + 1) - digamma(alpha_0 + 1)                    # digamma function
    
    # E[p_j] = α_j / α₀
    Exp_prob = alpha / alpha_0                                          # Expected class probabilities
    
    # aleatoric = -∑_j E[p_j] · ψ_diff_j
    au = -torch.sum(Exp_prob * term, dim=1)                             # Aleatoric uncertainty
    return au

def get_epistemic_uncertainty(alpha, eps=1e-10):
    """
    Computes epistemic uncertainty 
    I = H(E[p]) − E[H(P(y|p))]
    
    From paper: Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation (Tsiligkaridis)
    https://arxiv.org/abs/1910.04819 
    
    Args:
        alpha: Tensor of shape [B, C, H, W], Dirichlet parameters per class

    Returns:
        Tensor of shape [B, H, W], epistemic uncertainty of expected class probabilities
    """
    
    # epistemic = total predictive entropy − aleatoric uncertainty
    eu = get_predictive_entropy(alpha) - get_aleatoric_uncertainty(alpha)
    return eu

def mc_confidence_analytic(alpha: torch.Tensor,
                        y_true: torch.Tensor,
                        M: int = 64,
                        eps=1e-10):
    """
    alpha  : Tensor of shape [B, C, H, W], Dirichlet parameters per class
    y_true : Tensor of shape [B, 1, H, W], int labels
    M: number of Monte-Carla samples used for sampling Dirichlet distribution
    returns:
        conf  : Tensor of shape [B, 1, H, W]   
                fraction of Dirichlet samples whose prob for y_true
                is higher than the model's mean prob for y_true
                (0 near mode, 1 near tail)
    """
    
    # α₀ = ∑_j α_j, shape [B, C, H, W]
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True) + eps
    
    # Extract α_y, the concentration for the true class:
    alpha_y = torch.gather(alpha, dim=1, index=y_true)     # [B,1,H,W]
    
    # # Analytic mean per class: E[p_j] = α_j / α₀, shape [B, C, H, W]
    # Exp_prob = alpha/alpha_0
    
    # build Dirichlet distribution from alpha concentration parameters
    dist = alphas_to_Dirichlet(alpha)
    
    # Monte-Carlo Sampling of Dirichlet Distribution, shape [M, B, H, W, C]
    samples = dist.sample([M])
    samples = samples.permute(0, 1, 4, 2, 3)    # reverts order [M, B, H, W, C] -> [M, B, C, H, W]
    
    # Extract analytic expected (mean) probability for the true class, shape [B, 1, H, W]
    p_hat_y = alpha_y / alpha_0    
    #p_hat_y = torch.gather(Exp_prob, dim=-1, index=y_true)
    
    # Expand y_true so it lines up with samples' shape (samples: [n_samples, B, C, H, W])   
    y_expand = y_true.unsqueeze(0)                  # -> [1, B, 1, H, W]
    y_expand = y_expand.expand(M, -1, -1, -1, -1)   # -> [M, B, 1, H, W]

    # Gather the sampled probabilities for the true class
    samples_gt = samples.gather(dim=2, index=y_expand)  # -> [M, B, 1, H, W]
    
    # Compute the fraction "overshooting" the mean
    # A well-calibrated model should satisfy p(p_y > E[p_y])=1-E[p_y].
    # Example: If on average the model predicts 0.8 for the true class, we would expect only 20% of your samples to exceed that.
    conf = (samples_gt > p_hat_y.unsqueeze(0)).float().mean(0)
    return conf

def beta_confidence_analytic(alpha: torch.Tensor,
                            y_true: torch.Tensor,
                            eps: float = 1e-10) -> torch.Tensor:
    """
    Args:
        alpha  : [B,C,H,W] Dirichlet parameters
        y_true : [B,1,H,W] integer labels
    returns:
        conf   : [B,1,H,W] = P( p_y > E[p_y] ) exactly, no MC
    """
    # Total concentration α₀ = ∑_j α_j, shape [B,1,H,W]
    alpha_0 = alpha.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,H,W]

    # Extract α_y, the concentration for the true class:
    alpha_y = torch.gather(alpha, dim=1, index=y_true)     # [B,1,H,W]
    
    # Extract analytic expected (mean) probability for the true class
    p_hat_y = alpha_y / alpha_0                               # [B,1,H,W]
    
    # Define the second parameter b in the marginal Beta(a=α_y, b=α₀-α_y) function, where all other concentration mass is lumped together.
    # Because Dirichlet has Beta marginals for each coordinate p_j, we can reduce the calibration of p_y to a 1D-problem using the Beta marginals
    beta_b  = (alpha_0 - alpha_y).clamp_min(eps)                            # [B,1,H,W]

    # Prepare for calling the regularized incomplete Beta (the CDF)
    a = alpha_y.squeeze(1).cpu().numpy()   # [B,H,W]
    b = beta_b.squeeze(1).cpu().numpy()   # [B,H,W]
    x = p_hat_y.squeeze(1).cpu().numpy()    # [B,H,W]
    
    # regularized incomplete Beta function, i.e., calls closed-form CDF of Beta(a,b) at the point x (at the mean) directly,
    # without explicitely constructing the distribution (Dirichlet or Beta) itself.
    cdf_at_mean = scipy.special.betainc(a, b, x)
    
    # tail-area above the mean: 1-F_Beta(x;p̂_y, α₀-αy)=p(p>p̂)
    conf = (1.0 - cdf_at_mean)[:, None, ...]                 # [B,1,H,W]
    
    return conf                           # [B,1,H,W]

def reliability_dirichlet(alpha, y_true, coverages, n_samples: int=64, device="cpu"):
    # assure tensors are on same desired device
    alpha = alpha.to(device)
    y_true = y_true.detach().to(device)
    
    conf = mc_confidence_analytic(alpha, y_true, n_samples)
    #conf = beta_confidence_analytic(alpha, y_true, n_samples)
    if not isinstance(conf, torch.Tensor):
        conf = torch.tensor(conf)
    
    conf = conf.flatten()                                   # vector
    bins = torch.as_tensor(coverages, dtype=conf.dtype, device=conf.device)

    # empirical CDF F_hat(k) = p(conf <= k)
    F_hat = torch.stack([(conf <= k).float().mean() for k in bins])  # (K,)

    # ideal CDF is the 45° line -> reliability error
    rel_error = torch.abs(F_hat - bins)
    
    return {
        "avg_RLS": (1 - rel_error.mean()).item() * 100,
        "min_RLS": (1 - rel_error.max()).item()  * 100,
        "F_hat"  : F_hat.cpu().numpy(),           # for plotting if you like
        "bins"   : bins.cpu().numpy()
    }

def compute_ece_and_reliability(alpha: torch.Tensor,
                                y_true: torch.LongTensor,
                                n_bins: int = 10,
                                eps: float = 1e-10):
    """
    Args:
        alpha  : Tensor of shape [B, C, H, W], Dirichlet parameters per class
        y_true : Tensor of shape [B, 1, H, W], int labels
    Returns:
        confs, accs, ece
    """
    B, C, H, W = alpha.shape
    N = B * H * W

    ## Total concentration α_0 = ∑_j α_j
    alpha_0  = alpha.sum(dim=1, keepdim=True).clamp_min(eps)    # [B,1,H,W]
    
    ## Analytic mean per class: E[p_j] = α_j / α_0
    mean_probs = alpha / alpha_0    # [B,C,H,W]
    
    ## get max probability per sample (i.e., confidence for its choosen class) and get predicted label (positional)
    pred_conf_max, pred_labels = mean_probs.max(dim=1, keepdim=True)  # [B,1,H,W]
    # account for rounding errors to make sure conf is inside [0,1] and flattened
    pred_conf_max = pred_conf_max.flatten().clamp(min=eps, max=1 - eps) # [N]
    # correctness of the predictions
    ## Get floating-point boolean list of correct/false predictions
    correct = (pred_labels == y_true).float().flatten() # [N]

    # bin assignments
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins+1, device=alpha.device, dtype=alpha.dtype)    # np.linspace(0,1,n_bins+1)
    ## bucketize returns indices in [1...n_bins], subtract 1 -> [0...n_bins-1]
    bin_ids      = torch.bucketize(pred_conf_max, bin_edges, right=False) - 1  # [N]
    #bin_centers  = ((bin_edges[:-1] + bin_edges[1:]) / 2).cpu().numpy()           # [n_bins]
    ## assign mean probability for the choosen class to one of the N discrete bins
    #bin_ids = (pred_conf_max * n_bins).long()   # values in [0..n_bins-1]

    # for each bin, sum up confidences and correctness
    ## torch.bincount allows us to aggregate without loops
    counts       = torch.bincount(bin_ids, minlength=n_bins).float()      # [n_bins]
    sum_conf     = torch.bincount(bin_ids, weights=pred_conf_max, minlength=n_bins)
    sum_correct  = torch.bincount(bin_ids, weights=correct,  minlength=n_bins)
    ## avoid divide-by-zero but define set no counts to fill with NaNs for visualization purpose
    nonzero = counts > 0
    confs   = torch.full((n_bins,), fill_value=float("nan"), dtype=alpha.dtype, device=alpha.device)
    accs    = torch.full((n_bins,), fill_value=float("nan"), dtype=alpha.dtype, device=alpha.device)
    confs[nonzero] = sum_conf[nonzero] / counts[nonzero]    # bin-wise mean confidence
    accs[nonzero]  = sum_correct[nonzero] / counts[nonzero] # bin-wise accuracy
    # -> plotting accs against confs, example: when I say 80% confidence, I really am correct 80% of the time.
    
    # Expected Calibration Error
    # ECE = sum_k (n_k/N) * |accs[k] - confs[k]|
    weights = counts / N
    ece     = torch.nansum((weights * torch.abs(accs - confs)), dim=0).item()
    
    return np.asarray(confs.cpu()), np.asarray(accs.cpu()), ece

def sample_accuracy_check(alpha: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10, n_samples: int= 64, eps: float=1e-10):
    """MC-Sampling from Dirichlet distribution and check empirical frequency of argmax to ground truth class.

    Args:
        alpha (torch.Tensor): Dirichlet parameters per class, shape [B, C, H, W]
        y_true (torch.Tensor): Integer labels of ground truth class, shape [B, 1, H, W]
        n_samples (int): number of MC-samples
    Returns:
        counts
    """
    B, C, H, W = alpha.shape
    N = B * H * W
    # torch.cuda.mem_get_info()
    dist = alphas_to_Dirichlet(alpha)
    samples = dist.sample((n_samples,))                 # [M, B, H, W, C]
    max_samples = torch.argmax(samples, dim=-1)         # [M, B, H, W]
    
    # Expand y_true so it lines up with MC-samples' shape [M, B, H, W]
    y_expand = y_true.squeeze(1).unsqueeze(0)           # [1, B, H, W]
    y_expand = y_expand.expand(n_samples, -1, -1, -1)   # [M, B, H, W]

    # get per sample/pixel empirical accuracy over all MC-samples, make sure to be within ]0,1[ for use of bucketize
    correct_emp = (max_samples == y_expand)             # [M, B, H, W], bool
    correct_emp = correct_emp.float().mean(0)           # [B, H, W], fraction of M samples where argmax==y_true
    correct_emp = correct_emp.flatten().clamp(min=eps, max=1-eps)   # [N=B*H*W], and clamp 
    
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins+1, device=alpha.device, dtype=alpha.dtype)    # np.linspace(0,1,n_bins+1)
    # for each sample (pixel) assign empirical accuracy to its bin: bucketize -> indices in [1...n_bins], subtract 1 -> [0..n_bins-1]
    bin_ids = torch.bucketize(correct_emp, bin_edges, right=False) - 1  # [N]
    # count members in each bin
    counts = torch.bincount(bin_ids, minlength=n_bins).float()      # [n_bins]
    
    return counts.cpu().numpy()
    # observed frequency
    emp_freq = counts / N #counts.sum()
    
    bin_centers  = ((bin_edges[:-1] + bin_edges[1:]) / 2)           # [n_bins]

    return emp_freq.cpu().numpy(), bin_centers.cpu().numpy()

def save_empirical_reliability_plot(total_emp_freq: np.ndarray, bin_centers: np.ndarray, output_path: str="reliability_plot.png"):
    # get per-bin mean
    total_emp_freq_np = total_emp_freq.mean(axis=0)
    total_emp_freq_perc_np = total_emp_freq_np / total_emp_freq_np.sum()
    
    # Scale the dot sizes so they're visible:
    area_scale = 1000  # tweak this so the biggest dot is a nice size
    sizes = total_emp_freq_np / total_emp_freq_np.max() * area_scale

    plt.figure(figsize=(6,6))
    # 1) Plot the ideal calibration diagonal
    plt.plot([0,1],[0,1], '--', color='gray', label='Ideal Calibration')

    # 2) Scatter: x=bin_centers, y=empirical freq, s ~ counts
    plt.scatter(
        bin_centers,
        total_emp_freq_perc_np,
        s=sizes,
        c='C1',
        alpha=0.6,
        edgecolor='k',
        label='Empirical Frequency (dot size ~ bin count)'
    )

    # 3) Optionally fill under the curve to show "resolution"
    plt.fill_between(
        bin_centers,
        total_emp_freq_perc_np,
        0,
        color='C1',
        alpha=0.2
    )

    plt.xlabel('Forecast Probability (bin center)')
    plt.ylabel('Observed Frequency')
    plt.title('Reliability Diagram')
    plt.legend(loc='upper left', markerscale=0.2)
    plt.grid(True)
    plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# # get forcast probability
    # ## Total concentration α_0 = ∑_j α_j
    # alpha_0  = alpha.sum(dim=1, keepdim=True).clamp_min(eps)    # [B,1,H,W]
    
    # ## Analytic mean per class: E[p_j] = α_j / α_0
    # mean_probs = alpha / alpha_0    # [B,C,H,W]
    
    # ## get max probability per sample (i.e., confidence for its choosen class) and get predicted label (positional)
    # pred_conf_max, pred_labels = mean_probs.max(dim=1, keepdim=True)  # [B,1,H,W]
    # # account for rounding errors to make sure conf is inside ]0,1[ and flattened
    # pred_conf_max = pred_conf_max.flatten().clamp(min=eps, max=1 - eps) # [N]
    # pred_conf_bin_ids = torch.bucketize(pred_conf_max, bin_edges, right=False) - 1  # [N]
    # sum_conf = torch.bincount(pred_conf_bin_ids, weights=pred_conf_max, minlength=n_bins)
    # # count members in each bin
    # #counts_sum_conf = torch.bincount(pred_conf_bin_ids, minlength=n_bins).float()      # [n_bins]
    # confs   = torch.full((n_bins,), fill_value=float('nan'), dtype=alpha.dtype, device=alpha.device)
    # confs[sum_conf>0] = sum_conf[sum_conf>0] / counts[sum_conf>0]    # bin-wise mean confidence

    
    # pred_conf_ids = (pred_conf_max * n_bins).floor().long().clamp(0, n_bins-1)  # [N]

    # sum_conf     = torch.bincount(bin_ids, weights=pred_conf_max, minlength=n_bins)



# bin_centers  = ((bin_edges[:-1] + bin_edges[1:]) / 2).cpu().numpy()           # [n_bins]

    # # avoid divide-by-zero but define set no counts to fill with NaNs for visualization purpose
    # nonzero = counts > 0

    # # ## Extract α_y, the concentration for the true class:
    # # alpha_y = torch.gather(alpha, dim=1, index=y_true)  # [B,1,H,W]
    # # ## Extract analytic expected (mean) probability for the true class
    # # p_y   = (alpha_y / alpha_0).flatten() # [N]

    # # 2) correctness of the MAP prediction
    # ## Extract most likely prediction
    # pred_labels = mean_probs.argmax(dim=1, keepdim=True)    # [B,1,H,W]
    # ## Get floating-point boolean list of correct/false predictions
    
    
# OLD CODE:
# TODO: clean up

# def aleatoric_uncertainty(alpha, eps=1e-10):
#     """
#     Approximates aleatoric uncertainty (expected entropy) from Dirichlet parameters.

#     Args:
#         alpha: Tensor of shape [B, C, H, W]

#     Returns:
#         Tensor of shape [B, H, W]
#     """
#     alpha0 = torch.sum(alpha, dim=1, keepdim=True) + eps
#     term1 = digamma(alpha0 + 1)
#     term2 = torch.sum((alpha * digamma(alpha + 1)), dim=1, keepdim=True) / alpha0
#     expected_entropy = term1 - term2
#     return expected_entropy.squeeze(1)

# def logits_to_DirichletDist(predicted_logits: torch.Tensor,
#                             is_alpha: bool=False,
#                             eps: float=1e-10):
#     """
#     Converts logits to Dirchlet alpha concentration parameters and distribution

#     Args:
#         predicted_logits (Tensor): Tensor of shape [B, C, H, W], Dirichlet parameters (alpha > 0)
#         is_alpha (bool): bool indicating if already transfromed to Dirichlet alpha parameters
#         eps (float, optional): Small constant for numerical stability. Defaults to 0.01.

#     Returns:
#         tuple(Tensor, Tensor): 
#             - dist: Dirichlet distribution of type torch.distributions.Dirichlet
#             - alpha: alpha concentration parameters of shape [B, H, W, C] and
#     """
#     if is_alpha:
#         alpha = predicted_logits
#     else:
#         alpha = torch.nn.functional.softplus(predicted_logits)+1
    
#     alpha = alpha.permute(0, 2, 3, 1)   # [B, H, W, C] for torch.distributions
#     dist = Dirichlet(alpha)
    
#     return dist, alpha