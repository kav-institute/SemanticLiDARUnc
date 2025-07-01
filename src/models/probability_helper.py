import torch
from torch.special import digamma
from torch.distributions import Dirichlet

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

def logits_to_DirichletDist(predicted_logits, eps=0.01):
    """
    Converts logits to Dirchlet alpha concentration parameters and distribution

    Args:
        predicted_logits (Tensor): Tensor of shape [B, C, H, W], Dirichlet parameters (alpha > 0)
        eps (float, optional): Small constant for numerical stability. Defaults to 0.01.

    Returns:
        tuple(Tensor, Tensor): 
            - dist: Dirichlet distribution of type torch.distributions.Dirichlet
            - alpha: alpha concentration parameters of shape [B, H, W, C] and
    """
    alpha = torch.nn.functional.softplus(predicted_logits)+1
    # Ensure alpha is strictly positive
    alpha = alpha + eps
    
    alpha = alpha.permute(0, 2, 3, 1)   # [B, H, W, C] for torch.distributions
    dist = Dirichlet(alpha)
    
    return dist, alpha

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

def dirichlet_confidence(alpha: torch.Tensor,
                         y_true: torch.Tensor,
                         n_samples: int = 2048):
    """
    alpha  : [..., C] Dirichlet concentration
    y_true : [...]     int labels (same prefix shape)
    returns:
        conf  : [...]   fraction of Dirichlet samples whose prob for y_true
                        is *higher* than the model's mean prob for y_true
                        (0  near mode, 1 near tail)
    """
    C = alpha.shape[-1]
    p_mean = alpha / alpha.sum(-1, keepdim=True)          # [..., C]
    p_gt   = p_mean.gather(-1, y_true.unsqueeze(-1))      # [...]

    # Monte-Carlo
    dist = torch.distributions.Dirichlet(alpha)
    samples = dist.rsample(torch.Size([n_samples]))       # [N, ..., C]
    samples_gt = samples[..., y_true]   # broadcasting over N works

    conf = (samples_gt > p_gt.unsqueeze(0)).float().mean(0)  # [...]
    return conf                                              # smaller ⇒ higher confidence