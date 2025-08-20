# utils/mc_dropout.py
from __future__ import annotations
import contextlib
import torch
import torch.nn as nn

def set_dropout_mode(module: torch.nn.Module, train: bool) -> None:
    """
    Enable/disable *only* dropout layers; leave everything else (e.g., BatchNorm) unchanged.
    """
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                          nn.AlphaDropout, nn.FeatureAlphaDropout)):
            m.train(train)

@contextlib.contextmanager
def dropout_sampling(module: torch.nn.Module, enable: bool = True):
    """
    Context manager that flips dropout layers to train() for sampling and restores them after.
    """
    if enable:
        set_dropout_mode(module, True)
    try:
        yield
    finally:
        if enable:
            set_dropout_mode(module, False)

@torch.no_grad()
def mc_dropout_probs(
    model: torch.nn.Module,
    inputs,                 # whatever your model expects (list/tuple of tensors)
    T: int = 30,            # #MC samples
    temperature: float | None = None
):
    """
    MC-dropout sampling that keeps BN frozen (model.eval()) and enables only dropout layers.
    Returns a tensor of shape [T,B,C,H,W] with probabilities.
    """
    model.eval()  # keep BN frozen
    with dropout_sampling(model, enable=True):
        probs = []
        for _ in range(T):
            if hasattr(model, "forward_logits"):
                logits = model.forward_logits(*inputs)
                if temperature is not None:
                    logits = logits / max(1e-3, float(temperature))
                p = torch.softmax(logits, dim=1)
            else:
                # model already returns probs
                p = model(*inputs)
                if temperature is not None:
                    # apply T via logits ~= log(p)
                    p = p.clamp_min(1e-12)
                    logits = torch.log(p) / max(1e-3, float(temperature))
                    p = torch.softmax(logits, dim=1)
            probs.append(p.unsqueeze(0))
        return torch.cat(probs, dim=0)  # [T,B,C,H,W]

@torch.no_grad()
def predictive_entropy_mc(mc_probs: torch.Tensor, eps: float = 1e-12, normalize: bool = True):
    """
    mc_probs: [T,B,C,H,W] (probabilities).
    """
    mean_p = mc_probs.mean(dim=0).clamp_min(eps)  # [B,C,H,W]
    ent = -(mean_p * torch.log(mean_p)).sum(dim=1)  # [B,H,W]
    if not normalize:
        return ent
    C = mean_p.shape[1]
    return ent / float(torch.log(torch.tensor(C)).item())
