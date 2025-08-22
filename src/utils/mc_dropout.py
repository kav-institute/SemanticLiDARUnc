# utils/mc_dropout.py
from __future__ import annotations
import contextlib
import torch
import torch.nn as nn

# reuse your canonical detector
from models.losses import classify_output_kind

EPS = 1e-12


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
def _to_logits(out: torch.Tensor, kind: str, eps: float = EPS) -> torch.Tensor:
    """
    Convert a model forward() output to logits-like tensor.
    - 'logits'     -> as-is
    - 'probs'      -> log(p)
    - 'log_probs'  -> as-is (already log(p))
    """
    if kind == 'logits':
        return out
    elif kind == 'probs':
        return out.clamp_min(eps).log()
    elif kind == 'log_probs':
        return out
    else:
        raise ValueError(f"Unknown output kind: {kind}")


@torch.no_grad()
def mc_dropout_probs(
    model: torch.nn.Module,
    inputs,                 # whatever your model expects (list/tuple of tensors)
    T: int = 30,            # #MC samples
    temperature: float | None = None
):
    """
    MC-dropout sampling that keeps BN frozen (model.eval()) and enables only dropout layers.
    Returns a tensor of shape [T,B,C,H,W] with *probabilities*.

    Pipeline per sample:
      out = model(*inputs)            # logits OR probs OR log_probs
      kind = classify_output_kind(out)
      logits = to_logits(out, kind)   # ensure logits-like
      if temperature: logits /= T
      p = softmax(logits)
    """
    model.eval()  # keep BN frozen
    probs = []
    temp = None if temperature is None else max(1e-3, float(temperature))

    with dropout_sampling(model, enable=True):
        output_kind: str | None = None  # detect once to avoid overhead
        for _ in range(T):
            out = model(*inputs)
            assert not (isinstance(out, tuple) and len(out) >2), "Model returned/generated unexpectedly too many outputs"
            if isinstance(out, tuple) and len(out) == 2:
                logits_mean, logits_var = out
                out = logits_mean
            
            if output_kind is None:
                output_kind = classify_output_kind(out, class_dim=1)

            logits = _to_logits(out, output_kind)
            if temp is not None:
                logits = logits / temp
            p = torch.softmax(logits, dim=1)

            probs.append(p.unsqueeze(0))

    return torch.cat(probs, dim=0)  # [T,B,C,H,W]


@torch.no_grad()
def predictive_entropy_mc(mc_probs: torch.Tensor, eps: float = 1e-12, normalize: bool = True):
    """
    mc_probs: [T,B,C,H,W] (probabilities).
    Returns:
      entropy [B,H,W] (optionally normalized by log(C))
    """
    mean_p = mc_probs.mean(dim=0).clamp_min(eps)  # [B,C,H,W]
    ent = -(mean_p * torch.log(mean_p)).sum(dim=1)  # [B,H,W]
    if not normalize:
        return ent
    C = mean_p.shape[1]
    return ent / float(torch.log(torch.tensor(C, device=mean_p.device)).item())
