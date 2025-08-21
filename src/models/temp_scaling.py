# models/temp_scaling.py
from __future__ import annotations
import math, os, json, tqdm, torch
from utils.mc_dropout import set_dropout_mode
from models.losses import classify_output_kind  # reuse your existing helper

EPS = 1e-12

@torch.no_grad()
def _forward_to_probs(model: torch.nn.Module, inputs, kind_cache: dict | None = None):
    """
    Call model.forward(*inputs), detect output kind once (and cache),
    convert to probabilities [B,C,H,W].
    """
    out = model(*inputs)  # logits OR probs OR log_probs
    # detect kind once per model instance (cache by id)
    if kind_cache is not None:
        key = id(model)
        kind = kind_cache.get(key)
        if kind is None:
            kind = classify_output_kind(out, class_dim=1)
            kind_cache[key] = kind
    else:
        kind = classify_output_kind(out, class_dim=1)

    if kind == "logits":
        probs = torch.softmax(out, dim=1)
    elif kind == "probs":
        probs = out
    elif kind == "log_probs":
        probs = out.exp()
    else:
        raise ValueError(f"Unknown output kind: {kind}")

    return probs, kind

@torch.no_grad()
def cache_calib_logits(model, val_loader, device, cfg,
                    ignore_index: int = 255,
                    mode: str = "default",  # {"default","mc"}
                    mc_samples: int = 30):
    """
    Cache 'logits-like' + labels on CPU for temperature fitting.

    Default mode:
      - One forward pass per batch; convert to logits-like via:
          logits    -> logits
          probs     -> log(p)
          log_probs -> log(p) (already)
    MC mode:
      - Enable dropout-only; average probabilities across `mc_samples`;
        then take log(mean_p) as logits-like.
    Returns:
      logits_all [N,C] (CPU), labels_all [N] (CPU)
    """
    model.eval()
    kind_cache: dict[int, str] = {}
    if mode == "mc":
        set_dropout_mode(model, True)

    all_logits, all_labels = [], []

    for range_img, reflectivity, xyz, normals, labels in tqdm.tqdm(val_loader, desc=f"TS cache ({mode})"):
        range_img, reflectivity = range_img.to(device), reflectivity.to(device)
        xyz, normals = xyz.to(device), normals.to(device)
        # labels: [B,1,H,W] or [B,H,W] -> indices [B,H,W]
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = labels.long().to(device)

        # assemble inputs
        from utils.inputs import set_model_inputs
        inputs = set_model_inputs(range_img, reflectivity, xyz, normals, cfg)

        if mode == "default":
            # one pass -> probs -> logits_like = log(p)
            probs, _ = _forward_to_probs(model, inputs, kind_cache)
            logits_like = probs.clamp_min(EPS).log()
        else:
            # MC: average probabilities, then take log(mean_p)
            probs_sum = None
            for _ in range(mc_samples):
                p, _ = _forward_to_probs(model, inputs, kind_cache)
                probs_sum = p if probs_sum is None else (probs_sum + p)
            mean_p = (probs_sum / float(mc_samples)).clamp_min(EPS)
            logits_like = mean_p.log()

        B, C, H, W = logits_like.shape
        logits2d = logits_like.permute(0, 2, 3, 1).reshape(-1, C)
        labels1d = labels.reshape(-1)
        valid = labels1d != ignore_index
        if valid.any():
            all_logits.append(logits2d[valid].cpu())
            all_labels.append(labels1d[valid].cpu())

        # free VRAM
        del logits_like, logits2d, labels1d, valid, range_img, reflectivity, xyz, normals, labels

    if mode == "mc":
        set_dropout_mode(model, False)

    if not all_logits:
        raise ValueError("No valid pixels found in calibration loader.")
    return torch.cat(all_logits, 0), torch.cat(all_labels, 0)

def calibrate_temperature_from_cache(
    logits_cpu: torch.Tensor, labels_cpu: torch.Tensor,
    device, init_T: float | str = "auto",
    optimizer_type: str = "lbfgs", lr: float = 0.05, epochs: int = 2,
    chunk_size: int = 1_000_000, max_iter_lbfgs: int = 100,
    prev_T: float | None = None, save_path: str | None = None
) -> float:
    """
    Optimize a single scalar T on cached tensors (sum/mean objective).
    We assume cached 'logits_cpu' are log(p) (or logits) so that
    softmax(logits_cpu / T) yields calibrated probabilities.
    """
    if labels_cpu.dtype != torch.long:
        labels_cpu = labels_cpu.long()

    start_T = (prev_T if (init_T == "auto" and prev_T is not None) else
               (float(init_T) if init_T != "auto" else 1.0))
    log_T = torch.nn.Parameter(torch.log(torch.tensor([start_T], device=device)))
    N = labels_cpu.numel()
    crit = torch.nn.CrossEntropyLoss(reduction="sum")

    # optional pinned memory
    try:
        logits_cpu = logits_cpu.pin_memory()
        labels_cpu = labels_cpu.pin_memory()
        pinned = True
    except Exception:
        pinned = False

    if optimizer_type.lower() == "lbfgs":
        opt = torch.optim.LBFGS(
            [log_T], lr=1.0, max_iter=max_iter_lbfgs,
            line_search_fn="strong_wolfe", history_size=100,
            tolerance_grad=1e-10, tolerance_change=1e-12
        )
        @torch.enable_grad()
        def closure():
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                x = logits_cpu[i:j].to(device, non_blocking=pinned)
                y = labels_cpu[i:j].to(device, non_blocking=pinned)
                T = log_T.exp().clamp_min(1e-3)
                loss = crit(x / T, y) / N
                loss.backward()
                total += float(loss.item())
                del x, y, loss
            return torch.tensor(total, device=device)
        opt.step(closure)
    else:
        opt = torch.optim.Adam([log_T], lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                idx = perm[i:j]
                x = logits_cpu[idx].to(device, non_blocking=pinned)
                y = labels_cpu[idx].to(device, non_blocking=pinned)
                T = log_T.exp().clamp_min(1e-3)
                loss = crit(x / T, y) / (j - i)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                del x, y, loss

    T_value = float(log_T.exp().item())
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"temperature": T_value}, f)
    return T_value
