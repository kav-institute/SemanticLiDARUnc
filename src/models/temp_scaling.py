# models/temp_scaling.py
from __future__ import annotations
import math, os, json, tqdm, torch
from utils.mc_dropout import set_dropout_mode

@torch.no_grad()
def cache_calib_logits(model, val_loader, device, cfg,
                       ignore_index: int = 255,
                       mode: str = "default",  # {"default","mc"}
                       mc_samples: int = 30):
    """
    Cache raw logits + labels on CPU for temperature fitting.
    mode="default": single forward pass with model.eval()
    mode="mc":      average logits over `mc_samples` while enabling only dropout layers
    Returns:
      logits_all [N,C] (CPU), labels_all [N] (CPU)
    """
    model.eval()
    if mode == "mc":
        set_dropout_mode(model, True)

    all_logits, all_labels = [], []
    for range_img, reflectivity, xyz, normals, labels in tqdm.tqdm(val_loader, desc=f"TS cache ({mode})"):
        range_img, reflectivity = range_img.to(device), reflectivity.to(device)
        xyz, normals = xyz.to(device), normals.to(device)
        labels = labels.squeeze(1).long().to(device)

        # assemble inputs
        from utils.inputs import set_model_inputs
        inputs = set_model_inputs(range_img, reflectivity, xyz, normals, cfg)

        # get raw logits
        if not hasattr(model, "forward_logits"):
            raise RuntimeError("Model must expose forward_logits(...) to cache raw logits for TS.")

        if mode == "default":
            logits = model.forward_logits(*inputs)
        else:
            # mean of logits across MC passes (don't overwrite! keep the average)
            logits_sum = None
            for _ in range(mc_samples):
                l = model.forward_logits(*inputs)
                logits_sum = l if logits_sum is None else (logits_sum + l)
            logits = logits_sum / float(mc_samples)

        B, C, H, W = logits.shape
        logits2d = logits.permute(0, 2, 3, 1).reshape(-1, C)
        labels1d = labels.reshape(-1)
        valid = labels1d != ignore_index
        if valid.any():
            all_logits.append(logits2d[valid].cpu())
            all_labels.append(labels1d[valid].cpu())

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
    Optimize a single scalar T on cached tensors. (averaged to mean objective.)
    Returns the fitted T as float.
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
        num_chunks = math.ceil(N / chunk_size)
        @torch.enable_grad()
        def closure():
            opt.zero_grad(set_to_none=True)
            total_mean = 0.0
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                x = logits_cpu[i:j].to(device, non_blocking=pinned)
                y = labels_cpu[i:j].to(device, non_blocking=pinned)
                T = log_T.exp().clamp_min(1e-3)
                loss = crit(x / T, y) / N
                loss.backward()
                total_mean += float(loss.item())
                del x, y, loss
            return torch.tensor(total_mean, device=device)
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
