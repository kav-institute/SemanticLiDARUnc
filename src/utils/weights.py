# utils/weights.py
from __future__ import annotations
import os, time, torch
from typing import Tuple, Dict, Any

def _maybe_state_dict(obj: Any) -> Dict[str, torch.Tensor] | None:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    return None

def _strip_prefix(sd: dict, prefix: str) -> dict:
    if not sd:
        return sd
    plen = len(prefix)
    return { (k[plen:] if k.startswith(prefix) else k): v for k, v in sd.items() }

@torch.no_grad()
def load_pretrained_safely(
    model: torch.nn.Module,
    ckpt_path: str | None,
    device: torch.device | str = "cpu",
    *,
    ignore_keys_with: Tuple[str, ...] = ("logits.", "classifier.", "head."),
    copy_head_overlap: bool = False,   # copy first min(C_old, C_new) rows for heads
    verbose: bool = True,
) -> dict:
    """
    Loads as many params as possible from `ckpt_path` into `model`.
    - Keeps matching-shape params
    - Skips mismatches and any key containing substrings in `ignore_keys_with`
    - Robust to DDP prefixes ('module.' / 'model.') and checkpoints saved as dict(state_dict=...)
    Returns a small report dict.
    """
    report = {"ok": False, "loaded": 0, "ignored": 0, "mismatched": 0, "unexpected": 0, "path": ckpt_path}

    try:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            if verbose:
                print(f"[weights] No pretrained file at: {ckpt_path}")
            return report

        # Try safer torch.load first (newer PyTorch); fallback to legacy signature
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)  # keep broad compat
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)

        sd = _maybe_state_dict(ckpt)
        if sd is None:
            if verbose:
                print(f"[weights] Checkpoint does not contain a state_dict: {ckpt_path}")
            return report

        # Consider raw, stripped 'module.' and 'model.' variants—pick the best matching mapping
        cands = [sd, _strip_prefix(sd, "module."), _strip_prefix(sd, "model.")]
        model_sd = model.state_dict()
        best = max(cands, key=lambda d: sum(1 for k in d.keys() if k in model_sd))

        to_load = {}
        ignored, mismatched, unexpected = [], [], []

        for k, v in best.items():
            if any(substr in k for substr in ignore_keys_with):
                ignored.append(k)
                continue
            if k not in model_sd:
                unexpected.append(k)
                continue

            tgt = model_sd[k]
            if v.shape == tgt.shape:
                to_load[k] = v
            else:
                # optional: copy overlapping rows for classifier-like tensors (dim 0 differs only)
                if copy_head_overlap and len(v.shape) == len(tgt.shape) and v.shape[1:] == tgt.shape[1:]:
                    c = min(v.shape[0], tgt.shape[0])
                    if c > 0:
                        new_t = tgt.clone()
                        new_t[:c] = v[:c].to(new_t.dtype)
                        to_load[k] = new_t
                    else:
                        mismatched.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(tgt.shape)}")
                else:
                    mismatched.append(f"{k}: ckpt{tuple(v.shape)} vs model{tuple(tgt.shape)}")

        # Load the compatible subset
        missing_before = [k for k in model_sd.keys() if k not in to_load]
        model.load_state_dict(to_load, strict=False)

        report.update({
            "ok": True,
            "loaded": len(to_load),
            "ignored": len(ignored),
            "mismatched": len(mismatched),
            "unexpected": len(unexpected),
            "missing_after": len(missing_before),
        })
        if verbose:
            print(f"[weights] loaded={report['loaded']} ignored={report['ignored']} "
                  f"mismatched={report['mismatched']} unexpected={report['unexpected']}")
            if mismatched:
                print("[weights] mismatches:")
                for m in mismatched[:20]:
                    print("  -", m)
                if len(mismatched) > 20:
                    print(f"  ... and {len(mismatched)-20} more")
        return report

    except Exception as e:
        if verbose:
            print(f"[weights] Failed to load pretrained: {e}")
            print("No pretrained weights applied. Training from scratch...")
        time.sleep(1)
        return report