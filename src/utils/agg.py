from __future__ import annotations
from functools import wraps
from typing import Optional
import torch

def mean_aggregator():
    """
    Decorator that attaches O(1) mean-accumulation helpers to a function.

    The wrapped function `fn` keeps:
      - fn.add(x, mask=None): add a tensor/value to the running mean (optionally masked).
      - fn.accumulate(*args, mask=None, **kwargs): compute = fn(...), then add it.
      - fn.mean(reset=False): return current mean (sum/count); optionally reset.
      - fn.reset(): reset sum/count.
      - fn.sync_ddp(): (optional) all_reduce(sum,count) across distributed ranks.

    Notes:
      - If `x` is a tensor, we sum all elements (or only where mask==True).
      - If `x` is a scalar/float, we treat it as one item with count=1.
      - Safe with AMP: we upcast to float32 before summing.
    """
    def decorator(fn):
        stats = {"sum": 0.0, "count": 0}

        @wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        def _to_sum_and_count(x: torch.Tensor, mask: Optional[torch.Tensor]):
            if mask is not None:
                # ensure boolean mask and same shape (broadcast if needed)
                if not torch.is_tensor(mask):
                    raise TypeError("mask must be a torch.Tensor or None")
                m = mask
                # try to broadcast mask to x
                if m.shape != x.shape:
                    m = torch.broadcast_to(m, x.shape)
                s = x[m].float().sum().item()
                c = int(m.sum().item())
            else:
                s = x.float().sum().item()
                c = x.numel()
            return s, c

        def add(x, mask: Optional[torch.Tensor] = None):
            """Add a tensor (optionally masked) or a scalar to the running mean."""
            if torch.is_tensor(x):
                s, c = _to_sum_and_count(x.detach(), mask)
            else:
                # scalar value -> count = 1
                s, c = float(x), 1
            stats["sum"] += s
            stats["count"] += c

        def accumulate(*args, mask: Optional[torch.Tensor] = None, **kwargs):
            """
            Call fn(*args, **kwargs), then add its output.
            Returns the function's output so you can still use it downstream.
            """
            out = fn(*args, **kwargs)
            add(out, mask=mask)
            return out

        def mean(reset: bool = False) -> float:
            m = stats["sum"] / max(1, stats["count"])
            if reset:
                stats["sum"] = 0.0
                stats["count"] = 0
            return m

        def reset():
            stats["sum"] = 0.0
            stats["count"] = 0

        def sync_ddp():
            """Optionally aggregate mean across DDP ranks."""
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor([stats["sum"], float(stats["count"])], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            stats["sum"] = float(t[0].item())
            stats["count"] = int(t[1].item())

        wrapped.add = add
        wrapped.accumulate = accumulate
        wrapped.mean = mean
        wrapped.reset = reset
        wrapped.sync_ddp = sync_ddp
        return wrapped
    return decorator