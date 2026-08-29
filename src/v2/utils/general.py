"""Misc helpers: seeding, logging, timing, model stats."""
import os
import random
import time
import math
import contextlib
from pathlib import Path

import numpy as np
import torch


COLORS = {"black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
          "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
          "bright_black": "\033[90m", "bold": "\033[1m", "underline": "\033[4m", "end": "\033[0m"}


def colorstr(*args):
    *styles, string = args if len(args) > 1 else ("blue", "bold", args[0])
    return "".join(COLORS[s] for s in styles) + str(string) + COLORS["end"]


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True is a large speedup for fixed-size inputs; we re-enable
        # it explicitly because determinism costs ~30% throughput here.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        for n in range(2, 9999):
            p = Path(f"{path}{sep}{n}")
            if not p.exists():
                path = p
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def model_info(model, imgsz=640, device="cpu", verbose=True):
    """Return (n_params, n_gradients, GFLOPs, size_MB)."""
    n_p = sum(p.numel() for p in model.parameters())
    n_g = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    flops = 0.0
    try:
        import thop
        m = model
        im = torch.zeros(1, 3, imgsz, imgsz, device=next(m.parameters()).device)
        flops = thop.profile(m, inputs=(im,), verbose=False)[0] / 1e9 * 2  # GFLOPs
    except Exception:
        pass
    if verbose:
        print(f"{colorstr('Model')}  params={n_p:,}  grads={n_g:,}  "
              f"GFLOPs@{imgsz}={flops:.2f}  fp32_size={size_mb:.2f} MB  "
              f"(int8 ~{size_mb / 4:.2f} MB)")
    return n_p, n_g, flops, size_mb


def fuse_conv_bn(conv, bn):
    """Fold BatchNorm into the preceding Conv2d for export/inference."""
    fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                            conv.stride, conv.padding, conv.dilation, conv.groups,
                            bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, device=None):
        self.t = t
        self.device = device

    def __enter__(self):
        self.start = self._time()
        return self

    def __exit__(self, *exc):
        self.dt = self._time() - self.start
        self.t += self.dt

    def _time(self):
        if self.device is not None and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return time.time()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def cosine_lr(lrf, epochs):
    """Ultralytics-style cosine: lr multiplier from 1.0 -> lrf."""
    return lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1
