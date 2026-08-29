"""Optimizers: AdamW / SGD / MuSGD (Muon + SGD hybrid, YOLO26-style).

Muon reference: Keller Jordan, "Muon: an optimizer for hidden layers".
YOLO26 reports MuSGD = weighted mixture of a Muon update and an SGD update on
multi-dimensional params (conv kernels, linear weights), pure SGD on 1D params
(biases, norm scales).  The exact blend is not published, so this implementation
RMS-matches the Muon direction to the SGD update before blending; that keeps the
effective step size inside SGD's known-safe regime while still steering with
Muon's orthogonalised direction.  `muon_ratio=0.0` degenerates to plain SGD.
"""
import math

import torch
from torch import nn
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration approximating the orthogonalisation UV^T of G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class MuSGD(Optimizer):
    """SGD with an optional Muon-orthogonalised component on >=2D parameters."""

    def __init__(self, params, lr=1e-2, momentum=0.937, weight_decay=0.0,
                 nesterov=True, muon_ratio=0.5, ns_steps=5, use_muon=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, muon_ratio=muon_ratio,
                        ns_steps=ns_steps, use_muon=use_muon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            ratio = group["muon_ratio"] if group.get("use_muon", True) else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if nesterov else buf

                if ratio > 0.0 and p.ndim >= 2:
                    flat = upd.reshape(upd.shape[0], -1)
                    o = zeropower_via_newtonschulz5(flat, steps=group["ns_steps"])
                    # scale for non-square matrices (Muon's fan-in/out correction)
                    o = o * math.sqrt(max(1.0, flat.shape[0] / flat.shape[1]))
                    # RMS-match to the SGD update, then blend
                    o = o * (upd.norm() / (o.norm() + 1e-12))
                    upd = (1.0 - ratio) * upd + ratio * o.reshape_as(upd)

                p.add_(upd, alpha=-lr)
        return loss


def build_param_groups(model):
    """Split params into (norm/bias -> no decay, 1D) and (weights -> decay, >=2D)."""
    g_decay, g_nodecay = [], []
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias" or isinstance(module, norm_types) or p.ndim == 1:
                g_nodecay.append(p)
            else:
                g_decay.append(p)
    return g_decay, g_nodecay


def build_optimizer(model, name="AdamW", lr=1e-3, momentum=0.937, weight_decay=5e-4,
                    muon_ratio=0.5):
    """Create optimizer with 3 groups: [0]=weights(decay) [1]=norm/bias(no decay) [2]=head bias.

    Group order matters: the trainer warms up group 2 (biases) from a higher LR,
    exactly like the YOLOv5/v8 recipe.
    """
    g_decay, g_nodecay = build_param_groups(model)
    name = name.lower()

    if name == "adamw":
        opt = torch.optim.AdamW([
            {"params": g_decay, "weight_decay": weight_decay},
            {"params": g_nodecay, "weight_decay": 0.0},
        ], lr=lr, betas=(momentum, 0.999), eps=1e-8)
    elif name == "sgd":
        opt = torch.optim.SGD([
            {"params": g_decay, "weight_decay": weight_decay},
            {"params": g_nodecay, "weight_decay": 0.0},
        ], lr=lr, momentum=momentum, nesterov=True)
    elif name == "musgd":
        opt = MuSGD([
            {"params": g_decay, "weight_decay": weight_decay,
             "use_muon": True, "muon_ratio": muon_ratio},
            {"params": g_nodecay, "weight_decay": 0.0,
             "use_muon": False, "muon_ratio": 0.0},
        ], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"unknown optimizer '{name}' (adamw | sgd | musgd)")

    opt.opt_name = name
    return opt


def prog_alpha(epoch, epochs, a_init=0.8, a_final=0.1):
    """YOLO26 ProgLoss schedule: linear a_init -> a_final over training.

    total = alpha * L_one2many + (1 - alpha) * L_one2one
    """
    t = max(epochs - 1, 1)
    return max(1.0 - epoch / t, 0.0) * (a_init - a_final) + a_final
