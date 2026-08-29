"""Exponential moving average of model weights (train-time only)."""
import math
from copy import deepcopy

import torch


class ModelEMA:
    """Keeps a shadow copy of the model weights updated with a ramped EMA decay.

    decay(t) = base * (1 - exp(-t / tau))  -- fast early, slow later.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        self.updates += 1
        d = self.decay(self.updates)
        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        for k, v in de_parallel(model).__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            setattr(self.ema, k, v)


def de_parallel(model):
    return model.module if hasattr(model, "module") else model
