"""MiniDarknet-v2 backbone.

Capacity is deliberately biased toward P4/P5: this dataset's objects are large
(mean box = 41% of frame width, only 2.2% of boxes < 16 px), so deep semantic
stages matter far more than a high-resolution P2/P3 branch.
"""
import torch.nn as nn

from src.v2.models.blocks import Conv, C2f, SPPF


class MiniDarknetV2(nn.Module):
    """Returns P3 (/8), P4 (/16), P5 (/32)."""

    def __init__(self, in_ch=3, width=(16, 32, 64, 128, 256), depth=(1, 2, 2, 1)):
        super().__init__()
        w0, w1, w2, w3, w4 = width
        d1, d2, d3, d4 = depth

        self.stem = Conv(in_ch, w0, 3, 2)                     # /2
        self.stage1 = nn.Sequential(Conv(w0, w1, 3, 2), C2f(w1, w1, d1, shortcut=True))   # /4
        self.stage2 = nn.Sequential(Conv(w1, w2, 3, 2), C2f(w2, w2, d2, shortcut=True))   # /8  -> P3
        self.stage3 = nn.Sequential(Conv(w2, w3, 3, 2), C2f(w3, w3, d3, shortcut=True))   # /16 -> P4
        self.stage4 = nn.Sequential(Conv(w3, w4, 3, 2), C2f(w4, w4, d4, shortcut=True),
                                    SPPF(w4, w4, 5))                                      # /32 -> P5
        self.out_channels = (w2, w3, w4)

    def forward(self, x):
        x = self.stage1(self.stem(x))
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5
