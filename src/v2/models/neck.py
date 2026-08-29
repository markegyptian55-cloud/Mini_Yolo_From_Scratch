"""PAN-FPN neck (YOLOv8 layout: no 1x1 reduction convs -> fewer params)."""
import torch
import torch.nn as nn

from src.v2.models.blocks import Conv, C2f


class MiniPANv2(nn.Module):
    def __init__(self, ch=(64, 128, 256), depth=1):
        super().__init__()
        c3, c4, c5 = ch
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn_p4 = C2f(c5 + c4, c4, depth, shortcut=False)
        self.fpn_p3 = C2f(c4 + c3, c3, depth, shortcut=False)
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.pan_p4 = C2f(c3 + c4, c4, depth, shortcut=False)
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.pan_p5 = C2f(c4 + c5, c5, depth, shortcut=False)
        self.out_channels = (c3, c4, c5)

    def forward(self, p3, p4, p5):
        x = self.fpn_p4(torch.cat((self.up(p5), p4), 1))      # /16
        n3 = self.fpn_p3(torch.cat((self.up(x), p3), 1))      # /8
        n4 = self.pan_p4(torch.cat((self.down_p3(n3), x), 1))  # /16
        n5 = self.pan_p5(torch.cat((self.down_p4(n4), p5), 1))  # /32
        return n3, n4, n5
