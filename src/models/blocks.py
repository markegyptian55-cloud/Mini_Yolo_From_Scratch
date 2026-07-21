import torch
import torch.nn as nn

class ConvBNSiLU(nn.Module):
    """
    Standard Convolution -> Batch Normalization -> SiLU (Swish) activation block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # auto-padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Standard bottleneck residual block.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_, c2, 3, 1, padding=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """
    CSP Bottleneck with 2 Convolutions and Faster multi-scale feature split (used in YOLOv8).
    Processes features through parallel bottlenecks and concatenates split outputs.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        # First 1x1 Conv splits channels: outputs 2 * hidden channels
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, 1, 1)
        # Second 1x1 Conv merges split outputs: takes (2 + n) * hidden channels
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, 1, 1)
        # ModuleList of Bottlenecks
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        # Split along channels (dimension 1) into two tensors of size self.c
        y = list(self.cv1(x).chunk(2, 1))
        # Feed the last feature map recursively through the Bottlenecks and append outputs
        for m in self.m:
            y.append(m(y[-1]))
        # Concatenate all maps and pass through 1x1 output convolution
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) block by Glenn Jocher (YOLOv5).
    Saves computation while maintaining SPP's ability to pool features at multiple scales.
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))
