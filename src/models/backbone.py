import torch
import torch.nn as nn
from src.models.blocks import ConvBNSiLU, C2f, SPPF
from configs import config

class MiniDarknet(nn.Module):
    """
    Mini Darknet Backbone inspired by modern YOLO architectures (YOLOv8/YOLO11).
    Downsamples the input image by factors of 8, 16, and 32 to yield
    three feature maps (P3, P4, P5) representing different scales.

    ASCII Architecture Flow Diagram:
    
           Input Image (B, 3, H, W)
                     │
                     ▼
                [ Stem Conv ]  (Stride 2) ──────► (B, base_channels, H/2, W/2)
                     │
                     ▼
                [ Stage 1 ]    (Stride 2) ──────► (B, base_channels*2, H/4, W/4)
                     │
                     ▼
                [ Stage 2 ]    (Stride 2) ──────► P3: Stride 8 (B, base_channels*4, H/8, W/8)
                     │
                     ▼
                [ Stage 3 ]    (Stride 2) ──────► P4: Stride 16 (B, base_channels*8, H/16, W/16)
                     │
                     ▼
                [ Stage 4 ]    (Stride 2) ──────► P5: Stride 32 (B, base_channels*16, H/32, W/32)
    """
    def __init__(self, in_channels=3, base_channels=config.BASE_CHANNELS):
        super().__init__()
        
        # Stem: Stride 2 downsampling (H/2, W/2)
        # Input: (B, 3, 416, 416) -> Output: (B, 16, 208, 208)
        self.stem = ConvBNSiLU(in_channels, base_channels, kernel_size=3, stride=2)
        
        # Stage 1: Downsample to (H/4, W/4)
        # Output: (B, 32, 104, 104)
        self.dark1 = nn.Sequential(
            ConvBNSiLU(base_channels, base_channels * 2, kernel_size=3, stride=2),
            C2f(base_channels * 2, base_channels * 2, n=1)
        )
        
        # Stage 2 (P3 / Stride 8): Downsample to (H/8, W/8)
        # Output: (B, 64, 52, 52)
        self.dark2 = nn.Sequential(
            ConvBNSiLU(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),
            C2f(base_channels * 4, base_channels * 4, n=2)
        )
        
        # Stage 3 (P4 / Stride 16): Downsample to (H/16, W/16)
        # Output: (B, 128, 26, 26)
        self.dark3 = nn.Sequential(
            ConvBNSiLU(base_channels * 4, base_channels * 8, kernel_size=3, stride=2),
            C2f(base_channels * 8, base_channels * 8, n=3)
        )
        
        # Stage 4 (P5 / Stride 32): Downsample to (H/32, W/32)
        # Output: (B, 256, 13, 13)
        self.dark4 = nn.Sequential(
            ConvBNSiLU(base_channels * 8, base_channels * 16, kernel_size=3, stride=2),
            C2f(base_channels * 16, base_channels * 16, n=1),
            SPPF(base_channels * 16, base_channels * 16, k=5)
        )

    def forward(self, x):
        """
        Returns a dictionary of multi-scale feature maps:
        P3: stride 8   (e.g., 52x52 for 416x416 input)
        P4: stride 16  (e.g., 26x26 for 416x416 input)
        P5: stride 32  (e.g., 13x13 for 416x416 input)
        """
        # Input validation: Height and Width must be divisible by 32
        _, _, h, w = x.shape
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"❌ Input image resolution ({w}x{h}) must be divisible by 32.")

        x = self.stem(x)       # stride 2
        x = self.dark1(x)      # stride 4
        p3 = self.dark2(x)     # stride 8
        p4 = self.dark3(p3)    # stride 16
        p5 = self.dark4(p4)    # stride 32
        
        return {"P3": p3, "P4": p4, "P5": p5}
