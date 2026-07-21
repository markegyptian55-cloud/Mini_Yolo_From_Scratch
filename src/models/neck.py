import torch
import torch.nn as nn
from src.models.blocks import ConvBNSiLU, C2f

class MiniPANet(nn.Module):
    """
    Path Aggregation Network (PANet) style neck inspired by YOLOv5.
    Fuses multi-scale features top-down (FPN) and bottom-up (PANet) to provide
    highly informative spatial and semantic context to the detection head.
    
    Inputs:
        P3: stride 8   (channels: c_p3 = base * 4)
        P4: stride 16  (channels: c_p4 = base * 8)
        P5: stride 32  (channels: c_p5 = base * 16)
        
    Outputs:
        N3: stride 8   (channels: c_p3 = base * 4)
        N4: stride 16  (channels: c_p4 = base * 8)
        N5: stride 32  (channels: c_p5 = base * 16)
    """
    def __init__(self, base_channels=16):
        super().__init__()
        c_p3 = base_channels * 4   # 64
        c_p4 = base_channels * 8   # 128
        c_p5 = base_channels * 16  # 256

        # --- Top-Down Path (FPN) ---
        # Reduce P5 channels to match P4
        self.reduce_p5 = ConvBNSiLU(c_p5, c_p4, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Process concat(P5_upsampled, P4)
        self.c2f_fpn1 = C2f(c_p4 * 2, c_p4, n=1, shortcut=False)
        
        # Reduce P4_fused channels to match P3
        self.reduce_p4 = ConvBNSiLU(c_p4, c_p3, kernel_size=1, stride=1)
        # Process concat(P4_fused_upsampled, P3)
        self.c2f_fpn2 = C2f(c_p3 * 2, c_p3, n=1, shortcut=False)

        # --- Bottom-Up Path (PANet) ---
        # Downsample N3 to match N4
        self.downsample_n3 = ConvBNSiLU(c_p3, c_p3, kernel_size=3, stride=2)
        # Process concat(N3_downsampled, P4_fused)
        self.c2f_pan1 = C2f(c_p3 + c_p4, c_p4, n=1, shortcut=False)
        
        # Downsample N4 to match N5
        self.downsample_n4 = ConvBNSiLU(c_p4, c_p4, kernel_size=3, stride=2)
        # Process concat(N4_downsampled, P5_fused)
        self.c2f_pan2 = C2f(c_p4 + c_p5, c_p5, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        # 1. Top-Down path (FPN)
        p5_reduced = self.reduce_p5(p5)                    # (B, 128, 13, 13)
        p5_upsampled = self.upsample(p5_reduced)           # (B, 128, 26, 26)
        fpn_p4 = self.c2f_fpn1(torch.cat((p5_upsampled, p4), dim=1)) # (B, 128, 26, 26)

        fpn_p4_reduced = self.reduce_p4(fpn_p4)            # (B, 64, 26, 26)
        fpn_p4_upsampled = self.upsample(fpn_p4_reduced)   # (B, 64, 52, 52)
        n3 = self.c2f_fpn2(torch.cat((fpn_p4_upsampled, p3), dim=1)) # (B, 64, 52, 52) - N3 output

        # 2. Bottom-Up path (PANet)
        n3_downsampled = self.downsample_n3(n3)            # (B, 64, 26, 26)
        n4 = self.c2f_pan1(torch.cat((n3_downsampled, fpn_p4), dim=1)) # (B, 128, 26, 26) - N4 output

        n4_downsampled = self.downsample_n4(n4)            # (B, 128, 13, 13)
        n5 = self.c2f_pan2(torch.cat((n4_downsampled, p5), dim=1))     # (B, 256, 13, 13) - N5 output

        return n3, n4, n5
