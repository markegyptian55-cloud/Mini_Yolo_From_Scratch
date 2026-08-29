"""MiniYOLO-v2 (a.k.a. MiniYOLO-E): edge-first, anchor-free, NMS-free.

Box regression uses DFL (reg_max=16) by default since experiment 3; pass
reg_max=1 to rebuild the original scalar head for a pre-DFL checkpoint.
"""
import torch
import torch.nn as nn

from src.v2.models.backbone import MiniDarknetV2
from src.v2.models.blocks import Conv
from src.v2.models.head import DualDetect
from src.v2.models.neck import MiniPANv2
from src.v2.utils.general import model_info


SCALES = {
    # name: (backbone widths, backbone depths, neck depth)
    "p": ((8, 16, 32, 64, 128), (1, 1, 1, 1), 1),     # pico
    "t": ((12, 24, 48, 96, 192), (1, 1, 2, 1), 1),    # tiny
    "n": ((16, 32, 64, 128, 256), (1, 2, 2, 1), 1),   # nano  (default)
    "s": ((24, 48, 96, 192, 384), (1, 2, 3, 1), 2),   # small
}


class MiniYOLOv2(nn.Module):
    def __init__(self, nc=3, scale="n", strides=(8, 16, 32), e2e=True, max_det=300,
                 names=None, reg_max=16):
        super().__init__()
        assert scale in SCALES, f"scale must be one of {list(SCALES)}"
        width, depth, neck_depth = SCALES[scale]
        self.nc = nc
        self.scale = scale
        self.names = names or [str(i) for i in range(nc)]
        self.strides = strides
        self.reg_max = int(reg_max)

        self.backbone = MiniDarknetV2(3, width, depth)
        self.neck = MiniPANv2(self.backbone.out_channels, neck_depth)
        self.head = DualDetect(nc, self.neck.out_channels, strides, e2e=e2e,
                               max_det=max_det, reg_max=self.reg_max)
        self.is_fused = False

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return self.head([n3, n4, n5])

    # -- deployment helpers -------------------------------------------------
    def fuse(self):
        if not self.is_fused:
            for m in self.modules():
                if isinstance(m, Conv):
                    m.fuse()
            self.is_fused = True
        return self

    def export_ready(self, max_det=None):
        """Strip the one-to-many branch, fuse BN, force the NMS-free path."""
        self.head.strip_one2many()
        if max_det is not None:
            self.head.max_det = max_det
        self.fuse().eval()
        return self

    def info(self, imgsz=384, verbose=True):
        return model_info(self, imgsz=imgsz, verbose=verbose)


def build_model(nc=3, scale="n", imgsz=384, e2e=True, max_det=300, names=None,
                device="cpu", verbose=True, reg_max=16):
    model = MiniYOLOv2(nc=nc, scale=scale, e2e=e2e, max_det=max_det, names=names,
                       reg_max=reg_max).to(device)
    if verbose:
        model.info(imgsz=imgsz)
    return model


if __name__ == "__main__":
    for s in SCALES:
        m = MiniYOLOv2(nc=3, scale=s).eval()
        print(f"scale={s}")
        m.info(imgsz=384)
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 384, 384))
        print("   e2e out:", tuple(out.shape))
