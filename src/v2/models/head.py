"""Dual detection head: one-to-many (training richness) + one-to-one (NMS-free inference).

Design notes
------------
* Anchor-free.  Box regression is a **discrete distribution** over `reg_max` bins per
  ltrb side (Generalized Focal Loss / DFL), integrated back to a scalar distance in
  feature-cell units.  `reg_max=1` collapses this to the old DFL-free path (4 raw
  scalars) and is still supported for loading pre-DFL checkpoints.
  Trade-off, stated plainly: YOLO26 *removed* DFL because the 16-bin softmax+matmul is
  brittle across TFLite/NCNN compilers and hurts INT8 quantisation. We are trading that
  portability back for localisation accuracy -- experiment 2 showed mAP50 0.900 with
  mAP50-95 0.483 (ratio 0.54), and the scalar head is the remaining suspect.
  The integral is exported as a frozen 1x1 Conv + Softmax, both of which ONNX Runtime,
  TensorRT and OpenVINO handle natively.
* **No objectness branch.**  Since YOLOv8 the class score doubles as confidence and
  the task-aligned assigner supplies IoU-weighted soft targets.
* The one-to-one branch is a structural copy of the one-to-many branch.  Only the
  one-to-one branch survives export, so inference cost is identical to a single head.
* Class branch uses depthwise-separable convs (YOLO11 trick) -- this is where the v1
  head was burning most of its parameters.
"""
import copy
import math

import torch
import torch.nn as nn

from src.v2.models.blocks import Conv, DWConv
from src.v2.utils.boxes import make_anchors, dist2bbox


class DFL(nn.Module):
    """Integral of a discrete ltrb distribution -> one scalar distance per side.

    Weights are frozen to [0, 1, ..., reg_max-1], so this is literally the
    expectation of a softmax. Implemented as a 1x1 Conv rather than a matmul
    because Conv is the better-supported op in every edge compiler.
    """

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)

    def forward(self, x):
        """x: (B, 4*c1, A) raw logits -> (B, 4, A) distances."""
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DualDetect(nn.Module):
    e2e = True              # NMS-free path (one-to-one head only)
    max_det = 300
    # Emit the top-k selection inside the graph. Fine for ONNX Runtime / TensorRT /
    # OpenVINO, but TopK + GatherElements + Mod are not portable to NCNN or TFLite --
    # set False there and do the (trivial) selection in host code instead.
    export_postprocess = True

    def __init__(self, nc=3, ch=(64, 128, 256), strides=(8, 16, 32), e2e=True, max_det=300,
                 reg_max=16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = int(reg_max)          # 16 = DFL; 1 = legacy scalar head
        self.no = nc + 4 * self.reg_max
        self.e2e = e2e
        self.max_det = max_det
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float32))
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

        # Ultralytics widens the box trunk to 4*reg_max (=64) at every level. That costs
        # ~139k extra exported params here and pushes the FP16 ONNX past the 4.8 MB edge
        # budget this project is held to, so the trunk is widened to 2*reg_max instead:
        # enough that the final 1x1 is not squeezing 64 logits out of 16 channels, cheap
        # enough to stay inside budget. Measured cost is in AGENTS.md.
        c2 = max(16, ch[0] // 4, 2 * self.reg_max)    # box branch width
        c3 = max(32, min(ch[0], 96))                  # cls branch width

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(DWConv(x, c3, 3), DWConv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch)

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.bias_init()

    def bias_init(self):
        """Box bias -> 1.0; cls bias -> logit(0.01) prior.

        With reg_max>1 a constant box bias is uniform across bins, i.e. a uniform
        distribution whose expectation is (reg_max-1)/2 cells -- the same neutral
        start Ultralytics uses. With reg_max==1 it literally means "1 cell wide".
        """
        for cv2, cv3, s in zip(self.cv2, self.cv3, self.stride):
            cv2[-1].bias.data[:] = 1.0
            cv3[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)
        for cv2, cv3, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            cv2[-1].bias.data[:] = 1.0
            cv3[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)

    @staticmethod
    def _branch_forward(feats, cv2, cv3):
        return [torch.cat((cv2[i](f), cv3[i](f)), 1) for i, f in enumerate(feats)]

    def forward(self, feats):
        if self.training:
            # ProgLoss needs both raw branches; decoding happens inside the loss.
            return {"one2many": self._branch_forward(feats, self.cv2, self.cv3),
                    "one2one": self._branch_forward(feats, self.one2one_cv2, self.one2one_cv3)}

        if self.e2e:
            branch = self._branch_forward(feats, self.one2one_cv2, self.one2one_cv3)
            y = self._inference(branch)                       # (B, 4+nc, A)
            if not self.export_postprocess:
                return y
            return self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)

        assert hasattr(self, "cv2"), "one-to-many branch was stripped; e2e must stay True"
        branch = self._branch_forward(feats, self.cv2, self.cv3)
        return self._inference(branch)                        # (B, 4+nc, A) -> NMS

    def _inference(self, branch):
        """branch: list of (B, no, H, W) -> (B, 4 + nc, A) with xywh boxes in pixels."""
        shape = branch[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in branch], 2)
        if self.shape != shape or self.anchors.numel() == 0:
            anchors, strides = make_anchors(branch, self.stride, 0.5)
            self.anchors = anchors.transpose(0, 1)          # (2, A)
            self.strides = strides.transpose(0, 1)          # (1, A)
            self.shape = shape
        box, cls = x_cat.split((4 * self.reg_max, self.nc), 1)
        if self.reg_max > 1:
            box = self.dfl(box)                          # (B, 4*reg_max, A) -> (B, 4, A)
        dbox = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    @staticmethod
    def postprocess(preds, max_det, nc):
        """NMS-free selection (YOLOv10 style). preds: (B, A, 4+nc) xywh.

        Returns (B, max_det, 6) = [x, y, w, h, conf, cls] sorted by confidence.
        Pure gather/topk -> exports cleanly to ONNX / TFLite / NCNN.
        """
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        k = min(max_det, anchors)
        index = scores.amax(dim=-1).topk(k)[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        i = torch.arange(batch_size, device=preds.device)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None],
                          (index % nc)[..., None].float()], dim=-1)

    def strip_one2many(self):
        """Drop the training-only branch before export (halves head params)."""
        if hasattr(self, "cv2"):
            del self.cv2
            del self.cv3
        self.e2e = True
        return self
