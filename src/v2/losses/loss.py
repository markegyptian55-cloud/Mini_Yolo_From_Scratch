"""Detection loss for MiniYOLO-v2.

Four terms per branch (the DFL term is skipped when the head is built with reg_max=1):

    L_cls  = BCE(pred_logits, TAL soft target)          / sum(target_scores)
    L_box  = (1 - CIoU) * target_score                  / sum(target_scores)
    L_l1   = |pred_ltrb - target_ltrb| * target_score    / sum(target_scores)
    L_dfl  = distribution focal loss on the ltrb bins    / sum(target_scores)

L_dfl is the Generalized-Focal-Loss term: each ltrb distance is predicted as a softmax
over `reg_max` integer bins, and the loss is the linear interpolation of the cross
entropy against the two bins straddling the continuous target. CIoU supervises the
decoded box; DFL supervises the *shape* of the distribution behind it, which is what
sharpens localisation at the high IoU thresholds mAP50-95 averages over.

L_l1 is retained alongside it (it was YOLO26's DFL-free replacement) so experiment 3
changes exactly one thing versus experiment 2: the box representation. Dropping `l1`
is a separate, later experiment.

Two branches are trained jointly and blended by ProgLoss:

    L = alpha(t) * L_one2many + (1 - alpha(t)) * L_one2one,  alpha: 0.8 -> 0.1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.losses.tal import TaskAlignedAssigner
from src.v2.utils.boxes import bbox2dist, bbox_iou, dist2bbox, make_anchors, xywh2xyxy


class DetectionLoss:
    def __init__(self, model, hyp, device):
        head = model.head
        self.device = device
        self.nc = head.nc
        self.no = head.no
        self.reg_max = head.reg_max
        self.use_dfl = self.reg_max > 1
        # integration weights [0 .. reg_max-1] -- the same constant the head's DFL conv holds
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.stride = head.stride
        self.hyp = hyp
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.assigner_o2m = TaskAlignedAssigner(
            topk=int(hyp.get("tal_topk", 10)), num_classes=self.nc,
            alpha=float(hyp.get("tal_alpha", 0.5)), beta=float(hyp.get("tal_beta", 6.0)),
            stal=bool(hyp.get("stal", True)),
            stal_min_size=float(hyp.get("stal_min_size", 8.0)),
            stal_ref_size=float(hyp.get("stal_ref_size", 16.0)))
        self.assigner_o2o = TaskAlignedAssigner(
            topk=1, num_classes=self.nc,
            alpha=float(hyp.get("tal_alpha", 0.5)), beta=float(hyp.get("tal_beta", 6.0)),
            stal=bool(hyp.get("stal", True)),
            stal_min_size=float(hyp.get("stal_min_size", 8.0)),
            stal_ref_size=float(hyp.get("stal_ref_size", 16.0)))

    # ------------------------------------------------------------------ utils
    def preprocess(self, targets, batch_size, scale_tensor):
        """(n, 6)[img_idx, cls, xywhn] -> (b, n_max, 5)[cls, xyxy px] + mask."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, int(counts.max()), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = int(matches.sum())
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def dfl_expect(self, pred_dist):
        """(b, A, 4*reg_max) raw logits -> (b, A, 4) expected ltrb distances."""
        if not self.use_dfl:
            return pred_dist
        b, a, _ = pred_dist.shape
        return pred_dist.view(b, a, 4, self.reg_max).softmax(3).matmul(
            self.proj.to(pred_dist.dtype))

    @staticmethod
    def bbox_decode(anchor_points, pred_ltrb):
        return dist2bbox(pred_ltrb, anchor_points, xywh=False)

    @staticmethod
    def df_loss(pred_dist, target):
        """Distribution focal loss.

        pred_dist: (n*4, reg_max) logits.  target: (n*4,) continuous in [0, reg_max-1].
        Each target is split between its two neighbouring integer bins, weighted by the
        distance to each -- a target of 7.3 asks for 70% of the mass on bin 7 and 30%
        on bin 8.
        """
        top = pred_dist.shape[1] - 1
        tl = target.long()                          # bin below
        tr = tl + 1                                 # bin above
        wl = tr.to(target.dtype) - target           # weight on the lower bin
        wr = 1.0 - wl
        return (F.cross_entropy(pred_dist, tl.clamp(0, top), reduction="none") * wl
                + F.cross_entropy(pred_dist, tr.clamp(0, top), reduction="none") * wr)

    # ------------------------------------------------------------------ core
    def _branch_loss(self, feats, targets, assigner):
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((4 * self.reg_max, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   # (b, A, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   # (b, A, 4*reg_max) logits
        pred_ltrb = self.dfl_expect(pred_distri)                  # (b, A, 4) cell units

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_ltrb)  # xyxy, cell units

        _, target_bboxes, target_scores, fg_mask, _ = assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        loss_box = torch.zeros(1, device=self.device)
        loss_l1 = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                           xywh=False, CIoU=True)
            loss_box = ((1.0 - iou) * weight).sum() / target_scores_sum

            # With DFL the target must land inside the representable range [0, reg_max-1].
            # bbox2dist clamps to (arg - 0.01), so pass reg_max-1 -> 14.99 at reg_max=16,
            # which keeps the upper bin index at 15. Without DFL there is no range to
            # clamp to and the raw distance is used, exactly as before.
            target_ltrb = bbox2dist(anchor_points, target_bboxes,
                                    self.reg_max - 1 if self.use_dfl else None)
            loss_l1 = ((pred_ltrb[fg_mask] - target_ltrb[fg_mask]).abs() * weight).sum() \
                / target_scores_sum

            if self.use_dfl:
                loss_dfl = (self.df_loss(
                    pred_distri[fg_mask].view(-1, self.reg_max),
                    target_ltrb[fg_mask].view(-1),
                ).view(-1, 4).mean(-1, keepdim=True) * weight).sum() / target_scores_sum

        return (loss_box.squeeze(), loss_cls.squeeze(), loss_l1.squeeze(),
                loss_dfl.squeeze(), float(fg_mask.sum()))

    def __call__(self, preds, targets, alpha):
        """preds: dict with 'one2many'/'one2one' raw feature lists.
        targets: (n, 6) [img_idx, cls, xywhn]. alpha: ProgLoss weight for one2many.
        """
        feats_o2m = preds["one2many"]
        feats_o2o = preds["one2one"]
        batch_size = feats_o2m[0].shape[0]

        imgsz = torch.tensor(feats_o2m[0].shape[2:], device=self.device,
                             dtype=feats_o2m[0].dtype) * self.stride[0]
        scale_tensor = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]],
                                    device=self.device)
        tgt = self.preprocess(targets.to(self.device), batch_size, scale_tensor)

        b_m, c_m, l_m, d_m, n_m = self._branch_loss(feats_o2m, tgt, self.assigner_o2m)
        b_o, c_o, l_o, d_o, n_o = self._branch_loss(feats_o2o, tgt, self.assigner_o2o)

        gb, gc, gl = self.hyp["box"], self.hyp["cls"], self.hyp["l1"]
        gd = float(self.hyp.get("dfl", 1.5)) if self.use_dfl else 0.0
        loss_m = gb * b_m + gc * c_m + gl * l_m + gd * d_m
        loss_o = gb * b_o + gc * c_o + gl * l_o + gd * d_o
        total = alpha * loss_m + (1.0 - alpha) * loss_o

        items = {
            "box": float(alpha * gb * b_m + (1 - alpha) * gb * b_o),
            "cls": float(alpha * gc * c_m + (1 - alpha) * gc * c_o),
            "l1": float(alpha * gl * l_m + (1 - alpha) * gl * l_o),
            "dfl": float(alpha * gd * d_m + (1 - alpha) * gd * d_o),
            "o2m": float(loss_m),
            "o2o": float(loss_o),
            "npos": n_m,
        }
        return total * batch_size, items
