"""Task-Aligned Assigner (TOOD/YOLOv8) + STAL (YOLO26 small-target-aware assignment).

STAL, per the YOLO26 paper: candidate selection and box regression use *different*
geometry.  A ground-truth side shorter than s_min = 8 px can end up containing no
anchor centre after feature-map discretisation, so the object gets zero positives
and contributes no gradient at all.  STAL builds a surrogate box whose short sides
are inflated to s_ref = 16 px (the next stride level) and uses it **only** for the
inside-gt candidate mask.  Regression targets stay the original box.
"""
import torch
import torch.nn as nn

from src.v2.utils.boxes import bbox_iou


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """xy_centers (A, 2) px, gt_bboxes (b, n, 4) xyxy px -> (b, n, A) bool."""
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
    return deltas.view(bs, n_boxes, n_anchors, -1).amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """Resolve anchors matched to more than one GT by keeping the highest-IoU GT."""
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(1)
        is_max = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi, is_max, mask_pos).float()
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """align_metric = cls_score^alpha * CIoU^beta, top-k anchors per GT."""

    def __init__(self, topk=10, num_classes=3, alpha=0.5, beta=6.0, eps=1e-9,
                 stal=True, stal_min_size=8.0, stal_ref_size=16.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.stal = stal
        self.stal_min = stal_min_size
        self.stal_ref = stal_ref_size

    def _stal_boxes(self, gt_bboxes):
        """Inflate tiny GT sides to `stal_ref` px. Used for the candidate mask only."""
        x1, y1, x2, y2 = gt_bboxes.unbind(-1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = (x2 - x1), (y2 - y1)
        w = torch.where(w < self.stal_min, torch.full_like(w, self.stal_ref), w)
        h = torch.where(h < self.stal_min, torch.full_like(h, self.stal_ref), h)
        return torch.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), -1)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores  (b, A, nc) sigmoid probabilities
        pd_bboxes  (b, A, 4)  xyxy, image pixels
        anc_points (A, 2)     xy, image pixels
        gt_labels  (b, n, 1); gt_bboxes (b, n, 4) xyxy px; mask_gt (b, n, 1)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.bg_idx),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]).bool(),
                    torch.zeros_like(pd_scores[..., 0]).long())

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalise soft targets: scale the alignment metric by the best IoU per GT.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align = (align_metric * pos_overlaps /
                      (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        sel_boxes = self._stal_boxes(gt_bboxes) if self.stal else gt_bboxes
        mask_in_gts = select_candidates_in_gts(anc_points, sel_boxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        topk_mask = mask_gt.expand(-1, -1, self.topk).bool()
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=topk_mask)
        return mask_topk * mask_in_gts * mask_gt, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype,
                               device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype,
                                  device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False,
                                     CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(self.bs, dtype=torch.int64,
                                 device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1],
                                     self.num_classes), dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores
