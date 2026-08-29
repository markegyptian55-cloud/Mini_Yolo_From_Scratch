"""Detection metrics: COCO-style AP@0.5, AP@0.5:0.95, per-class P/R/F1."""
import numpy as np
import torch

from src.v2.utils.boxes import box_iou


def compute_ap(recall, precision):
    """101-point interpolated AP (COCO)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") \
        else np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """tp: (n, niou) bool. Returns tp_c, fp_c, p, r, f1, ap (nc, niou), unique_classes."""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros(nc)
    r = np.zeros(nc)
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc)
        r[ci] = np.interp(-0.1, -conf[i], recall[:, 0], left=0)
        p[ci] = np.interp(-0.1, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])[0]

    f1 = 2 * p * r / (p + r + eps)
    return p, r, f1, ap, unique_classes.astype(int)


def match_predictions(pred_classes, true_classes, iou, iouv):
    """Greedy IoU matching per threshold. Returns (n_pred, niou) bool."""
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
    iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else np.asarray(iou)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                m_iou = iou[matches[:, 0], matches[:, 1]]
                matches = matches[m_iou.argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


class DetMetrics:
    """Accumulates per-image stats and produces the summary table."""

    def __init__(self, nc, names, device="cpu"):
        self.nc = nc
        self.names = names
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
        self.results = None

    def process_batch(self, detections, labels):
        """detections (n, 6) xyxy conf cls; labels (m, 5) cls xyxy -- both pixel space."""
        npr = detections.shape[0]
        nl = labels.shape[0]
        if npr == 0:
            if nl:
                self.stats["tp"].append(np.zeros((0, self.iouv.numel()), dtype=bool))
                self.stats["conf"].append(np.zeros(0))
                self.stats["pred_cls"].append(np.zeros(0))
                self.stats["target_cls"].append(labels[:, 0].cpu().numpy())
            return
        if nl == 0:
            self.stats["tp"].append(np.zeros((npr, self.iouv.numel()), dtype=bool))
            self.stats["conf"].append(detections[:, 4].cpu().numpy())
            self.stats["pred_cls"].append(detections[:, 5].cpu().numpy())
            self.stats["target_cls"].append(np.zeros(0))
            return
        iou = box_iou(labels[:, 1:5], detections[:, :4])
        correct = match_predictions(detections[:, 5].cpu().numpy(),
                                    labels[:, 0].cpu().numpy(), iou, self.iouv)
        self.stats["tp"].append(correct)
        self.stats["conf"].append(detections[:, 4].cpu().numpy())
        self.stats["pred_cls"].append(detections[:, 5].cpu().numpy())
        self.stats["target_cls"].append(labels[:, 0].cpu().numpy())

    def compute(self):
        if not self.stats["tp"]:
            self.results = None
            return {}
        tp = np.concatenate(self.stats["tp"], 0)
        conf = np.concatenate(self.stats["conf"], 0)
        pred_cls = np.concatenate(self.stats["pred_cls"], 0)
        target_cls = np.concatenate(self.stats["target_cls"], 0)
        if tp.shape[0] == 0 or target_cls.shape[0] == 0:
            self.results = None
            return {}
        p, r, f1, ap, cls_idx = ap_per_class(tp, conf, pred_cls, target_cls)
        self.results = dict(p=p, r=r, f1=f1, ap=ap, cls_idx=cls_idx,
                            nt=np.bincount(target_cls.astype(int), minlength=self.nc))
        return {
            "precision": float(p.mean()) if len(p) else 0.0,
            "recall": float(r.mean()) if len(r) else 0.0,
            "mAP50": float(ap[:, 0].mean()) if len(ap) else 0.0,
            "mAP50-95": float(ap.mean()) if len(ap) else 0.0,
            "fitness": float(0.1 * (ap[:, 0].mean() if len(ap) else 0.0)
                             + 0.9 * (ap.mean() if len(ap) else 0.0)),
        }

    def table(self):
        if self.results is None:
            return "no predictions"
        rr = self.results
        lines = [f"{'Class':>14s} {'Images':>8s} {'Instances':>10s} {'P':>8s} {'R':>8s} "
                 f"{'mAP50':>8s} {'mAP50-95':>9s}"]
        lines.append(f"{'all':>14s} {'-':>8s} {int(rr['nt'].sum()):>10d} "
                     f"{rr['p'].mean():>8.3f} {rr['r'].mean():>8.3f} "
                     f"{rr['ap'][:, 0].mean():>8.3f} {rr['ap'].mean():>9.3f}")
        for i, c in enumerate(rr["cls_idx"]):
            name = self.names[c] if c < len(self.names) else str(c)
            lines.append(f"{name:>14s} {'-':>8s} {int(rr['nt'][c]):>10d} "
                         f"{rr['p'][i]:>8.3f} {rr['r'][i]:>8.3f} "
                         f"{rr['ap'][i, 0]:>8.3f} {rr['ap'][i].mean():>9.3f}")
        return "\n".join(lines)
