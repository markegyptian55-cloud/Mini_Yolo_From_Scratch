import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

from src.utils.boxes import box_iou

def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the average precision (AP) per class given true positives,
    confidence scores, predicted classes, and target classes.
    Source: YOLOv5/v8 style metric calculation.
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)  # number of classes

    # Create Precision-Recall curve and compute AP for each class
    ap = np.zeros((nc, tp.shape[1]))  # columns correspond to different IoU thresholds
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels for this class
        n_p = i.sum()  # number of predictions for this class

        if n_p == 0 or n_l == 0:
            continue

        # Accumulate True Positives and False Positives
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)
        # Precision
        precision = tpc / (tpc + fpc)

        # AP calculation: Area under PR curve
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    return ap, unique_classes

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute the average precision, given the recall and precision curves.
    Uses the COCO-style 101-point or all-points interpolation.
    """
    # Append sentinel values at beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope (monotonic decrease)
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # Integrate area under curve
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)

def evaluate_predictions(
    predictions: List[torch.Tensor],
    targets: torch.Tensor,
    iou_thresholds: Optional[Union[List[float], np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate predicted bounding boxes against ground truth targets.
    predictions: List of Tensors, each of shape (num_predictions, 6) -> [xyxy, conf, class_id]
    targets: Tensor of shape (num_objects, 6) -> [batch_idx, class_id, x_center, y_center, width, height]
             (normalized xywh coordinates)
    iou_thresholds: List or array of IoU thresholds to evaluate
    
    Returns:
        tp: true positives matrix of shape (num_predictions, num_thresholds)
        conf: confidence scores
        pred_cls: predicted classes
        target_cls: target classes
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # COCO thresholds

    iou_thresholds_tensor = torch.tensor(iou_thresholds, device=targets.device)
    stats = []

    for si, pred in enumerate(predictions):
        # Find targets for this image
        img_targets = targets[targets[:, 0] == si]
        nl = len(img_targets)
        
        if len(pred) == 0:
            if nl > 0:
                stats.append((
                    np.zeros((0, len(iou_thresholds)), dtype=bool),
                    np.zeros(0),
                    np.zeros(0),
                    img_targets[:, 1].cpu().numpy()
                ))
            continue

        # Sort predictions by confidence in descending order
        pred = pred[pred[:, 4].argsort(descending=True)]

        # Extract predictions
        pred_boxes = pred[:, :4]  # xyxy (normalized)
        pred_scores = pred[:, 4]
        pred_cls = pred[:, 5]

        tp = torch.zeros((len(pred), len(iou_thresholds)), dtype=torch.bool, device=pred.device)

        # Ground truth class and box
        target_cls = img_targets[:, 1] if nl > 0 else torch.zeros((0,), device=pred.device)
        target_boxes = img_targets[:, 2:6] if nl > 0 else torch.zeros((0, 4), device=pred.device)
        
        # Convert target boxes from xywh to xyxy
        if nl > 0:
            x_c, y_c, w, h = target_boxes.unbind(-1)
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2
            target_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)
        else:
            target_xyxy = torch.zeros((0, 4), device=pred.device)

        if nl > 0:
            # Compute IoU matrix between predictions and targets
            # pred_boxes: (N, 4), target_xyxy: (M, 4) -> ious: (N, M)
            ious = box_iou(pred_boxes, target_xyxy)
            
            for ti, iou_threshold in enumerate(iou_thresholds_tensor):
                detected = set()
                
                # Check each prediction
                for pi in range(len(pred)):
                    p_ious = ious[pi]
                    
                    # Find GTs of the same class
                    same_class_mask = (target_cls == pred_cls[pi])
                    if not same_class_mask.any():
                        continue
                    
                    # Mask out classes that don't match
                    p_ious_same_class = p_ious * same_class_mask
                    
                    # Find the best GT of the same class
                    best_iou_val, best_gt_idx = p_ious_same_class.max(0)
                    best_gt_idx = best_gt_idx.item()
                    
                    # Match if best IoU exceeds threshold and GT hasn't been matched
                    if best_iou_val >= iou_threshold and best_gt_idx not in detected:
                        tp[pi, ti] = True
                        detected.add(best_gt_idx)

        # Convert to numpy for list concatenation and return compatibility
        stats.append((
            tp.cpu().numpy(),
            pred_scores.cpu().numpy(),
            pred_cls.cpu().numpy(),
            target_cls.cpu().numpy()
        ))

    # Zip stats and concatenate
    if len(stats) > 0:
        tp, conf, pred_cls, target_cls = [np.concatenate(x, 0) for x in zip(*stats)]
    else:
        tp = np.zeros((0, len(iou_thresholds)), dtype=bool)
        conf = np.zeros(0)
        pred_cls = np.zeros(0)
        target_cls = np.zeros(0)

    return tp, conf, pred_cls, target_cls
