import torch
from typing import List, Optional
from src.utils.boxes import xywh2xyxy

try:
    import torchvision
    _has_torchvision = True
except ImportError:
    _has_torchvision = False

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    Supports torchvision.ops.nms when available for optimized C++/CUDA execution.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    if _has_torchvision:
        return torchvision.ops.nms(boxes, scores, iou_threshold)

    # Fallback to custom pure PyTorch NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        
        i = order[0].item()
        keep.append(i)

        xx1 = torch.clamp(x1[order[1:]], min=x1[i])
        yy1 = torch.clamp(y1[order[1:]], min=y1[i])
        xx2 = torch.clamp(x2[order[1:]], max=x2[i])
        yy2 = torch.clamp(y2[order[1:]], max=y2[i])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        ids = torch.where(iou <= iou_threshold)[0]
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    max_det: int = 300
) -> List[torch.Tensor]:
    """
    Runs Non-Maximum Suppression (NMS) on inference results.
    prediction: Tensor of shape [batch_size, num_predictions, 4 + 1 + num_classes]
                 where the 4 coords are in xywh (center x, center y, width, height)
    Returns:
        List of detections, each of shape [num_dets, 6] -> [xyxy, conf, cls]
    """
    # Create independent output tensors for each image to prevent sharing references
    output = [torch.zeros((0, 6), device=prediction.device) for _ in range(prediction.shape[0])]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply confidence threshold filter (using objectness)
        x = x[x[:, 4] > conf_thres]

        if not x.shape[0]:
            continue

        # Compute class confidence: class_score = objectness * class_probability
        box = x[:, :4]
        obj = x[:, 4:5]
        cls_probs = x[:, 5:]
        
        scores, class_ids = torch.max(cls_probs * obj, dim=1, keepdim=True)

        # Filter boxes below confidence threshold based on final score
        keep_mask = scores.squeeze(1) > conf_thres
        box = box[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        if not box.shape[0]:
            continue

        # Filter by classes if specified (vectorized)
        if classes is not None:
            classes_tensor = torch.tensor(classes, device=class_ids.device)
            class_mask = (class_ids == classes_tensor).any(dim=1)
            box = box[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]

            if not box.shape[0]:
                continue

        # Limit candidate boxes before NMS to speed up performance
        max_nms = 30000
        if box.shape[0] > max_nms:
            _, sort_indices = scores.squeeze(1).sort(descending=True)
            sort_indices = sort_indices[:max_nms]
            box = box[sort_indices]
            scores = scores[sort_indices]
            class_ids = class_ids[sort_indices]

        # Convert box coordinates from xywh to xyxy
        box_xyxy = xywh2xyxy(box)

        # Apply class-specific NMS
        # Add class offset to box coordinates so NMS runs independently for each class
        offset = class_ids * 0.0 if agnostic else class_ids * 4096.0
        boxes_for_nms = box_xyxy + offset
        
        keep_indices = nms(boxes_for_nms, scores.squeeze(1), iou_thres)

        # Limit detections to max_det
        if keep_indices.numel() > max_det:
            keep_indices = keep_indices[:max_det]

        if keep_indices.numel() > 0:
            output[xi] = torch.cat((box_xyxy[keep_indices], scores[keep_indices], class_ids[keep_indices].float()), 1)

    return output
