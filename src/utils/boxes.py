import torch
import math
from typing import Union, Tuple, List

def xywh2xyxy(x: Union[torch.Tensor, object]) -> Union[torch.Tensor, object]:
    """
    Convert bounding box coordinates from [x_center, y_center, width, height]
    to [x_min, y_min, x_max, y_max].
    Works for both PyTorch Tensors and NumPy arrays.
    """
    xy = x[..., :2]  # center x, y
    wh = x[..., 2:4]  # width, height
    
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    
    if isinstance(x, torch.Tensor):
        return torch.cat([xy_min, xy_max], dim=-1)
    else:
        import numpy as np
        return np.concatenate([xy_min, xy_max], axis=-1)

def xyxy2xywh(x: Union[torch.Tensor, object]) -> Union[torch.Tensor, object]:
    """
    Convert bounding box coordinates from [x_min, y_min, x_max, y_max]
    to [x_center, y_center, width, height].
    """
    xy_min = x[..., :2]
    xy_max = x[..., 2:4]
    
    xy = (xy_min + xy_max) / 2
    wh = xy_max - xy_min
    
    if isinstance(x, torch.Tensor):
        return torch.cat([xy, wh], dim=-1)
    else:
        import numpy as np
        return np.concatenate([xy, wh], axis=-1)

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) of two sets of boxes.
    box1: Tensor of shape (N, 4) in [x_min, y_min, x_max, y_max] format
    box2: Tensor of shape (M, 4) in [x_min, y_min, x_max, y_max] format
    Returns: Tensor of shape (N, M) containing IoU values
    """
    # Get coordinates of intersection boxes
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Calculate areas of both sets of boxes
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    union = area1[:, None] + area2 - inter + eps

    return inter / union

def bbox_iou_loss(box1: torch.Tensor, box2: torch.Tensor, iou_type: str = "ciou", eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate IoU, GIoU, or CIoU loss for batch bounding boxes.
    box1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format (predictions)
    box2: Tensor of shape (N, 4) in [x1, y1, x2, y2] format (targets)
    Returns: iou, loss
    """
    # Overlap area coordinates
    x11, y11, x12, y12 = box1.unbind(-1)
    x21, y21, x22, y22 = box2.unbind(-1)

    # Intersection area
    inter = (torch.min(x12, x22) - torch.max(x11, x21)).clamp(min=0) * \
            (torch.min(y12, y22) - torch.max(y11, y21)).clamp(min=0)

    # Union area
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if iou_type in ["giou", "ciou"]:
        # Enclosing box coordinates
        cw = torch.max(x12, x22) - torch.min(x11, x21)
        ch = torch.max(y12, y22) - torch.min(y11, y21)

        if iou_type == "giou":
            c_area = cw * ch + eps
            return iou, 1.0 - (iou - (c_area - union) / c_area)
        
        elif iou_type == "ciou":
            # Coefficent c2 (diagonal length squared of enclosing box) and rho2 (distance between box centers squared)
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((x11 + x12 - x21 - x22) ** 2 + (y11 + y12 - y21 - y22) ** 2) / 4

            # Aspect ratio consistency v and trade-off parameter alpha
            # Using h.clamp to prevent division by zero or negative values
            h2_clamp = h2.clamp(min=1e-7)
            h1_clamp = h1.clamp(min=1e-7)
            v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2_clamp) - torch.atan(w1 / h1_clamp), 2)
            alpha = v / (1 - iou + v + eps)
            alpha = alpha.detach()

            return iou, 1.0 - (iou - (rho2 / c2 + v * alpha))

    return iou, 1.0 - iou
