import torch
import torch.nn as nn
from src.utils.boxes import bbox_iou_loss, xywh2xyxy
from configs import config

class BCEWithLogitsFocalLoss(nn.Module):
    """
    Focal Loss wrapper for binary classification with logits.
    Focal Loss = -alpha * (1 - p)^gamma * log(p) for positive samples,
                 -(1 - alpha) * p^gamma * log(1 - p) for negative samples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        # Calculate p_t (probability of the target class)
        p_t = p * targets + (1 - p) * (1 - targets)
        # Apply focal scaling factor
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class MiniYOLOLoss(nn.Module):
    """
    Loss function for Mini YOLO.
    Consists of three parts:
        1. Bounding box regression loss (CIoU loss) on positive grid cells.
        2. Objectness loss (Focal Loss) on all grid cells.
        3. Classification loss (Focal Loss) on positive grid cells.
    """
    def __init__(self, num_classes, strides=config.STRIDES, 
                 box_weight=config.BOX_WEIGHT, obj_weight=config.OBJ_WEIGHT, cls_weight=config.CLS_WEIGHT,
                 label_smoothing=config.LABEL_SMOOTHING):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.label_smoothing = label_smoothing
        
        # Focal Loss criteria (alpha=0.25, gamma=2.0)
        self.focal_obj = BCEWithLogitsFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        self.focal_cls = BCEWithLogitsFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    def forward(self, predictions, targets, img_size):
        """
        predictions: Dict from MiniYOLO forward pass.
            - "pred": (B, M, 5 + num_classes) -> [tx, ty, tw, th, obj, cls1, cls2...]
            - "decoded_box": (B, M, 4) -> [bx, by, bw, bh] (normalized) [Optional during training]
            - "grid": (M, 2) -> [grid_x, grid_y]
            - "stride": (M, 1) -> [stride]
        targets: Tensor of shape (total_objects_in_batch, 6)
            - [batch_idx, class_id, x_center, y_center, width, height] (normalized)
        img_size: int (image resolution, e.g., 416)
        """
        pred_raw = predictions["pred"]
        pred_decoded = predictions.get("decoded_box", None)
        device = pred_raw.device
        batch_size = pred_raw.shape[0]
        total_points = pred_raw.shape[1]

        # 1. Initialize target tensors matching prediction shape
        # target box: (B, M, 4)
        t_box = torch.zeros((batch_size, total_points, 4), device=device)
        # target objectness: (B, M, 1)
        t_obj = torch.zeros((batch_size, total_points, 1), device=device)
        # target class probabilities: (B, M, num_classes)
        t_cls = torch.zeros((batch_size, total_points, self.num_classes), device=device)
        
        # Keep track of which cells are positive (contain an object center or neighbor)
        pos_mask = torch.zeros((batch_size, total_points), dtype=torch.bool, device=device)

        # 2. Build target assignment (map ground truths to the appropriate grid cells)
        grid_sizes = [img_size // s for s in self.strides]
        
        # Offsets in the flattened M points array for each scale
        offsets = []
        curr_offset = 0
        for sz in grid_sizes:
            offsets.append(curr_offset)
            curr_offset += sz * sz

        # Loop through each batch item to match ground truth boxes
        for b in range(batch_size):
            img_gts = targets[targets[:, 0] == b]
            if len(img_gts) == 0:
                continue

            for gt in img_gts:
                class_id = int(gt[1])
                gx, gy, gw, gh = gt[2:6]  # normalized coordinates

                # Match this ground truth box to grid cells at each scale (strides 8, 16, 32)
                for scale_idx, stride in enumerate(self.strides):
                    sz = grid_sizes[scale_idx]
                    offset = offsets[scale_idx]

                    # Grid coordinates of ground truth center
                    col = int(gx * sz)
                    row = int(gy * sz)

                    # Multi-Positive Assignment candidates: center, up, down, left, right
                    candidates = [
                        (col, row),
                        (col, row - 1),
                        (col, row + 1),
                        (col - 1, row),
                        (col + 1, row)
                    ]

                    # Assign all valid neighbor cells within spatial bounds
                    for c_col, c_row in candidates:
                        if 0 <= c_col < sz and 0 <= c_row < sz:
                            flat_idx = offset + c_row * sz + c_col
                            t_box[b, flat_idx] = gt[2:6]
                            t_obj[b, flat_idx, 0] = 1.0
                            t_cls[b, flat_idx, class_id] = 1.0
                            pos_mask[b, flat_idx] = True

        # 3. Calculate Losses
        loss_box = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        mean_iou = 0.0

        # Count positive matches
        n_pos = pos_mask.sum()

        if n_pos > 0:
            # Extract positive predictions and targets
            if pred_decoded is not None:
                pos_pred_boxes = pred_decoded[pos_mask]  # (n_pos, 4)
            else:
                # Decode ONLY the positive matched boxes (Optimizes training memory)
                pos_pred_raw_reg = pred_raw[pos_mask][:, :4]  # (n_pos, 4)
                tx, ty, tw, th = pos_pred_raw_reg.unbind(-1)
                
                # Fetch matching grid cell indices
                _, m_idx = torch.where(pos_mask)
                pos_grid = predictions["grid"][m_idx]       # (n_pos, 2)
                pos_stride = predictions["stride"][m_idx]   # (n_pos, 1)
                
                # Decoded coordinates normalized to [0, 1]
                bx = (pos_grid[:, 0] + torch.sigmoid(tx)) * pos_stride[:, 0] / img_size
                by = (pos_grid[:, 1] + torch.sigmoid(ty)) * pos_stride[:, 0] / img_size
                bw = torch.exp(tw.clamp(-5.0, 5.0)) * pos_stride[:, 0] / img_size
                bh = torch.exp(th.clamp(-5.0, 5.0)) * pos_stride[:, 0] / img_size
                
                pos_pred_boxes = torch.stack((bx, by, bw, bh), dim=-1)

            pos_target_boxes = t_box[pos_mask]        # (n_pos, 4)

            # Compute box regression loss (CIoU)
            pos_pred_xyxy = xywh2xyxy(pos_pred_boxes)
            pos_target_xyxy = xywh2xyxy(pos_target_boxes)

            iou, iou_loss = bbox_iou_loss(pos_pred_xyxy, pos_target_xyxy, iou_type="ciou")
            loss_box = iou_loss.mean()
            mean_iou = iou.mean().item()

            # Compute classification loss with optional label smoothing
            pos_pred_cls_logits = pred_raw[pos_mask][:, 5:]   # (n_pos, num_classes)
            pos_target_cls = t_cls[pos_mask]                  # (n_pos, num_classes)
            
            if self.label_smoothing > 0.0:
                pos_target_cls = pos_target_cls * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
                
            loss_cls = self.focal_cls(pos_pred_cls_logits, pos_target_cls)

        # Compute objectness loss across all predictions (Focal Loss)
        pred_obj_logits = pred_raw[..., 4:5]                  # (B, M, 1)
        loss_obj = self.focal_obj(pred_obj_logits, t_obj)

        # Total Loss
        total_loss = (
            self.box_weight * loss_box + 
            self.obj_weight * loss_obj + 
            self.cls_weight * loss_cls
        )

        return total_loss, {
            "loss": total_loss.item(),
            "box_loss": loss_box.item(),
            "obj_loss": loss_obj.item(),
            "cls_loss": loss_cls.item(),
            "n_pos": n_pos.item(),
            "positive_samples": n_pos.item(),
            "mean_iou": mean_iou
        }
