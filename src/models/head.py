import torch
import torch.nn as nn
from src.models.blocks import ConvBNSiLU
from configs import config

class DecoupledHead(nn.Module):
    """
    Decoupled Detection Head for Mini YOLO inspired by YOLOv8/YOLO11.
    For each scale (stride 8, 16, 32), it passes PANet neck features through separate:
        - Classification branch (using ConvBNSiLU blocks)
        - Bounding Box Regression branch (using ConvBNSiLU blocks)
    
    Predicts:
        - Box coordinates: [tx, ty, tw, th] (4 channels)
        - Objectness score: [obj] (1 channel)
        - Class probabilities: [cls1, cls2, ...] (num_classes channels)
        
    Layout:
                   PANet Input (N_i)
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      [Cls ConvBNSiLU]     [Reg ConvBNSiLU]
      [Cls ConvBNSiLU]     [Reg ConvBNSiLU]
             │                   │
         ┌───┴───┐               │
         ▼       ▼               ▼
      [ClsPred] [ObjPred]     [RegPred]
    """
    def __init__(self, num_classes, base_channels=config.BASE_CHANNELS, strides=config.STRIDES):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.num_scales = len(strides)
        
        # Input channel sizes corresponding to fused PANet outputs N3, N4, N5
        in_channels = [base_channels * 4, base_channels * 8, base_channels * 16] # [64, 128, 256]

        # Define ModuleLists for decoupled layers across scales
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for c in in_channels:
            # 1. Classification branch: 2 sequential ConvBNSiLU layers
            self.cls_convs.append(nn.Sequential(
                ConvBNSiLU(c, c, kernel_size=3, stride=1, padding=1),
                ConvBNSiLU(c, c, kernel_size=3, stride=1, padding=1)
            ))
            # 2. Bounding Box Regression branch: 2 sequential ConvBNSiLU layers
            self.reg_convs.append(nn.Sequential(
                ConvBNSiLU(c, c, kernel_size=3, stride=1, padding=1),
                ConvBNSiLU(c, c, kernel_size=3, stride=1, padding=1)
            ))
            
            # 3. Final prediction conv layers (1x1 kernels)
            self.cls_preds.append(nn.Conv2d(c, num_classes, kernel_size=1))
            self.reg_preds.append(nn.Conv2d(c, 4, kernel_size=1))
            self.obj_preds.append(nn.Conv2d(c, 1, kernel_size=1))  # Objectness prediction branch

        # 4. Initialize grid cache dictionary to accelerate meshgrid generation
        self.grid_cache = {}

        # 5. Apply weights initialization automatically
        self._init_weights()

    def _init_weights(self):
        """
        Private weight initialization helper:
        - Kaiming Normal initialization for Conv2d layers.
        - BatchNorm2d initialized with weight = 1, bias = 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get_grid(self, h, w, device):
        """
        Generates meshgrid coordinates for a given spatial resolution (h, w)
        in a fully compile-friendly and trace-safe manner (avoiding graph breaks).
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device), 
            torch.arange(w, device=device), 
            indexing="ij"
        )
        return torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).to(torch.float32)

    def forward_single_scale(self, x, i):
        """
        Helper method to process prediction outputs for a single scale.
        """
        stride = self.strides[i]
        batch_size, _, h, w = x.shape
        
        # 1. Forward features through decoupled convolutional branches
        cls_feat = self.cls_convs[i](x)
        reg_feat = self.reg_convs[i](x)
        
        # 2. Get predictions
        pred_cls = self.cls_preds[i](cls_feat)  # Class logits: (B, num_classes, H, W)
        pred_reg = self.reg_preds[i](reg_feat)  # Box regression logits: (B, 4, H, W)
        pred_obj = self.obj_preds[i](cls_feat)  # Objectness logits: (B, 1, H, W)
        
        # 3. Reshape/Permute outputs to (B, H*W, C)
        pred_reg = pred_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        pred_obj = pred_obj.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)

        # 4. Concatenate raw predictions for this scale: [tx, ty, tw, th, obj, cls1, cls2...]
        pred_scale = torch.cat((pred_reg, pred_obj, pred_cls), dim=-1)
        
        # 5. Retrieve cached grid coordinates
        grid = self.get_grid(h, w, x.device)
        
        # 6. Build stride mapping tensor
        strides_scale = torch.full((h * w, 1), stride, device=x.device, dtype=torch.float32)

        # 7. Decode bounding boxes ONLY during inference/evaluation (Optimizes training memory)
        decoded_box = None
        if not self.training:
            tx, ty, tw, th = pred_reg.unbind(-1)
            img_h = h * stride
            img_w = w * stride
            
            bx = (grid[:, 0] + torch.sigmoid(tx)) * stride / img_w
            by = (grid[:, 1] + torch.sigmoid(ty)) * stride / img_h
            bw = torch.exp(tw.clamp(-5.0, 5.0)) * stride / img_w
            bh = torch.exp(th.clamp(-5.0, 5.0)) * stride / img_h
            
            decoded_box = torch.stack((bx, by, bw, bh), dim=-1)  # (B, H*W, 4)

        return pred_scale, grid, strides_scale, decoded_box

    def forward(self, feats):
        """
        feats: List of feature maps [N3, N4, N5] from PANet Neck
        Returns:
            Dictionary containing combined predictions, grids, strides, and decoded boxes.
        """
        preds_all = []
        grids_all = []
        strides_all = []
        decoded_boxes_all = []

        # Iterate over scales (strides 8, 16, 32)
        for i, x in enumerate(feats):
            pred_scale, grid, strides_scale, decoded_box = self.forward_single_scale(x, i)
            
            preds_all.append(pred_scale)
            grids_all.append(grid)
            strides_all.append(strides_scale)
            if decoded_box is not None:
                decoded_boxes_all.append(decoded_box)

        # Concatenate scale prediction outputs along sequence dimension (total_points)
        predictions = torch.cat(preds_all, dim=1)                 # (B, total_points, 5 + num_classes)
        grids = torch.cat(grids_all, dim=0)                        # (total_points, 2)
        strides = torch.cat(strides_all, dim=0)                    # (total_points, 1)

        out = {
            "pred": predictions,
            "grid": grids,
            "stride": strides
        }

        # Only decode and return decoded_box when not training
        if not self.training:
            out["decoded_box"] = torch.cat(decoded_boxes_all, dim=1)   # (B, total_points, 4)

        return out
