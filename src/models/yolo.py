import torch
import torch.nn as nn
from src.models.backbone import MiniDarknet
from src.models.neck import MiniPANet
from src.models.head import DecoupledHead
from configs import config

class MiniYOLO(nn.Module):
    """
    Mini YOLO Object Detection Model.
    Combines the MiniDarknet backbone, MiniPANet neck, and DecoupledHead detection head.

    ASCII Architecture Pipeline Diagram:
    
           Input Image (B, 3, H, W)
                     │
                     ▼
             [ MiniDarknet ] (Backbone) ──────► P3, P4, P5 (Feature Maps)
                     │
                     ▼
              [ MiniPANet ] (Neck Feature Fusion) ──► N3, N4, N5 (Fused Maps)
                     │
                     ▼
             [ DecoupledHead ] (Detection Head)
                     │
             ┌───────┴───────┐
             ▼               ▼
        (Training)      (Evaluation/Inference)
            │                │
            ▼                ▼
       Raw Projs       [ inference() ]
       (head_outputs)  Sigmoid Activations
                       Concatenated predictions
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = config.BASE_CHANNELS
        self.strides = config.STRIDES
        self.names = config.CLASS_NAMES
        self.export = False
        
        # 1. Feature Extractor (Backbone)
        self.backbone = MiniDarknet(in_channels=3, base_channels=self.base_channels)
        
        # 2. Feature Fusion Neck (PANet)
        self.neck = MiniPANet(base_channels=self.base_channels)
        
        # 3. Detection Head (Decoupled Head)
        self.head = DecoupledHead(num_classes=self.num_classes, base_channels=self.base_channels, strides=self.strides)

    def forward(self, x: torch.Tensor) -> dict | torch.Tensor:
        """
        x: Tensor of shape (B, 3, H, W)
        Returns:
            If training:
                Dictionary containing "pred", "grid", "stride"
            If inference:
                Tensor of shape (B, total_points, 5 + num_classes)
        """
        # Input Validation
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise ValueError(
                f"❌ Input x must be a 4D PyTorch Tensor (B, C, H, W). "
                f"Got shape {x.shape if isinstance(x, torch.Tensor) else type(x)}"
            )

        # 1. Backbone features
        backbone_feats = self.backbone(x)
        p3 = backbone_feats["P3"]
        p4 = backbone_feats["P4"]
        p5 = backbone_feats["P5"]
        
        # 2. Neck features
        n3, n4, n5 = self.neck(p3, p4, p5)
        
        # 3. Head predictions
        head_outputs = self.head([n3, n4, n5])
        
        if self.training:
            return head_outputs
        else:
            return self.inference(head_outputs)

    def inference(self, head_outputs: dict) -> torch.Tensor:
        """
        Private-like evaluation/inference helper:
        - Applies sigmoid to objectness and class scores.
        - Concatenates decoded boxes + objectness + class probabilities.
        - Returns the final prediction tensor.
        """
        # Get decoded boxes
        decoded_boxes = head_outputs["decoded_box"]  # (B, total_points, 4) -> [x, y, w, h]
        
        # Extract raw obj and class scores
        raw_pred = head_outputs["pred"]  # (B, total_points, 5 + num_classes)
        raw_obj = raw_pred[..., 4:5]
        raw_cls = raw_pred[..., 5:]
        
        # Apply activation functions
        obj_prob = torch.sigmoid(raw_obj)
        cls_prob = torch.sigmoid(raw_cls)
        
        # Combine into a single inference tensor
        inference_out = torch.cat((decoded_boxes, obj_prob, cls_prob), dim=-1)
        return inference_out
