import torch
import numpy as np
from PIL import Image
import math
from typing import List, Tuple, Union
from configs import config

class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms: List[object]) -> None:
        self.transforms = transforms

    def __call__(self, image: Union[Image.Image, np.ndarray], boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[Image.Image, torch.Tensor], Union[torch.Tensor, np.ndarray]]:
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes

class ToTensor:
    """
    Convert a PIL Image or numpy.ndarray to tensor and normalize to [0, 1].
    Converts boxes to torch.Tensor.
    """
    def __call__(self, image: Union[Image.Image, np.ndarray], boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image, dtype=np.float32) / 255.0
        else:
            img = image.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        
        # Convert boxes to torch.FloatTensor
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
            
        return img, boxes

class Resize:
    """
    Resize image to a given size (img_size, img_size).
    Supports:
        - Stretch Resize (default)
        - LetterBox Resize (preserves aspect ratio by padding borders)
    """
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image: Union[Image.Image, np.ndarray], boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[Image.Image, Union[torch.Tensor, np.ndarray]]:
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        target_w, target_h = self.size
        src_w, src_h = image.size

        if config.LETTERBOX:
            # Preserves aspect ratio, pads remaining margins with neutral gray color (114, 114, 114)
            r = min(target_w / src_w, target_h / src_h)
            new_w = int(round(src_w * r))
            new_h = int(round(src_h * r))
            
            # Resize image maintaining aspect ratio
            resized_image = image.resize((new_w, new_h), Image.BILINEAR)
            
            # Paste onto padded canvas
            canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
            dw = (target_w - new_w) // 2
            dh = (target_h - new_h) // 2
            canvas.paste(resized_image, (dw, dh))
            image = canvas
            
            # Recalculate normalized coordinates for labels
            if len(boxes) > 0:
                # Convert coords relative to padding and new resolution
                x_center = (boxes[:, 1] * src_w * r + dw) / target_w
                y_center = (boxes[:, 2] * src_h * r + dh) / target_h
                width = (boxes[:, 3] * src_w * r) / target_w
                height = (boxes[:, 4] * src_h * r) / target_h
                
                boxes[:, 1] = x_center
                boxes[:, 2] = y_center
                boxes[:, 3] = width
                boxes[:, 4] = height
        else:
            # Traditional stretch resize (aspect ratio not preserved)
            image = image.resize(self.size, Image.BILINEAR)
            
        return image, boxes

class RandomHorizontalFlip:
    """
    Horizontally flip the given Image and boxes randomly with a given probability.
    Uses torch.rand() for reproducible results.
    """
    def __init__(self, p: float = config.HFLIP_PROB) -> None:
        self.p = p

    def __call__(self, image: Union[Image.Image, np.ndarray], boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[Image.Image, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        if torch.rand(1).item() < self.p:
            # Flip image
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = np.fliplr(image)
            
            # Flip boxes (x_center = 1.0 - x_center)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]
                
        return image, boxes

class RandomHSV:
    """
    Randomly adjusts Hue, Saturation, and Value (Brightness) of the image.
    Inspired by YOLOv8 data augmentations.
    """
    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4, p: float = config.HSV_PROB) -> None:
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        self.p = p

    def __call__(self, image: Image.Image, boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[Image.Image, Union[torch.Tensor, np.ndarray]]:
        if torch.rand(1).item() < self.p:
            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            img_hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            
            # Generate random gains via torch.rand
            r = (torch.rand(3) * 2.0 - 1.0)  # three random values in [-1, 1]
            h_rand = r[0].item() * self.h_gain
            s_rand = r[1].item() * self.s_gain + 1.0
            v_rand = r[2].item() * self.v_gain + 1.0
            
            # Extract HSV channels
            h = img_hsv[..., 0].astype(np.int32)
            s = img_hsv[..., 1].astype(np.int32)
            v = img_hsv[..., 2].astype(np.int32)
            
            # Apply color adjustments with boundary clipping
            h = (h + int(h_rand * 255)) % 256
            s = np.clip(s * s_rand, 0, 255)
            v = np.clip(v * v_rand, 0, 255)
            
            # Store updated channels
            img_hsv[..., 0] = h.astype(np.uint8)
            img_hsv[..., 1] = s.astype(np.uint8)
            img_hsv[..., 2] = v.astype(np.uint8)
            
            # Convert back to PIL RGB
            image = Image.fromarray(img_hsv, "HSV").convert("RGB")
            
        return image, boxes

class RandomAffine:
    """
    Applies random translation, rotation, and scaling to the image and updates
    bounding box coordinates accordingly.
    """
    def __init__(self, degrees: float = 10.0, translate: float = 0.05, scale: float = 0.05, p: float = config.AFFINE_PROB) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(self, image: Image.Image, boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[Image.Image, Union[torch.Tensor, np.ndarray]]:
        if torch.rand(1).item() < self.p:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            w, h = image.size
            cx, cy = w / 2, h / 2
            
            # Sample parameters
            angle = (torch.rand(1).item() * 2.0 - 1.0) * self.degrees
            angle_rad = angle * math.pi / 180.0
            
            tx = (torch.rand(1).item() * 2.0 - 1.0) * self.translate * w
            ty = (torch.rand(1).item() * 2.0 - 1.0) * self.translate * h
            
            s = torch.rand(1).item() * (self.scale * 2.0) + (1.0 - self.scale)
            
            # 1. Forward Affine parameters (Source -> Target mapping)
            A = s * math.cos(angle_rad)
            B = -s * math.sin(angle_rad)
            C = cx + tx - A * cx - B * cy
            D = s * math.sin(angle_rad)
            E = s * math.cos(angle_rad)
            F = cy + ty - D * cx - E * cy
            
            # 2. Inverse Affine parameters (Target -> Source mapping for PIL)
            # Inverse of 2D matrix
            det = A * E - B * D
            if abs(det) > 1e-5:
                inv_a = E / det
                inv_b = -B / det
                inv_c = (B * F - E * C) / det
                inv_d = -D / det
                inv_e = A / det
                inv_f = (D * C - A * F) / det
                
                # Apply transformation to image
                image = image.transform((w, h), Image.AFFINE, (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f), resample=Image.BILINEAR)
                
                # Transform bounding boxes
                if len(boxes) > 0:
                    keep_indices = []
                    new_boxes = []
                    
                    for box in boxes:
                        class_id = box[0]
                        bx_c, by_c, bw, bh = box[1:5]
                        
                        # Convert normalized xywh to absolute corners
                        x1 = (bx_c - bw / 2) * w
                        y1 = (by_c - bh / 2) * h
                        x2 = (bx_c + bw / 2) * w
                        y2 = (by_c + bh / 2) * h
                        
                        # 4 corner coordinates of the box
                        corners = [
                            (x1, y1), (x2, y1),
                            (x1, y2), (x2, y2)
                        ]
                        
                        # Forward transform each corner point
                        new_corners_x = []
                        new_corners_y = []
                        for px, py in corners:
                            nx = A * px + B * py + C
                            ny = D * px + E * py + F
                            new_corners_x.append(nx)
                            new_corners_y.append(ny)
                            
                        # Bounding box of the transformed corners
                        x1_new = max(0, min(new_corners_x))
                        y1_new = max(0, min(new_corners_y))
                        x2_new = min(w, max(new_corners_x))
                        y2_new = min(h, max(new_corners_y))
                        
                        # Filter out boxes that are translated completely out of bounds or collapsed
                        w_new = x2_new - x1_new
                        h_new = y2_new - y1_new
                        if w_new > 1.0 and h_new > 1.0:
                            # Convert back to normalized coordinates
                            bx_c_new = (x1_new + w_new / 2) / w
                            by_c_new = (y1_new + h_new / 2) / h
                            bw_new = w_new / w
                            bh_new = h_new / h
                            
                            new_boxes.append([class_id, bx_c_new, by_c_new, bw_new, bh_new])
                            
                    if len(new_boxes) > 0:
                        if isinstance(boxes, torch.Tensor):
                            boxes = torch.tensor(new_boxes, dtype=torch.float32, device=boxes.device)
                        else:
                            boxes = np.array(new_boxes, dtype=np.float32)
                    else:
                        if isinstance(boxes, torch.Tensor):
                            boxes = torch.zeros((0, 5), dtype=torch.float32, device=boxes.device)
                        else:
                            boxes = np.zeros((0, 5), dtype=np.float32)
                            
        return image, boxes

class Normalize:
    """
    Normalize a tensor image with mean and standard deviation.
    Reads defaults from config.py if not specified.
    """
    def __init__(self, mean: List[float] = config.MEAN, std: List[float] = config.STD) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor, boxes: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, Union[torch.Tensor, np.ndarray]]:
        # Shift means to match image device
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        image = (image - mean) / std
        return image, boxes
