import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
from configs import config

class YOLODataset(Dataset):
    """
    YOLO Dataset loader that parses images and label files in YOLO format:
    class_idx x_center y_center width height (all coordinates normalized between 0 and 1)
    """
    def __init__(self, img_dir: str | Path, label_dir: str | Path, transform: Optional[Callable] = None) -> None:
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        # 1. Scan and validate every image file
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.img_paths: List[Path] = []
        
        print(f"🔍 Scanning and validating images in: {self.img_dir}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"❌ Image directory '{self.img_dir}' does not exist.")
            
        for path in sorted(self.img_dir.iterdir()):
            if path.suffix.lower() in valid_extensions:
                try:
                    # Quick integrity check without loading full pixel data
                    with Image.open(path) as temp_img:
                        temp_img.verify()
                    self.img_paths.append(path)
                except Exception as e:
                    print(f"⚠️ Warning: Skipping corrupted or unreadable image '{path.name}': {e}")
                    
        # 2. Validate and pre-cache all label files
        self.label_cache: Dict[Path, np.ndarray] = {}
        print(f"🔍 Parsing and validating labels in: {self.label_dir}")
        
        for img_path in self.img_paths:
            label_path = self.label_dir / (img_path.stem + ".txt")
            boxes = []
            
            if label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) != 5:
                                if len(parts) > 5:
                                    try:
                                        class_idx = int(parts[0])
                                        if not (0 <= class_idx < config.NUM_CLASSES):
                                            print(f"⚠️ Warning: Class index {class_idx} in '{label_path.name}' line {line_num} is out of bounds (0-{config.NUM_CLASSES-1}). Skipping line.")
                                            continue
                                        
                                        all_coords = [float(x) for x in parts[1:]]
                                        if len(all_coords) % 2 != 0:
                                            all_coords = all_coords[:-1]
                                            
                                        if len(all_coords) >= 4:
                                            xs = all_coords[0::2]
                                            ys = all_coords[1::2]
                                            
                                            xmin, xmax = min(xs), max(xs)
                                            ymin, ymax = min(ys), max(ys)
                                            
                                            xc = (xmin + xmax) / 2.0
                                            yc = (ymin + ymax) / 2.0
                                            w = xmax - xmin
                                            h = ymax - ymin
                                            
                                            coords = [xc, yc, w, h]
                                            if any(c < 0.0 or c > 1.0 for c in coords):
                                                print(f"⚠️ Warning: Converted coordinate value(s) {coords} out of bounds [0, 1] in '{label_path.name}' line {line_num}. Skipping line.")
                                                continue
                                                
                                            boxes.append([class_idx, xc, yc, w, h])
                                            continue
                                    except ValueError:
                                        pass
                                print(f"⚠️ Warning: Invalid label format in '{label_path.name}' line {line_num} (expected 5 values, got {len(parts)}). Skipping line.")
                                continue
                            try:
                                class_idx = int(parts[0])
                                if not (0 <= class_idx < config.NUM_CLASSES):
                                    print(f"⚠️ Warning: Class index {class_idx} in '{label_path.name}' line {line_num} is out of bounds (0-{config.NUM_CLASSES-1}). Skipping line.")
                                    continue
                                    
                                coords = [float(x) for x in parts[1:5]]
                                if any(c < 0.0 or c > 1.0 for c in coords):
                                    print(f"⚠️ Warning: Coordinate value(s) {coords} out of bounds [0, 1] in '{label_path.name}' line {line_num}. Skipping line.")
                                    continue
                                    
                                boxes.append([class_idx] + coords)
                            except ValueError as e:
                                print(f"⚠️ Warning: Failed to parse values in '{label_path.name}' line {line_num}: {e}. Skipping line.")
                                continue
                except Exception as e:
                    print(f"⚠️ Warning: Error reading label file '{label_path.name}': {e}. Skipping file.")

            # Store validated labels in cache (handles empty-label images as well)
            if not boxes:
                self.label_cache[img_path] = np.zeros((0, 5), dtype=np.float32)
            else:
                self.label_cache[img_path] = np.array(boxes, dtype=np.float32)

        # 3. Optional image caching in memory
        self.image_cache: Dict[Path, Image.Image] = {}
        if config.CACHE_IMAGES:
            print("💾 Caching dataset images in memory...")
            for img_path in self.img_paths:
                try:
                    self.image_cache[img_path] = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to cache image '{img_path.name}': {e}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        
        # 1. Retrieve image (from cache or disk)
        if img_path in self.image_cache:
            image = self.image_cache[img_path].copy()
        else:
            image = Image.open(img_path).convert("RGB")
            
        # 2. Retrieve pre-cached label
        boxes = self.label_cache[img_path].copy()
        
        # 3. Apply transformations
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        return image, boxes

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    Combines images and labels into batch tensors.
    
    Returns:
        images: Tensor of shape (batch_size, 3, H, W)
        targets: Tensor of shape (total_objects_in_batch, 6)
                 where columns are [batch_idx, class_id, x, y, w, h]
    """
    images, targets = zip(*batch)
    
    # Stack images along batch dimension
    images = torch.stack(images, dim=0)
    
    # Prepend batch index to targets
    batch_targets = []
    for i, target in enumerate(targets):
        if len(target) > 0:
            # target is shape (num_objs, 5) -> [class_id, x, y, w, h]
            num_objs = target.shape[0]
            batch_idx = torch.full((num_objs, 1), i, dtype=torch.float32)
            
            # Convert target list or numpy array to tensor if needed
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)
                
            # Concatenate batch_idx with class_id and coords
            target_with_idx = torch.cat((batch_idx, target), dim=1)
            batch_targets.append(target_with_idx)
            
    if batch_targets:
        targets_tensor = torch.cat(batch_targets, dim=0)
    else:
        targets_tensor = torch.zeros((0, 6), dtype=torch.float32)
        
    return images, targets_tensor
