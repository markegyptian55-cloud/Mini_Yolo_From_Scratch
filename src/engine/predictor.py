import os
import time
import json
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# Import local modules
from configs import config
from src.models.yolo import MiniYOLO
from src.utils.nms import non_max_suppression
from src.utils.visualization import draw_predictions
from src.data.transforms import Compose, Resize, ToTensor, Normalize

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[MiniYOLO, List[str]]:
    """
    Load the MiniYOLO model from a saved checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Checkpoint file '{checkpoint_path}' not found.\n"
            f"Please run training first to save the model: python train.py"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    chk_config = checkpoint["config"]
    
    model = MiniYOLO(
        num_classes=chk_config["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded checkpoint '{checkpoint_path}' from epoch {checkpoint['epoch']+1} "
          f"(Best Validation mAP@50: {checkpoint['best_map50']:.4f})")
    
    return model, chk_config["class_names"]

@torch.no_grad()
def predict_image(
    image_path: str,
    model: MiniYOLO,
    class_names: List[str],
    device: torch.device,
    conf_thres: float = config.CONF_THRESHOLD,
    iou_thres: float = config.NMS_IOU_THRESHOLD,
    classes: Optional[List[int]] = config.FILTER_CLASSES,
    agnostic: bool = config.AGNOSTIC_NMS
) -> Image.Image:
    """
    Runs MiniYOLO inference on an input image, applies NMS,
    and returns a PIL image with drawn bounding boxes.
    """
    # 0. Measure total time start
    t_total_start = time.time()

    # 1. Preprocess: load and run transforms identically to validation
    t_prep_start = time.time()
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size

    val_transform = Compose([
        Resize(config.IMG_SIZE),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # We pass a dummy numpy array for boxes to satisfy signature
    dummy_boxes = np.zeros((0, 5), dtype=np.float32)
    img_tensor, _ = val_transform(original_img, dummy_boxes)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    t_prep = time.time() - t_prep_start

    # 2. Model Inference
    t_inf_start = time.time()
    # Runs inside autocast block if AMP is enabled
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    try:
        autocast_context = torch.amp.autocast(device_type=device_type, enabled=config.USE_AMP)
    except AttributeError:
        autocast_context = torch.cuda.amp.autocast(enabled=config.USE_AMP)

    with autocast_context:
        predictions = model(img_tensor)
    t_inf = time.time() - t_inf_start

    # 3. Non-Maximum Suppression
    t_nms_start = time.time()
    nms_predictions = non_max_suppression(
        predictions, 
        conf_thres=conf_thres, 
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic,
        max_det=config.MAX_DETECTIONS
    )
    detections = nms_predictions[0]  # shape: (num_dets, 6) -> [xyxy_norm, conf, class_id]
    t_nms = time.time() - t_nms_start
    t_total = time.time() - t_total_start

    # Print speeds if enabled
    if config.PRINT_SPEED:
        print(f"\n⚡ Speed / Latency Summary:")
        print(f"  • Preprocess time: {t_prep*1000:.2f} ms")
        print(f"  • Inference time:  {t_inf*1000:.2f} ms")
        print(f"  • NMS time:        {t_nms*1000:.2f} ms")
        print(f"  • Total latency:   {t_total*1000:.2f} ms")

    # Get absolute coordinates for drawing/saving
    detections_np = detections.cpu().numpy()
    abs_detections = []
    
    image_name = os.path.basename(image_path)
    print(f"\n🔍 Detections for '{image_name}':")
    if len(detections_np) == 0:
        print("  No objects detected.")
    else:
        for det in detections_np:
            x1_n, y1_n, x2_n, y2_n, conf, class_id = det
            class_id = int(class_id)
            class_name = class_names[class_id]

            # Scale to original dimensions
            x1 = x1_n * orig_w
            y1 = y1_n * orig_h
            x2 = x2_n * orig_w
            y2 = y2_n * orig_h
            
            abs_detections.append([x1, y1, x2, y2, conf, class_id])
            print(f"  Class: {class_name:12s} | Conf: {conf:.4f} | Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    abs_detections = np.array(abs_detections) if abs_detections else np.zeros((0, 6))

    # 4. Drawing overlay (call utility)
    draw_img = draw_predictions(original_img, abs_detections, class_names)

    # 5. Output directories setup
    pred_dir = os.path.join(config.RUNS_DIR, "predictions")
    img_save_dir = os.path.join(pred_dir, "images")
    lbl_save_dir = os.path.join(pred_dir, "labels")
    json_save_dir = os.path.join(pred_dir, "json")
    
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)

    base_name = os.path.splitext(image_name)[0]

    # Save outputs based on configuration flags
    if config.SAVE_PRED_IMAGES:
        img_path = os.path.join(img_save_dir, f"{base_name}_prediction.jpg")
        draw_img.save(img_path)
        print(f"💾 Saved annotated image to: {img_path}")

    if config.SAVE_TXT:
        txt_path = os.path.join(lbl_save_dir, f"{base_name}.txt")
        with open(txt_path, "w") as f:
            for det in detections_np:
                x1_n, y1_n, x2_n, y2_n, conf, class_id = det
                # Save in normalized format [class, x_center, y_center, w, h]
                x_center = (x1_n + x2_n) / 2.0
                y_center = (y1_n + y2_n) / 2.0
                w_norm = x2_n - x1_n
                h_norm = y2_n - y1_n
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        print(f"💾 Saved YOLO txt labels to: {txt_path}")

    if config.SAVE_JSON:
        json_path = os.path.join(json_save_dir, f"{base_name}.json")
        json_data = []
        for det in abs_detections:
            x1, y1, x2, y2, conf, class_id = det
            json_data.append({
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "confidence": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"💾 Saved JSON detections to: {json_path}")

    return draw_img
