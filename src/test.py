import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Import local modules
import config
from models.yolo import MiniYOLO
from utils.nms import non_max_suppression

def load_model(checkpoint_path, device):
    """
    Load the MiniYOLO model from a saved checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Checkpoint file '{checkpoint_path}' not found.\n"
            f"Please run training first: python src/train.py"
        )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    chk_config = checkpoint["config"]
    
    model = MiniYOLO(
        num_classes=chk_config["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from checkpoint '{checkpoint_path}' (Epoch {checkpoint['epoch']+1})")
    return model, chk_config["class_names"]

@torch.no_grad()
def run_test_inference(test_img_dir, output_dir, model, class_names, device, conf_thres=0.3, iou_thres=0.45):
    """
    Runs batch inference on all images in test_img_dir and saves visual predictions
    with bounding boxes drawn in output_dir.
    """
    # 1. Verify and create directories
    if not os.path.exists(test_img_dir):
        print(f"❌ Test images directory '{test_img_dir}' does not exist.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Get all images in directory
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(test_img_dir) 
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_files:
        print(f"❌ No valid images found in '{test_img_dir}'.")
        return

    print(f"📷 Found {len(image_files)} images for testing.")
    print(f"🏃 Running inference and saving annotated images to: '{output_dir}'...\n")

    # Font setup
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    # Colors for the classes
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200)]

    # 3. Iterate through images and run inference
    for img_name in tqdm(image_files, desc="Testing"):
        img_path = os.path.join(test_img_dir, img_name)
        original_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = original_img.size

        # Preprocess
        resized_img = original_img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        img_np = np.array(resized_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Forward pass
        predictions = model(img_tensor)

        # NMS
        nms_predictions = non_max_suppression(
            predictions, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres,
            max_det=config.MAX_DETECTIONS
        )
        detections = nms_predictions[0]

        # Draw detections
        draw_img = original_img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        detected_list = []

        if len(detections) > 0:
            for det in detections:
                x1_n, y1_n, x2_n, y2_n = det[:4].cpu().numpy()
                conf = det[4].item()
                class_id = int(det[5].item())
                class_name = class_names[class_id]

                # Convert normalized to absolute coordinates
                x1, y1 = x1_n * orig_w, y1_n * orig_h
                x2, y2 = x2_n * orig_w, y2_n * orig_h
                
                detected_list.append(f"{class_name} ({conf:.2%})")

                # Choose color and draw bounding box
                color = colors[class_id % len(colors)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label
                label_text = f"{class_name} {conf:.2f}"
                if font:
                    text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
                    draw.text((x1 + 3, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)
                else:
                    draw.text((x1 + 2, y1 - 10), label_text, fill=color)

        # Print detection summaries to console if needed
        # (Commented out to prevent terminal flooding for large directories, but summary stats are recorded)
        
        # Save output image
        output_path = os.path.join(output_dir, f"det_{img_name}")
        draw_img.save(output_path)

    print(f"\n🎉 Testing finished! Annotated images saved successfully to '{output_dir}'.")

def main():
    device = config.DEVICE
    checkpoint_path = config.MODEL_SAVE_PATH
    
    # Check if a custom test directory path is provided or use default test dir under dataset
    test_img_dir = os.path.join(config.DATA_DIR, "test", "images")
    output_dir = os.path.join(config.RUNS_DIR, "test")
    
    # Fallback to validation images if test directory does not exist
    if not os.path.exists(test_img_dir):
        print(f"⚠️ Test folder '{test_img_dir}' not found.")
        val_fallback = os.path.join(config.DATA_DIR, "val", "images")
        if os.path.exists(val_fallback):
            print(f"🔄 Falling back to running testing on validation folder: '{val_fallback}'")
            test_img_dir = val_fallback
            output_dir = os.path.join(config.RUNS_DIR, "val_test")

    try:
        model, class_names = load_model(checkpoint_path, device)
        run_test_inference(test_img_dir, output_dir, model, class_names, device, conf_thres=0.3, iou_thres=0.45)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
