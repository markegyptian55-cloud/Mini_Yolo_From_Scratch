import os
from configs import config
from src.engine.predictor import load_model, predict_image

def main() -> None:
    device = config.DEVICE
    checkpoint_path = config.MODEL_SAVE_PATH
    
    # 1. Load model
    try:
        model, class_names = load_model(checkpoint_path, device)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Select image to predict on
    image_path = None
    if os.path.exists(config.VAL_IMG_DIR):
        files = [f for f in os.listdir(config.VAL_IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            image_path = os.path.join(config.VAL_IMG_DIR, files[0])
            
    if image_path is None:
        print("❌ No test images found in validation directory.")
        return

    # 3. Run prediction
    result_img = predict_image(image_path, model, class_names, device, conf_thres=config.CONF_THRESHOLD, iou_thres=config.NMS_IOU_THRESHOLD)
    
    # 4. Save result
    output_path = os.path.join(config.RUNS_DIR, "prediction_output.jpg")
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    result_img.save(output_path)
    print(f"\n🎉 Prediction visualization saved successfully to: '{output_path}'")

if __name__ == "__main__":
    main()
