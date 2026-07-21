from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Optional

def draw_predictions(
    image: Image.Image,
    detections: np.ndarray,  # shape: (num_dets, 6) -> [x1, y1, x2, y2, conf, class_id] (absolute coordinates)
    class_names: List[str]
) -> Image.Image:
    """
    Draws bounding boxes and labels onto the image and returns the annotated image.
    """
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Try to load default font, otherwise fallback to standard text drawing
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    # Colors for different classes (Cycle through a list of 10 nice colors)
    colors = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)
    ]

    for det in detections:
        x1, y1, x2, y2 = det[:4]
        conf = det[4]
        class_id = int(det[5])
        class_name = class_names[class_id]

        # Choose color
        color = colors[class_id % len(colors)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label background and text
        label_text = f"{class_name} {conf:.2f}"
        if font:
            try:
                # Get text bounding box size
                text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                
                draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
                draw.text((x1 + 3, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)
            except Exception:
                # Fallback to drawing simple text if textbbox is unsupported
                draw.text((x1 + 2, y1 - 10), label_text, fill=color)
        else:
            # Fallback label drawing
            draw.text((x1 + 2, y1 - 10), label_text, fill=color)

    return draw_img
