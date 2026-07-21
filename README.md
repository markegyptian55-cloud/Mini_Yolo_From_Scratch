# MiniYOLO: Real-time Fatigue Expression Detector
*A custom, lightweight, production-ready MiniYOLO object detector modeled after the Ultralytics YOLOv8/YOLO11 architecture, specialized in identifying human expression and fatigue signals (`closed_eye`, `open_eye`, `yawning`).*

---

## 🏗️ System Architecture Overview
The system is divided into modular package components to maximize code reusability, training performance, and compilation compatibility:

```mermaid
graph TD
    subgraph Data Pipeline
        dataset[dataset.py <br> Polygon Converter & Cache]
        transforms[transforms.py <br> HSV, Affine, Flip Augmentations]
    end

    subgraph MiniYOLO Model
        backbone[backbone.py <br> Darknet Feature Extractor]
        neck[neck.py <br> PANet Multi-Scale Fuser]
        head[head.py <br> Decoupled Head]
        yolo[yolo.py <br> Model Wrappers]
        yolo --> backbone
        yolo --> neck
        yolo --> head
    end

    subgraph Loss & Metrics
        loss[yolo_loss.py <br> CIoU & Focal Loss]
        evaluator[evaluator.py <br> Common AP Calculations]
    end

    subgraph Execution Launchers
        train[train.py]
        validate[validate.py]
        predict[predict.py]
    end

    train --> Data Pipeline
    train --> MiniYOLO Model
    train --> loss
    validate --> evaluator
    predict --> evaluator
```

---

## 📂 Project Directory Structure

```
mini_yolo/
├── configs/
│   └── config.py              # Central configurations & central hyperparameter overrides
├── info/
│   ├── book.md                # Technical engineering documentation book
│   └── first report for train 1/ # Performance logs, charts, and predictions
├── runs/
│   ├── train/                 # Checkpoints (*.pth) and training logs
│   └── predictions/           # Inference outputs (annotated images)
├── src/
│   ├── data/
│   │   ├── dataset.py         # YOLO Dataset loader & dynamic polygon converter
│   │   └── transforms.py      # Augmentation pipeline classes
│   ├── engine/
│   │   ├── evaluator.py       # Centered validation matching & AP computation
│   │   ├── predictor.py       # High-performance FP16 inference engine
│   │   ├── trainer.py         # Training loop, optimizer/scheduler step manager
│   │   └── validator.py       # Standalone checkpoint evaluation launcher
│   ├── losses/
│   │   └── yolo_loss.py       # Multi-positive target matcher & loss functions
│   ├── models/
│   │   ├── backbone.py        # Darknet multi-scale feature extractor (P3, P4, P5)
│   │   ├── blocks.py          # ConvBNSiLU, Bottleneck, C2f, and SPPF modules
│   │   ├── head.py            # Decoupled bounding box, class, and obj head
│   │   ├── neck.py            # PANet neck multi-scale feature fuser
│   │   └── yolo.py            # Unified MiniYOLO network wrapper
│   ├── utils/
│   │   ├── boxes.py           # Geometric coordinate utilities (IoU, CIoU loss, etc.)
│   │   ├── logger.py          # Formatted console outputs
│   │   ├── metrics.py         # Confusion matrix and AP calculation helpers
│   │   ├── misc.py            # Extra system utilities
│   │   ├── nms.py             # Torchvision-accelerated class-agnostic NMS
│   │   ├── seed.py            # Reproducibility seed initializer
│   │   └── visualization.py   # Prediction box visualization & label renderer
│   ├── predict.py             # Inference launcher
│   ├── train.py               # Main training launcher
│   └── validate.py            # Main validation launcher
└── requirements.txt           # Dependency requirements
```

---

## 🛠️ Step 1: Environment Setup

We manage dependencies cleanly through a Python virtual environment.

1.  **Navigate to the project root directory**:
    ```powershell
    cd C:\Users\Admin\Desktop\mini_yolo
    ```
2.  **Activate the Virtual Environment**:
    *   **PowerShell**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **CMD**:
        ```cmd
        venv\Scripts\activate.bat
        ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 Step 2: Dataset Configuration

Your dataset follows the standard Ultralytics YOLO format located under the `dataset/` directory:

```
dataset/
├── train/
│   ├── images/          # Training images (.jpg, .png, etc.)
│   └── labels/          # YOLO label files (.txt) containing bounding boxes
└── val/
    ├── images/          # Validation images
    └── labels/          # YOLO label files
```

### Bounding Box Label Format
Each image has a corresponding `.txt` file of the same name (e.g., `image_001.jpg` matches `image_001.txt`). 
```
<class_id> <x_center> <y_center> <width> <height>
```
*   `class_id`: Integer index representing `0`: `closed_eye`, `1`: `open_eye`, `2`: `yawning`.
*   Coordinates are normalized between `0.0` and `1.0` relative to image dimensions.
*   **Automatic Polygon Support**: The dataset loader automatically detects and converts Roboflow polygon segmentation coordinates (`len(coords) > 5`) into standard bounding box coordinates on the fly.

---

## ✏️ Step 3: Match Configuration to Your Dataset

All training and inference parameters are configured inside **`configs/config.py`**. Key configurations include:
*   **`CLASS_NAMES`**: Configured to `['closed_eye', 'open_eye', 'yawning']`.
*   **`IMG_SIZE`**: Set to `416` (must be a multiple of 32).
*   **`OPTIMIZER`**: Options include `"AdamW"` or `"SGD"`.
*   **`SCHEDULER`**: Options include `"CosineAnnealingLR"`, `"CosineAnnealingWarmRestarts"`, or `None`.
*   **`RESUME` & `CHECKPOINT_PATH`**: Set `RESUME = True` and point `CHECKPOINT_PATH` to a saved checkpoint (e.g., `"runs/train/mini_yolo_epoch_20.pth"`) to continue training seamlessly.

---

## 🚀 Step 4: Run Training, Validation & Inference

All modules are executed as Python packages to ensure clean module path resolution:

### 1. Start/Resume Training:
```powershell
python -m src.train
```
*   Verifies dataset directories and begins training. Saves checkpoint weights periodically to `runs/train/` and automatically saves the best performing weights to `runs/train/mini_yolo_best.pth`.

### 2. Standalone Model Evaluation:
```powershell
python -m src.validate
```
*   Loads the best model weights checkpoint, evaluates precision and recall metrics on the validation dataset, and logs a formatted results table.

### 3. Single-Image Bounding Box Prediction:
```powershell
python -m src.predict
```
*   Loads the trained model, performs inference on sample validation images, and saves visual box overlays directly to `runs/predictions/images/`.

---

## 📊 Chapter 5: Training Reports & Performance Logs

Detailed training statistics, validation precision/recall progression curves, and engineering reports are documented inside the **`info/`** folder:
*   **`info/book.md`**: Complete technical guide detailing the project history, compilation modifications, and structural refactorings.
*   **`info/first report for train 1/`**: Performance logs, validation mAP charts (`map_curve.png`), and sample inference screenshots for your first 20-epoch training session.
