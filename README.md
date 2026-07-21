# Mini YOLO from Scratch: Training & Inference Guide

This guide walks you through setting up your environment, configuring your dataset, and running training and prediction on your eye/yawn detection dataset.

---

## 🛠️ Step 1: Environment Setup

We recommend setting up a Python virtual environment to manage dependencies cleanly.

1.  **Open your terminal** in the project root directory (`C:\Users\Admin\Desktop\mini_yolo`).
2.  **Activate the Virtual Environment**:
    The virtual environment has already been initialized in the `env` folder. Run:
    *   **CMD**:
        ```cmd
        env\Scripts\activate.bat
        ```
    *   **PowerShell**:
        ```powershell
        env\Scripts\activate.ps1
        ```
3.  **Install Required Libraries Offline**:
    Since all required offline wheels are stored inside `env/wheels/`, install them offline:
    ```bash
    pip install --no-index --find-links=env/wheels -r requirements.txt
    ```

---

## 📂 Step 2: Dataset Configuration

Your dataset follows the standard Ultralytics YOLO format located under `C:\Users\Admin\Desktop\mini_yolo\dataset`:

```
dataset/
├── train/
│   ├── images/          # Training images (.jpg, .png, etc.)
│   └── labels/          # Matching label text files for training images
├── val/
│   ├── images/          # Validation images
│   └── labels/          # Matching label text files for validation images
└── dataset.yaml         # Dataset classes configuration metadata
```

### Bounding Box Label Format
Each image has a corresponding `.txt` file of the same name (e.g., `image_001.jpg` matches `image_001.txt`). Each line represents one object:
```
<class_id> <x_center> <y_center> <width> <height>
```
*   `class_id`: Integer index of the class:
    *   `0`: `closed_eye`
    *   `1`: `open_eye`
    *   `2`: `yawning`
*   Coordinates are normalized between `0.0` and `1.0` relative to the image dimensions.

---

## ✏️ Step 3: Match Configuration to Your Dataset

Open `src/config.py` and modify these parameters as needed:
1.  **`CLASS_NAMES`**: Configured to `['closed_eye', 'open_eye', 'yawning']`.
2.  **`NUM_CLASSES`**: Automatically set to `3`.
3.  **`IMG_SIZE`**: Default is `416` (must be a multiple of 32).
4.  **`EPOCHS`**: Currently set to `1` for your test run. For a full training sequence, increase this (e.g., `50` or `100`).
5.  **`BATCH_SIZE`**: Default is `8`. If using a GPU, you can increase this to `16` or `32`.
6.  **`SAVE_EVERY`**: Set this (e.g., `5`) to save periodic checkpoint files (`runs/train/mini_yolo_epoch_N.pth`) every N epochs.
7.  **`RESUME` & `CHECKPOINT_PATH`**: Set `RESUME = True` and point `CHECKPOINT_PATH` to a saved checkpoint (e.g., `"runs/train/mini_yolo_epoch_5.pth"`) to restore the model, optimizer, scheduler, epoch index, and best mAP, continuing training seamlessly.

---

## 🚀 Step 4: Run Training, Validation & Testing

Run these commands from the project root folder:

### 1. Start Training:
```bash
python src/train.py
```
Trains the model on the training set and evaluates it on the validation set after each epoch. The best weights are saved to `runs/train/mini_yolo_best.pth`.

### 2. Standalone Model Evaluation:
```bash
python src/validation.py
```
Loads the best checkpoint and computes mAP@50 and mAP@50-95 metrics class-by-class on the entire validation set. It prints a detailed table summary of the results.

### 3. Run Batch Inference on Test Set:
```bash
python src/test.py
```
Scans `dataset/test/images/`, runs inference on all images, applies NMS, and saves the drawn box visualization results to `runs/test/`. (If no test folder exists, it automatically falls back to validation images and saves outputs to `runs/val_test/`).

### 4. Run Single-Image Prediction:
```bash
python src/predict.py
```
Loads the best checkpoint, performs object detection on a single validation image, prints detections, and saves the visual result to `runs/prediction_output.jpg`.
