# MiniYOLO Engineering Book: Architecture & Evolution
*A comprehensive guide to the refactoring, optimization, and training history of the MiniYOLO object detection pipeline.*

---

## Table of Contents
1. **Chapter 1**: Project Background & Reorganization Goals
2. **Chapter 2**: Target Project Architecture
3. **Chapter 3**: Core Module Refactorings & Technical Justifications
4. **Chapter 4**: Compilation Compatibility & Graph Break Resolution
5. **Chapter 5**: Data Pipeline & Polygon Conversion Logic
6. **Chapter 6**: Hyperparameter & Configuration Evolution
7. **Chapter 7**: Training Progress & Validation Metrics History
8. **Chapter 8**: CPU Training Performance & Time Analysis
9. **Chapter 9**: Recommended Future Optimizations
10. **Chapter 10**: Fine-Tuning Setup & Hyperparameter Adjustments

---

## Chapter 1: Project Background & Reorganization Goals

The MiniYOLO project was initiated to create a custom, lightweight, production-grade object detector specialized in detecting human expressions and fatigue signals (`closed_eye`, `open_eye`, `yawning`). 

Initially, the repository was structured as a flat set of script utilities with duplicated code, hardcoded math constants, and manual dependency setups. To reach the architectural quality of modern, state-of-the-art vision frameworks (like Ultralytics YOLOv8 and YOLO11), a major modernization program was executed.

### Core Objectives:
*   **Decouple Training and Validation**: Extract common validation loops and evaluation math into a unified engine to prevent duplicate code.
*   **Production Standardization**: Reorganize file hierarchies into discrete modules (`data/`, `models/`, `losses/`, `engine/`, `utils/`).
*   **Clean Packages**: Remove custom shell-level path manipulations (`sys.path.insert`) in favor of proper absolute package imports.
*   **Modern Pipeline Augmentations**: Implement rich transforms (RandomAffine, HSV color scale distortion, horizontal flips) managed natively by global configuration constants.
*   **Graph Tracing Compatibility**: Ensure all parts of the forward model and loss functions are fully compatible with `torch.compile` by removing graph breaks.

---

## Chapter 2: Target Project Architecture

The directory layout has been streamlined into a modular package format. All `__init__.py` files and `__pycache__` folders were completely deleted from the workspace, converting the directories into clean Python implicit namespace packages.

```
mini_yolo/
├── configs/
│   └── config.py              # Central configurations & central hyperparameter overrides
├── info/
│   ├── book.md                # [This File] Complete engineering documentation
│   └── first report for train 1/ # Performance logs, charts, and predictions
├── runs/
│   ├── train/                 # Checkpoints (*.pth) and training logs
│   └── predictions/           # Inference outputs (annotated images)
├── src/
│   ├── data/
│   │   ├── dataset.py         # YOLO Dataset class, cache loader, & polygon converter
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
│   │   ├── generate_report_visuals.py # Automated report visual generator
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

## Chapter 3: Core Module Refactorings & Technical Justifications

### 1. Reusable Evaluator Engine (`evaluator.py`)
*   **Refactor**: Extracted duplicate evaluation code from the trainer and validator into a single `Evaluator` class in `src/engine/evaluator.py`.
*   **Justification**: Consolidates AP and timing calculation logic. It parses validation images, runs inference under an autocast context, computes box matching via metric matching, and outputs results in a clean table. Both training-time validation and standalone validation leverage this single code path.

### 2. High-Performance Predictor (`predictor.py`)
*   **Refactor**: Upgraded prediction pipeline in `src/engine/predictor.py`.
*   **Justification**: Leverages identical preprocessing transforms (`Resize`, `ToTensor`, `Normalize`) to match validation statistics. Added automatic input reshaping, custom class filtering (`FILTER_CLASSES`), class-agnostic NMS (`AGNOSTIC_NMS`), and automatic output subdirectory generation.

### 3. Coordinate Handling and NMS Acceleration (`boxes.py` & `nms.py`)
*   **Refactor**: Converted coordinates transformations (`xywh2xyxy` / `xyxy2xywh`) to vectorized formats and replaced custom NMS loops with `torchvision.ops.nms`.
*   **Justification**: Vectorized array operations prevent slow in-place array updates that break autograd tracking. Leveraging `torchvision.ops.nms` shifts NMS overhead to compiled CUDA kernels and reduces execution time.

### 4. Modernized AMP API (`trainer.py`, `evaluator.py`, `predictor.py`)
*   **Refactor**: Replaced deprecated `torch.cuda.amp.autocast(...)` and `torch.cuda.amp.GradScaler(...)` calls.
*   **Justification**: Modernized syntax to use the unified `torch.amp` namespaces:
    *   `torch.amp.GradScaler(device_type, enabled=...)`
    *   `torch.amp.autocast(device_type=..., enabled=...)`
    This silences warnings and ensures compatibility with PyTorch 2.6+ while handling CPU/GPU fallback dynamically.

---

## Chapter 4: Compilation Compatibility & Graph Break Resolution

When utilizing `torch.compile` to run the model at maximum speed on modern GPUs, standard Python structures inside the model forward execution tree can trigger compiler **graph breaks**, which drop execution back to the slower Python interpreter.

### 1. Removing Dictionary Caching in decoupled head (`head.py`)
*   **Old Code**: Checked grid cache using shape keys inside a Python dictionary:
    ```python
    key = (h, w, str(device))
    if key not in self.grid_cache:
        ...
    ```
*   **Problem**: Accessing a Python dictionary using dynamic properties and stringifying `device` values breaks compiler tracing.
*   **New Code**: Generates meshgrid coordinate arrays dynamically on-the-fly inside the compiled execution graph:
    ```python
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).to(torch.float32)
    ```

### 2. Removing Context Managers in Loss Functions (`boxes.py`)
*   **Old Code**: Computed aspect ratio consistency parameter `alpha` using context managers:
    ```python
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ```
*   **Problem**: `with torch.no_grad():` state changes during graph execution disrupt tracing.
*   **New Code**: Replaced with clean tensor detaches:
    ```python
    alpha = v / (1 - iou + v + eps)
    alpha = alpha.detach()
    ```

---

## Chapter 5: Data Pipeline & Polygon Conversion Logic

Roboflow datasets frequently export labels containing segmented polygon vertices instead of standard bounding box formats.

### 1. Dynamic Polygon Converter (`dataset.py`)
*   **Old Code**: Skipped any label txt lines that contained more than 5 elements, showing invalid label format warnings.
*   **Problem**: Polygon labels were discarded, lowering active target counts.
*   **New Code**: Implemented a parser fallback. When a line length exceeds 5 values (e.g., length 11, 13, 15, 17, 19, 21), it alternates the floating points to identify the polygon’s $(x, y)$ vertices, computes the enclosing rectangle boundaries, and transforms them into standard YOLO bounding coordinates:
    ```python
    all_coords = [float(x) for x in parts[1:]]
    xs = all_coords[0::2]
    ys = all_coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    ```

---

## Chapter 6: Hyperparameter & Configuration Evolution

Here is the trace of configuration changes implemented:

| Hyperparameter | Baseline Settings | Initial Train Settings | Fine-Tuning Settings (Current) | Technical Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Optimizer** | SGD | AdamW | **AdamW** | Offers fast convergence on multi-class topologies. |
| **Scheduler** | None | CosineAnnealingLR | **CosineAnnealingLR** | Smoothly decays learning rate over extended epochs. |
| **Learning Rate** | `1e-3` | `1e-3` | **`5e-4`** | Lower learning rate prevents destructive updates during fine-tuning. |
| **Epochs Limit** | `1` | `20` | **`50`** | Extends training up to 50 epochs for deeper convergence. |
| **Box Weight** | `1.0` | `5.0` | **`7.5`** | Increases box regression focus for precise eye/mouth boundary fitting. |
| **Class Weight** | `1.0` | `1.0` | **`1.25`** | Improves distinction between `open_eye` and `closed_eye` states. |
| **Resume Mode** | `False` | `False` | **`True`** | Resumes training from `mini_yolo_best.pth`. |

---

## Chapter 7: Training Progress & Validation Metrics History

### 📈 Historical Metric Progress
The model is trained on a dataset containing 3 classes (`closed_eye`, `open_eye`, `yawning`).

*   **Initial Baseline (Before refactorings)**:
    *   *Best mAP@50*: `0.0649`
*   **New Refactored Training Run (Initial 20 Epochs)**:
    *   **Epoch 5**: `0.4284` mAP@50
    *   **Epoch 10**: `0.5067` mAP@50
    *   **Epoch 15**: `0.5817` mAP@50 (🥇 Peak validation accuracy)
    *   **Epoch 20**: `0.5817` mAP@50 (Completed stage 1)

---

## Chapter 8: CPU Training Performance & Time Analysis

Due to hardware availability, training was run using the **CPU** rather than a GPU. Because of the size of the dataset, this introduces significant computational latency.

### 1. Dataset Dimensions & Load Metrics
*   **Training Images**: 33,365
*   **Validation Images**: 5,477
*   **Batch Size**: 8
*   **Total Batches (Iterations) per Epoch**: 4,170 (calculated as $33,365 / 8$)

### 2. Time Calculations
*   **Average Processing Speed**: $\sim 1.45 \text{ iterations (batches) per second}$
*   **Total Seconds per Epoch**: $\sim 2,875 \text{ seconds}$
*   **Total Minutes per Epoch**: $\sim 48.0 \text{ minutes}$ (roughly **$0.8\text{ hours}$**)
*   **Completed Stage 1 Duration (to Epoch 20)**: $\sim 960 \text{ minutes}$ (**$16.0\text{ hours}$**)
*   **Remaining Fine-Tuning Duration (Epochs 21 to 85, 65 Epochs)**: $\sim 3,120 \text{ minutes}$ (**$52.0\text{ hours}$**)
*   **Total Projected Duration for 85 Epochs**: $\sim 4,080 \text{ minutes}$ (**$68.0\text{ hours}$**)

---

## Chapter 9: Recommended Future Optimizations

To push the model's accuracy past `0.70` mAP@50 and improve compute speed, we recommend the following next steps:

1.  **Run with GPU (CUDA)**: Training on a CUDA-enabled GPU would increase batch iteration speed to $\sim 50\text{-}100\text{ iterations/second}$, cutting epoch training time down from **48 minutes** to **under 2 minutes**.
2.  **Add Model EMA (Exponential Moving Average)**: Keeping a moving average of weights during gradient updates smooths out validation fluctuations.
3.  **Adjust Image Size to 640**: Modern YOLO models are optimized for 640x640 resolution. Upgrading `IMG_SIZE` from 416 to 640 in `configs/config.py` will help resolve smaller object details.

---

## Chapter 10: Fine-Tuning Setup & Hyperparameter Adjustments

To achieve optimal fine-tuning performance without adding code complexity to the model architecture:

1.  **Checkpoint Resumption**: `RESUME` is enabled in `configs/config.py` pointing to `runs/train/mini_yolo_best.pth`.
2.  **Learning Rate Refinement**: `LEARNING_RATE` is adjusted to `5e-4` (half of initial rate) to refine features without disturbing established pre-trained weights. `train.py` dynamically updates optimizer parameter group learning rates upon loading checkpoints.
3.  **Loss Rebalancing**: `BOX_WEIGHT` is increased to `7.5` and `CLS_WEIGHT` to `1.25` to sharpen bounding box edges around subtle eye regions.
4.  **Code Simplicity Preserved**: Model architecture definitions (`src/models/yolo.py`, `backbone.py`, `head.py`, `neck.py`) remain untouched, ensuring the code stays lightweight, clean, and easy to study.
5.  **Crash-Proof Auto-Resumption (`auto_resume_train.ps1`)**: Integrated dynamic checkpoint resolution (`mini_yolo_last.pth` preferred over `mini_yolo_best.pth`) and created an automated PowerShell monitoring loop (`auto_resume_train.ps1`). If training is interrupted by power failure or OS events, it automatically restarts and resumes seamlessly from the latest completed epoch.
