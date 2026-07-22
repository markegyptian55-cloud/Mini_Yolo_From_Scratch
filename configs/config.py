import torch
import os
import random
import numpy as np

# 1. Automatic device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Random seed configuration
SEED = 42

def set_seed(seed=SEED):
    """
    Seed random number generators for random, numpy, torch, and torch.cuda.
    Enables deterministic mode when possible to guarantee reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Run seeding automatically on import
set_seed(SEED)

# 4. Inference parameters
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300

# Image parameters
IMG_SIZE = 416  # standard YOLO input size (must be multiple of 32)
CHANNELS = 3
BASE_CHANNELS = 16

# Training parameters
BATCH_SIZE = 8
EPOCHS = 85
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # 0 is safest on Windows to avoid multiprocessing issues

# 8. Checkpoint saving configuration
SAVE_EVERY = 5  # Optionally save checkpoints every N epochs
RESUME = True
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "runs", "train", "mini_yolo_best.pth")

# 9 & 10. DataLoader parameters
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0

# 5 & 6. Optimization parameters
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"

# 7. Automatic Mixed Precision (AMP)
USE_AMP = True if torch.cuda.is_available() else False

# Model parameters
# Strides at which predictions are made (Downsampling factors of Backbone)
STRIDES = [8, 16, 32] 

# Project Root (one level up from configs folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATA_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val", "images")
VAL_LABEL_DIR = os.path.join(DATA_DIR, "val", "labels")

# Save paths
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(RUNS_DIR, "train")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "mini_yolo_best.pth")

# Class names (Eye/yawn detection classes)
CLASS_NAMES = ['closed_eye', 'open_eye', 'yawning']

# Loss weights and parameters
BOX_WEIGHT = 7.5
OBJ_WEIGHT = 1.0
CLS_WEIGHT = 1.25
LABEL_SMOOTHING = 0.0
CACHE_IMAGES = False

# Inference & Visualization Saving Parameters
SAVE_PRED_IMAGES = True
SAVE_TXT = False
SAVE_JSON = False
AGNOSTIC_NMS = False
FILTER_CLASSES = None
BENCHMARK = False
PRINT_SPEED = True

# Data Augmentation parameters
LETTERBOX = True
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
HFLIP_PROB = 0.5
HSV_PROB = 0.5
AFFINE_PROB = 0.5

# 11. Automatically computed number of classes
NUM_CLASSES = len(CLASS_NAMES)

# 12. Simple validation checks
if IMG_SIZE % 32 != 0:
    raise ValueError(f"❌ Invalid Config: IMG_SIZE ({IMG_SIZE}) must be divisible by 32.")
if NUM_CLASSES <= 0:
    raise ValueError(f"❌ Invalid Config: NUM_CLASSES ({NUM_CLASSES}) must be greater than 0.")
if STRIDES != sorted(STRIDES):
    raise ValueError(f"❌ Invalid Config: STRIDES ({STRIDES}) must be sorted in ascending order.")
if BATCH_SIZE <= 0:
    raise ValueError(f"❌ Invalid Config: BATCH_SIZE ({BATCH_SIZE}) must be greater than 0.")

# 13. Print configuration function
def print_config():
    """
    Prints all important settings in a clean table format before training starts.
    Includes CUDA hardware details if available.
    """
    print("\n" + "=" * 60)
    print(f"{'🔧 MINI YOLO CONFIGURATION':^60s}")
    print("=" * 60)
    print(f"{'Parameter':25s} | {'Value'}")
    print("-" * 60)
    print(f"{'DEVICE':25s} | {DEVICE.type.upper()}")
    if DEVICE.type == "cuda":
        print(f"{'GPU Name':25s} | {torch.cuda.get_device_name(0)}")
        print(f"{'CUDA Version':25s} | {torch.version.cuda}")
        print(f"{'Number of GPUs':25s} | {torch.cuda.device_count()}")
    print(f"{'SEED':25s} | {SEED}")
    print(f"{'IMG_SIZE':25s} | {IMG_SIZE} (divisible by 32: Yes)")
    print(f"{'BATCH_SIZE':25s} | {BATCH_SIZE}")
    print(f"{'EPOCHS':25s} | {EPOCHS}")
    print(f"{'LEARNING_RATE':25s} | {LEARNING_RATE}")
    print(f"{'NUM_CLASSES':25s} | {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"{'OPTIMIZER':25s} | {OPTIMIZER}")
    print(f"{'SCHEDULER':25s} | {SCHEDULER}")
    print(f"{'USE_AMP':25s} | {USE_AMP}")
    print(f"{'PIN_MEMORY':25s} | {PIN_MEMORY}")
    print(f"{'PERSISTENT_WORKERS':25s} | {PERSISTENT_WORKERS}")
    print(f"{'SAVE_EVERY':25s} | {SAVE_EVERY} epochs")
    print(f"{'RESUME':25s} | {RESUME}")
    print(f"{'CHECKPOINT_PATH':25s} | {CHECKPOINT_PATH}")
    print(f"{'LETTERBOX':25s} | {LETTERBOX}")
    print(f"{'HFLIP_PROB':25s} | {HFLIP_PROB}")
    print(f"{'HSV_PROB':25s} | {HSV_PROB}")
    print(f"{'AFFINE_PROB':25s} | {AFFINE_PROB}")
    print(f"{'SAVE_PRED_IMAGES':25s} | {SAVE_PRED_IMAGES}")
    print(f"{'SAVE_TXT':25s} | {SAVE_TXT}")
    print(f"{'SAVE_JSON':25s} | {SAVE_JSON}")
    print(f"{'AGNOSTIC_NMS':25s} | {AGNOSTIC_NMS}")
    print(f"{'FILTER_CLASSES':25s} | {FILTER_CLASSES}")
    print(f"{'BENCHMARK':25s} | {BENCHMARK}")
    print(f"{'PRINT_SPEED':25s} | {PRINT_SPEED}")
    print("=" * 60 + "\n")
