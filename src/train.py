import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from pathlib import Path

# Import local modules
from configs import config
from src.data.dataset import YOLODataset, collate_fn
from src.data.transforms import Compose, Resize, ToTensor, RandomHorizontalFlip, RandomHSV, RandomAffine, Normalize
from src.models.yolo import MiniYOLO
from src.losses.yolo_loss import MiniYOLOLoss
from src.engine.trainer import YOLOTrainer

def verify_dataset_consistency() -> None:
    """
    Verify dataset integrity and folder consistency before training starts.
    Checks:
    - Directories exist.
    - Dataset is not empty.
    - Coordinates are present.
    Raises informative errors if anything is wrong.
    """
    train_img_dir = Path(config.TRAIN_IMG_DIR)
    train_lbl_dir = Path(config.TRAIN_LABEL_DIR)
    val_img_dir = Path(config.VAL_IMG_DIR)
    val_lbl_dir = Path(config.VAL_LABEL_DIR)
    
    # Check that required folders exist
    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if not directory.exists():
            raise FileNotFoundError(f"❌ Required dataset directory '{directory}' does not exist.")
            
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Retrieve files list
    train_images = sorted([f for f in train_img_dir.iterdir() if f.suffix.lower() in valid_extensions])
    val_images = sorted([f for f in val_img_dir.iterdir() if f.suffix.lower() in valid_extensions])
    
    # Check for empty datasets
    if len(train_images) == 0:
        raise ValueError(f"❌ Empty dataset: No valid images found in training path '{train_img_dir}'.")
    if len(val_images) == 0:
        raise ValueError(f"❌ Empty dataset: No valid images found in validation path '{val_img_dir}'.")
        
    print(f"✅ Dataset consistency checks passed: Found {len(train_images)} train and {len(val_images)} val images.")

def build_model() -> MiniYOLO:
    """
    Instantiates the MiniYOLO model and maps it immediately to the active hardware device.
    """
    model = MiniYOLO(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    return model

def build_optimizer(model: nn.Module) -> optim.Optimizer:
    """
    Builds the selected optimizer from options specified in configuration (AdamW, SGD).
    """
    opt_type = config.OPTIMIZER.upper()
    if opt_type == "ADAMW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif opt_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"❌ Unsupported optimizer configured: '{config.OPTIMIZER}'. Supported: AdamW, SGD.")
    return optimizer

def build_scheduler(optimizer: optim.Optimizer) -> Optional[object]:
    """
    Instantiates the learning rate scheduler based on configurations.
    Supports CosineAnnealingLR, CosineAnnealingWarmRestarts, and None.
    """
    sched_type = config.SCHEDULER
    if sched_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    elif sched_type == "CosineAnnealingWarmRestarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif sched_type is None or str(sched_type).lower() == "none":
        return None
    else:
        raise ValueError(
            f"❌ Unsupported scheduler configured: '{config.SCHEDULER}'. "
            f"Supported: CosineAnnealingLR, CosineAnnealingWarmRestarts, None."
        )

def build_loss() -> MiniYOLOLoss:
    """
    Constructs the custom MiniYOLOLoss loss criterion using weights defined in config.
    """
    return MiniYOLOLoss(
        num_classes=config.NUM_CLASSES,
        strides=config.STRIDES,
        box_weight=config.BOX_WEIGHT,
        obj_weight=config.OBJ_WEIGHT,
        cls_weight=config.CLS_WEIGHT,
        label_smoothing=config.LABEL_SMOOTHING
    )

def build_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Instantiates datasets and PyTorch DataLoaders equipped with modern YOLOv8-inspired augmentations.
    """
    # 1. Training Augmentations Pipeline
    train_transform = Compose([
        Resize(config.IMG_SIZE),
        RandomAffine(p=config.AFFINE_PROB),
        RandomHorizontalFlip(p=config.HFLIP_PROB),
        RandomHSV(p=config.HSV_PROB),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # 2. Validation Pipeline (No augmentation)
    val_transform = Compose([
        Resize(config.IMG_SIZE),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    # 3. Datasets
    train_dataset = YOLODataset(
        img_dir=config.TRAIN_IMG_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        transform=train_transform
    )
    
    val_dataset = YOLODataset(
        img_dir=config.VAL_IMG_DIR,
        label_dir=config.VAL_LABEL_DIR,
        transform=val_transform
    )
    
    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    return train_loader, val_loader

def print_model_summary(model: nn.Module) -> None:
    """
    Helper to print training run, model parameter, and device details.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 50)
    print(f"{'📊 TRAINING PIPELINE SUMMARY':^50s}")
    print("=" * 50)
    print(f"{'Model Name':25s} | MiniYOLO")
    print(f"{'Total Parameters':25s} | {total_params:,}")
    print(f"{'Trainable Parameters':25s} | {trainable_params:,}")
    print(f"{'Input Image Size':25s} | {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"{'Batch Size':25s} | {config.BATCH_SIZE}")
    print(f"{'Number of Classes':25s} | {config.NUM_CLASSES} ({', '.join(config.CLASS_NAMES)})")
    print(f"{'Training Device':25s} | {config.DEVICE.type.upper()}")
    print("=" * 50 + "\n")

def main() -> None:
    # 1. Print configuration parameters
    config.print_config()

    # 2. Verify dataset consistency
    verify_dataset_consistency()

    # 3. Setup CuDNN performance properties
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

    # 4. Build Model & summary
    model = build_model()
    print_model_summary(model)

    # 5. Build components
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    loss_fn = build_loss()
    train_loader, val_loader = build_dataloaders()

    # 6. Resume training support
    start_epoch = 0
    best_map50 = 0.0

    if config.RESUME:
        if config.CHECKPOINT_PATH and os.path.exists(config.CHECKPOINT_PATH):
            print(f"🔄 Resuming training from checkpoint: {config.CHECKPOINT_PATH}")
            # Load checkpoint safely on active device
            checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
            start_epoch = checkpoint["epoch"] + 1
            best_map50 = checkpoint["best_map50"]
            print(f"✅ Successfully resumed! Last completed epoch: {checkpoint['epoch']+1}. "
                  f"Best mAP@50 restored: {best_map50:.4f}. Continuing from epoch {start_epoch+1}...")
        else:
            print(f"⚠️ Warning: Resume checkpoint '{config.CHECKPOINT_PATH}' not found or path is empty. "
                  f"Starting a new training run from scratch.")

    # 7. Instantiate Trainer
    trainer = YOLOTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        config=config,
        scheduler=scheduler
    )
    trainer.best_map50 = best_map50

    # Restore GradScaler state if resuming and key exists
    if config.RESUME and "checkpoint" in locals() and "scaler_state_dict" in checkpoint:
        if checkpoint["scaler_state_dict"] is not None:
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # 8. Start training!
    trainer.fit(start_epoch=start_epoch)

if __name__ == "__main__":
    main()
