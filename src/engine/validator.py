import os
import torch
from torch.utils.data import DataLoader

# Import local modules
from configs import config
from src.data.dataset import YOLODataset, collate_fn
from src.data.transforms import Compose, Resize, ToTensor, Normalize
from src.models.yolo import MiniYOLO
from src.engine.evaluator import Evaluator

def run_validation(checkpoint_path: str, device: torch.device) -> dict:
    """
    Loads a checkpoint, builds DataLoader, instantiates Evaluator, runs metrics,
    and displays timing and accuracy summaries.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ Checkpoint file '{checkpoint_path}' not found.\n"
            f"Please run training first: python src/train.py"
        )

    # 1. Load checkpoint details
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    chk_config = checkpoint["config"]
    class_names = chk_config["class_names"]
    num_classes = chk_config["num_classes"]
    
    print(f"🥇 Loading best model checkpoint from epoch {checkpoint['epoch']+1}...")
    print(f"📊 Training best validation mAP@50 was: {checkpoint['best_map50']:.4f}\n")

    # 2. Build Model and Load Weights
    model = MiniYOLO(
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 3. Setup Validation DataLoader
    val_transform = Compose([
        Resize(config.IMG_SIZE),
        ToTensor(),
        Normalize(mean=config.MEAN, std=config.STD)
    ])

    val_dataset = YOLODataset(
        img_dir=config.VAL_IMG_DIR,
        label_dir=config.VAL_LABEL_DIR,
        transform=val_transform
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

    print(f"📦 Loaded {len(val_dataset)} validation images.")
    print("🧪 Running evaluation loop...")

    # 4. Instantiate Evaluator & run evaluation
    evaluator = Evaluator(class_names=class_names, num_classes=num_classes)
    metrics, timings = evaluator.evaluate(model, val_loader, device)

    # 5. Print summaries
    evaluator.print_results(metrics)
    evaluator.summarize(metrics, timings)

    return metrics
