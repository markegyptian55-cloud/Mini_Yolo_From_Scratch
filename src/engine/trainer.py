import os
import torch
from tqdm import tqdm
import numpy as np

from src.engine.evaluator import Evaluator

class YOLOTrainer:
    """
    Trainer class that orchestrates training epochs, validations,
    metric computation (mAP), and checkpoint saving.
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, config, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.epochs = config.EPOCHS
        
        # Best metric tracker
        self.best_map50 = 0.0

        # Evaluator
        self.evaluator = Evaluator(class_names=self.config.CLASS_NAMES, num_classes=self.config.NUM_CLASSES)

        # AMP GradScaler initialization
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        try:
            self.scaler = torch.amp.GradScaler(device_type, enabled=self.config.USE_AMP)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.USE_AMP)

        # Create checkpoint directory if not exists
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP autocast block
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            try:
                autocast_context = torch.amp.autocast(device_type=device_type, enabled=self.config.USE_AMP)
            except AttributeError:
                autocast_context = torch.cuda.amp.autocast(enabled=self.config.USE_AMP)

            with autocast_context:
                outputs = self.model(images)
                loss, loss_items = self.loss_fn(outputs, targets, img_size=self.config.IMG_SIZE)

            # Detect and handle NaN/Inf loss values safely
            if not torch.isfinite(loss):
                print(f"\n⚠️ Warning: Loss is non-finite ({loss.item()}). Skipping step to protect weights.")
                continue

            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()
            
            # Unscale gradients before clipping for accurate norm measurement
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            # Optimize step and update scaler parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss terms
            epoch_loss += loss_items["loss"]
            epoch_box_loss += loss_items["box_loss"]
            epoch_obj_loss += loss_items["obj_loss"]
            epoch_cls_loss += loss_items["cls_loss"]

            # Update progress bar description
            pbar.set_postfix({
                "loss": f"{loss_items['loss']:.4f}",
                "box": f"{loss_items['box_loss']:.4f}",
                "obj": f"{loss_items['obj_loss']:.4f}",
                "cls": f"{loss_items['cls_loss']:.4f}",
                "n_pos": loss_items["n_pos"]
            })

        num_batches = len(self.train_loader)
        stats = {
            "loss": epoch_loss / num_batches,
            "box_loss": epoch_box_loss / num_batches,
            "obj_loss": epoch_obj_loss / num_batches,
            "cls_loss": epoch_cls_loss / num_batches
        }
        return stats

    @torch.no_grad()
    def validate(self):
        metrics, timings = self.evaluator.evaluate(self.model, self.val_loader, self.device)
        return metrics

    def fit(self, start_epoch=0):
        print(f"Training started on device: {self.device}")
        
        for epoch in range(start_epoch, self.epochs):
            # 1. Train one epoch
            train_stats = self.train_epoch(epoch)
            
            # Print epoch training stats
            print(
                f"Epoch {epoch+1} Results - "
                f"Loss: {train_stats['loss']:.4f} | "
                f"Box Loss: {train_stats['box_loss']:.4f} | "
                f"Obj Loss: {train_stats['obj_loss']:.4f} | "
                f"Cls Loss: {train_stats['cls_loss']:.4f}"
            )

            # 2. Run validation (every epoch or periodically. Let's do every epoch)
            val_stats = self.validate()
            
            map50 = val_stats["mAP50"]
            map50_95 = val_stats["mAP50-95"]
            print(f"Validation - mAP@50: {map50:.4f} | mAP@50:95: {map50_95:.4f}\n")

            # Update scheduler if present
            if self.scheduler is not None:
                self.scheduler.step()

            # Prepare checkpoint dict
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
                "best_map50": self.best_map50,
                "config": {
                    "img_size": self.config.IMG_SIZE,
                    "num_classes": self.config.NUM_CLASSES,
                    "class_names": self.config.CLASS_NAMES
                }
            }

            # 3. Checkpoint saving
            # Save best checkpoint
            if map50 > self.best_map50:
                self.best_map50 = map50
                checkpoint_data["best_map50"] = self.best_map50
                torch.save(checkpoint_data, self.config.MODEL_SAVE_PATH)
                print(f"🥇 New best model saved to {self.config.MODEL_SAVE_PATH} (mAP@50: {map50:.4f})")

            # Save last checkpoint (always saved for training resumption robustness)
            last_path = os.path.join(self.config.CHECKPOINT_DIR, "mini_yolo_last.pth")
            torch.save(checkpoint_data, last_path)

            # Save periodic checkpoints every N epochs
            epoch_idx = epoch + 1
            if self.config.SAVE_EVERY > 0 and epoch_idx % self.config.SAVE_EVERY == 0:
                epoch_path = os.path.join(self.config.CHECKPOINT_DIR, f"mini_yolo_epoch_{epoch_idx}.pth")
                torch.save(checkpoint_data, epoch_path)
                print(f"💾 Saved epoch checkpoint to {epoch_path}")
                
        print("Training completed! Best Validation mAP@50: {:.4f}".format(self.best_map50))
