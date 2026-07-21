import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from src.utils.nms import non_max_suppression
from src.utils.metrics import evaluate_predictions, ap_per_class
from configs import config

class Evaluator:
    """
    Evaluator class that handles running validation metrics computation
    (Precision, Recall, mAP50, mAP50-95) and printing structured summaries.
    """
    def __init__(self, class_names: List[str], num_classes: int = config.NUM_CLASSES) -> None:
        self.class_names = class_names
        self.num_classes = num_classes

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], dict]:
        """
        Runs the evaluation loop over the dataloader and returns validation stats and execution timings.
        """
        model.eval()
        stats = []
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # COCO thresholds

        # Measure times
        start_time = time.time()
        inference_times = []
        nms_times = []

        # Predict and evaluate
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Evaluating"):
                images = images.to(device)
                
                # Measure inference time
                t0 = time.time()
                device_type = "cuda" if "cuda" in str(device) else "cpu"
                try:
                    autocast_context = torch.amp.autocast(device_type=device_type, enabled=config.USE_AMP)
                except AttributeError:
                    autocast_context = torch.cuda.amp.autocast(enabled=config.USE_AMP)

                with autocast_context:
                    predictions = model(images)
                inference_times.append(time.time() - t0)
                
                # Measure NMS time
                t1 = time.time()
                nms_predictions = non_max_suppression(
                    predictions, 
                    conf_thres=0.001,  # use lower threshold for metric curves
                    iou_thres=0.5,
                    max_det=config.MAX_DETECTIONS
                )
                nms_times.append(time.time() - t1)

                # Evaluate predictions against targets
                tp, conf, pred_cls, target_cls = evaluate_predictions(
                    nms_predictions, targets, iou_thresholds
                )
                stats.append((tp, conf, pred_cls, target_cls))

        eval_duration = time.time() - start_time
        
        # Merge stats across batches
        stats_zipped = zip(*stats)
        stats_merged = [np.concatenate(x, 0) for x in stats_zipped] if stats else []
        
        metrics = self.compute_metrics(stats_merged)
        
        # Compute timings
        avg_inf_time = np.mean(inference_times) if inference_times else 0.0
        avg_nms_time = np.mean(nms_times) if nms_times else 0.0
        fps = len(dataloader.dataset) / eval_duration if eval_duration > 0 else 0.0

        timings = {
            "eval_time": eval_duration,
            "avg_inference_time": avg_inf_time,
            "avg_nms_time": avg_nms_time,
            "fps": fps
        }

        return metrics, timings

    def compute_metrics(self, stats_merged: list) -> Dict[str, float]:
        """
        Computes precision, recall, mAP50, and mAP50-95 from merged statistics.
        """
        map50 = 0.0
        map50_95 = 0.0
        precision = 0.0
        recall = 0.0
        num_targets = 0
        num_detections = 0

        # Class-wise metrics placeholders
        class_ap = []

        if len(stats_merged) > 0 and stats_merged[0].any():
            tp, conf, pred_cls, target_cls = stats_merged
            num_targets = len(target_cls)
            num_detections = len(pred_cls)

            # Compute Average Precision (AP) per class
            ap, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls)
            
            # Mean AP at IoU 0.5 (first threshold)
            map50 = ap[:, 0].mean()
            # Mean AP at IoU 0.5:0.95 (mean over all 10 thresholds)
            map50_95 = ap.mean()

            # Precision & Recall calculations at IoU=0.5
            tp_at_50 = tp[:, 0]
            num_tp = tp_at_50.sum()
            precision = num_tp / num_detections if num_detections > 0 else 0.0
            recall = num_tp / num_targets if num_targets > 0 else 0.0

            # Store class indices and stats for summarization
            # Count targets per class
            unique_targets, target_counts = np.unique(target_cls, return_counts=True)
            target_counts_dict = dict(zip(unique_targets, target_counts))

            for idx, class_idx in enumerate(unique_classes):
                class_ap.append({
                    "class_idx": int(class_idx),
                    "name": self.class_names[int(class_idx)],
                    "targets": target_counts_dict.get(class_idx, 0),
                    "ap50": ap[idx, 0],
                    "ap50_95": ap[idx].mean()
                })

        return {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "num_targets": num_targets,
            "num_detections": num_detections,
            "class_ap": class_ap
        }

    def print_results(self, metrics: Dict[str, float]) -> None:
        """
        Prints the evaluation table in a clean, human-readable format.
        """
        print("\n" + "=" * 75)
        print(f"{'Class Name':20s} | {'Images':8s} | {'Targets':8s} | {'AP@50':10s} | {'AP@50:95':10s}")
        print("=" * 75)

        for cap in metrics.get("class_ap", []):
            print(f"{cap['name']:20s} | {'all':8s} | {cap['targets']:8d} | {cap['ap50']:.4f}     | {cap['ap50_95']:.4f}")

        print("-" * 75)
        print(f"{'ALL (Summary)':20s} | {'all':8s} | {metrics['num_targets']:8d} | {metrics['mAP50']:.4f}     | {metrics['mAP50-95']:.4f}")
        print("=" * 75 + "\n")

    def summarize(self, metrics: Dict[str, float], timings: dict) -> None:
        """
        Displays summary speed metrics and classification accuracy statistics.
        """
        print("📊 EVALUATION SPEED SUMMARY")
        print(f"  • FPS: {timings['fps']:.2f}")
        print(f"  • Average Inference Time: {timings['avg_inference_time']*1000:.2f} ms")
        print(f"  • Average NMS Time: {timings['avg_nms_time']*1000:.2f} ms")
        print(f"  • Total Evaluation Duration: {timings['eval_time']:.2f} seconds")
        print("\n📊 EVALUATION ACCURACY METRICS")
        print(f"  • Precision: {metrics['precision']:.4f}")
        print(f"  • Recall: {metrics['recall']:.4f}")
        print(f"  • mAP@50: {metrics['mAP50']:.4f}")
        print(f"  • mAP@50:95: {metrics['mAP50-95']:.4f}")
        print(f"  • Detections / Targets: {metrics['num_detections']} / {metrics['num_targets']}\n")

    def save_results(self, metrics: Dict[str, float], filepath: str) -> None:
        """
        Placeholder method for saving validation results.
        """
        pass

    # Placeholders for future plotting / evaluation capabilities
    def plot_confusion_matrix(self) -> None:
        """Placeholder for future Confusion Matrix plotting."""
        pass

    def plot_pr_curve(self) -> None:
        """Placeholder for future Precision-Recall Curve plotting."""
        pass

    def plot_recall_curve(self) -> None:
        """Placeholder for future Recall Curve plotting."""
        pass

    def plot_f1_curve(self) -> None:
        """Placeholder for future F1 Curve plotting."""
        pass

    def export_csv(self) -> None:
        """Placeholder for exporting results to CSV."""
        pass

    def export_json(self) -> None:
        """Placeholder for exporting results to JSON."""
        pass
