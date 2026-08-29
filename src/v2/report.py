"""Full evaluation report for a MiniYOLO-v2 checkpoint: metrics + 10 plots + annotated video.

    python -m src.v2.report --weights runs/v2/v2_n384/weights/best.pt --split test \
        --video "dataset/VIDEO FOR TEST/15-MaleGlasses.mp4" --out info/v2_n384
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import torch                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.data.build import build_dataloader, build_dataset          # noqa: E402
from src.v2.data.augment import letterbox                              # noqa: E402
from src.v2.utils.boxes import non_max_suppression, xywh2xyxy, scale_boxes  # noqa: E402
from src.v2.utils.metrics import DetMetrics, ap_per_class, compute_ap  # noqa: E402
from src.v2.utils.general import colorstr                              # noqa: E402
from src.v2.utils.plots import plot_results                            # noqa: E402
from src.v2.train import resolve_data, split_dirs                      # noqa: E402
from src.v2.val import load_model                                      # noqa: E402
from src.v2.predict import infer, draw                                 # noqa: E402
from src.v2 import hud                                                 # noqa: E402
from src.v2.temporal import DriverStateMonitor                         # noqa: E402


@torch.no_grad()
def run_validation(model, dl, nc, names, device, conf, iou, max_det, e2e):
    """Same as Validator, but also returns per-image dets/labels for a confusion matrix."""
    model.eval()
    model.head.e2e = e2e
    model.head.max_det = max_det
    metrics = DetMetrics(nc, names, device=device)
    confmat = np.zeros((nc + 1, nc + 1), dtype=int)  # +1 row/col = background
    t_infer, n_img = 0.0, 0

    for imgs, targets, _paths, _shapes in dl:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        bs, _, h, w = imgs.shape
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        preds = model(imgs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer += time.time() - t0
        n_img += bs

        if e2e:
            dets = []
            for b in range(bs):
                d = preds[b].float()
                d = d[d[:, 4] > conf]
                if d.shape[0]:
                    d = torch.cat((xywh2xyxy(d[:, :4]), d[:, 4:]), 1)
                dets.append(d)
        else:
            dets = non_max_suppression(preds.float(), conf, iou, max_det=max_det)

        targets = targets.to(device)
        scale = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
        for b in range(bs):
            t = targets[targets[:, 0] == b]
            lb = torch.cat((t[:, 1:2], xywh2xyxy(t[:, 2:6] * scale)), 1) if t.shape[0] \
                else torch.zeros((0, 5), device=device)
            d = dets[b]
            metrics.process_batch(d, lb)

            # -- confusion matrix: greedy IoU>=0.5 match between GT and preds --
            if lb.shape[0] and d.shape[0]:
                from src.v2.utils.boxes import box_iou
                ious = box_iou(lb[:, 1:5], d[:, :4]).cpu().numpy()
                gt_cls = lb[:, 0].cpu().numpy().astype(int)
                pr_cls = d[:, 5].cpu().numpy().astype(int)
                matched_pred = set()
                for gi in range(len(gt_cls)):
                    j = ious[gi].argmax() if ious.shape[1] else -1
                    if j >= 0 and ious[gi, j] >= 0.5 and j not in matched_pred:
                        confmat[gt_cls[gi], pr_cls[j]] += 1
                        matched_pred.add(j)
                    else:
                        confmat[gt_cls[gi], nc] += 1  # missed (false negative)
                for j in range(len(pr_cls)):
                    if j not in matched_pred:
                        confmat[nc, pr_cls[j]] += 1  # spurious (false positive)
            elif lb.shape[0]:
                for c in lb[:, 0].cpu().numpy().astype(int):
                    confmat[c, nc] += 1
            elif d.shape[0]:
                for c in d[:, 5].cpu().numpy().astype(int):
                    confmat[nc, c] += 1

    res = metrics.compute()
    res["ms_per_image"] = 1000.0 * t_infer / max(n_img, 1)
    res["fps"] = max(n_img, 1) / max(t_infer, 1e-9)
    return metrics, res, confmat


def write_evaluation_report(out_path, ckpt, weights_path, splits):
    """One consolidated evaluation file for the whole experiment -- checkpoint
    metadata, then a section per split (images/boxes, per-class table, speed).
    `splits` is {split_name: (metrics, res, n_images, n_boxes)}."""
    lines = [
        "MiniYOLO-v2 Evaluation Report",
        "=================================",
        f"weights     : {Path(weights_path).resolve()}",
        f"checkpoint  : epoch {ckpt.get('epoch')}  best_fitness {ckpt.get('best_fitness')}",
        f"scale       : {ckpt.get('scale')}",
        f"classes     : {ckpt.get('names')}",
        "",
    ]
    for split, (metrics, res, n_images, n_boxes) in splits.items():
        lines += [
            f"--- {split} split " + "-" * (60 - len(split)),
            f"images: {n_images}   boxes: {n_boxes}",
            "",
            metrics.table(),
            "",
            f"speed: {res['ms_per_image']:.2f} ms/img ({res['fps']:.0f} img/s)",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_full_epoch_log(csv_path, out_path):
    """Full per-epoch training numbers (all epochs), formatted as a plain table."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    lines = [f"MiniYOLO-v2 -- full per-epoch log ({len(df)} epochs)",
             f"source: {csv_path}", ""]
    hdr = (f"{'epoch':>5} {'box_loss':>9} {'cls_loss':>9} {'l1_loss':>8} {'loss_o2m':>9} "
           f"{'loss_o2o':>9} {'lr':>10} {'precision':>9} {'recall':>7} {'mAP50':>7} "
           f"{'mAP50-95':>9} {'fitness':>8} {'ms/img':>7}")
    lines += [hdr, "-" * len(hdr)]
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['epoch']):>5} {r['box_loss']:>9.4f} {r['cls_loss']:>9.4f} "
            f"{r['l1_loss']:>8.4f} {r['loss_o2m']:>9.4f} {r['loss_o2o']:>9.4f} "
            f"{r['lr']:>10.6f} {r['precision']:>9.4f} {r['recall']:>7.4f} "
            f"{r['mAP50']:>7.4f} {r['mAP50_95']:>9.4f} {r['fitness']:>8.4f} "
            f"{r['ms_per_image']:>7.3f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(confmat, names, out_path):
    labels = list(names) + ["background"]
    norm = confmat / np.maximum(confmat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6), tight_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalized, IoU>=0.5)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if confmat[i, j] > 0:
                ax.text(j, i, str(confmat[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_curves(metrics, names, out_dir):
    """PR curve, F1/P/R-vs-confidence curves, all classes + macro average."""
    stats = metrics.stats
    if not stats["tp"]:
        return
    tp = np.concatenate(stats["tp"], 0)
    conf = np.concatenate(stats["conf"], 0)
    pred_cls = np.concatenate(stats["pred_cls"], 0)
    target_cls = np.concatenate(stats["target_cls"], 0)
    if tp.shape[0] == 0 or target_cls.shape[0] == 0:
        return

    i = np.argsort(-conf)
    tp_s, conf_s, pred_s = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls.astype(int), return_counts=True)
    px = np.linspace(0, 1, 1000)

    py_pr = []      # PR curve (recall on x, precision on y) at IoU 0.5
    p_curve = np.zeros((len(unique_classes), 1000))
    r_curve = np.zeros((len(unique_classes), 1000))
    f1_curve = np.zeros((len(unique_classes), 1000))

    for ci, c in enumerate(unique_classes):
        m = pred_s == c
        n_l, n_p = nt[ci], m.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp_s[m, 0]).cumsum(0)
        tpc = tp_s[m, 0].cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        r_curve[ci] = np.interp(-px, -conf_s[m], recall, left=0)
        p_curve[ci] = np.interp(-px, -conf_s[m], precision, left=1)
        f1_curve[ci] = 2 * p_curve[ci] * r_curve[ci] / (p_curve[ci] + r_curve[ci] + 1e-16)
        py_pr.append(np.interp(px, recall, precision, left=1))

    def _plot(curve, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        for ci, c in enumerate(unique_classes):
            name = names[c] if c < len(names) else str(c)
            ax.plot(px, curve[ci], linewidth=1.4, label=name)
        ax.plot(px, curve.mean(0), linewidth=2.4, color="black", linestyle="--",
                label="all (macro avg)")
        ax.set_xlabel("confidence" if "confidence" in title.lower() else "recall")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.savefig(out_dir / fname, dpi=140)
        plt.close(fig)

    # PR curve
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    for ci, c in enumerate(unique_classes):
        name = names[c] if c < len(names) else str(c)
        ap = compute_ap(np.linspace(0, 1, 1000), py_pr[ci])[0]
        ax.plot(px, py_pr[ci], linewidth=1.4, label=f"{name} (AP50={ap:.3f})")
    mean_pr = np.mean(py_pr, axis=0)
    ax.plot(px, mean_pr, linewidth=2.4, color="black", linestyle="--", label="all")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall Curve (IoU=0.5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "03_pr_curve.png", dpi=140)
    plt.close(fig)

    _plot(f1_curve, "F1", "04_f1_confidence.png", "F1 - Confidence Curve")
    _plot(p_curve, "Precision", "05_precision_confidence.png", "Precision - Confidence Curve")
    _plot(r_curve, "Recall", "06_recall_confidence.png", "Recall - Confidence Curve")


def plot_ap_per_class(metrics, names, out_path):
    if metrics.results is None:
        return
    r = metrics.results
    cls_idx = r["cls_idx"]
    labels = [names[c] if c < len(names) else str(c) for c in cls_idx]
    ap50 = r["ap"][:, 0]
    ap5095 = r["ap"].mean(1)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    w = 0.35
    ax.bar(x - w / 2, ap50, w, label="mAP50")
    ax.bar(x + w / 2, ap5095, w, label="mAP50-95")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP")
    ax.set_title("Per-class Average Precision")
    for xi, v in zip(x - w / 2, ap50):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, ap5095):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def count_labels(label_dir, nc):
    counts = np.zeros(nc, dtype=int)
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            c = int(line.split()[0])
            if 0 <= c < nc:
                counts[c] += 1
    return counts


def plot_class_distribution(root, data, names, nc, out_path):
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    x = np.arange(len(names))
    w = 0.25
    for i, split in enumerate(["train", "val", "test"]):
        rel = data.get(split, f"images/{split}")
        try:
            img_dir, lab_dir = split_dirs(root, rel)
            counts = count_labels(lab_dir, nc)
        except Exception:
            counts = np.zeros(nc, dtype=int)
        ax.bar(x + (i - 1) * w, counts, w, label=split)
        for xi, v in zip(x + (i - 1) * w, counts):
            if v:
                ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylabel("instance count")
    ax.set_title("Class Distribution per Split")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_speed(res_by_split, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), tight_layout=True)
    splits = list(res_by_split.keys())
    ms = [res_by_split[s]["ms_per_image"] for s in splits]
    fps = [res_by_split[s]["fps"] for s in splits]
    axes[0].bar(splits, ms, color="#4c72b0")
    axes[0].set_title("Latency (ms/image)")
    axes[0].set_ylabel("ms")
    for i, v in enumerate(ms):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(splits, fps, color="#55a868")
    axes[1].set_title("Throughput (img/s)")
    axes[1].set_ylabel("FPS")
    for i, v in enumerate(fps):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def sample_predictions_grid(model, img_dir, names, device, imgsz, conf_thres, out_path, n=9):
    files = sorted(Path(img_dir).glob("*"))
    files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not files:
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    cols = 3
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), tight_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, pick):
        im0 = cv2.imread(str(files[idx]))
        if im0 is None:
            continue
        dets, _ = infer(model, im0, imgsz, device, conf_thres)
        im0 = draw(im0, dets, names)
        ax.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
        ax.set_title(files[idx].name, fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample Test Predictions")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def write_video_analysis(out_path, vres, names, video_in, weights_path, conf_thres):
    """One text file per annotated clip: throughput, class mix, and the temporal
    driver-state readout from src/v2/temporal.py.

    Every number here is measured on this run. Nothing is a target or an estimate.
    """
    t = vres["temporal"]
    n = max(vres["frames"], 1)
    total_det = sum(vres["detections_per_class"].values())
    L = [
        "=" * 74,
        "ANNOTATED VIDEO ANALYSIS",
        "=" * 74,
        f"source          : {video_in}",
        f"weights         : {weights_path}",
        f"conf threshold  : {conf_thres}",
        "",
        "-- throughput -------------------------------------------------------",
        f"frames processed        : {vres['frames']}",
        f"source FPS              : {vres['source_fps']:.2f}",
        f"clip duration           : {t['duration_s']:.2f} s",
        f"inference               : {vres['avg_ms']:.2f} ms/frame  "
        f"({vres['avg_fps']:.1f} FPS achievable)",
        f"real-time headroom      : {vres['avg_fps'] / max(vres['source_fps'], 1e-6):.2f}x "
        f"source rate",
        "",
        "-- detections by class ----------------------------------------------",
        f"total detections        : {total_det}",
    ]
    for name in names:
        c = vres["detections_per_class"].get(name, 0)
        L.append(f"  {name:<12s}          : {c:6d}  "
                 f"({100.0 * c / max(total_det, 1):5.1f}% of detections, "
                 f"{c / n:5.2f} per frame)")
    L += [
        "",
        "-- temporal driver state (src/v2/temporal.py) -----------------------",
        "PERCLOS is measured only over frames where an eye was detected; the",
        "coverage line says how often that was. Thresholds are the conventional",
        "DMS values and are NOT validated against drowsiness ground truth --",
        "this dataset has none. Read these as instrumentation, not diagnosis.",
        "",
        f"eye-detection coverage  : {t['eye_coverage'] * 100:.1f}% of frames",
        f"PERCLOS (final window)  : "
        + ("n/a (no eye detected)" if t["perclos_final"] is None
           else f"{t['perclos_final'] * 100:.1f}%"),
        f"blinks (100-400 ms)     : {t['blinks']}  ({t['blinks_per_min']:.1f}/min)",
        f"yawns (>= 400 ms)       : {t['yawns']}  ({t['yawns_per_min']:.1f}/min)",
        f"microsleeps (>= 1.5 s)  : {t['microsleeps']}",
        f"longest eye closure     : {t['longest_closure_s']:.2f} s",
        "",
        "-- alert level, frames ----------------------------------------------",
    ]
    for lvl in ("SAFE", "WARNING", "CRITICAL"):
        c = vres["alert_frames"].get(lvl, 0)
        L.append(f"  {lvl:<9s}             : {c:6d}  ({100.0 * c / n:5.1f}% of frames)")
    L.append("")
    Path(out_path).write_text("\n".join(L), encoding="utf-8")
    return out_path


def annotate_video(model, names, device, imgsz, conf_thres, video_in, video_out,
                   model_name="MiniYOLO-v2", half=False):
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        print(colorstr("red", "bold", "error"), f"could not open {video_in}")
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    for _ in range(3):
        model(torch.zeros(1, 3, imgsz, imgsz, device=device))

    n, times = 0, []
    class_counts = {name: 0 for name in names}
    state_history = deque(maxlen=hud.STATE_HISTORY_SIZE)
    fatigue_tracker = hud.FatigueTracker()
    # Timing comes from the clip's own FPS, not from inference speed, so the temporal
    # numbers describe the driver in the video rather than how fast this GPU ran.
    monitor = DriverStateMonitor(fps=fps_in, conf_thres=conf_thres)
    alert_frames = {"SAFE": 0, "WARNING": 0, "CRITICAL": 0}
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        dets, dt = infer(model, frame, imgsz, device, conf_thres, half)
        times.append(dt)
        for d in dets:
            c = names[int(d[5])] if int(d[5]) < len(names) else str(int(d[5]))
            class_counts[c] = class_counts.get(c, 0) + 1
        fps_now = 1000 / max(np.mean(times[-30:]), 1e-6)
        _, tstate = hud.render_frame(frame, dets, names, model_name, n + 1, fps_now,
                                     state_history, fatigue_tracker, monitor=monitor)
        alert_frames[tstate["alert_level"]] += 1
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    return {"frames": n, "avg_ms": float(np.mean(times)) if times else 0.0,
            "avg_fps": float(1000 / max(np.mean(times), 1e-6)) if times else 0.0,
            "detections_per_class": class_counts,
            "source_fps": float(fps_in),
            "temporal": monitor.summary(),
            "alert_frames": alert_frames}


def parse_args():
    ap = argparse.ArgumentParser("MiniYOLO-v2 full report")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--vid-conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--out", type=str, required=True,
                    help='report destination. Per AGENTS.md Rule 1 this must be '
                         '"checkpoints/Expi-<N>-imagez-<Size>/REPORTS EXPI-<N>"')
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    out_dir = Path(args.out)
    plots_dir = out_dir / "plots"
    video_dir = out_dir / "video"
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(args.weights, device)
    names = ckpt.get("names") or [str(i) for i in range(ckpt["nc"])]
    nc = ckpt["nc"]
    root, data, _, _ = resolve_data(args.data)

    res_by_split = {}
    eval_splits = {}
    for split in ["val", "test"]:
        rel = data.get(split, f"images/{split}")
        img_dir, lab_dir = split_dirs(root, rel)
        if not Path(img_dir).exists():
            continue
        ds = build_dataset(img_dir, lab_dir, nc, args.imgsz, ckpt.get("hyp", {}),
                           augment=False, prefix=colorstr(f"{split}: "))
        n_images = ds.n
        n_boxes = sum(len(x) for x in ds.labels)
        dl = build_dataloader(ds, args.batch, args.workers, shuffle=False)
        metrics, res, confmat = run_validation(model, dl, nc, names, device,
                                               args.conf, args.iou, args.max_det, e2e=True)
        print(f"\n[{split}]\n{metrics.table()}\nspeed: {res['ms_per_image']:.2f} ms/img "
              f"({res['fps']:.0f} FPS)")
        eval_splits[split] = (metrics, res, n_images, n_boxes)
        res_by_split[split] = res

        if split == "test":
            plot_confusion_matrix(confmat, names, plots_dir / "02_confusion_matrix.png")
            plot_curves(metrics, names, plots_dir)
            plot_ap_per_class(metrics, names, plots_dir / "07_ap_per_class.png")

    write_evaluation_report(out_dir / "evaluation.txt", ckpt, args.weights, eval_splits)

    # 01 training curves + full per-epoch log (from the run directory this checkpoint came from)
    run_dir = Path(args.weights).resolve().parents[1]
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        plot_results(csv_path, plots_dir / "01_training_curves.png")
        write_full_epoch_log(csv_path, out_dir / "full_epoch_log.txt")

    # 08 class distribution across splits
    plot_class_distribution(root, data, names, nc, plots_dir / "08_class_distribution.png")

    # 09 speed benchmark
    if res_by_split:
        plot_speed(res_by_split, plots_dir / "09_speed_benchmark.png")

    # 10 sample predictions grid
    test_rel = data.get("test", "images/test")
    test_img_dir, _ = split_dirs(root, test_rel)
    if Path(test_img_dir).exists():
        sample_predictions_grid(model, test_img_dir, names, device, args.imgsz,
                                args.vid_conf, plots_dir / "10_sample_predictions.png")

    # annotated test video
    if args.video:
        video_in = Path(args.video)
        if video_in.exists():
            video_out = video_dir / f"{video_in.stem}_annotated.mp4"
            model_name = f"MiniYOLO-v2 ({ckpt.get('scale', 'n')}{args.imgsz})"
            vres = annotate_video(model, names, device, args.imgsz, args.vid_conf,
                                  video_in, video_out, model_name=model_name)
            if vres:
                write_video_analysis(video_dir / "analysis.txt", vres, names,
                                     video_in, args.weights, args.vid_conf)
                t = vres["temporal"]
                print(f"\n{colorstr('done')} annotated video -> {video_out}")
                print(f"  {vres['frames']} frames @ {vres['source_fps']:.1f} fps source, "
                      f"{vres['avg_ms']:.2f} ms/frame inference")
                print(f"  PERCLOS "
                      + ("n/a" if t["perclos_final"] is None
                         else f"{t['perclos_final'] * 100:.1f}%")
                      + f"  blinks={t['blinks']}  yawns={t['yawns']}  "
                        f"microsleeps={t['microsleeps']}")
                print(f"  analysis -> {video_dir / 'analysis.txt'}")
        else:
            print(colorstr("red", "bold", "warning"), f"video not found: {video_in}")

    print(f"\n{colorstr('done')} report -> {out_dir}")


if __name__ == "__main__":
    main()
