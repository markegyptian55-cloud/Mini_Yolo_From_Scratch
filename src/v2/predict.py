"""Run a trained MiniYOLO-v2 on images, a video file, or a live camera.

    python -m src.v2.predict --weights runs/v2/exp/weights/best.pt --source dataset/images/test
    python -m src.v2.predict --weights ... --source 0 --show      # webcam stream
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.data.augment import letterbox                    # noqa: E402
from src.v2.utils.boxes import scale_boxes, xywh2xyxy        # noqa: E402
from src.v2.utils.general import colorstr, increment_path    # noqa: E402
from src.v2.val import load_model                            # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}
PALETTE = [(56, 56, 255), (49, 210, 207), (10, 249, 72), (255, 157, 151), (255, 112, 31)]


def draw(im, dets, names):
    """Thin boxes with the label alternating above/below and left/right-anchored
    by detection order, so two adjacent boxes (e.g. left/right eye) don't stack
    their labels on top of each other."""
    h, w = im.shape[:2]
    order = sorted(range(len(dets)), key=lambda i: dets[i][0])  # left-to-right by x1
    for rank, i in enumerate(order):
        x1, y1, x2, y2, conf, cls = dets[i]
        c = int(cls)
        color = PALETTE[c % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(im, p1, p2, color, 1, cv2.LINE_AA)

        label = f"{names[c] if c < len(names) else c} {conf:.2f}"
        tw, th = cv2.getTextSize(label, 0, 0.42, 1)[0]
        above = (rank % 2 == 0)
        anchor_left = (p1[0] < w - p2[0])  # more room on the left side of the frame -> hang label left
        lx = p1[0] if anchor_left else max(0, p2[0] - tw - 3)
        ly_top = p1[1] - th - 5 if above else p2[1]
        ly_bot = p1[1] if above else p2[1] + th + 5
        ly_top, ly_bot = max(0, ly_top), min(h, ly_bot)
        cv2.rectangle(im, (lx, ly_top), (lx + tw + 3, ly_bot), color, -1, cv2.LINE_AA)
        ty = ly_bot - 4 if above else ly_bot - 3
        cv2.putText(im, label, (lx + 1, ty), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def draw_panel(im, lines, origin=(10, 10)):
    """Small translucent stats sidebar in the corner (3-4 short lines)."""
    x0, y0 = origin
    pad, lh = 8, 18
    w = max(cv2.getTextSize(t, 0, 0.48, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    overlay = im.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, im, 0.45, 0, im)
    cv2.rectangle(im, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1, cv2.LINE_AA)
    for i, t in enumerate(lines):
        cv2.putText(im, t, (x0 + pad, y0 + pad + (i + 1) * lh - 5), 0, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return im


@torch.no_grad()
def infer(model, im0, imgsz, device, conf_thres, half=False):
    im, r, pad = letterbox(im0, imgsz, scaleup=False)
    x = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)[::-1])).to(device)
    x = (x.half() if half else x.float())[None] / 255.0
    t0 = time.perf_counter()
    pred = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000

    d = pred[0].float()
    d = d[d[:, 4] > conf_thres]
    if d.shape[0]:
        boxes = xywh2xyxy(d[:, :4])
        boxes = scale_boxes(boxes, im.shape[:2], im0.shape[:2], ratio_pad=(r, pad))
        d = torch.cat((boxes, d[:, 4:]), 1)
    return d.cpu().numpy(), dt


def parse_args():
    ap = argparse.ArgumentParser("MiniYOLO-v2 predict")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True,
                    help="image file/dir, video file, or a camera index")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-dir", type=str, default="runs/v2/predict")
    ap.add_argument("--max-frames", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    model, ckpt = load_model(args.weights, device)
    if args.half:
        model.half()
    model.fuse()
    names = ckpt.get("names") or [str(i) for i in range(ckpt["nc"])]
    save_dir = increment_path(Path(args.save_dir), mkdir=True)

    src = args.source
    is_cam = src.isdigit()
    is_vid = is_cam or Path(src).suffix.lower() in VID_EXT

    # warm up so the first timing is not a lie
    for _ in range(3):
        model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device,
                          dtype=torch.half if args.half else torch.float))

    if is_vid:
        cap = cv2.VideoCapture(int(src) if is_cam else src)
        writer, n, times = None, 0, []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            dets, dt = infer(model, frame, args.imgsz, device, args.conf, args.half)
            times.append(dt)
            frame = draw(frame, dets, names)
            panel = [f"FPS: {1000 / max(np.mean(times[-30:]), 1e-6):.0f}  ({np.mean(times[-30:]):.1f} ms)",
                     f"frame: {n + 1}",
                     f"detections: {len(dets)}"]
            frame = draw_panel(frame, panel)
            if args.show:
                cv2.imshow("MiniYOLO-v2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if not is_cam:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(save_dir / "out.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
                writer.write(frame)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"{colorstr('done')} {n} frames, {np.mean(times):.2f} ms/frame "
              f"({1000 / max(np.mean(times), 1e-6):.0f} FPS)")
    else:
        p = Path(src)
        files = sorted(f for f in glob.glob(str(p / "*")) if Path(f).suffix.lower() in IMG_EXT) \
            if p.is_dir() else [str(p)]
        times = []
        for f in files:
            im0 = cv2.imread(f)
            if im0 is None:
                continue
            dets, dt = infer(model, im0, args.imgsz, device, args.conf, args.half)
            times.append(dt)
            cv2.imwrite(str(save_dir / Path(f).name), draw(im0, dets, names))
        print(f"{colorstr('done')} {len(times)} images -> {save_dir}  "
              f"({np.mean(times):.2f} ms/img)")


if __name__ == "__main__":
    main()
