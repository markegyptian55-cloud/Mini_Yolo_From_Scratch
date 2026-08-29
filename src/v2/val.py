"""Evaluate a MiniYOLO-v2 checkpoint.

    python -m src.v2.val --weights runs/v2/exp/weights/best.pt --split test
    python -m src.v2.val --weights ... --no-e2e      # compare against the NMS path
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.data.build import build_dataloader, build_dataset    # noqa: E402
from src.v2.engine.validator import Validator                    # noqa: E402
from src.v2.models.yolo import MiniYOLOv2                        # noqa: E402
from src.v2.train import resolve_data, split_dirs                # noqa: E402
from src.v2.utils.general import colorstr                        # noqa: E402


def load_model(weights, device, use_ema=True):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    # reg_max default 1: checkpoints written before experiment 3 have no such key and
    # were trained with the scalar (DFL-free) head. Guessing 16 there would silently
    # build the wrong architecture and load nothing.
    model = MiniYOLOv2(nc=ckpt["nc"], scale=ckpt.get("scale", "n"),
                       names=ckpt.get("names"),
                       reg_max=int(ckpt.get("reg_max", 1))).to(device)
    sd = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ckpt


def parse_args():
    ap = argparse.ArgumentParser("MiniYOLO-v2 validation")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="dataset/data.yaml")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--no-e2e", action="store_true", help="use one-to-many head + NMS")
    ap.add_argument("--raw", action="store_true", help="evaluate raw weights, not EMA")
    ap.add_argument("--half", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    model, ckpt = load_model(args.weights, device, use_ema=not args.raw)
    if args.half:
        model.half()
    hyp = ckpt.get("hyp", {})

    root, data, nc, names = resolve_data(args.data)
    img_dir, lab_dir = split_dirs(root, data.get(args.split, f"images/{args.split}"))
    ds = build_dataset(img_dir, lab_dir, nc, args.imgsz, hyp, augment=False,
                       prefix=colorstr(f"{args.split}: "))
    dl = build_dataloader(ds, args.batch, args.workers, shuffle=False)

    v = Validator(dl, nc, names, device, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, e2e=not args.no_e2e)
    res = v(model, half=args.half, desc=args.split)
    mode = "one-to-many + NMS" if args.no_e2e else "one-to-one (NMS-free)"
    print(f"\n{colorstr('mode')} {mode}   weights={'raw' if args.raw else 'EMA'}"
          f"   imgsz={args.imgsz}   half={args.half}")
    print(v.metrics.table())
    print(f"\nspeed: {res['ms_per_image']:.2f} ms/img  ({res['fps']:.0f} img/s, "
          f"batch={args.batch})")
    return res


if __name__ == "__main__":
    main()
