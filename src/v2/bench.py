"""Latency / size sweep across scales and input resolutions.

    python -m src.v2.bench                       # all scales, GPU fp16 + fp32
    python -m src.v2.bench --device cpu --scales n
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.models.yolo import MiniYOLOv2, SCALES     # noqa: E402
from src.v2.utils.general import colorstr             # noqa: E402


@torch.no_grad()
def timeit(model, imgsz, device, half, n=60, warmup=15):
    x = torch.zeros(1, 3, imgsz, imgsz, device=device,
                    dtype=torch.half if half else torch.float)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser("MiniYOLO-v2 benchmark")
    ap.add_argument("--scales", nargs="*", default=list(SCALES))
    ap.add_argument("--imgsz", nargs="*", type=int, default=[320, 384, 448, 512, 640])
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--nc", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          and args.device != "cpu" else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"{colorstr('device')} {dev_name}\n")
    print(f"{'scale':>6} {'params':>10} {'fp32 MB':>8} {'int8 MB':>8} " +
          " ".join(f"{s:>10}" for s in args.imgsz))

    for s in args.scales:
        m = MiniYOLOv2(nc=args.nc, scale=s).export_ready().to(device)
        n_p, _, _, size = m.info(imgsz=args.imgsz[0], verbose=False)
        row = [f"{s:>6} {n_p:>10,} {size:>8.2f} {size / 4:>8.2f}"]
        for sz in args.imgsz:
            if device.type == "cuda":
                mh = MiniYOLOv2(nc=args.nc, scale=s).export_ready().to(device).half()
                ms = timeit(mh, sz, device, True)
                del mh
            else:
                ms = timeit(m, sz, device, False)
            row.append(f"{ms:>9.2f}m")
        print(" ".join(row))
    print(f"\n(GPU numbers are fp16 PyTorch eager, batch 1, one-to-one head, no NMS. "
          f"TensorRT is typically 2-3x faster still.)")


if __name__ == "__main__":
    main()
