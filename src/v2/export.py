"""Export a trained checkpoint to ONNX (and optionally TorchScript).

The exported graph:
  * has the one-to-many training branch removed,
  * has BatchNorm folded into the preceding convolutions,
  * ends in the NMS-free one-to-one head -> a fixed (1, max_det, 6) output
    [x, y, w, h, conf, cls] with boxes in input-image pixels.

    python -m src.v2.export --weights checkpoints/Expi-3-imagez-384/weights/best.pt --imgsz 384
    python -m src.v2.export --weights ... --opset 12 --max-det 100   # NCNN/TFLite

The .onnx is written beside the .pt. Per AGENTS.md Rule 1 the deployed file is named
`best.onnx` in that same `weights/` folder -- pass --name best to have it written
directly, rather than renaming a `best_<imgsz>.onnx` afterwards.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.models.yolo import MiniYOLOv2       # noqa: E402
from src.v2.utils.general import colorstr       # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser("MiniYOLO-v2 export")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=13,
                    help="13 for TensorRT/ORT, 12 for the widest NCNN/TFLite support")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--dynamic", action="store_true", help="dynamic batch axis")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--torchscript", action="store_true")
    ap.add_argument("--raw", action="store_true", help="export raw weights, not EMA")
    ap.add_argument("--raw-head", action="store_true",
                    help="skip the in-graph top-k. Output becomes (B, 4+nc, A) of raw "
                         "xywh + class scores -- required for NCNN / TFLite, which do "
                         "not support TopK / GatherElements / Mod")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--name", type=str, default="",
                    help='output stem beside the weights. Default keeps the historical '
                         '"<weights>_<imgsz>.onnx"; pass "best" for the Rule 1 layout')
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    # See val.py: absent reg_max means a pre-experiment-3, DFL-free checkpoint.
    model = MiniYOLOv2(nc=ckpt["nc"], scale=ckpt.get("scale", "n"),
                       names=ckpt.get("names"),
                       reg_max=int(ckpt.get("reg_max", 1)))
    # strict=False: checkpoints carry stray thop profiling buffers (total_ops /
    # total_params) left behind by bench.py. They are not weights.
    model.load_state_dict(ckpt["ema" if (not args.raw and "ema" in ckpt) else "model"],
                          strict=False)
    model.export_ready(max_det=args.max_det).to(device)
    model.head.export_postprocess = not args.raw_head

    im = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)
    if args.half:
        model.half()
        im = im.half()

    with torch.no_grad():
        y = model(im)
    if args.raw_head:
        print(f"{colorstr('export')} output {tuple(y.shape)}  "
              f"(B, 4+nc, anchors): rows 0-3 = xywh in pixels of a {args.imgsz}px input, "
              f"rows 4.. = per-class scores (already sigmoid). "
              f"Host code: score = max over classes; keep score > conf. "
              f"No NMS needed -- the one-to-one head is already non-redundant.")
    else:
        print(f"{colorstr('export')} output {tuple(y.shape)}  "
              f"[x, y, w, h, conf, cls], boxes in pixels of a {args.imgsz}px input")
    model.info(imgsz=args.imgsz)

    out = Path(args.weights).with_suffix("")
    if args.name:
        onnx_path = str(out.parent / f"{args.name}{'_rawhead' if args.raw_head else ''}.onnx")
    else:
        onnx_path = f"{out}_{args.imgsz}{'_rawhead' if args.raw_head else ''}.onnx"
    dynamic = {"images": {0: "batch"}, "det": {0: "batch"}} if args.dynamic else None
    torch.onnx.export(model, im, onnx_path, opset_version=args.opset,
                      input_names=["images"], output_names=["det"],
                      dynamic_axes=dynamic, do_constant_folding=True)

    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"{colorstr('green', 'bold', 'ok')} ONNX opset={args.opset} -> {onnx_path} "
              f"({Path(onnx_path).stat().st_size / 1e6:.2f} MB)")
        ops = sorted({n.op_type for n in m.graph.node})
        print(f"     ops: {', '.join(ops)}")
    except Exception as e:
        print(f"onnx check skipped: {e}")

    if args.torchscript:
        ts_path = f"{out}_{args.imgsz}.torchscript"
        torch.jit.trace(model, im, strict=False).save(ts_path)
        print(f"{colorstr('green', 'bold', 'ok')} TorchScript -> {ts_path}")

    print("\nnext steps:")
    print(f"  TensorRT : trtexec --onnx={onnx_path} --fp16 --saveEngine=model.engine")
    print(f"  OpenVINO : ovc {onnx_path}")
    print(f"  NCNN     : onnx2ncnn {onnx_path} model.param model.bin   (use --opset 12)")


if __name__ == "__main__":
    main()
