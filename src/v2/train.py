"""MiniYOLO-v2 training entry point.

    python -m src.v2.train --data dataset/data.yaml --scale n --imgsz 384 --batch 64 `
        --project checkpoints --name Expi-3-imagez-384 --exist-ok

Runs live in `checkpoints/Expi-<N>-imagez-<Size>/`; the trainer writes `weights/` there
and `src/v2/report.py --out "<same dir>/REPORTS EXPI-<N>"` fills in the rest. See
AGENTS.md "Rule 1".
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2.data.build import build_dataloader, build_dataset          # noqa: E402
from src.v2.engine.trainer import Trainer                              # noqa: E402
from src.v2.models.yolo import build_model                             # noqa: E402
from src.v2.utils.general import colorstr, increment_path, init_seeds  # noqa: E402


def resolve_data(data_yaml):
    """Read a YOLO data.yaml and return (root, nc, names).

    `path:` is honoured when it exists, otherwise we fall back to the directory
    containing the yaml -- which is what you want after moving a dataset between
    machines (this project's data.yaml still points at an old drive).
    """
    p = Path(data_yaml).resolve()
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", "")) if d.get("path") else p.parent
    if not root.exists():
        print(f"{colorstr('yellow', 'bold', 'note')} data.yaml path '{root}' not found; "
              f"using '{p.parent}'")
        root = p.parent
    names = d.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    nc = int(d.get("nc", len(names)))
    return root, d, nc, names


def split_dirs(root, rel):
    img_dir = (root / rel).resolve()
    parts = list(img_dir.parts)
    # .../images/train -> .../labels/train
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    label_dir = Path(*parts)
    if not label_dir.exists():
        alt = img_dir.parent / "labels"
        label_dir = alt if alt.exists() else label_dir
    return img_dir, label_dir


def parse_args():
    ap = argparse.ArgumentParser("MiniYOLO-v2 trainer")
    ap.add_argument("--data", type=str, default="DATASET-CHAPTER 2/data.yaml")
    ap.add_argument("--hyp", type=str, default="src/v2/cfg/hyp.yaml")
    ap.add_argument("--scale", type=str, default="n", choices=["p", "t", "n", "s"])
    ap.add_argument("--reg-max", type=int, default=1,
                    help="box-regression bins per ltrb side. 1 = the scalar DFL-free "
                         "head (default: the proven baseline). 16 = DFL, which "
                         "Chapter 1 experiment 3 tested and REFUTED -- it lost 0.017 "
                         "mAP50 and 0.005 mAP50-95 while costing +0.19 MB and ~7% "
                         "more time. Do not re-enable without a new hypothesis.")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6,
                    help="train dataloader workers, auto-capped by free RAM. "
                         "Each one is a separate torch process on Windows")
    ap.add_argument("--optimizer", type=str, default="musgd",
                    choices=["musgd", "sgd", "adamw"])
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="checkpoints",
                    help="root for experiment folders; see AGENTS.md Rule 1")
    ap.add_argument("--name", type=str, default="exp",
                    help="experiment folder name, e.g. Expi-3-imagez-384")
    ap.add_argument("--exist-ok", action="store_true",
                    help="write into --name even if the folder already exists, instead of "
                         "forking to <name>2. Needed when the target folder was pre-created. "
                         "Still refuses to overwrite a folder that already holds weights.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=60, help="0 disables early stopping")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--cache", action="store_true", help="cache all images in RAM")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-e2e", action="store_true",
                    help="validate through the one-to-many head + NMS instead")
    ap.add_argument("--val-conf", type=float, default=0.001)
    ap.add_argument("--val-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--val-workers", type=int, default=0,
                    help="0 keeps validation in-process (recommended: saves memory)")
    ap.add_argument("--lr0", type=float, default=None, help="override hyp lr0")
    ap.set_defaults(amp=True)
    return ap.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    init_seeds(args.seed, deterministic=False)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available()
                          and args.device not in ("cpu",) else "cpu")

    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)
    if args.optimizer == "adamw" and args.lr0 is None:
        hyp["lr0"] = hyp.get("lr0_adamw", 0.001)
        print(f"{colorstr('hyp')} optimizer=adamw -> lr0={hyp['lr0']}")
    if args.lr0 is not None:
        hyp["lr0"] = args.lr0

    root, data, nc, names = resolve_data(args.data)
    tr_img, tr_lab = split_dirs(root, data.get("train", "images/train"))
    va_img, va_lab = split_dirs(root, data.get("val", "images/val"))

    if args.resume:
        # keep logging into the run we are resuming instead of forking a new folder
        save_dir = Path(args.resume).resolve().parents[1]
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        target = Path(args.project) / args.name
        if args.exist_ok and (target / "weights" / "best.pt").exists():
            # A pre-created empty folder is fine to reuse; a finished experiment is not.
            raise SystemExit(
                f"refusing to overwrite: {target / 'weights' / 'best.pt'} already exists. "
                "Rename or move that experiment first, or drop --exist-ok to fork a new "
                "folder.")
        save_dir = increment_path(target, exist_ok=args.exist_ok, mkdir=True)
    # Rule 1: keep the run's exact configuration with its report, not loose in the
    # experiment root. Falls back to the root when there is no REPORTS folder yet.
    _rep = next(iter(sorted(save_dir.glob("REPORTS EXPI*"))), None)
    cfg_dir = _rep or save_dir
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "hyp.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(cfg_dir / "args.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    print(f"{colorstr('data')} {root}  nc={nc}  names={names}")
    train_ds = build_dataset(tr_img, tr_lab, nc, args.imgsz, hyp, augment=True,
                             cache_ram=args.cache, prefix=colorstr("train: "))
    val_ds = build_dataset(va_img, va_lab, nc, args.imgsz, hyp, augment=False,
                           cache_ram=False, prefix=colorstr("val: "))
    train_loader = build_dataloader(train_ds, args.batch, args.workers, shuffle=True,
                                    seed=args.seed, drop_last=True)
    # Validation is letterbox-only (no mosaic), ~2 ms/image single-process, so it runs
    # in-process. Persistent val workers would otherwise hold N extra torch imports in
    # memory for the entire run for no throughput gain.
    val_loader = build_dataloader(val_ds, max(args.batch, 32), args.val_workers,
                                  shuffle=False, seed=args.seed)

    model = build_model(nc=nc, scale=args.scale, imgsz=args.imgsz,
                        e2e=not args.no_e2e, max_det=args.max_det, names=names,
                        device=device, reg_max=args.reg_max)
    print(f"{colorstr('head')} reg_max={args.reg_max} "
          f"({'DFL' if args.reg_max > 1 else 'scalar, DFL-free'})")

    trainer = Trainer(model, train_loader, val_loader, hyp, args, device, save_dir)
    if args.resume:
        trainer.resume(args.resume)
    print(f"{colorstr('results')} {save_dir}")
    try:
        trainer.train()
    except RuntimeError as e:
        if "worker" in str(e).lower() or "shared file mapping" in str(e).lower():
            print("\n" + colorstr("red", "bold", "dataloader workers died")
                  + " -- this is memory, not a code bug."
                  "\n  * lower --workers (try 4, then 2, then 0)"
                  "\n  * close memory-heavy apps (a browser can hold 8+ GB)"
                  "\n  * your pagefile is on the D: HDD; moving it to the C: SSD removes"
                  " the stall that turns a commit spike into a crash")
        raise


if __name__ == "__main__":
    main()
