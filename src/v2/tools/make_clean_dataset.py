"""Build a convention-consistent copy of the dataset.

HISTORICAL -- kept for the record, no longer runnable as written.
==============================================================
On 2026-08-29 the three dataset directories were consolidated into a single
canonical `dataset/` (the validated former `dataset_final_v1`). The inputs this
script expects -- the 28,170-image contaminated `dataset/` and the `dataset_clean/`
it produced -- no longer exist. It is preserved because it documents how the
experiment-2 dataset was derived, and the lineage CSVs it read are still on disk at
`dataset/metadata/lineage/`.

Do not re-run it against the current `dataset/`: that directory is already filtered,
so the script would be a no-op at best and would shadow the validated build at worst.
The current dataset is built and checked by `build_final_dataset.py` +
`validate_dataset.py`.

Diagnosis (see info/experiment 1 baseline/): the dataset merges two
incompatible annotation conventions under the same class names --

  * object-level : one small box per eye        (median width ~0.10-0.19)
  * face-level   : one big box over the whole face, labelled with the eye
                   state it depicts             (median width ~0.62-0.78)

For closed_eye/open_eye this is a genuine contradiction: the same visual
content is labelled at two box scales 3-8x apart, so the regressor can
never be tight and mAP50-95 is capped. `yawning` is unaffected -- a yawn is
a face-scale event, and 100% of yawning boxes live in images that this
filter keeps.

The filter drops any IMAGE that contains a face-scale EYE box (whole image,
not just the box -- dropping a box alone would turn a real eye into
unlabelled background and actively teach the model to miss it).

    python -m src.v2.tools.make_clean_dataset --out dataset_clean   # historical
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EYE_CLASSES = (0, 1)          # closed_eye, open_eye
FACE_SCALE_W = 0.4            # boxes wider than this are the face-level convention
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def link_or_copy(src, dst):
    """Hardlink when possible (instant, no extra disk), else copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def find_image(img_dir, stem):
    for ext in IMG_EXT:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    hits = list(img_dir.glob(f"{stem}.*"))
    return hits[0] if hits else None


def is_contaminated(rows):
    """True if any eye-class box is drawn at face scale."""
    return any(int(r[0]) in EYE_CLASSES and float(r[3]) > FACE_SCALE_W for r in rows)


def process_split(src_root, dst_root, split, names):
    src_img = src_root / "images" / split
    src_lab = src_root / "labels" / split
    if not src_img.exists():
        return None
    dst_img = dst_root / "images" / split
    dst_lab = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    kept_boxes = [0] * len(names)
    dropped_boxes = [0] * len(names)

    for lab in sorted(src_lab.glob("*.txt")):
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        if is_contaminated(rows):
            dropped += 1
            for r in rows:
                dropped_boxes[int(r[0])] += 1
            continue
        img = find_image(src_img, lab.stem)
        if img is None:
            continue
        link_or_copy(img, dst_img / img.name)
        link_or_copy(lab, dst_lab / lab.name)
        kept += 1
        for r in rows:
            kept_boxes[int(r[0])] += 1

    return {"kept": kept, "dropped": dropped,
            "kept_boxes": kept_boxes, "dropped_boxes": dropped_boxes}


def main():
    ap = argparse.ArgumentParser("build convention-consistent dataset")
    ap.add_argument("--src", type=str, default="dataset")
    ap.add_argument("--out", type=str, default="dataset_clean")
    args = ap.parse_args()

    src_root = (ROOT / args.src).resolve()
    dst_root = (ROOT / args.out).resolve()

    # Guard: the consolidated `dataset/` is already the filtered build. Re-running this
    # on it would produce a redundant copy that no report or checkpoint refers to.
    if (src_root / "metadata" / "build_info.json").exists():
        raise SystemExit(
            f"refusing to run: {src_root} is the consolidated, validated dataset "
            "(metadata/build_info.json present), not the raw contaminated source this "
            "script was written for. See the HISTORICAL note at the top of this file.")

    with open(src_root / "data.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    dst_root.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "val", "test"):
        r = process_split(src_root, dst_root, split, names)
        if r:
            report[split] = r
            kb = "  ".join(f"{names[i]}={r['kept_boxes'][i]}" for i in range(len(names)))
            db = "  ".join(f"{names[i]}={r['dropped_boxes'][i]}" for i in range(len(names)))
            print(f"{split:5s}  kept {r['kept']:6d} imgs   {kb}")
            print(f"       drop {r['dropped']:6d} imgs   {db}")

    # No absolute `path:` key -- resolve_data() falls back to the yaml's own
    # folder when it is absent, so the dataset survives being moved between
    # drives without emitting a stale-path warning.
    out_yaml = {"train": "images/train", "val": "images/val",
                "test": "images/test", "nc": len(names),
                "names": {i: n for i, n in enumerate(names)}}
    header = (
        "# Convention-consistent subset of ../dataset, built by\n"
        "# src/v2/tools/make_clean_dataset.py\n"
        "#\n"
        "# Images containing a FACE-SCALE eye box (width > 0.4) are removed --\n"
        "# the source data mixed per-eye boxes with whole-face boxes under the\n"
        "# same class names, which caps mAP50-95. All yawning boxes survive.\n"
        "# NOTE: val/test here are SUBSETS of the original splits, so numbers are\n"
        "# not directly comparable to a model evaluated on ../dataset. Evaluate\n"
        "# both models on THIS test split for an apples-to-apples comparison.\n"
    )
    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    print(f"\ndone -> {dst_root}")


if __name__ == "__main__":
    main()
