"""Build dataset_final_v1 -- the locked baseline dataset for BASELINE_V1.

Deterministic, hardlinked, never mutates the source. See DATASET_REPORT.md (which
this script writes) for the full rationale.

Selection rule, applied per IMAGE, with an A-F reason code:

  E  remove  unreadable / zero-byte image, or a syntactically invalid label
  C  remove  any eye-class box wider than tau -- ambiguous population containing
             BOTH legitimate eye close-up crops AND face-scale mislabels. They
             cannot be separated automatically with the tooling available offline,
             and neither is deployment-relevant for a dashcam, so both go.
  A  keep    all eye boxes at eye scale; yawning kept at any scale (face-scale is
             the correct convention for that class)

Splits are INHERITED from the source, never recomputed: the existing assignment
was verified leakage-free (no visual_group_id spans two splits) and reshuffling
would destroy that property. Intra-split near-duplicates are RETAINED and flagged
in the manifest -- redundancy is not leakage.

NOTE (2026-08-29): the output of this script WAS `dataset_final_v1/`, which has since
been promoted to the project's single canonical `dataset/`. The `--src` this expects
is the pre-consolidation contaminated dataset, which no longer exists on this machine;
the build is reproducible only from a backup of it plus the lineage CSVs now at
`dataset/metadata/lineage/`.

    python -m src.v2.tools.build_final_dataset --src <raw> --out dataset_final_v1 --tau 0.40
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EYE = (0, 1)
NAMES = ["closed_eye", "open_eye", "yawning"]
SPLITS = ("train", "val", "test")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def find_image(img_dir, stem):
    for e in IMG_EXT:
        p = img_dir / f"{stem}{e}"
        if p.exists():
            return p
    hits = list(img_dir.glob(stem + ".*"))
    return hits[0] if hits else None


def manifest_maps(src_root):
    p = src_root / "curated_dataset_manifest.csv"
    if not p.exists():
        return {}, {}
    m = pd.read_csv(p)
    k = m[m.keep_or_remove == "KEEP"]
    stems = k.new_path.apply(lambda q: os.path.splitext(os.path.basename(str(q)))[0])
    return dict(zip(stems, k.source)), dict(zip(stems, k.visual_group_id))


def parse_label(path):
    """Return (rows, error). rows = list of (cls,x,y,w,h)."""
    rows = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5:
            return None, f"line {ln}: expected 5 fields, got {len(p)}"
        try:
            c = int(p[0])
            x, y, w, h = (float(v) for v in p[1:])
        except ValueError:
            return None, f"line {ln}: non-numeric field"
        if c not in (0, 1, 2):
            return None, f"line {ln}: class id {c} out of range"
        if not all(np.isfinite(v) for v in (x, y, w, h)):
            return None, f"line {ln}: NaN/inf coordinate"
        if w <= 0 or h <= 0:
            return None, f"line {ln}: non-positive box size"
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w <= 1.0 and h <= 1.0):
            return None, f"line {ln}: coordinate outside [0,1]"
        rows.append((c, x, y, w, h))
    return rows, None


def probe_image(path):
    """(ok, width, height, gray_flag, brightness, blur_var, err)."""
    try:
        if path.stat().st_size == 0:
            return False, 0, 0, "unknown", -1.0, -1.0, "zero-byte file"
        im = cv2.imread(str(path))
        if im is None or im.size == 0:
            return False, 0, 0, "unknown", -1.0, -1.0, "unreadable by OpenCV"
        h, w = im.shape[:2]
        b, g, r = (c.astype(np.int16) for c in cv2.split(im))
        gray = bool(np.abs(b - g).mean() < 6 and np.abs(g - r).mean() < 6)
        gy = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return True, w, h, ("grayscale_or_ir" if gray else "rgb"), float(gy.mean()), \
            float(cv2.Laplacian(gy, cv2.CV_64F).var()), ""
    except Exception as e:  # noqa: BLE001
        return False, 0, 0, "unknown", -1.0, -1.0, f"{type(e).__name__}: {e}"


def lighting_bucket(brightness, gray_flag):
    if brightness < 0:
        return "unknown"
    if brightness < 60:
        return "night_or_very_low_light"
    if brightness < 100:
        return "low_light"
    if brightness > 190:
        return "bright_or_backlit"
    return "normal_daylight"


def main():
    ap = argparse.ArgumentParser("build dataset_final_v1")
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out", default="dataset_final_v1")
    ap.add_argument("--tau", type=float, default=0.40)
    ap.add_argument("--blur-p", type=float, default=10.0,
                    help="percentile of Laplacian variance below which an image is flagged blurry")
    args = ap.parse_args()

    src_root = (ROOT / args.src).resolve()
    dst_root = (ROOT / args.out).resolve()
    src_map, vg_map = manifest_maps(src_root)

    # ---------------- pass 1: inspect every source image ----------------
    records = []
    for split in SPLITS:
        lab_dir = src_root / "labels" / split
        img_dir = src_root / "images" / split
        if not lab_dir.exists():
            continue
        for f in sorted(lab_dir.glob("*.txt")):
            stem = f.stem
            rec = dict(image_id=stem, source_dataset=src_map.get(stem, "unknown"),
                       split=split, visual_group_id=vg_map.get(stem, -1),
                       subject_id="unknown", video_id="unknown", session_id="unknown")
            img = find_image(img_dir, stem)
            if img is None:
                rec.update(decision="REMOVE", reason_code="E",
                           reason="orphan label: no matching image", quality_status="missing_image")
                records.append(rec)
                continue
            rows, err = parse_label(f)
            ok, w, h, gray, bright, blur, ierr = probe_image(img)
            rec.update(original_path=str(img), image_width=w, image_height=h,
                       gray_flag=gray, brightness=round(bright, 2), blur_var=round(blur, 2),
                       lighting_condition=lighting_bucket(bright, gray),
                       glasses_status="unknown", blur_status="unknown")
            if not ok:
                rec.update(decision="REMOVE", reason_code="E", reason=ierr,
                           quality_status="corrupt")
                records.append(rec)
                continue
            if rows is None:
                rec.update(decision="REMOVE", reason_code="E",
                           reason=f"invalid label: {err}", quality_status="invalid_label")
                records.append(rec)
                continue
            cls = [r[0] for r in rows]
            eyes = [r[3] for r in rows if r[0] in EYE]
            max_eye_w = max(eyes) if eyes else 0.0
            rec.update(class_count=len(rows),
                       classes_present="|".join(sorted({NAMES[c] for c in cls})) or "none",
                       n_closed=cls.count(0), n_open=cls.count(1), n_yawn=cls.count(2),
                       max_eye_box_width=round(max_eye_w, 4), quality_status="ok",
                       annotation_status="eye_scale" if max_eye_w <= args.tau else "over_tau")
            if max_eye_w > args.tau:
                rec.update(decision="REMOVE", reason_code="C",
                           reason=("eye box wider than tau: eye close-up crop or face-scale "
                                   "mislabel; ambiguous and not deployment-relevant"))
            else:
                rec.update(decision="KEEP", reason_code="A",
                           reason="eye boxes at eye scale; deployment-relevant framing")
            records.append(rec)

    df = pd.DataFrame(records)
    kept = df[df.decision == "KEEP"].copy()

    # ---------------- near-duplicate flags (retained, not removed) ----------------
    vg_counts = kept[kept.visual_group_id >= 0].visual_group_id.value_counts()
    dup_groups = set(vg_counts[vg_counts > 1].index)
    kept["duplicate_status"] = np.where(kept.visual_group_id.isin(dup_groups),
                                        "intra_split_near_duplicate", "unique")
    # blur flag from the kept distribution
    bv = kept.loc[kept.blur_var >= 0, "blur_var"]
    thr = float(np.percentile(bv, args.blur_p)) if len(bv) else -1.0
    kept["blur_status"] = np.where(kept.blur_var < 0, "unknown",
                                   np.where(kept.blur_var < thr, "blurry", "sharp"))

    # ---------------- write dataset ----------------
    for split in SPLITS:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    meta = dst_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    final_paths = []
    for r in kept.itertuples():
        img = Path(r.original_path)
        lab = src_root / "labels" / r.split / f"{r.image_id}.txt"
        di = dst_root / "images" / r.split / img.name
        dl = dst_root / "labels" / r.split / lab.name
        link_or_copy(img, di)
        link_or_copy(lab, dl)
        final_paths.append(str(di))
    kept["final_path"] = final_paths

    # ---------------- data.yaml (no absolute path key: survives moves) ----------------
    ycontent = (
        "# dataset_final_v1 -- locked baseline dataset for BASELINE_V1\n"
        "# Built by src/v2/tools/build_final_dataset.py. See DATASET_REPORT.md.\n"
        f"# tau = {args.tau} (eye-box scale rule, justified in dataset/metadata/analysis/threshold_study/)\n"
        "# No absolute 'path:' key: resolve_data() falls back to this file's own folder,\n"
        "# so the dataset survives being moved between drives.\n")
    with open(dst_root / "data.yaml", "w", encoding="utf-8") as fh:
        fh.write(ycontent)
        yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test",
                        "nc": 3, "names": {i: n for i, n in enumerate(NAMES)}},
                       fh, sort_keys=False)

    # ---------------- metadata ----------------
    man_cols = ["image_id", "source_dataset", "original_path", "final_path", "subject_id",
                "video_id", "session_id", "split", "visual_group_id", "class_count",
                "classes_present", "n_closed", "n_open", "n_yawn", "image_width",
                "image_height", "max_eye_box_width", "quality_status", "duplicate_status",
                "annotation_status", "lighting_condition", "gray_flag", "brightness",
                "blur_var", "blur_status", "glasses_status", "reason_code", "reason"]
    kept.reindex(columns=man_cols).to_csv(meta / "dataset_manifest.csv", index=False)
    df.reindex(columns=[c for c in man_cols if c in df.columns] + ["decision"]).to_csv(
        meta / "quality_report.csv", index=False)

    kept.groupby("source_dataset").agg(
        images=("image_id", "size"), closed=("n_closed", "sum"),
        open_eye=("n_open", "sum"), yawning=("n_yawn", "sum")).reset_index().to_csv(
        meta / "source_manifest.csv", index=False)

    kept.groupby(["split", "source_dataset"]).agg(
        images=("image_id", "size")).reset_index().to_csv(meta / "split_manifest.csv", index=False)

    cs = []
    for split in SPLITS:
        s = kept[kept.split == split]
        for i, col in enumerate(["n_closed", "n_open", "n_yawn"]):
            cs.append(dict(split=split, class_id=i, class_name=NAMES[i],
                           box_count=int(s[col].sum()),
                           image_count_containing_class=int((s[col] > 0).sum())))
    pd.DataFrame(cs).to_csv(meta / "class_statistics.csv", index=False)

    # ---------------- lock the test set ----------------
    lock = []
    for p in sorted((dst_root / "images" / "test").glob("*")):
        lock.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  images/test/{p.name}")
    for p in sorted((dst_root / "labels" / "test").glob("*.txt")):
        lock.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  labels/test/{p.name}")
    (meta / "test_set_lock.sha256").write_text("\n".join(lock) + "\n", encoding="utf-8")

    build_info = dict(
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=str(src_root), tau=args.tau, seed=0,
        images_inspected=int(len(df)), images_kept=int(len(kept)),
        removed_by_reason={k: int(v) for k, v in
                           df[df.decision == "REMOVE"].reason_code.value_counts().items()},
        test_files_locked=len(lock), tool=str(Path(__file__).relative_to(ROOT)))
    (meta / "build_info.json").write_text(json.dumps(build_info, indent=2), encoding="utf-8")

    print(json.dumps(build_info, indent=2))
    for split in SPLITS:
        s = kept[kept.split == split]
        print(f"  {split:5s} images={len(s):6d} boxes={int(s.class_count.sum()):6d} "
              f"closed={int(s.n_closed.sum()):6d} open={int(s.n_open.sum()):6d} "
              f"yawn={int(s.n_yawn.sum()):6d}")
    print(f"\nwrote {dst_root}")


if __name__ == "__main__":
    main()
