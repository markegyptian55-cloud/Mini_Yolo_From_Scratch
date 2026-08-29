"""Independent validator for a built dataset. Reads ONLY the dataset directory.

Deliberately does not import the builder: if the builder has a bug, a validator
sharing its logic would inherit it. Everything here is re-derived from files on
disk, with the manifest used only as a cross-check target.

    python -m src.v2.tools.validate_dataset --data dataset_final_v1
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPLITS = ("train", "val", "test")
NAMES = ["closed_eye", "open_eye", "yawning"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Gate:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))
        if status == "FAIL":
            self.failed = True

    def render(self):
        w = max(len(r[0]) for r in self.rows)
        out = []
        for n, s, d in self.rows:
            tag = {"PASS": "[PASS]", "FAIL": "[FAIL]",
                   "LIMIT": "[DOCUMENTED LIMITATION]"}[s]
            out.append(f"{tag:26s} {n.ljust(w)}  {d}")
        return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_final_v1")
    ap.add_argument("--full-hash", action="store_true",
                    help="re-hash every test file against the lock (slower)")
    args = ap.parse_args()
    root = (ROOT / args.data).resolve()
    g = Gate()

    # ---------- yaml ----------
    ypath = root / "data.yaml"
    try:
        y = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        names = y.get("names")
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
        ok = (names == NAMES and int(y.get("nc", -1)) == 3
              and all(y.get(s) == f"images/{s}" for s in SPLITS))
        g.add("YAML consistency", "PASS" if ok else "FAIL",
              f"nc={y.get('nc')} names={names} abs_path_key={'path' in y}")
        g.add("class mapping", "PASS" if names == NAMES else "FAIL",
              "0=closed_eye 1=open_eye 2=yawning" if names == NAMES else f"got {names}")
    except Exception as e:  # noqa: BLE001
        g.add("YAML consistency", "FAIL", str(e))
        g.add("class mapping", "FAIL", "yaml unreadable")

    # ---------- images / labels ----------
    img_bad, lab_bad, orphan_lab, orphan_img = [], [], [], []
    counts, boxes = {}, {}
    per_class = defaultdict(int)
    all_stems = {}
    for sp in SPLITS:
        idir, ldir = root / "images" / sp, root / "labels" / sp
        imgs = {p.stem: p for p in idir.glob("*") if p.suffix.lower() in IMG_EXT}
        labs = {p.stem: p for p in ldir.glob("*.txt")}
        orphan_lab += [f"{sp}/{s}" for s in labs.keys() - imgs.keys()]
        orphan_img += [f"{sp}/{s}" for s in imgs.keys() - labs.keys()]
        counts[sp] = len(imgs)
        nb = 0
        for s, p in imgs.items():
            all_stems[s] = sp
            if p.stat().st_size == 0:
                img_bad.append(f"{sp}/{p.name}: zero-byte")
                continue
            im = cv2.imread(str(p))
            if im is None or im.size == 0:
                img_bad.append(f"{sp}/{p.name}: unreadable")
        for s, p in labs.items():
            for ln, line in enumerate(p.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                f = line.split()
                if len(f) != 5:
                    lab_bad.append(f"{sp}/{p.name}:{ln} field count {len(f)}")
                    continue
                try:
                    c = int(f[0]); x, yy, w, h = (float(v) for v in f[1:])
                except ValueError:
                    lab_bad.append(f"{sp}/{p.name}:{ln} non-numeric")
                    continue
                if c not in (0, 1, 2):
                    lab_bad.append(f"{sp}/{p.name}:{ln} class {c}")
                elif not all(np.isfinite(v) for v in (x, yy, w, h)):
                    lab_bad.append(f"{sp}/{p.name}:{ln} NaN/inf")
                elif w <= 0 or h <= 0:
                    lab_bad.append(f"{sp}/{p.name}:{ln} non-positive wh")
                elif not (0 <= x <= 1 and 0 <= yy <= 1 and w <= 1 and h <= 1):
                    lab_bad.append(f"{sp}/{p.name}:{ln} out of [0,1]")
                else:
                    nb += 1
                    per_class[c] += 1
        boxes[sp] = nb

    g.add("image integrity", "PASS" if not img_bad else "FAIL",
          f"{sum(counts.values())} images, {len(img_bad)} bad" + (f" e.g. {img_bad[:2]}" if img_bad else ""))
    g.add("label integrity", "PASS" if not lab_bad else "FAIL",
          f"{sum(boxes.values())} boxes, {len(lab_bad)} invalid" + (f" e.g. {lab_bad[:2]}" if lab_bad else ""))
    g.add("split integrity", "PASS" if not (orphan_lab or orphan_img) else "FAIL",
          f"orphan labels={len(orphan_lab)} orphan images={len(orphan_img)}")

    # ---------- duplicate / leakage ----------
    dupe = [s for s, c in Counter(
        [p.stem for sp in SPLITS for p in (root / "images" / sp).glob("*")]).items() if c > 1]
    g.add("duplicate integrity", "PASS" if not dupe else "FAIL",
          f"{len(dupe)} stems appear in more than one split")

    man_p = root / "metadata" / "dataset_manifest.csv"
    if man_p.exists():
        man = pd.read_csv(man_p)
        vg = man[man.visual_group_id >= 0]
        span = vg.groupby("visual_group_id")["split"].nunique()
        n_span = int((span > 1).sum())
        g.add("cross-split near-duplicate leakage = zero", "PASS" if n_span == 0 else "FAIL",
              f"{n_span} visual groups span >1 split")
        intra = int((man.duplicate_status == "intra_split_near_duplicate").sum())
        g.add("near-duplicate accounting", "PASS",
              f"{intra} images flagged intra_split_near_duplicate (retained by design)")
        mism = []
        for sp in SPLITS:
            if int((man.split == sp).sum()) != counts[sp]:
                mism.append(f"{sp}: manifest {int((man.split == sp).sum())} vs disk {counts[sp]}")
        g.add("manifest consistency", "PASS" if not mism else "FAIL", "; ".join(mism) or "counts match disk")
        # annotation consistency: no eye box above tau survived
        bi = root / "metadata" / "build_info.json"
        tau = json.loads(bi.read_text())["tau"] if bi.exists() else 0.40
        over = int((man.max_eye_box_width > tau).sum())
        g.add("annotation consistency", "PASS" if over == 0 else "FAIL",
              f"{over} kept images have an eye box wider than tau={tau}")
    else:
        for n in ("cross-split near-duplicate leakage = zero", "manifest consistency",
                  "annotation consistency", "near-duplicate accounting"):
            g.add(n, "FAIL", "metadata/dataset_manifest.csv missing")

    # ---------- test set lock ----------
    lock = root / "metadata" / "test_set_lock.sha256"
    if lock.exists():
        lines = [l for l in lock.read_text().splitlines() if l.strip()]
        n_test_files = len(list((root / "images" / "test").glob("*"))) + \
            len(list((root / "labels" / "test").glob("*.txt")))
        ok = len(lines) == n_test_files
        detail = f"{len(lines)} hashes for {n_test_files} test files"
        if ok and args.full_hash:
            bad = 0
            for l in lines:
                hsh, rel = l.split("  ", 1)
                p = root / rel
                if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != hsh:
                    bad += 1
            ok = bad == 0
            detail += f", {bad} mismatched"
        g.add("test set lock", "PASS" if ok else "FAIL", detail)
    else:
        g.add("test set lock", "FAIL", "metadata/test_set_lock.sha256 missing")

    # ---------- reproducibility ----------
    bi = root / "metadata" / "build_info.json"
    need = ["dataset_manifest.csv", "source_manifest.csv", "split_manifest.csv",
            "quality_report.csv", "class_statistics.csv"]
    miss = [n for n in need if not (root / "metadata" / n).exists()]
    g.add("reproducibility", "PASS" if bi.exists() and not miss else "FAIL",
          f"build_info={'yes' if bi.exists() else 'no'} missing_metadata={miss or 'none'}")

    # ---------- deployment-domain audit ----------
    if man_p.exists() and "lighting_condition" in man.columns:
        lit = man.lighting_condition.value_counts().to_dict()
        gray = int((man.gray_flag == "grayscale_or_ir").sum())
        g.add("deployment-domain audit", "PASS",
              f"lighting={lit}, grayscale/IR={gray}")
    else:
        g.add("deployment-domain audit", "FAIL", "manifest lacks lighting fields")

    # ---------- documented limitations ----------
    if man_p.exists():
        unk = int((man.subject_id == "unknown").sum())
        g.add("subject/video-disjoint split", "LIMIT",
              f"subject_id unknown for {unk}/{len(man)} images; session_id absent in source "
              "manifest, so subject-disjointness cannot be established")
        g.add("glasses / head-pose coverage", "LIMIT",
              "no per-image glasses or pose metadata exists in any source; reported as unknown")

    g.add("final dataset report", "PASS" if (root / "DATASET_REPORT.md").exists() else "FAIL",
          "DATASET_REPORT.md")

    # ---------- summary ----------
    print("=" * 78)
    print("FINAL DATASET VALIDATION")
    print("=" * 78)
    print(g.render())
    print()
    print(f"Status: {'FAIL' if g.failed else 'PASS'}")
    print()
    print("Images:")
    for sp in SPLITS:
        print(f"  {sp:6s} {counts.get(sp, 0)}")
    print(f"  Total  {sum(counts.values())}")
    print()
    print("Boxes:")
    for sp in SPLITS:
        print(f"  {sp:6s} {boxes.get(sp, 0)}")
    print(f"  Total  {sum(boxes.values())}")
    print()
    print("Classes:")
    for i, n in enumerate(NAMES):
        print(f"  {n:12s} {per_class[i]}")
    print()
    print(f"Baseline readiness: {'NOT READY' if g.failed else 'READY'}")
    return 1 if g.failed else 0


if __name__ == "__main__":
    sys.exit(main())
