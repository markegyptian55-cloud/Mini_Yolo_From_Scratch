"""Three-way comparison: dataset/ vs dataset_clean/ vs candidate_final.

HISTORICAL -- kept for the record, no longer runnable as written.
==============================================================
This ran BEFORE dataset_final_v1 was written, so that the build decision rested on
evidence rather than assertion. Its output is preserved at
`dataset/metadata/analysis/dataset_comparison.md` and its conclusion is quoted in
`dataset/DATASET_REPORT.md`.

On 2026-08-29 the three directories were consolidated into one canonical `dataset/`
(the validated former `dataset_final_v1`), so two of the three inputs this script
compares are gone. The lineage manifest it reads still exists, moved to
`dataset/metadata/lineage/curated_dataset_manifest.csv`.

To re-derive the comparison you would need the pre-consolidation directories back
from a backup. Do not point `--src` at the current `dataset/` and read the result as
a comparison: it would be comparing the final dataset against itself.

    python -m src.v2.tools.compare_datasets --tau 0.40      # historical
"""
import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EYE = (0, 1)
NAMES = ["closed_eye", "open_eye", "yawning"]
SPLITS = ("train", "val", "test")
OUT = ROOT / "dataset" / "metadata" / "analysis"
SAMPLE_N = 400          # per split, for pixel-level domain probes
RNG = np.random.default_rng(0)


def manifest_maps(src_root):
    # Post-consolidation the lineage CSVs moved into metadata/lineage/; check both.
    p = src_root / "curated_dataset_manifest.csv"
    if not p.exists():
        p = src_root / "metadata" / "lineage" / "curated_dataset_manifest.csv"
    if not p.exists():
        return {}, {}
    m = pd.read_csv(p)
    k = m[m.keep_or_remove == "KEEP"]
    stems = k.new_path.apply(lambda q: os.path.splitext(os.path.basename(str(q)))[0])
    return dict(zip(stems, k.source)), dict(zip(stems, k.visual_group_id))


def scan(root, src_map=None, vg_map=None):
    """Per-image records for a YOLO dataset dir."""
    src_map = src_map or {}
    vg_map = vg_map or {}
    recs = []
    for split in SPLITS:
        lab = root / "labels" / split
        if not lab.exists():
            continue
        for f in sorted(lab.glob("*.txt")):
            rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
            cls = [int(r[0]) for r in rows]
            eyes = [float(r[3]) for r in rows if int(r[0]) in EYE]
            recs.append(dict(
                stem=f.stem, split=split, n_boxes=len(rows),
                n_closed=cls.count(0), n_open=cls.count(1), n_yawn=cls.count(2),
                max_eye_w=max(eyes) if eyes else 0.0,
                source=src_map.get(f.stem, "unknown"),
                vg=vg_map.get(f.stem, -1),
                boxes=[(int(r[0]), float(r[3]), float(r[4])) for r in rows]))
    return pd.DataFrame(recs)


def domain_probe(root, df, n=SAMPLE_N):
    """Sampled pixel statistics: grayscale/IR share, brightness, blur."""
    out = {}
    for split in SPLITS:
        sub = df[df.split == split]
        if sub.empty:
            continue
        sel = sub.sample(min(n, len(sub)), random_state=0)
        gray = dark = 0
        br, blur = [], []
        for r in sel.itertuples():
            p = list((root / "images" / split).glob(r.stem + ".*"))
            if not p:
                continue
            im = cv2.imread(str(p[0]))
            if im is None:
                continue
            b, g, rr = (c.astype(np.int16) for c in cv2.split(im))
            if np.abs(b - g).mean() < 6 and np.abs(g - rr).mean() < 6:
                gray += 1
            gy = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            m = float(gy.mean())
            br.append(m)
            if m < 60:
                dark += 1
            blur.append(float(cv2.Laplacian(gy, cv2.CV_64F).var()))
        n_ok = len(br)
        out[split] = dict(n=n_ok, gray_pct=100 * gray / max(n_ok, 1),
                          dark_pct=100 * dark / max(n_ok, 1),
                          bright_mean=float(np.mean(br)) if br else 0.0,
                          bright_p5=float(np.percentile(br, 5)) if br else 0.0,
                          bright_p95=float(np.percentile(br, 95)) if br else 0.0,
                          blur_median=float(np.median(blur)) if blur else 0.0,
                          blur_p10=float(np.percentile(blur, 10)) if blur else 0.0)
    return out


def box_stats(df):
    w = defaultdict(list)
    ar = defaultdict(list)
    for r in df.itertuples():
        for c, bw, bh in r.boxes:
            w[c].append(bw)
            ar[c].append(bw / max(bh, 1e-9))
    return {NAMES[c]: dict(n=len(w[c]), med_w=float(np.median(w[c])),
                           p10_w=float(np.percentile(w[c], 10)),
                           p90_w=float(np.percentile(w[c], 90)),
                           med_ar=float(np.median(ar[c]))) for c in sorted(w)}


def dup_stats(df):
    vg = df[df.vg >= 0]
    if vg.empty:
        return dict(groups=0, pairs=0, imgs_in_pairs=0, cross_split=0)
    g = vg.groupby("vg")["split"].agg(["nunique", "count"])
    return dict(groups=int(len(g)),
                pairs=int((g["count"] > 1).sum()),
                imgs_in_pairs=int(g[g["count"] > 1]["count"].sum()),
                cross_split=int((g["nunique"] > 1).sum()))


def block(name, df, dom, root):
    b = box_stats(df)
    d = dup_stats(df)
    L = [f"### {name}", "",
         f"- images: **{len(df)}**  |  boxes: **{int(df.n_boxes.sum())}**",
         "- split: " + ", ".join(f"{s}={int((df.split == s).sum())}" for s in SPLITS),
         f"- boxes per class: closed_eye={int(df.n_closed.sum())}, "
         f"open_eye={int(df.n_open.sum())}, yawning={int(df.n_yawn.sum())}",
         "- images containing class: " + ", ".join(
             f"{NAMES[i]}={int((df[c] > 0).sum())}"
             for i, c in enumerate(['n_closed', 'n_open', 'n_yawn'])),
         "", "| class | boxes | med width | p10 w | p90 w | med aspect |",
         "|---|---|---|---|---|---|"]
    for c, v in b.items():
        L.append(f"| {c} | {v['n']} | {v['med_w']:.3f} | {v['p10_w']:.3f} "
                 f"| {v['p90_w']:.3f} | {v['med_ar']:.2f} |")
    L += ["", "- source distribution: " + ", ".join(
        f"{k}={v}" for k, v in df.source.value_counts().items()), ""]
    L += [f"- near-duplicate groups: {d['groups']}, intra-split pairs: {d['pairs']} "
          f"({d['imgs_in_pairs']} images), **cross-split groups: {d['cross_split']}**", ""]
    if dom:
        L += ["| split | n | grayscale/IR % | dark(<60) % | brightness mean | p5 | p95 | blur median |",
              "|---|---|---|---|---|---|---|---|"]
        for s, v in dom.items():
            L.append(f"| {s} | {v['n']} | {v['gray_pct']:.1f} | {v['dark_pct']:.1f} "
                     f"| {v['bright_mean']:.0f} | {v['bright_p5']:.0f} | {v['bright_p95']:.0f} "
                     f"| {v['blur_median']:.0f} |")
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.40)
    args = ap.parse_args()

    src_root = ROOT / "dataset"
    clean_root = ROOT / "dataset_clean"
    if (src_root / "metadata" / "build_info.json").exists():
        raise SystemExit(
            f"refusing to run: {src_root} is the consolidated, validated dataset, not "
            "the raw pre-consolidation source. Comparing it against itself is "
            "meaningless. See the HISTORICAL note at the top of this file; the original "
            "output is at dataset/metadata/analysis/dataset_comparison.md.")
    src_map, vg_map = manifest_maps(src_root)

    orig = scan(src_root, src_map, vg_map)
    cand = orig[orig.max_eye_w <= args.tau].copy()
    clean = scan(clean_root, src_map, vg_map) if clean_root.exists() else pd.DataFrame()

    L = ["# Dataset comparison: `dataset/` vs `dataset_clean/` vs `candidate_final`", "",
         f"Candidate rule: drop image if any eye-class box width > **tau = {args.tau}** "
         "(chosen in `dataset/metadata/analysis/threshold_study/`).", "",
         "Domain figures are sampled (see `n` column), not exhaustive.", ""]
    L += block("dataset/ (Experiment 1)", orig, domain_probe(src_root, orig), src_root)
    if not clean.empty:
        L += block("dataset_clean/ (Experiment 2)", clean, domain_probe(clean_root, clean), clean_root)
    L += block("candidate_final (proposed)", cand, domain_probe(src_root, cand), src_root)

    # equivalence check against dataset_clean
    L += ["## Equivalence check: candidate_final vs dataset_clean", ""]
    if not clean.empty:
        a, b = set(cand.stem), set(clean.stem)
        same = a == b
        L += [f"- identical image selection: **{same}**",
              f"- in candidate only: {len(a - b)}  |  in dataset_clean only: {len(b - a)}", ""]
        if same:
            L += ["> The image selection is equivalent to dataset_clean; the improvement is the",
                  "> reproducible curation, validation, metadata, and documented rationale.", ""]
    else:
        L += ["- dataset_clean/ not present", ""]

    # decision table
    L += ["## Decision table", "",
          "| category | KEEP / REMOVE | reason | evidence |",
          "|---|---|---|---|",
          f"| Eye box width <= {args.tau} on a face/scene | KEEP | deployment case: dashcam sees "
          "faces at scene scale | band grid 0.30-0.40 is mostly correct eye-level boxes |",
          f"| Eye box width > {args.tau}, eye close-up crop | REMOVE | not deployment-relevant; a "
          "dashcam never frames a single eye | band grid 0.40-0.50 is ~15/16 eye crops |",
          f"| Eye box width > {args.tau}, whole-face box | REMOVE | genuine mislabel (incl. faces "
          "in sunglasses tagged open_eye) | visual audit of 30 flagged images |",
          "| yawning boxes (any scale) | KEEP | face-scale is the correct convention for this "
          "class | 0 yawning boxes removed at every tau |",
          "| Intra-split near-duplicate pairs | KEEP | redundancy, not leakage; preserve "
          "information first | 0 cross-split groups, verified twice |",
          "| Existing split assignment | KEEP | already leakage-free; reshuffling would destroy "
          "a verified property | visual_group_id spans 0 splits |",
          "| Grayscale/IR images | KEEP | real low-light/IR diversity, scarce and valuable | "
          "train 11.2% vs test 5.2% (documented mismatch) |", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_comparison.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
