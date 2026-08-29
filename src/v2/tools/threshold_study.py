"""Threshold (tau) study for the eye-box scale rule used to build dataset_final_v1.

The rule removes an IMAGE when any eye-class box (closed_eye/open_eye) exceeds a
normalized-width threshold tau. Two distinct populations sit above that line and
neither is deployment-relevant, so both are removed -- but for different reasons:

  * eye close-up crops   : the image IS an eye, so a frame-filling box is CORRECT,
                           but a dashcam never sees such a frame.
  * face-scale mislabels : an eye class drawn over a whole face; genuinely wrong.

They cannot be separated automatically with the tools available offline (see
DATASET_REPORT.md). This study therefore picks tau from statistics + visual
evidence rather than from width alone.

HISTORICAL -- this chose tau=0.40 for the dataset that is now `dataset/`. Re-running
it against the consolidated dataset measures the *result* of the filter, not the
population it was chosen from, so the histogram will show no tail above tau by
construction. Output preserved at `dataset/metadata/analysis/threshold_study/`.

    python -m src.v2.tools.threshold_study        # historical
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EYE = (0, 1)
NAMES = ["closed_eye", "open_eye", "yawning"]
TAUS = [0.25, 0.30, 0.35, 0.40, 0.45]
OUT = ROOT / "dataset" / "metadata" / "analysis" / "threshold_study"


def load_source_map(src_root):
    """stem -> source and stem -> visual_group_id, from the curation manifest."""
    # Post-consolidation the lineage CSVs moved into metadata/lineage/; check both.
    mpath = src_root / "curated_dataset_manifest.csv"
    if not mpath.exists():
        mpath = src_root / "metadata" / "lineage" / "curated_dataset_manifest.csv"
    if not mpath.exists():
        return {}, {}
    m = pd.read_csv(mpath)
    k = m[m.keep_or_remove == "KEEP"]
    stems = k.new_path.apply(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
    return dict(zip(stems, k.source)), dict(zip(stems, k.visual_group_id))


def collect(src_root):
    """One row per BOX, with the parent image's max eye width attached."""
    src_map, vg_map = load_source_map(src_root)
    rows = []
    for split in ("train", "val", "test"):
        for f in sorted((src_root / "labels" / split).glob("*.txt")):
            recs = [l.split() for l in f.read_text().splitlines() if l.strip()]
            if not recs:
                continue
            eyes = [r for r in recs if int(r[0]) in EYE]
            max_eye_w = max((float(r[3]) for r in eyes), default=0.0)
            for r in recs:
                c = int(r[0])
                x, y, w, h = (float(v) for v in r[1:5])
                rows.append(dict(
                    stem=f.stem, split=split, cls=c, cls_name=NAMES[c],
                    w=w, h=h, area=w * h, ar=w / max(h, 1e-9),
                    cx=x, cy=y, max_eye_w=max_eye_w,
                    source=src_map.get(f.stem, "unknown"),
                    vg=vg_map.get(f.stem, -1)))
    return pd.DataFrame(rows)


def per_image(df):
    return df.groupby(["stem", "split", "source"], as_index=False).agg(
        max_eye_w=("max_eye_w", "first"), n_boxes=("cls", "size"))


def hist_table(df):
    eye = df[df.cls.isin(EYE)]
    edges = [0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .60, .70, .80, .90, 1.01]
    out = ["## Eye-box width histogram (the valley test)", "",
           "| width bin | closed_eye | open_eye |", "|---|---|---|"]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        a = int(((eye.cls == 0) & (eye.w >= lo) & (eye.w < hi)).sum())
        b = int(((eye.cls == 1) & (eye.w >= lo) & (eye.w < hi)).sum())
        out.append("| {:.2f}-{:.2f} | {} | {} |".format(lo, hi, a, b))
    return "\n".join(out) + "\n"


def signal_table(df):
    """Multi-signal profile of eye boxes on each side of the 0.40 line."""
    eye = df[df.cls.isin(EYE)].copy()
    eye["band"] = np.where(eye.w > 0.40, "above 0.40", "below 0.40")
    g = eye.groupby(["cls_name", "band"]).agg(
        n=("w", "size"), med_w=("w", "median"), med_h=("h", "median"),
        med_area=("area", "median"), med_ar=("ar", "median"),
        p10_w=("w", lambda s: s.quantile(.10)), p90_w=("w", lambda s: s.quantile(.90)))
    out = ["## Multi-signal profile of eye boxes", "",
           "| class | band | n | med w | med h | med area | med aspect | p10 w | p90 w |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (c, b), r in g.iterrows():
        out.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.4f} | {:.2f} | {:.3f} | {:.3f} |".format(
            c, b, int(r.n), r.med_w, r.med_h, r.med_area, r.med_ar, r.p10_w, r.p90_w))
    return "\n".join(out) + "\n"


def tau_table(df, imgs):
    out = ["## Candidate thresholds", "",
           "| tau | imgs removed | imgs remaining | closed removed | open removed "
           "| closed left | open left | yawning left |",
           "|---|---|---|---|---|---|---|---|"]
    for t in TAUS:
        drop = set(imgs.loc[imgs.max_eye_w > t, "stem"])
        rem = df[~df.stem.isin(drop)]
        rmv = df[df.stem.isin(drop)]
        out.append("| {:.2f} | {} | {} | {} | {} | {} | {} | {} |".format(
            t, len(drop), len(imgs) - len(drop),
            int((rmv.cls == 0).sum()), int((rmv.cls == 1).sum()),
            int((rem.cls == 0).sum()), int((rem.cls == 1).sum()), int((rem.cls == 2).sum())))
    out += ["", "### Source distribution of REMAINING images, per tau", ""]
    srcs = sorted(imgs.source.unique())
    out.append("| tau | " + " | ".join(srcs) + " |")
    out.append("|" + "---|" * (len(srcs) + 1))
    for t in TAUS:
        vc = imgs[imgs.max_eye_w <= t].source.value_counts()
        out.append("| {:.2f} | ".format(t) + " | ".join(str(int(vc.get(s, 0))) for s in srcs) + " |")
    out += ["", "### Per-split effect, per tau", "",
            "| tau | train keep | val keep | test keep |", "|---|---|---|---|"]
    for t in TAUS:
        keep = imgs[imgs.max_eye_w <= t]
        vc = keep.split.value_counts() if "split" in keep else None
        out.append("| {:.2f} | {} | {} | {} |".format(
            t, int(vc.get("train", 0)), int(vc.get("val", 0)), int(vc.get("test", 0))))
    return "\n".join(out) + "\n"


def render_band(src_root, df, lo, hi, fname, n=16):
    """Sample images whose max eye width lands in [lo,hi) and draw their eye boxes."""
    imgs = per_image(df)
    sel = imgs[(imgs.max_eye_w >= lo) & (imgs.max_eye_w < hi)]
    if sel.empty:
        return 0
    sel = sel.sample(min(n, len(sel)), random_state=0)
    tiles = []
    for r in sel.itertuples():
        p = list((src_root / "images" / r.split).glob(r.stem + ".*"))
        if not p:
            continue
        im = cv2.imread(str(p[0]))
        if im is None:
            continue
        H, W = im.shape[:2]
        for b in df[(df.stem == r.stem) & (df.cls.isin(EYE))].itertuples():
            col = (0, 0, 255) if b.cls == 0 else (0, 255, 0)
            cv2.rectangle(im,
                          (int((b.cx - b.w / 2) * W), int((b.cy - b.h / 2) * H)),
                          (int((b.cx + b.w / 2) * W), int((b.cy + b.h / 2) * H)), col, 4)
        tiles.append(cv2.resize(im, (150, 150)))
    if not tiles:
        return 0
    while len(tiles) % 8:
        tiles.append(np.zeros((150, 150, 3), np.uint8))
    grid = cv2.vconcat([cv2.hconcat(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
    cv2.imwrite(str(OUT / fname), grid)
    return len(sel)


def main():
    src_root = ROOT / "dataset"
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect(src_root)
    imgs = per_image(df)
    # keep split on the per-image frame for the per-split table
    imgs = imgs.merge(df[["stem", "split"]].drop_duplicates("stem"), on="stem",
                      how="left", suffixes=("", "_y"))
    df.to_csv(OUT / "box_signals.csv", index=False)

    md = ["# Threshold (tau) study - eye-box scale rule", "",
          "Source: `dataset/` - {} images, {} boxes.".format(len(imgs), len(df)), "",
          "The rule drops an IMAGE when any eye box width > tau. Both populations above the",
          "line (legitimate eye close-up crops, and face-scale mislabels) are removed: neither",
          "is deployment-relevant for a dashcam, and they cannot be separated automatically.", "",
          hist_table(df), signal_table(df), tau_table(df, imgs)]
    (OUT / "threshold_study.md").write_text("\n".join(md), encoding="utf-8")

    n1 = render_band(src_root, df, 0.30, 0.40, "band_just_below_0.30-0.40.png")
    n2 = render_band(src_root, df, 0.40, 0.50, "band_just_above_0.40-0.50.png")
    n3 = render_band(src_root, df, 0.20, 0.30, "band_well_below_0.20-0.30.png")
    print((OUT / "threshold_study.md").read_text(encoding="utf-8"))
    print("rendered bands: below={} above={} wellbelow={} -> {}".format(n1, n2, n3, OUT))


if __name__ == "__main__":
    main()
