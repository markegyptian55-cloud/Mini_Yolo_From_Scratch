# DATASET_FINAL_V1 — Report

Built `2026-08-29` by `src/v2/tools/build_final_dataset.py` from `dataset/` with
τ = 0.40. Deterministic (seed 0), hardlinked, source never mutated.

Every figure below is measured from this dataset or its manifest. Where a fact could
not be established it says so; nothing is estimated and presented as measured.

---

## Executive Summary

`dataset_final_v1` is the locked baseline dataset for `BASELINE_V1`: **18,447 images,
28,864 boxes**, one annotation convention, zero cross-split near-duplicate leakage,
full provenance metadata, and a SHA-256-locked test set.

**Its image selection is very nearly identical to `dataset_clean/`.** It differs by
exactly **two images** — two zero-box background negatives (`yawn_new_1229_*`) that
`dataset_clean`'s builder silently skipped and this build deliberately retains.
Stated plainly, as required:

> The image selection is equivalent to dataset_clean; the improvement is the
> reproducible curation, validation, metadata, and documented rationale.

That is the honest claim. This dataset is not better because it contains better
images. It is better because the selection rule is now justified by evidence rather
than assumption, every image carries provenance, the leakage-free property is
verified rather than assumed, and the test set is cryptographically locked.

## Source Datasets

Provenance comes from `dataset/curated_dataset_manifest.csv`, produced by an earlier
curation pass. That pass reduced `Dataset-Main` (50,654 images) to `Dataset-Curated`
(28,170) by collapsing near-duplicate clusters of size ≥3 (22,099 removed) and
resolving cross-split near-duplicate leakage (385 removed). **`Dataset-Main` is not
present on this machine** — the path in `dataset/data.yaml` does not exist — so its
contents could not be re-verified.

| source | images kept | closed_eye | open_eye | yawning |
|---|---|---|---|---|
| unmatched | 6,804 | 4,529 | 5,043 | 1,113 |
| bare_numeric | 5,976 | 5,857 | 119 | 116 |
| yawn_new | 3,531 | 0 | 0 | 7,201 |
| istockphoto | 862 | 4 | 1,168 | 6 |
| dd_v1 | 593 | 383 | 337 | 210 |
| img | 358 | 0 | 0 | 0 |
| Yimg | 246 | 0 | 0 | 0 |
| session | 77 | 47 | 40 | 44 |

(`img`/`Yimg` contribute images whose boxes are counted under their split totals; see
`metadata/source_manifest.csv` for the authoritative per-source figures.)

## Cleaning — what was removed and why

**9,723 images removed, all under reason code `C`.** Zero images were removed for
corruption or invalid labels — every one of the 28,170 source images decoded cleanly
and every label parsed.

The rule: drop an image when any eye-class box exceeds **τ = 0.40** normalized width.
Above that line sit two distinct populations:

1. **Legitimate eye close-up crops.** The image *is* an eye, so a frame-filling box is
   correct annotation — but a dashcam never frames a single eye at 640×640.
2. **Face-scale mislabels.** An `open_eye`/`closed_eye` box drawn over an entire face.
   Visual audit found genuine errors here, including two faces wearing **sunglasses**
   labelled `open_eye`.

Both are removed, for two different reasons: (1) is not deployment-relevant, (2) is
wrong. **They cannot be separated automatically** — Haar cascades detect faces in only
4% of these images (they fail on rotated, dark and occluded faces), a skin-tone
heuristic fails because the `session` source is grayscale/IR and scores 0.000
regardless of framing, and no pretrained face detector is available offline. Per the
governing rule for this build, ambiguous samples are removed rather than relabelled.
**No ground truth was invented.**

### Why τ = 0.40

Justified by statistics *and* visual inspection, not a convenient histogram cut.

- **Bimodality.** Eye-box widths are strongly bimodal. `open_eye` counts per bin fall
  to 70–118 across 0.20–0.40, then climb to 1,857 in 0.90–1.01.
- **Multi-signal separation.** Above 0.40, median aspect ratio is **exactly 1.00** and
  median area is 0.49–0.55 of the frame. Below 0.40, aspect ratio is 1.05–1.28 and
  median area is 0.008–0.025 — a **~20× area separation**, not a marginal cut.
- **Visual check at the boundary.** The 0.40–0.50 band is ~15 of 16 eye close-up
  crops. The 0.30–0.40 band is predominantly correct eye-level boxes on faces.
- **Alternatives tested.** τ ∈ {0.25, 0.30, 0.35, 0.40, 0.45} were all measured
  (`dataset/metadata/analysis/threshold_study/threshold_study.md`). Lower values discard valid
  eye-level annotations from the valley; 0.45 begins admitting the large-box mode.

`yawning` is untouched at every τ — **0 yawning boxes removed** — because face-scale
is the correct convention for that class.

## Annotation Standard

```
0 = closed_eye    eye-level bounding box
1 = open_eye      eye-level bounding box
2 = yawning       face-scale bounding box (established project convention)
```

Verified against `dataset/data.yaml` before any label was touched; the mapping was
not assumed. **No label file was modified by this build** — labels are hardlinked
byte-identical from the source. Consistency was achieved purely by image selection.

## Duplicate Handling

The authoritative identity is `visual_group_id` (perceptual near-duplicate clustering
from the original curation), **not** the filename. Filename stems are unreliable:
5,173 of 8,179 stems map to more than one visual group — e.g. four `000007_jpg.rf.*`
files are one man's face plus three close-ups of a different person's eye.

- Near-duplicate groups: **16,007**
- **Intra-split** near-duplicate pairs: 4,880 images flagged
  `intra_split_near_duplicate` in the manifest — train 3,800, val 540, test 540
- **These are retained, not deleted.** A near-duplicate inside one split is
  redundancy, not leakage. Information is preserved; removal would need demonstrated
  evidence that evaluation is degraded, which has not been shown.

## Leakage Prevention

**Zero cross-split near-duplicate leakage**, verified twice independently — once on
the source manifest, once on the built dataset by the validator.

The existing split assignment was **inherited unchanged**, not recomputed. It was
already leakage-free, and reshuffling would have destroyed a verified property for a
cosmetically rounder percentage. The invariant re-asserted is: *no `visual_group_id`
may appear in more than one split.*

An earlier analysis in this project claimed 71% test-set leakage based on filename
stems. **That claim was wrong and is retracted** — it treated a shared numbering
scheme as an image identity.

## Dataset Statistics

| split | images | boxes | closed_eye | open_eye | yawning |
|---|---|---|---|---|---|
| train | 14,442 | 22,632 | 10,435 | 5,418 | 6,779 |
| val | 2,101 | 3,223 | 1,551 | 663 | 1,009 |
| test | 1,904 | 3,009 | 1,391 | 716 | 902 |
| **total** | **18,447** | **28,864** | **13,377** | **6,797** | **8,690** |

Split ratio 78.3 / 11.4 / 10.3 — close to the 80/10/10 target, and inherited rather
than re-cut so that leakage-freedom is preserved.

Box geometry after cleaning (train):

| class | median width | p10 | p90 | median aspect |
|---|---|---|---|---|
| closed_eye | 0.184 | 0.050 | 0.283 | 1.28 |
| open_eye | 0.097 | 0.044 | 0.169 | 1.05 |
| yawning | 0.729 | 0.159 | 1.000 | 1.13 |

**Class balance is reported, not forced.** closed_eye outnumbers open_eye ~2:1. No
images were deleted to equalise counts — that would destroy realistic event
frequencies. If this proves harmful, address it at training time with class weighting
or sampling, not by mutating the dataset.

## Domain Coverage

### REAL DATA COVERAGE (measured on every image, exhaustive — not sampled)

| lighting | train | val | test | total |
|---|---|---|---|---|
| normal_daylight | 8,839 | 1,219 | 1,254 | 11,312 |
| low_light | 3,961 | 652 | 440 | 5,053 |
| bright_or_backlit | 974 | 124 | 125 | 1,223 |
| night_or_very_low_light | 668 | 106 | 85 | 859 |

| attribute | train | val | test |
|---|---|---|---|
| grayscale / IR | 868 (6.0%) | 103 (4.9%) | 109 (5.7%) |
| RGB | 13,574 | 1,998 | 1,795 |
| blurry (bottom decile of Laplacian variance) | 1,433 | 214 | 198 |
| sharp | 13,009 | 1,887 | 1,706 |

Class × lighting (images containing the class):

| | normal_daylight | low_light | bright/backlit | night |
|---|---|---|---|---|
| closed_eye | 4,936 | 1,500 | 603 | 296 |
| open_eye | 2,171 | 1,286 | 31 | 137 |
| yawning | 4,985 | 2,476 | 747 | 456 |

Lighting is derived from mean image luminance, a proxy — not a human-verified label.
An earlier sampled probe in this project suggested a train/test grayscale mismatch of
11.2% vs 5.2%; the exhaustive count above (6.0% vs 5.7%) shows **that was sampling
noise and the splits are in fact well matched**.

### SYNTHETIC / AUGMENTED COVERAGE

**None of the coverage above is synthetic.** All figures are real captured images.

The source images carry Roboflow export augmentation baked in (visible as the
`.rf.<hash>` filename suffix); the degree is not recoverable from the data. Training
augmentation (mosaic, HSV, affine, blur, erasing) is applied at train time by
`src/v2/data/augment.py` and is **not** counted anywhere in this section.
Augmentation is not a substitute for real diversity and is not claimed as such.

## Quality Audit

| check | scope | result |
|---|---|---|
| image decode (PIL/OpenCV) | all 28,170 source images | 0 corrupt, 0 zero-byte |
| label syntax, class range, coordinate range | all 28,170 label files | 0 invalid |
| eye-box scale rule | all 28,170 | 9,723 removed (code `C`) |
| visual audit of flagged images | 30 inspected at 3 width bands | ~half eye crops, ~half face-scale mislabels |
| threshold sensitivity | 5 candidate τ values | recorded in `dataset/metadata/analysis/threshold_study/` |
| cross-split leakage | 16,007 visual groups | 0 spanning splits |

## Final Split

train 14,442 · val 2,101 · test 1,904.

**The test set is locked**: SHA-256 of all 3,808 test files (1,904 images + 1,904
labels) in `metadata/test_set_lock.sha256`. Re-verify with
`validate_dataset.py --full-hash`. The test set must not be consulted for any further
dataset decision.

## Known Limitations

Stated as limitations, not passes.

1. **Not subject-disjoint, and cannot be made so.** `session_id` is empty for all
   50,654 rows of the source manifest. The only source encoding subject identity in
   filenames (`session`, MRL-style `s0001_…`) contributes just 77 images after
   cleaning. Splits are near-duplicate-disjoint, **not** subject-disjoint. The same
   person may appear in train and test. Any claim of subject-independent evaluation
   would be false.
2. **Two removal reasons are conflated.** Eye close-up crops and face-scale mislabels
   are both removed under code `C` and cannot be told apart with available tooling.
   The count of genuine annotation errors is therefore unknown; the ~50/50 estimate
   comes from a 30-image visual sample, not a full audit.
3. **Ambiguous samples were discarded, not corrected.** Some legitimate imagery was
   lost alongside genuine errors. A face detector would allow recovering the crops.
4. **Lighting labels are a luminance proxy**, not human-verified. Backlighting,
   shadow, and head pose are not separately measured.
5. **Glasses / sunglasses / head-pose / driver identity are `unknown`** for every
   image. No source provides them; none were invented.
6. **Intra-split redundancy remains** — 4,880 images (26.5%) are near-duplicates of
   another image in the same split, including 540 in test. Evaluation on this test set
   is therefore slightly less independent than the raw image count suggests.
7. **`Dataset-Main` is unavailable**, so the upstream curation could not be re-verified
   from primary data; its manifest was trusted for provenance.
8. **`open_eye` is under-represented in bright/backlit conditions** (31 images) — the
   thinnest cell in the coverage table.

## Dataset Decision

1. **Why better than `dataset/`?** One consistent eye-box convention instead of two
   contradictory ones; 9,723 ambiguous or mislabelled images removed; full provenance
   and quality metadata; verified leakage-freedom; locked test set.
2. **Why better or more defensible than `dataset_clean/`?** The image selection is
   equivalent to dataset_clean (± 2 background images); the improvement is the
   reproducible curation, validation, metadata, and documented rationale. In
   particular, `dataset_clean` was justified by a claim about "annotation
   contamination" that turned out to be only half correct — the stronger and correct
   justification is deployment relevance, established here by evidence.
3. **What was removed?** 9,723 images whose eye boxes exceed τ = 0.40.
4. **What was retained?** 18,447 images including all 8,690 yawning boxes, all
   grayscale/IR and night imagery, all intra-split near-duplicates, and 2 background
   negatives.
5. **What remains ambiguous?** The crop-vs-mislabel split within the removed set;
   subject identity; glasses and head pose.
6. **What limitations remain?** See the eight items above — chiefly the absence of
   subject-disjoint splitting.
7. **Ready for `BASELINE_V1`?** See below.

## Baseline Readiness

```
READY FOR BASELINE TRAINING
```

Subject to Limitation 1: this dataset supports **near-duplicate-disjoint** evaluation,
not subject-independent evaluation. Baseline metrics should be reported with that
qualification, and the domain breakdown above should be used for per-condition
evaluation rather than relying on a single aggregate mAP.

Train `BASELINE_V1` **from scratch** on this dataset — not fine-tuned from any earlier
checkpoint — so its result is attributable to the data.
