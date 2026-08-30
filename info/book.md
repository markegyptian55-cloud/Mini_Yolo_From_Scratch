# MiniYOLO Engineering Book

Record of the datasets, the model, and every training experiment run on this project.

**Rules this document follows.** Every number here is either (a) **[measured]** — produced by
a command run on this machine, with the source file named, or (b) **[from record]** — read out
of a file this session did not create. Where two surviving records disagree, the disagreement is
written down as a disagreement and **not** resolved by guessing. Where something cannot be
recovered, it says so instead of estimating. Nothing in this document is inferred and presented
as fact.

Last updated **2026-08-30**. On that date the record was restructured: the dataset history and
all three Foundation Experiments (exp 1-3, including exp 3's completed run) were consolidated
into one comprehensive **Chapter 1**, so the entire empirical record survives the deletion of
the underlying experiment directories intact and in one place. Process/tooling material that is
not itself a model result (temporal monitoring, the checkpoints layout, bugs fixed) moved to
**Appendix C**. Chapter 2 is reserved for the next phase of the project, which begins with a
new dataset.

**Chapter 0 corrects four statements originally made in the pre-restructuring text.** Where a
later fact overturns an earlier one, the earlier text is left standing with a pointer, not
quietly edited -- a record that rewrites itself is not a record. This applies across the
2026-08-30 restructuring too: nothing was deleted, only reorganised and consolidated.

---

## Table of Contents

- **Chapter 0** -- Corrections: statements this document got wrong
- **Chapter 1** -- The Foundation Experiments (Exp 1 - Exp 3): dataset, all three experiments,
  comparative tables, direct answers, and what remains unresolved
- **Appendix A** -- The v1 CPU model: an unresolved conflict in the record
- **Appendix B** -- Original v1 engineering notes, preserved verbatim
- **Appendix C** -- Infrastructure built alongside the Foundation Experiments (temporal
  monitoring, the checkpoints layout, bugs fixed, root cleanup)

---

## Chapter 0: Corrections — four things this document got wrong

Added 2026-08-29. Each item names where the wrong claim sits, what is actually true, and how
it was checked. None of these were caught by the user; all four surfaced during the dataset
audit (now Chapter 1, section 1.1), three of them while trying to verify my own earlier work.

**C-1. A claimed 71% train/test leakage does not exist. Retracted in full.**
During the audit I found that 1,787 filename stems appear in more than one split and reported
this as catastrophic leakage. That was wrong. The stem before Roboflow's rf-hash suffix is a
shared numbering scheme, not an image identity: [measured] 5,173 of 8,179 stems map to more
than one visual_group_id. Concretely, the four 000007_jpg files at that stem are one man's face
plus three close-ups of a *different* person's eye — four distinct visual groups.
The authoritative identity is visual_group_id (perceptual near-duplicate clustering, in the
lineage manifest under dataset/metadata/lineage/). [measured] **Zero visual groups span
two splits**, checked twice and re-checked by the dataset validator. True redundancy is 1.15x,
not 3.44x. The original curation was sound. This claim never reached a report or a metric.

**C-2. The original rationale for experiment 2 (now section 1.3.1) was half wrong.**
That section and 1.1.2 originally presented the filtered images as "annotation contamination" —
mislabels. A visual audit of 30 flagged images shows the flagged set is roughly **half genuine
mislabels** (an open_eye box drawn over a whole face, including two faces wearing sunglasses)
and roughly **half legitimate eye close-up crops**, where a frame-filling box is *correct*
because the image genuinely is an eye. The removal still stands, but the honest justification is
**deployment relevance**, not error: a dashcam never sees a 640x640 single-eye crop, so neither
population belongs in this dataset. Two different reasons, one filter. Section 1.1.2's width
statistics are unaffected and remain correct as measured.

**C-3. The grayscale/IR train-vs-test mismatch was sampling noise.**
An early 250-image probe reported 11.2% grayscale/IR in train against 5.2% in test and I wrote
that up as a domain mismatch. [measured] The exhaustive count over all 18,447 images gives
**6.0% train vs 5.7% test**. The splits are well matched. There is no mismatch to fix.

**C-4. The original "3.8% dark" figure (now section 1.7, item 3) is superseded, not wrong.**
It was measured on the pre-consolidation dataset. [measured] On today's dataset: 859 of 18,447
images are night_or_very_low_light (4.7%) and 1,080 are grayscale/IR (5.9%). The conclusion is
unchanged and if anything better supported: low-light driving remains untested (section 1.7,
item 3).

One earlier error inside this document was already corrected in place on 2026-08-27: the
epoch-52 metrics (now section 1.2.3) had been read off a shifted CSV column. The corrected
values are P 0.514 / R 0.927 / mAP50 0.849 / mAP50-95 0.479 / fitness 0.516.

---

## Chapter 1: The Foundation Experiments (Exp 1 – Exp 3)

This chapter is the complete empirical record of the project's first phase: three
training runs on the driver-fatigue detection problem, the datasets they used, and what
each one proved or refuted. It is written to survive the deletion of the experiment
directories themselves — every number a future reader could need is transcribed here.

**Phase verdict, stated up front.** Three experiments established a working
architecture and a validated dataset, and closed off two hypotheses about the
localisation ceiling. **Neither hypothesis held.** The project ends Chapter 1 with
mAP50 ≈ 0.90, mAP50-95 ≈ 0.48 (ratio 0.54, where healthy is above 0.70), and the cause
of that ceiling still unidentified. Experiment 2's best checkpoint remains the best
model produced.

---

### 1.1 Dataset foundations

Three classes throughout, in this index order: `closed_eye` (0), `open_eye` (1),
`yawning` (2).

#### 1.1.1 Lineage

| name | images | role | status |
|---|---:|---|---|
| **Dataset-Main** | ~50,654 manifest rows | original aggregate from multiple sources | **gone** from this machine; never on it during this phase |
| **Dataset-Curated** | 28,170 | near-duplicate collapse + cross-split leakage resolution of Dataset-Main | **deleted** 2026-08-29 after consolidation |
| **`dataset/`** (validated) | **18,447** | the single canonical dataset from 2026-08-29 onward | **current** |

Dataset-Main is not recoverable here. The only surviving record of it is
`dataset/metadata/lineage/curated_dataset_manifest.csv` (50,654 rows), which is why that
file was preserved rather than deleted during the consolidation.

#### 1.1.2 The annotation-convention defect (the 3–8× scale contradiction)

[measured] Box-width distributions in Dataset-Curated are **bimodal**: the same class
name was used for boxes at two scales differing by 3–8×.

| class | total boxes | width > 0.4 (face-scale) | width ≤ 0.4 (object-scale) |
|---|---:|---:|---:|
| closed_eye | 17,556 | **23.7%** | 76.3% |
| open_eye | 12,365 | **45.0%** | 55.0% |
| yawning | 8,690 | 69.5% | 30.5% |

[measured] Median box width, within one class, comparing images carrying one eye box
against images carrying two:

| class | 1-box images | 2-box images | ratio |
|---|---:|---:|---:|
| closed_eye | 0.620 | 0.188 | **3.3×** |
| open_eye | 0.778 | 0.098 | **8.0×** |

Two conventions were mixed under identical labels:

- **object-level** — one small box per eye (sources matching `closed_eye_#-jpg_face_#`,
  `dd_v#_closed-*`, `istockphoto-*`)
- **face-level** — one large box over the whole face, labelled with the eye state it
  depicts (sources matching `s#_#_#_#_#_#_#_#_png.rf.*`)

For `closed_eye` / `open_eye` this is a genuine contradiction — the regressor was asked
to cover widths from 0.05 to 1.0 under one class name, so it could never be tight at high
IoU. For `yawning` it is **not** a contradiction: a yawn is inherently a face-scale event
and the large-box mode is correct for that class.

[measured] All images in both groups are 640×640 — Roboflow resized everything — so the
two conventions cannot be separated by image dimensions.

> **See Chapter 0, C-2.** The framing of the removed images as *mislabels* is only about
> half right. Roughly half are legitimate eye close-up crops, correctly annotated but
> irrelevant to a dashcam. The filter is justified by **deployment relevance**, not error.

#### 1.1.3 The validated dataset

Built by `src/v2/tools/build_final_dataset.py`, deterministic (seed 0), hardlinked, never
mutating its source. [measured] From `dataset/metadata/build_info.json`:

```
images_inspected  = 28,170        tau  = 0.40        seed = 0
images_kept       = 18,447
removed_by_reason = {'C': 9,723}  -- 0 removed for corruption or invalid labels
test_files_locked = 3,808
```

| split | images | boxes | closed_eye | open_eye | yawning |
|---|---:|---:|---:|---:|---:|
| train | 14,442 | 22,632 | 10,435 | 5,418 | 6,779 |
| val | 2,101 | 3,223 | 1,551 | 663 | 1,009 |
| test | 1,904 | 3,009 | 1,391 | 716 | 902 |
| **total** | **18,447** | **28,864** | **13,377** | **6,797** | **8,690** |

**The τ = 0.40 threshold, justified three ways.** [measured] The two populations separate
on more than width — median aspect ratio above the line is *exactly* 1.00 (a square box,
i.e. a frame-filling annotation), and box **area** separates them by roughly **20×**:

| class | band | n | med width | med area | med aspect |
|---|---|---:|---:|---:|---:|
| closed_eye | above 0.40 | 4,155 | 0.703 | 0.4907 | **1.00** |
| closed_eye | below 0.40 | 13,401 | 0.184 | 0.0250 | 1.28 |
| open_eye | above 0.40 | 5,568 | 0.808 | 0.5452 | **1.00** |
| open_eye | below 0.40 | 6,797 | 0.097 | 0.0077 | 1.05 |

[measured] Sensitivity across all five candidate thresholds:

| τ | images removed | images remaining | closed removed | open removed | yawning left |
|---|---:|---:|---:|---:|---:|
| 0.25 | 11,925 | 16,243 | 7,601 | 5,855 | 8,690 |
| 0.30 | 10,705 | 17,463 | 5,436 | 5,776 | 8,690 |
| 0.35 | 10,183 | 17,985 | 4,650 | 5,688 | 8,690 |
| **0.40** | **9,723** | **18,445** | **4,179** | **5,568** | **8,690** |
| 0.45 | 9,221 | 18,947 | 3,823 | 5,399 | 8,690 |

**`yawning` loses exactly zero boxes at every τ** — the control this rule needed. Visual
inspection: the 0.40–0.50 band is ~15 of 16 eye close-up crops; the 0.30–0.40 band is
mostly correct eye-level boxes on visible faces.

**What could not be automated, and was therefore not faked.** The two populations above τ —
legitimate eye crops and genuine face-scale mislabels — could not be told apart automatically
on this machine. Three attempts failed: **Haar cascades** (4% detection rate; misses rotated,
dark and occluded faces), **skin-tone fraction** (fails outright — the `session` source scores
0.000 because it is grayscale/IR, not because it is a crop), and **mediapipe** (segfaults on a
protobuf mismatch, `'MessageFactory' object has no attribute 'GetPrototype'`; no pretrained
face detector was cached and the network was restricted). Both populations were therefore
removed together, for two stated reasons, and **no image was given an invented label**.

**Before the consolidation, three overlapping directories existed:**

| directory | images | role |
|---|---:|---|
| `dataset/` (old) | 28,170 | Dataset-Curated — experiment 1's data, mixed conventions |
| `dataset_clean/` | 18,445 | the §1.1.2 filtered subset — experiment 2's data |
| `dataset_final_v1/` | 18,447 | built during this audit |

#### 1.1.4 The 14 validation gates

`src/v2/tools/validate_dataset.py` deliberately does **not** import the builder — a
validator sharing the builder's logic inherits the builder's bugs. Everything is
re-derived from files on disk. [measured] **Status: PASS. Baseline readiness: READY.**

| # | gate | result |
|---|---|---|
| 1 | YAML consistency | PASS — nc=3, names correct, no absolute `path:` key |
| 2 | class mapping | PASS — 0=closed_eye, 1=open_eye, 2=yawning |
| 3 | image integrity | PASS — 18,447 decoded, **0 corrupt** |
| 4 | label integrity | PASS — 28,864 boxes, **0 invalid** |
| 5 | split integrity | PASS — 0 orphan labels, 0 orphan images |
| 6 | duplicate integrity | PASS — 0 stems in more than one split |
| 7 | cross-split near-duplicate leakage | PASS — **0 visual groups span a split** |
| 8 | near-duplicate accounting | PASS — 4,880 flagged `intra_split_near_duplicate`, retained by design |
| 9 | manifest consistency | PASS — counts match disk |
| 10 | annotation consistency | PASS — 0 kept images exceed τ |
| 11 | **test set lock** | PASS — **3,808 SHA-256 hashes, 0 mismatched** |
| 12 | reproducibility | PASS — build_info.json + 5 metadata CSVs present |
| 13 | deployment-domain audit | PASS — lighting and grayscale/IR distribution recorded |
| 14 | final dataset report | PASS — `DATASET_REPORT.md` present |

Two items are marked `[DOCUMENTED LIMITATION]` and deliberately **not** PASS:

1. **Subject/video-disjoint split cannot be established.** `session_id` is empty for all
   50,654 lineage rows; `subject_id` is unknown for 18,447 of 18,447 images. The same
   person may appear in train and test. The splits are near-duplicate-disjoint, which is
   a strictly weaker guarantee. **Every metric in this chapter carries that caveat.**
2. **Glasses and head-pose coverage is unknown.** No source carries per-image metadata
   for either; reported as unknown rather than estimated.

**Test-set lock.** The 3,808 hashes (1,904 images + 1,904 labels) in
`dataset/metadata/test_set_lock.sha256` were re-verified after the project moved drives
*and* after the `dataset_final_v1` → `dataset` rename: **0 mismatches both times**. Any
future claim that a model was scored on "the same test set" is checkable against that
file, not taken on trust.

[measured] Deployment-domain distribution: `normal_daylight` 11,312 · `low_light` 5,053 ·
`bright_or_backlit` 1,223 · `night_or_very_low_light` 859 (4.7%) · grayscale/IR 1,080
(5.9%, split as 6.0% train vs 5.7% test — well matched, see Chapter 0 C-3).

#### 1.1.5 The consolidation — three directories to one, 2026-08-29

Performed only after validation passed, on explicit confirmation, never before. `dataset/VIDEO
FOR TEST/` was moved into the surviving dataset first so `report.py` and `hud.py` kept
working; lineage was preserved into `dataset/metadata/lineage/` (`curated_dataset_manifest.csv`
50,654 rows, `EDA/`, `duplicate_report.csv`, `quality_report.csv`, `class_statistics.csv`,
`source_statistics.csv`, `dataset_summary.txt`, and the old `data.yaml` renamed
`source_data.yaml`) — without these the build would no longer be reproducible or auditable.
Old `dataset/` and `dataset_clean/` were then deleted and `dataset_final_v1/` renamed to
`dataset/`.

[measured] The three directories were hardlinked to one another, so deleting two only dropped
link counts — a probe file went from `nlink=3` to `nlink=1` with content intact, confirmed by
re-running the full 3,808-hash test-set lock afterward with **0 mismatches**. The validator was
re-run against the renamed directory: still PASS, still READY.

**Consequence for the numbers in this chapter:** today's `dataset/` test split **is**
experiment 2's test split (differing only by two *train* images), so exp 2's and exp 3's
numbers are directly comparable to anything scored on `dataset/` now. Experiment 1's are not
— it was scored on the larger, now-deleted, contaminated split — which is why §1.3.5 exists.

---

### 1.2 Experiment 1 — Baseline

**Folder:** `checkpoints/Expi-1-imagez-384/` · **Dataset:** Dataset-Curated (28,170 imgs)

#### 1.2.1 Architecture

MiniYOLO-v2, built from scratch (not Ultralytics). Scale `n`.

| component | detail |
|---|---|
| Backbone | `MiniDarknetV2` — stem Conv(3→16, s2), 4 stages of Conv(s2)+C2f, SPPF. Widths (16,32,64,128,256), depths (1,2,2,1). Outputs P3/8, P4/16, P5/32 |
| Neck | `MiniPANv2` — FPN top-down (upsample+concat), then PAN bottom-up (stride-2 conv+concat) |
| Head | `DualDetect` — decoupled, anchor-free, **DFL-free (`reg_max = 1`)**: 4 raw ltrb scalars per location. **No objectness branch** (class score doubles as confidence). Class branch uses depthwise-separable convs |
| Dual branch | one2many (training richness) + one2one (**NMS-free** inference). Only one2one survives export, so inference cost equals a single head |
| Loss | CIoU + L1 (box) · BCE against task-aligned soft targets (cls) |
| Assignment | Task-aligned assigner + **STAL** (small-target-aware, `stal_min_size=8.0`) |
| Schedule | **ProgLoss** — one2many weight α annealed 0.8 → 0.1 across the run |
| Optimizer | **MuSGD** (`muon_ratio = 0.5`) |
| Size | 2,501,882 params train / 2,375,157 exported · 2.00 GFLOPs @384 · 9.50 MB fp32 |

This is the YOLO26 recipe (DFL-free + MuSGD + STAL + ProgLoss) implemented in full.

#### 1.2.2 Configuration

```
scale n | imgsz 384 (multi-scale 320-512) | batch 64 | epochs 300
optimizer musgd | seed 0 | patience 60 | amp true | workers 6
val_conf 0.001 | val_iou 0.7 | max_det 300 | e2e (NMS-free) validation
device: NVIDIA RTX 2000 Ada Generation, 16 GB
```

**Hyperparameters** (`hyp.yaml`) — optimisation and loss:

| key | value | | key | value |
|---|---|---|---|---|
| lr0 | 0.005 | | box gain | 7.5 |
| lrf | 0.05 | | cls gain | 0.7 |
| momentum | 0.937 | | l1 gain | 1.0 |
| weight_decay | 0.0005 | | tal_topk | 10 |
| warmup_epochs | 3.0 | | tal_alpha | 0.5 |
| warmup_momentum | 0.8 | | tal_beta | 6.0 |
| warmup_bias_lr | 0.1 | | stal | true |
| nominal_batch | 64 | | stal_min_size | 8.0 |
| muon_ratio | 0.5 | | stal_ref_size | 16.0 |
| ema_decay | 0.9999 | | prog_alpha_init | 0.8 |
| ema_tau | 2000 | | prog_alpha_final | 0.1 |

**The 16 augmentations:**

| augmentation | value | rationale |
|---|---|---|
| `mosaic` | 0.85 | 4-image mosaic; the main source of scale diversity |
| `close_mosaic` | 20 | mosaic/mixup/erasing disabled for the final 20 epochs |
| `mixup` | 0.10 | |
| `scale` | 0.5 | ±50% random resize |
| `degrees` | 10.0 | head tilt is real in a car cabin — higher than COCO's ~0 |
| `shear` | 2.0 | |
| `translate` | 0.1 | |
| `perspective` | 0.0005 | camera mounting angle varies |
| `fliplr` | 0.5 | |
| `flipud` | 0.0 | a driver is never upside-down |
| `hsv_h` | 0.015 | |
| `hsv_s` | 0.7 | |
| `hsv_v` | 0.5 | cabin lighting swings hard (tunnel, sun, night) |
| `gray` | 0.10 | IR / monochrome driver cameras |
| `blur` | 0.05 | motion blur + defocus on a live feed |
| `erasing` | 0.25 | occlusion: hand over mouth, sunglasses, wheel |
| *(plus)* `multi_scale_lo/hi` | 0.84 / 1.34 | 384 → 320–512 training window |

> **Note on reading `training_summary.txt`:** that file is written at the *end* of a run,
> so it reports `mixup 0.0` and `erasing 0.0` — the post-`close_mosaic` state. The
> authoritative starting values are in `hyp.yaml`, written before the first batch.

#### 1.2.3 The interruption

[from record] Training was interrupted at **epoch 52** by a **Windows Update forced
reboot** — not a crash. The Windows Event Log confirms planned restarts by
`MoUsoCoreWorker.exe` / `TrustedInstaller.exe`. The run was resumed from `last.pt` and
completed all 300 epochs.

[measured] Epoch 52's metrics, for the record: **P 0.514 · R 0.927 · mAP50 0.849 ·
mAP50-95 0.479 · fitness 0.516**. (These were mis-transcribed once from a shifted CSV
column and corrected — see Chapter 0.)

This interruption is why the checkpoint format keeps optimizer state, and why every epoch
writes `last.pt`. It is also why the timing telemetry added later reports two totals
(§1.5).

#### 1.2.4 Results — best epoch 294, `best_fitness` 0.5730903592115826

[measured] `checkpoints/Expi-1-imagez-384/REPORTS EXPI-1/evaluation.txt`.
**Scored on the Dataset-Curated (mixed-convention) splits**, which no longer exist.

**val split** — 3,082 images, 4,206 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 4,206 | 0.644 | 0.950 | 0.907 | 0.536 |
| closed_eye | 1,932 | 0.581 | 0.935 | 0.888 | 0.439 |
| open_eye | 1,265 | 0.537 | 0.942 | 0.865 | 0.531 |
| yawning | 1,009 | 0.815 | 0.974 | 0.966 | 0.638 |

speed: 1.11 ms/img (899 img/s)

**test split** — 2,873 images, 3,981 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 3,981 | **0.653** | **0.944** | **0.907** | **0.541** |
| closed_eye | 1,767 | 0.588 | 0.939 | 0.880 | 0.429 |
| open_eye | 1,312 | 0.579 | 0.925 | 0.881 | 0.543 |
| yawning | 902 | 0.793 | 0.968 | 0.961 | 0.650 |

speed: 1.05 ms/img (956 img/s)

**Wall clock: 12.12 h** [from record, console output] — spanning the reboot. No per-epoch
timing was recorded; that instrumentation did not exist yet.

#### 1.2.5 Video inference

[measured] `demo_video/analysis.txt` — `15-MaleGlasses.mp4`, 640×480, native 30 FPS,
2,639 frames (every frame), conf threshold 0.25:

- **Latency 12.23 ms/frame → 81.8 FPS** (~2.7× real-time)
- Total detections **6,165**: `open_eye` 5,350 (86.78%) · `closed_eye` 553 (8.97%) ·
  `yawning` 262 (4.25%)

⚠️ **This latency figure is not comparable to experiments 2 and 3.** Experiment 2
measured 5.81 ms/frame on the *identical* architecture and input size. Nothing about the
model got faster — exp 1's measurement was taken while other applications were loading
the machine. Treat the difference as measurement conditions, not a speedup, until both
checkpoints are re-benchmarked back to back on an idle machine. **That re-benchmark has
never been run.**

---

### 1.3 Experiment 2 — Clean-data ablation

**Folder:** `checkpoints/Expi-2-imagez-384/` · **Dataset:** `dataset_clean` (18,445 imgs)

#### 1.3.1 Hypothesis

> Removing the annotation-convention contradiction (§1.1.2) will lift accuracy, and
> **most of all `mAP50-95`**, because the box regressor will no longer be asked to cover
> a 3–8× scale range under one class name. The largest gain should appear in `open_eye`,
> the most contaminated class at 45.0%.

#### 1.3.2 The isolated variable

**Dataset only.** [measured] `hyp.yaml` was byte-identical to experiment 1 across all 42
keys; `args.yaml` differed only in the `--data` path. Scale, imgsz, batch, epochs,
optimizer, seed, patience, AMP, every loss gain and all 16 augmentations held constant.

```
dataset/ -> dataset_clean/       train 22,215 -> 14,440 | val 3,082 -> 2,101 | test 2,873 -> 1,904
```

Built by `make_clean_dataset.py`: drops every **image** containing a face-scale eye box
(w > 0.4) — the whole image, not just the box, because deleting the box alone would turn
a real eye into unlabelled background and actively train the model to miss it. Zero
`yawning` boxes removed. Face-scale eye boxes after filtering: **0.0%** (from 23.7% /
45.0%).

**Trained from scratch, not fine-tuned** from experiment 1. [measured] Epoch 1 mAP50 =
0.0003 proves it — a fine-tune would have started near exp 1's converged accuracy. The
reasoning is in §1.6.1.

#### 1.3.3 Run

**300 epochs, 6.09 h** [from record], no interruption. Best epoch **283**,
`best_fitness` 0.5241072334190018.

#### 1.3.4 Results

[measured] `checkpoints/Expi-2-imagez-384/REPORTS EXPI-2/evaluation.txt`.
Scored on the clean splits — which **are** today's `dataset/` splits (they differ by two
zero-box background negatives, both in *train*).

**val split** — 2,101 images, 3,223 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 3,223 | 0.657 | 0.942 | 0.891 | 0.483 |
| closed_eye | 1,551 | 0.652 | 0.927 | 0.892 | 0.417 |
| open_eye | 663 | 0.508 | 0.922 | 0.812 | 0.397 |
| yawning | 1,009 | 0.811 | 0.977 | 0.969 | 0.636 |

speed: 1.02 ms/img (983 img/s)

**test split** — 1,904 images, 3,009 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 3,009 | **0.664** | **0.940** | **0.900** | **0.483** |
| closed_eye | 1,391 | 0.662 | 0.927 | 0.900 | 0.415 |
| open_eye | 716 | 0.530 | 0.919 | 0.836 | 0.382 |
| yawning | 902 | 0.799 | 0.976 | 0.964 | 0.652 |

speed: 0.91 ms/img (1,098 img/s)

#### 1.3.5 The valid comparison, and the +0.073 `open_eye` gain

Experiments 1 and 2 were scored on **different** test splits, so their headline rows are
not directly comparable. Both checkpoints were therefore re-evaluated on the **same clean
test split** ([measured], `BASELINE_for_comparison.txt`):

| | exp 1 | exp 2 | Δ |
|---|---:|---:|---:|
| mAP50 | 0.870 | **0.900** | **+0.030** |
| mAP50-95 | 0.473 | **0.483** | **+0.010** |

Per class, mAP50:

| class | exp 1 | exp 2 | Δ | contamination removed |
|---|---:|---:|---:|---|
| `open_eye` | — | — | **+0.073** | 45.0% (the most contaminated) |
| `yawning` | — | — | **+0.002** | 0% — **the control** |

The gain is concentrated exactly where the contamination was, and the untouched class
moved by +0.002 — behaving correctly as a control. **The filter did what it was designed
to do.**

#### 1.3.6 What was refuted

**The headline prediction was wrong.** mAP50-95 moved **+0.010** (0.473 → 0.483), not
the substantial rise predicted. mAP50 gained three times as much. The ratio
mAP50-95/mAP50 went from 0.54 to 0.54 — **unchanged**.

> **Conclusion: the annotation-convention contradiction was not the cause of the
> localisation ceiling.** It was a real defect and fixing it was worthwhile — precision
> rose, `open_eye` gained 7.3 points of mAP50 — but the thing it was predicted to fix
> did not move. This refutation is what motivated experiment 3.

#### 1.3.7 Video inference

[measured] Same clip, 2,639 frames, conf 0.25:

- **Latency 5.81 ms/frame → 172.0 FPS** (~5.7× real-time)
- Total detections **5,637** (vs exp 1's 6,165 — **528 fewer**): `open_eye` 4,958
  (87.96%) · `closed_eye` 373 (6.62%, **−180**) · `yawning` 306 (5.43%)

The direction matches the measured precision gain (0.653 → 0.664): fewer spurious
detections. **Without frame-level ground truth for this clip it cannot be proven that the
removed boxes were all false positives** — only that the direction is consistent.

---

### 1.4 Experiment 3 — DFL head ablation

**Folder:** `checkpoints/Expi-3-imagez-384/` · **Dataset:** `dataset/` (18,447 imgs)

#### 1.4.1 Hypothesis

> Experiment 2 eliminated annotation convention as the cause of the localisation ceiling.
> The remaining suspect is the **box representation**: the head regresses four raw scalars
> per location (`reg_max = 1`) under CIoU + L1 — the weakest representation available.
> Replacing it with a **discrete distribution over 16 bins per ltrb side**, supervised by
> Distribution Focal Loss, should sharpen localisation and lift mAP50-95, which averages
> IoU thresholds up to 0.95 and is therefore exactly the metric a coarse box
> representation caps.

#### 1.4.2 Architecture changes

- **`src/v2/models/head.py`** — new `DFL` module. Each ltrb side is predicted as a
  softmax over `reg_max = 16` integer bins and integrated back to a scalar by a **frozen
  1×1 convolution whose weights are `[0, 1, …, 15]`** — literally the expectation of the
  distribution. Implemented as a Conv rather than a matmul because Conv is the
  better-supported op in every edge compiler. `cv2` now emits `4 × reg_max = 64` channels
  instead of 4.
- **`src/v2/losses/loss.py`** — new `L_dfl` term: the linear interpolation of the
  cross-entropy against the two integer bins straddling each continuous target. A target
  of 7.3 asks for 70% of the mass on bin 7 and 30% on bin 8. CIoU supervises the decoded
  box; DFL supervises the shape of the distribution behind it.
- **`src/v2/cfg/hyp.yaml`** — new gain `dfl: 1.5` (the YOLOv8/v11 value, not tuned here).
- Box trunk widened to `2 × reg_max` rather than Ultralytics' `4 × reg_max`, specifically
  to protect the export size budget; the wider trunk measured ~139k more exported params.

**Backward compatibility.** `reg_max = 1` still builds the original head — [measured]
parameter counts identical to before the change (2,501,882 train / 2,375,157 exported) —
and every loader defaults to `reg_max = 1` when a checkpoint lacks the key, because every
pre-experiment-3 checkpoint used the scalar head. Defaulting to 16 would have silently
built the wrong architecture with `strict=False` hiding the failure.

#### 1.4.3 The isolated variable

[measured] Across all 42 `hyp.yaml` keys, **exactly one differs** between experiments 2
and 3: `dfl: 1.5` (absent in exp 2). Every augmentation, every loss gain, every
optimisation setting identical. `args.yaml` differs in `reg_max` (1 → 16) and `workers`
(6 → 4, a memory accommodation with no accuracy effect).

**CIoU and L1 were both retained** alongside DFL — deliberately, to keep the experiment to
a single variable. Ultralytics drops L1 entirely when using DFL.

#### 1.4.4 The L1 vs DFL competition — a pre-registered concern

[measured] `l1_loss` at initialisation is **7.15** with the DFL head against **2.20** with
the scalar head — **3.3× higher**. The cause is structural: a uniform distribution over 16
bins has expectation 7.5 cells, whereas the old head's bias started boxes at 1 cell. With
`l1: 1.0` that term became the largest in the loss early in training and competed with DFL
for the same gradient.

[measured] Trajectory over the run:

| epoch | `l1_loss` | `dfl_loss` |
|---|---:|---:|
| 1 | 6.13 | 4.06 |
| 300 | 1.23 | 1.35 |

By the end the two terms are roughly balanced, so **the competition did not destabilise
training** — but it may have cost the early epochs something never recovered. This was
written down *before* results were known, and remains the leading candidate explanation
for the negative outcome.

#### 1.4.5 Run and cost

**300/300 epochs completed**, no interruption. Best `best_fitness` 0.5187640260940111 at
epoch 299 (checkpoint saved at epoch 299; the final epoch 300 val row is the best on
`fitness`, 0.518764).

[measured] `training_summary.txt` — the first run with per-epoch timing:

```
this process            : 6.565 h  (393.9 min)
sum of epoch timings    : 6.520 h
epochs completed        : 300 of 300
mean epoch              : 78.2 s
fastest / slowest       : 74.2 s / 142.9 s
first / last epoch      : 142.9 s / 77.4 s
```

The 142.9 s first epoch against a 74.2 s minimum is first-touch disk reads plus CUDA
warm-up — a 1.9× spread that makes any single-epoch timing estimate unreliable.

**Export size regression** [measured], FP16 ONNX at `--imgsz 384`, opset 13:

| head | exported params | FP16 ONNX |
|---|---:|---:|
| `reg_max = 1` | 2,375,157 | **4.85 MB** |
| `reg_max = 8` | 2,376,593 | **4.86 MB** |
| `reg_max = 16` | 2,466,649 | **5.04 MB** |

`reg_max = 16` costs **+0.19 MB** over the DFL-free head and lands **0.24 MB above the
4.8 MB edge budget**. `reg_max = 8` would have bought DFL for +0.01 MB.

#### 1.4.6 Results

[measured] `checkpoints/Expi-3-imagez-384/REPORTS EXPI-3/evaluation.txt`.

**val split** — 2,101 images, 3,223 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 3,223 | 0.650 | 0.935 | 0.884 | 0.478 |
| closed_eye | 1,551 | 0.632 | 0.910 | 0.870 | 0.406 |
| open_eye | 663 | 0.517 | 0.921 | 0.815 | 0.395 |
| yawning | 1,009 | 0.801 | 0.974 | 0.965 | 0.634 |

speed: 1.12 ms/img (891 img/s)

**test split** — 1,904 images, 3,009 boxes:

| Class | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **all** | 3,009 | **0.658** | **0.931** | **0.883** | **0.478** |
| closed_eye | 1,391 | 0.657 | 0.914 | 0.882 | 0.415 |
| open_eye | 716 | 0.531 | 0.912 | 0.811 | 0.372 |
| yawning | 902 | 0.786 | 0.967 | 0.956 | 0.648 |

speed: 0.98 ms/img (1,021 img/s)

#### 1.4.7 Formal refutation of the DFL hypothesis

Experiments 2 and 3 were scored on the **same** test split (1,904 images / 3,009 boxes),
so this comparison is direct and needs no cross-evaluation:

| metric | exp 2 (`reg_max=1`) | exp 3 (`reg_max=16`) | Δ |
|---|---:|---:|---:|
| Precision | 0.664 | 0.658 | **−0.006** |
| Recall | 0.940 | 0.931 | **−0.009** |
| mAP50 | 0.900 | 0.883 | **−0.017** |
| mAP50-95 | 0.483 | 0.478 | **−0.005** |

Per class, mAP50-95:

| class | exp 2 | exp 3 | Δ |
|---|---:|---:|---:|
| closed_eye | 0.415 | 0.415 | **0.000** |
| open_eye | 0.382 | 0.372 | **−0.010** |
| yawning | 0.652 | 0.648 | **−0.004** |

> **The hypothesis is refuted. Every metric moved down or stayed flat; nothing improved.**

Three observations:

1. **The bar was set in advance and missed.** The pre-registered criterion was "if
   experiment 3 does not clearly beat 0.483, revert". It came in at **0.478** — below,
   not above.
2. **The damage is in mAP50, not mAP50-95.** The metric DFL was supposed to lift moved
   −0.005 (noise-scale). The metric it was not supposed to touch moved **−0.017**,
   concentrated in `open_eye` (mAP50 0.836 → 0.811, −2.5 points). `closed_eye` mAP50-95
   is *exactly* flat at 0.415. DFL did not merely fail to help localisation — it cost
   classification confidence on the class that was already weakest.
3. **It cost more to get less.** 6.52 h vs 6.09 h (~7% slower) and 5.04 MB vs 4.85 MB
   exported.

**This also walked back a YOLO26 design decision knowingly.** YOLO26 removed DFL on
purpose — the 16-bin softmax hurts INT8 quantisation and is brittle in TFLite/NCNN
compilers. This project implements the rest of that recipe (MuSGD, STAL, ProgLoss) and
stepped out of line with it on this one point. The result does not justify the deviation.

**Recommended disposition:** revert to `reg_max = 1`. Experiment 2's checkpoint remains
the best model this project has produced. The one loose thread worth pulling before
abandoning DFL entirely is **exp 3b: identical run with `l1: 0.0`** (§1.4.4) — not a
larger `reg_max`.

---

### 1.5 Comparative summary tables

#### 1.5.1 Architecture and loss

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| Backbone | MiniDarknetV2 (n) | *identical* | *identical* |
| Neck | MiniPANv2 | *identical* | *identical* |
| Head | DualDetect, anchor-free, no objectness | *identical* | *identical* |
| **`reg_max`** | **1 (DFL-free)** | **1 (DFL-free)** | **16 (DFL)** |
| Box regression | 4 raw ltrb scalars | 4 raw ltrb scalars | 16-bin softmax → frozen 1×1 conv integral |
| Box loss | CIoU + L1 | CIoU + L1 | **CIoU + L1 + DFL** |
| `dfl` gain | — | — | **1.5** |
| Cls loss | BCE, TAL soft targets | *identical* | *identical* |
| Assignment | TAL + STAL | *identical* | *identical* |
| Inference | NMS-free (one2one) | *identical* | *identical* |
| Train params | 2,501,882 | 2,501,882 | **2,685,042** |
| Export params | 2,375,157 | 2,375,157 | **2,466,649** |

#### 1.5.2 Hyperparameters and augmentations

[measured] Across all 42 `hyp.yaml` keys, **`dfl` is the only key that differs between
any two experiments.** Everything below was held constant across all three runs:

| group | settings (identical in Exp 1, 2, 3) |
|---|---|
| Optimisation | `lr0` 0.005 · `lrf` 0.05 · `momentum` 0.937 · `weight_decay` 0.0005 · `warmup_epochs` 3.0 · `warmup_momentum` 0.8 · `warmup_bias_lr` 0.1 · `nominal_batch` 64 · `muon_ratio` 0.5 |
| EMA | `ema_decay` 0.9999 · `ema_tau` 2000 |
| Loss gains | `box` 7.5 · `cls` 0.7 · `l1` 1.0 |
| Assignment | `tal_topk` 10 · `tal_alpha` 0.5 · `tal_beta` 6.0 · `stal` true · `stal_min_size` 8.0 · `stal_ref_size` 16.0 |
| ProgLoss | `prog_alpha_init` 0.8 → `prog_alpha_final` 0.1 |
| **Augmentation (16)** | `mosaic` 0.85 · `close_mosaic` 20 · `mixup` 0.10 · `scale` 0.5 · `degrees` 10.0 · `shear` 2.0 · `translate` 0.1 · `perspective` 0.0005 · `fliplr` 0.5 · `flipud` 0.0 · `hsv_h` 0.015 · `hsv_s` 0.7 · `hsv_v` 0.5 · `gray` 0.10 · `blur` 0.05 · `erasing` 0.25 |
| Multi-scale | `multi_scale_lo` 0.84 · `multi_scale_hi` 1.34 (384 → 320–512) |

Run arguments:

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| dataset | Dataset-Curated (28,170) | dataset_clean (18,445) | dataset (18,447) |
| scale / imgsz / batch | n / 384 / 64 | n / 384 / 64 | n / 384 / 64 |
| epochs | 300 | 300 | 300 |
| optimizer / seed | musgd / 0 | musgd / 0 | musgd / 0 |
| patience / amp | 60 / true | 60 / true | 60 / true |
| workers | 6 | 6 | 4 |
| `reg_max` | 1 | 1 | **16** |

#### 1.5.3 Training cost

| | Exp 1 | Exp 2 | Exp 3 |
|---|---:|---:|---:|
| Wall clock | **12.12 h** ¹ | **6.09 h** | **6.52 h** |
| Epochs | 300 | 300 | 300 |
| Best epoch | 294 | 283 | 299 |
| `best_fitness` | 0.5730903592 | 0.5241072334 | 0.5187640261 |
| Mean epoch | not recorded ² | not recorded ² | **78.2 s** |
| Fastest / slowest epoch | not recorded ² | not recorded ² | 74.2 s / 142.9 s |
| Interruption | **yes** — Windows Update, epoch 52 | no | no |
| FP32 ONNX | 9.50 MB | 9.50 MB | 9.87 MB |
| **FP16 ONNX** | **4.85 MB** | **4.85 MB** | **5.04 MB** |
| Checkpoint `best.pt` | 30.69 MB | 30.69 MB | 32.89 MB |

¹ Spans the reboot; larger dataset (28,170 vs ~18,446 images) also contributes.
² Per-epoch timing instrumentation did not exist until experiment 3. **No per-epoch
timing may be quoted for experiments 1 or 2 — none was measured.**

#### 1.5.4 Unified evaluation — the clean test split

The only three-way comparable numbers. Exp 2 and Exp 3 were scored natively on this
split (1,904 images / 3,009 boxes); **Exp 1 was re-evaluated onto it** for comparability.

**Overall:**

| metric | Exp 1 ¹ | Exp 2 | Exp 3 |
|---|---:|---:|---:|
| Precision | 0.642 ² | **0.664** | 0.658 |
| Recall | — ² | **0.940** | 0.931 |
| **mAP50** | 0.870 | **0.900** | 0.883 |
| **mAP50-95** | 0.473 | **0.483** | 0.478 |
| mAP50-95 / mAP50 | 0.54 | 0.54 | 0.54 |

¹ Re-evaluated, from `BASELINE_for_comparison.txt`.
² Only the metrics recorded in that comparison file are given; blanks were not measured
and are not estimated.

**Per class, on each experiment's own native test split** (Exp 1's split is the
mixed-convention one and its rows are therefore *not* comparable to the other two):

| class | metric | Exp 1 (mixed split) | Exp 2 (clean split) | Exp 3 (clean split) |
|---|---|---:|---:|---:|
| **closed_eye** | P | 0.588 | 0.662 | 0.657 |
| | R | 0.939 | 0.927 | 0.914 |
| | mAP50 | 0.880 | **0.900** | 0.882 |
| | mAP50-95 | 0.429 | 0.415 | 0.415 |
| **open_eye** | P | 0.579 | 0.530 | 0.531 |
| | R | 0.925 | 0.919 | 0.912 |
| | mAP50 | 0.881 | **0.836** | 0.811 |
| | mAP50-95 | 0.543 | 0.382 | 0.372 |
| **yawning** | P | 0.793 | 0.799 | 0.786 |
| | R | 0.968 | 0.976 | 0.967 |
| | mAP50 | 0.961 | **0.964** | 0.956 |
| | mAP50-95 | 0.650 | 0.652 | 0.648 |

> **Reading the `open_eye` rows.** Exp 1's apparently superior 0.543 mAP50-95 is an
> artefact of its test split, which still contained the large face-scale boxes — big
> boxes are far easier to localise at high IoU. On the shared clean split exp 1 scores
> **below** exp 2. This is precisely why §1.3.5's re-evaluation exists and why the mixed
> split was retired.

**The constant across all three: mAP50-95 / mAP50 = 0.54.** Two deliberate interventions —
fixing the labels, then changing the box representation — moved it by nothing. Whatever
caps localisation in this project was untouched by either.

#### 1.5.5 Video deployment telemetry

Same clip throughout: `15-MaleGlasses.mp4`, 640×480, native 30 FPS, 2,639 frames (every
frame), conf threshold 0.25, RTX 2000 Ada.

| | Exp 1 | Exp 2 | Exp 3 |
|---|---:|---:|---:|
| Latency | 12.23 ms ¹ | **5.81 ms** | 6.47 ms |
| Throughput | 81.8 FPS ¹ | **172.0 FPS** | 154.5 FPS |
| Real-time headroom | 2.7× ¹ | 5.7× | 5.15× |
| Total detections | 6,165 | 5,637 | 6,134 |
| `open_eye` | 5,350 (86.78%) | 4,958 (87.96%) | 5,447 (88.8%) |
| `closed_eye` | 553 (8.97%) | 373 (6.62%) | 438 (7.1%) |
| `yawning` | 262 (4.25%) | 306 (5.43%) | 249 (4.1%) |

¹ **Not comparable** — measured on a loaded machine. See §1.2.5. The three architectures
are near-identical in cost; exp 1 was not 2× slower.

**Temporal telemetry** (exp 3 only — `src/v2/temporal.py` did not exist for exps 1–2):

```
eye-detection coverage  : 97.9% of frames
PERCLOS (final window)  : 10.7%
blinks (100-400 ms)     : 16  (10.9/min)
yawns (>= 400 ms)       : 4  (2.7/min)
microsleeps (>= 1.5 s)  : 0
longest eye closure     : 0.83 s
alert frames            : SAFE 1,279 (48.5%) | WARNING 1,360 (51.5%) | CRITICAL 0
```

⚠️ Every threshold here is a conventional DMS-literature default. **None is validated
against this project's data** — the dataset has no drowsiness ground truth, so nothing
was or could be tuned against a label. This is instrumentation, never a diagnosis.

**The measured domain gap.** [measured] Experiment 2's model, conf 0.25 — 600 sampled
test stills against every third frame of the demo video:

| source | detections | mean conf | median | > 0.7 | > 0.9 |
|---|---:|---:|---:|---:|---:|
| test stills | 1,108 | 0.645 | 0.720 | 52.4% | **11.4%** |
| demo video | 1,886 | 0.585 | 0.647 | 36.7% | **0.0%** |

**Not one detection in 1,886 video frames exceeds 0.9 confidence, against 11.4% on
stills.** The model is never confident on real footage.

[measured] **Lighting is not the cause for this clip.** Mean-grey brightness: training set
p5 65.0 / median 116.1 / p95 196.8; demo video p5 96.2 / median 102.7 / p95 110.8 — the
video sits well inside the training range. The remaining candidates are motion blur,
video compression, continuous capture versus curated stills, and the subject's glasses.
**None of these has been isolated experimentally.**

---

### 1.6 Direct answers

#### 1.6.1 Why was experiment 2 trained from scratch instead of fine-tuning exp 1's `best.pt`?

Because fine-tuning would have destroyed the experiment. The purpose was to measure the
effect of **one variable** — the dataset. A model initialised from exp 1's weights carries
exp 1's learned response to the contaminated boxes: it has already been trained to emit
face-scale boxes for `open_eye`. Any subsequent measurement would confound "what the clean
data teaches" with "what the contaminated data already taught, partially unlearned". The
resulting number would not answer the question.

Training from scratch cost 6.09 h and produced an attributable result. That was the right
trade. [measured] Epoch 1 mAP50 = 0.0003 confirms the run genuinely started from random
initialisation.

#### 1.6.2 Why is `best.pt` ~30 MB when the model is nano-class?

[measured] `best.pt` is 30,690,602 bytes because it holds **three copies** of the network
plus optimiser state:

| component | size | needed for inference? |
|---|---:|---|
| `model` (raw weights) | 10.33 MB | no |
| `ema` (EMA weights — what is actually evaluated) | 10.33 MB | **yes** |
| `optimizer` state | 10.08 MB | no — needed for `--resume` |

The model itself is 2,375,157 parameters. The optimiser state exists so `--resume` works,
which is what rescued experiment 1 from the Windows Update reboot. The deployable artefact
is the ONNX export: **9.50 MB fp32 / 4.85 MB fp16**.

⚠️ The training banner prints an "int8 ~2.38 MB" figure. That is **arithmetic**
(params ÷ 4), not a measured quantised export. **No int8 model has ever been built or
evaluated in this project. Do not quote that number.**

#### 1.6.3 What is the best model this project has produced?

**Experiment 2's checkpoint** — `checkpoints/Expi-2-imagez-384/weights/best.pt`, epoch
283, mAP50 0.900 / mAP50-95 0.483 on the clean test split. Experiment 3 did not beat it.

---

### 1.7 What Chapter 1 leaves unresolved

1. **The localisation ceiling — still unexplained.** mAP50-95/mAP50 = 0.54 across all
   three experiments, against a healthy 0.70+. Two hypotheses tested, both refuted:
   annotation convention (exp 2) and box representation (exp 3). **The cause is not
   known.** The strongest untested lead is *label tightness* — published fatigue work
   reports ~0.94 mAP50-95 on other datasets against 0.483 here. That is a different
   dataset and not a valid comparison, but a gap that size is unlikely to be architecture
   alone. **Nobody has hand-audited a sample of boxes for tightness.** That is the
   cheapest remaining experiment and it has not been run.
2. **The train/deploy domain gap — measured, uncaused.** 0 of 1,886 video detections
   exceed 0.9 confidence against 11.4% on stills (§1.5.5). Lighting is ruled out for this
   clip. Motion blur, compression, and continuous capture remain unseparated.
3. **Night driving — untested.** 4.7% of the dataset is `night_or_very_low_light`. No
   result in this document tests it; the only demo clip is daylight.
4. **Not subject-disjoint, and cannot be made so** from the available metadata (§1.1.4).
   Every metric in this chapter carries that caveat.
5. **Exp 1's video latency was never re-benchmarked** against exps 2–3 on an idle machine
   (§1.2.5).
6. **`reg_max = 8` was never trained.** It would have bought DFL for +0.01 MB instead of
   +0.19 MB. Only 1 and 16 were run.
7. **Exp 3b (`l1: 0.0`) was never run** — the one remaining fair test of DFL (§1.4.4).

---

## Appendix C: Infrastructure built alongside the Foundation Experiments

Everything in this appendix is tooling and process, not a model result — it belongs beside
Chapter 1 rather than inside it because none of it moved a metric. Dated 2026-08-29 to
2026-08-30, spanning the period between experiment 2 and the writing of this record.

### C.1 Temporal driver monitoring — `src/v2/temporal.py`

**Why the detector is not enough.** The detector is per-frame and stateless. Fatigue is
not. A single closed-eye frame means nothing; a closed-eye frame that is the 45th in a row
means the driver is asleep. Every metric in Chapter 1 is a per-frame class score, which
cannot express that difference. The earlier HUD heuristic (`FatigueTracker`, a 45-frame
blend of two class fractions) is not a fatigue measure — it has no notion of event duration
at all. The temporal logic is kept out of both the model and the renderer so it can be
tested without a frame buffer, replayed offline, and reused by any front end.

**The three signals.**

- **PERCLOS** — fraction of time the eyes are closed over a rolling 60 s window, the most
  validated drowsiness proxy in the literature. Computed **only over frames where an eye
  was actually detected**; frames where the detector saw no eye go to a separate `coverage`
  figure instead of silently biasing the score. This matters specifically here: §1.5.5
  documents that the model is never confident on real video, so dropped frames are expected
  and must not be quietly counted as "eyes open".
- **Blinks versus microsleeps** — the same visual event, separated by duration alone.
  100–400 ms is a natural blink; a closure at or past 1.5 s is a microsleep, and it fires
  **the moment the threshold is crossed, not when the eyes reopen**. A lost face does
  **not** end a closure — only a confirmed `open_eye` does, so the alarm stays latched if
  the detector drops the driver mid-event.
- **Yawn frequency** — continuous duration plus occurrences per minute; a run must exceed
  400 ms to count, so a two-frame flicker is treated as noise.

Alert ladder SAFE → WARNING → CRITICAL. An active microsleep outranks every windowed
statistic, because the windows are averages and the microsleep is happening now.

**Tests.** [measured] `python -m src.v2.tests.test_temporal` — **27 assertions, all
passing**, over synthetic 30 fps timelines so expected frame indices are exact. Covered: a
200 ms blink counted as a blink and not a microsleep; a 67 ms flicker rejected; a 500 ms
closure counted as neither; a 2 s closure firing exactly one microsleep at **frame 45**
(= 1.5 s); the alarm latching through a lost face and clearing only on a confirmed open
eye; PERCLOS excluding blind frames from its denominator while coverage drops to 0.5; yawn
counting and rate; end-of-stream totals.

**Rendering.** `src/v2/hud.py` draws the telemetry panel (PERCLOS, blinks/min, yawns/min,
live closure timer) and a full-width alarm strip during a microsleep. A label-overlap
regression was introduced and fixed in the same session: the taller telemetry panel buried
detection-box labels in the top-left corner. `draw_detection_boxes` now takes an `avoid`
rectangle and relocates a buried label below the box, then right of the panel.

**The limitation that matters most.** Every threshold here is a conventional value from
the driver-monitoring literature, and **none of them has been validated on this project's
data** — there is no drowsiness ground truth anywhere in the dataset, so nothing was or
could be tuned against a label. Output is instrumentation, never a diagnosis. This warning
is repeated in the module docstring, in every generated `demo_video/analysis.txt`, and in
`AGENTS.md`.

### C.2 The `checkpoints/` reorganisation

Working rule: **less files is better to understand.** Before, one experiment was split
across two trees — `runs/v2/<name>/` for weights and the epoch CSV, `info/experiment N
<name>/` for the report — tied together only by memory. Now every experiment is one
self-contained folder with exactly two subdirectories:

```
checkpoints/Expi-<N>-imagez-<Size>/
  weights/            best.pt  last.pt  best.onnx
  REPORTS EXPI-<N>/   evaluation.txt  training_log.txt  training_log.csv
                      training_summary.txt  full_epoch_log.txt  args.yaml  hyp.yaml
                      plots/  demo_video/
```

[measured] `checkpoints/Expi-1-imagez-384/` 21 files / 90 MB; `checkpoints/Expi-2-imagez-384/`
22 files / 90 MB (also carries `BASELINE_for_comparison.txt`, the cross-evaluation behind
§1.3.5); `checkpoints/Expi-3-imagez-384/` now holds the full trained result (§1.4). `runs/`
was deleted after migration. Experiment 1 had no ONNX export, so one was generated during
the migration so all three experiments share identical structure.

Three things were **moved rather than deleted**, because deleting them would have broken
citations in documents that are themselves the record: `analysis/` →
`dataset/metadata/analysis/` (the τ study and 3-way comparison behind §1.1.3, ten
references across six files repointed); `info/experiment 1 baseline/historical_v1_cpu_baseline/`
→ `info/historical_v1_cpu_baseline/` (v1 is not a v2 experiment); each run's `args.yaml` /
`hyp.yaml` → into its report folder (the only surviving statement of what configuration
produced those numbers). One file was deleted outright: each run's `results.png`,
regenerated from `training_log.csv` by `src/v2/utils/plots.plot_results`.

The layout is enforced in code. `train.py` defaults to `--project checkpoints`;
`--exist-ok` lets a run land in a pre-created folder instead of forking to `<name>2`,
while still refusing to start if `weights/best.pt` is already there. When a `REPORTS
EXPI-*` folder exists in the target, the trainer writes `training_log.csv`, `args.yaml`,
`hyp.yaml` and `plots/01_training_curves.png` straight into it. [measured] Verified twice:
a 1-epoch smoke test (six files, nothing loose, no forked directory) and, for real, the
full 300-epoch experiment 3 run (§1.4), which produced the exact expected layout.

**Bug found and fixed while writing this record (2026-08-30).** `src/v2/report.py` still
hardcoded `video_dir = out_dir / "video"`, writing outside the Rule 1 `demo_video/`
subdirectory it was supposed to use — experiment 1 and 2's folders had been renamed to
`demo_video/` by hand during the migration, but the source was never updated, so
experiment 3's report silently created a second `video/` folder alongside the empty
pre-created `demo_video/`. Fixed at the source (`video_dir = out_dir / "demo_video"`) and
the two folders were merged for experiment 3. This is exactly the kind of drift Rule 1
exists to prevent, and it slipped through because the rule was enforced by convention in
one place (the migration) and not yet in the code that runs after it.

### C.3 Configuration and timing capture — Rule 1a

Three gaps in the record were closed before experiment 3 started:

**Augmentations were never in any report.** `hyp.yaml`'s sixteen augmentation parameters
(§1.5.2) were previously only in the live `src/v2/cfg/hyp.yaml`, which drifts the moment
anything is tuned. Both `args.yaml` and `hyp.yaml` are now snapshotted into
`REPORTS EXPI-<N>/` at startup, before the first batch.

**Per-epoch time was never recorded.** `training_log.csv` now carries `epoch_seconds`,
`train_seconds`, `val_seconds`, `elapsed_hours`. Train and validation are split
deliberately: an augmentation or hyperparameter change moves the training portion, while
validation is near-fixed overhead, and a single total conceals which one moved.

**Total wall clock lived only in scrollback.** `training_summary.txt` is now written at
run end with total duration, epochs completed, mean/fastest/slowest/first/last epoch, best
fitness, final validation metrics, and the full configuration and augmentation block.

[measured] Verified with a 2-epoch smoke test before the real run: epoch 1 took 125.2 s,
epoch 2 took 86.6 s — a 38.6 s (45%) spread between two identically configured epochs, from
first-touch disk reads and CUDA warm-up. That spread is itself the argument for recording
every epoch rather than sampling one. This instrumentation then ran for real across
experiment 3's full 300 epochs (§1.4.5).

One asymmetry is recorded rather than papered over: **experiments 1 and 2 have no
per-epoch timings at all** — the instrumentation did not exist when they ran. Their totals
(12.12 h, 6.09 h) are process wall clock read from console output; experiment 1's
additionally spans a Windows Update reboot. **No per-epoch timing may be quoted for
experiments 1 or 2.** For this reason a resumed run now reports two totals — process wall
clock, and the sum of per-epoch timings recovered from the CSV, which is the honest one
across an interruption.

### C.4 Bugs fixed across this phase

- **`src/v2/val.py` / `src/v2/export.py`** — `load_state_dict` was strict, and every
  checkpoint carries 418 stray `total_ops` / `total_params` buffers left behind by `thop`
  profiling in `bench.py`. Loading any checkpoint raised `RuntimeError`; export was
  completely blocked. Fixed with `strict=False` — those keys are not weights.
- **`src/v2/report.py`** (`plot_curves`) — indexed a name list with a `numpy.float32`,
  raising `TypeError`. Fixed with an `int` cast.
- **`src/v2/report.py`** (`annotate_video`) — hardcoded `video/` instead of the Rule 1
  `demo_video/` output directory. See C.2. Fixed 2026-08-30.

### C.5 Root directory cleanup

Two root folders were deleted outright, on explicit confirmation: `env/` (47 MB — an
in-project virtualenv with no torch in it; the project runs on the conda env `AI-3.11`)
and `graphify-out/` (2.1 MB — a knowledge-graph run from 2026-08-24 predating most of the
current code, regenerable via `/graphify --update`). [measured] Neither is imported
anywhere in `src/` or `configs/`, and the active interpreter was confirmed to be the conda
one, not `env/`, before deleting.

`configs/` was deliberately **left in place** — it is v1's configuration and 10 files
under `src/` still do `from configs import config`; `src/v2/` never reads it. Removing it
would mean editing frozen v1 code for tidiness alone. The project root therefore has five
directories, not four, and the reason is written down here rather than left to be
rediscovered.

Both standing rules (directory layout, mandatory logging) are written at the top of
`AGENTS.md`. Rule 2 is why this appendix, and Chapter 1, exist.

---

## Appendix A: The v1 (`src/`) CPU model — an unresolved conflict in the record

The original model in `src/` was trained on CPU before any of the v2 work. **Its weights do not
exist.** [measured] A search of the whole `C:\ssd projects` tree (the project root at the time) for any `.pth` or
`mini_yolo_best*` file belonging to this model returns nothing; the only `.pth` files present
belong to unrelated projects. No `results.csv` and no raw log survive either. The model therefore
cannot be reloaded, re-evaluated, or compared against v2 on any shared test split.

**Two surviving records of this run disagree with each other, and there is no way to adjudicate
between them.**

| | `info/.../historical_v1_cpu_baseline/original_report.md` | `book.md` Chapter 7–8 (Appendix B below) |
|---|---|---|
| epochs | 20 | 85 |
| peak mAP50 | 0.5817 at epoch 15 | 0.6805 at epoch 70 |
| duration | 15 h 58 m | ~68 h projected for 85 epochs |
| train images | not stated | 33,365 |
| val images | not stated | 5,477 |

[measured] The git history offers partial corroboration for the larger figure: commit `6a33595`
is titled "Production-ready MiniYOLO: Architecture, Pipeline & 85-Epoch Training Reports".

The reading most consistent with both documents is that the 20-epoch report covers a first stage
and training later continued to 85 epochs — but **this is an inference, not a verified fact**, and
it is recorded here as such. Neither peak mAP50 figure can be confirmed.

Note also that the v1 image counts (33,365 train / 5,477 val) do not match the current
Dataset-Curated counts (22,215 / 3,082), which is consistent with v1 having been trained on
Dataset-Main before the near-duplicate collapse described in §1.1.

**No v1 number in this document should be cited as a benchmark.** v1 is not comparable to v2:
different dataset, no surviving weights, and no test-split evaluation was ever recorded for it.

---

## Appendix B: Original v1 engineering notes, preserved verbatim

Everything below is the previous contents of `book.md`, written before the v2 work and preserved
unchanged so nothing is lost. It documents the v1 refactoring effort. **Its numbers have not been
re-verified by this session**, and Chapters 7–8 of it are one side of the conflict described in
Appendix A above.

---

## MiniYOLO Engineering Book: Architecture & Evolution
*A comprehensive guide to the refactoring, optimization, and training history of the MiniYOLO object detection pipeline.*

---

### Table of Contents
1. **Chapter 1**: Project Background & Reorganization Goals
2. **Chapter 2**: Target Project Architecture
3. **Chapter 3**: Core Module Refactorings & Technical Justifications
4. **Chapter 4**: Compilation Compatibility & Graph Break Resolution
5. **Chapter 5**: Data Pipeline & Polygon Conversion Logic
6. **Chapter 6**: Hyperparameter & Configuration Evolution
7. **Chapter 7**: Training Progress & Validation Metrics History
8. **Chapter 8**: CPU Training Performance & Time Analysis
9. **Chapter 9**: Recommended Future Optimizations
10. **Chapter 10**: Fine-Tuning Setup & Hyperparameter Adjustments

---

### Chapter 1: Project Background & Reorganization Goals

The MiniYOLO project was initiated to create a custom, lightweight, production-grade object detector specialized in detecting human expressions and fatigue signals (`closed_eye`, `open_eye`, `yawning`). 

Initially, the repository was structured as a flat set of script utilities with duplicated code, hardcoded math constants, and manual dependency setups. To reach the architectural quality of modern, state-of-the-art vision frameworks (like Ultralytics YOLOv8 and YOLO11), a major modernization program was executed.

#### Core Objectives:
*   **Decouple Training and Validation**: Extract common validation loops and evaluation math into a unified engine to prevent duplicate code.
*   **Production Standardization**: Reorganize file hierarchies into discrete modules (`data/`, `models/`, `losses/`, `engine/`, `utils/`).
*   **Clean Packages**: Remove custom shell-level path manipulations (`sys.path.insert`) in favor of proper absolute package imports.
*   **Modern Pipeline Augmentations**: Implement rich transforms (RandomAffine, HSV color scale distortion, horizontal flips) managed natively by global configuration constants.
*   **Graph Tracing Compatibility**: Ensure all parts of the forward model and loss functions are fully compatible with `torch.compile` by removing graph breaks.

---

### Chapter 2: Target Project Architecture

The directory layout has been streamlined into a modular package format. All `__init__.py` files and `__pycache__` folders were completely deleted from the workspace, converting the directories into clean Python implicit namespace packages.

```
mini_yolo/
├── configs/
│   └── config.py              # Central configurations & central hyperparameter overrides
├── info/
│   ├── book.md                # [This File] Complete engineering documentation
│   └── first report for train 1/ # Performance logs, charts, and predictions
├── runs/
│   ├── train/                 # Checkpoints (*.pth) and training logs
│   └── predictions/           # Inference outputs (annotated images)
├── src/
│   ├── data/
│   │   ├── dataset.py         # YOLO Dataset class, cache loader, & polygon converter
│   │   └── transforms.py      # Augmentation pipeline classes
│   ├── engine/
│   │   ├── evaluator.py       # Centered validation matching & AP computation
│   │   ├── predictor.py       # High-performance FP16 inference engine
│   │   ├── trainer.py         # Training loop, optimizer/scheduler step manager
│   │   └── validator.py       # Standalone checkpoint evaluation launcher
│   ├── losses/
│   │   └── yolo_loss.py       # Multi-positive target matcher & loss functions
│   ├── models/
│   │   ├── backbone.py        # Darknet multi-scale feature extractor (P3, P4, P5)
│   │   ├── blocks.py          # ConvBNSiLU, Bottleneck, C2f, and SPPF modules
│   │   ├── head.py            # Decoupled bounding box, class, and obj head
│   │   ├── neck.py            # PANet neck multi-scale feature fuser
│   │   └── yolo.py            # Unified MiniYOLO network wrapper
│   ├── utils/
│   │   ├── boxes.py           # Geometric coordinate utilities (IoU, CIoU loss, etc.)
│   │   ├── generate_report_visuals.py # Automated report visual generator
│   │   ├── logger.py          # Formatted console outputs
│   │   ├── metrics.py         # Confusion matrix and AP calculation helpers
│   │   ├── misc.py            # Extra system utilities
│   │   ├── nms.py             # Torchvision-accelerated class-agnostic NMS
│   │   ├── seed.py            # Reproducibility seed initializer
│   │   └── visualization.py   # Prediction box visualization & label renderer
│   ├── predict.py             # Inference launcher
│   ├── train.py               # Main training launcher
│   └── validate.py            # Main validation launcher
└── requirements.txt           # Dependency requirements
```

---

### Chapter 3: Core Module Refactorings & Technical Justifications

#### 1. Reusable Evaluator Engine (`evaluator.py`)
*   **Refactor**: Extracted duplicate evaluation code from the trainer and validator into a single `Evaluator` class in `src/engine/evaluator.py`.
*   **Justification**: Consolidates AP and timing calculation logic. It parses validation images, runs inference under an autocast context, computes box matching via metric matching, and outputs results in a clean table. Both training-time validation and standalone validation leverage this single code path.

#### 2. High-Performance Predictor (`predictor.py`)
*   **Refactor**: Upgraded prediction pipeline in `src/engine/predictor.py`.
*   **Justification**: Leverages identical preprocessing transforms (`Resize`, `ToTensor`, `Normalize`) to match validation statistics. Added automatic input reshaping, custom class filtering (`FILTER_CLASSES`), class-agnostic NMS (`AGNOSTIC_NMS`), and automatic output subdirectory generation.

#### 3. Coordinate Handling and NMS Acceleration (`boxes.py` & `nms.py`)
*   **Refactor**: Converted coordinates transformations (`xywh2xyxy` / `xyxy2xywh`) to vectorized formats and replaced custom NMS loops with `torchvision.ops.nms`.
*   **Justification**: Vectorized array operations prevent slow in-place array updates that break autograd tracking. Leveraging `torchvision.ops.nms` shifts NMS overhead to compiled CUDA kernels and reduces execution time.

#### 4. Modernized AMP API (`trainer.py`, `evaluator.py`, `predictor.py`)
*   **Refactor**: Replaced deprecated `torch.cuda.amp.autocast(...)` and `torch.cuda.amp.GradScaler(...)` calls.
*   **Justification**: Modernized syntax to use the unified `torch.amp` namespaces:
    *   `torch.amp.GradScaler(device_type, enabled=...)`
    *   `torch.amp.autocast(device_type=..., enabled=...)`
    This silences warnings and ensures compatibility with PyTorch 2.6+ while handling CPU/GPU fallback dynamically.

---

### Chapter 4: Compilation Compatibility & Graph Break Resolution

When utilizing `torch.compile` to run the model at maximum speed on modern GPUs, standard Python structures inside the model forward execution tree can trigger compiler **graph breaks**, which drop execution back to the slower Python interpreter.

#### 1. Removing Dictionary Caching in decoupled head (`head.py`)
*   **Old Code**: Checked grid cache using shape keys inside a Python dictionary:
    ```python
    key = (h, w, str(device))
    if key not in self.grid_cache:
        ...
    ```
*   **Problem**: Accessing a Python dictionary using dynamic properties and stringifying `device` values breaks compiler tracing.
*   **New Code**: Generates meshgrid coordinate arrays dynamically on-the-fly inside the compiled execution graph:
    ```python
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).to(torch.float32)
    ```

#### 2. Removing Context Managers in Loss Functions (`boxes.py`)
*   **Old Code**: Computed aspect ratio consistency parameter `alpha` using context managers:
    ```python
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ```
*   **Problem**: `with torch.no_grad():` state changes during graph execution disrupt tracing.
*   **New Code**: Replaced with clean tensor detaches:
    ```python
    alpha = v / (1 - iou + v + eps)
    alpha = alpha.detach()
    ```

---

### Chapter 5: Data Pipeline & Polygon Conversion Logic

Roboflow datasets frequently export labels containing segmented polygon vertices instead of standard bounding box formats.

#### 1. Dynamic Polygon Converter (`dataset.py`)
*   **Old Code**: Skipped any label txt lines that contained more than 5 elements, showing invalid label format warnings.
*   **Problem**: Polygon labels were discarded, lowering active target counts.
*   **New Code**: Implemented a parser fallback. When a line length exceeds 5 values (e.g., length 11, 13, 15, 17, 19, 21), it alternates the floating points to identify the polygon’s $(x, y)$ vertices, computes the enclosing rectangle boundaries, and transforms them into standard YOLO bounding coordinates:
    ```python
    all_coords = [float(x) for x in parts[1:]]
    xs = all_coords[0::2]
    ys = all_coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    ```

---

### Chapter 6: Hyperparameter & Configuration Evolution

Here is the trace of configuration changes implemented:

| Hyperparameter | Baseline Settings | Initial Train Settings | Fine-Tuning Settings (Current) | Technical Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Optimizer** | SGD | AdamW | **AdamW** | Offers fast convergence on multi-class topologies. |
| **Scheduler** | None | CosineAnnealingLR | **CosineAnnealingLR** | Smoothly decays learning rate over extended epochs. |
| **Learning Rate** | `1e-3` | `1e-3` | **`5e-4`** | Lower learning rate prevents destructive updates during fine-tuning. |
| **Epochs Limit** | `1` | `20` | **`50`** | Extends training up to 50 epochs for deeper convergence. |
| **Box Weight** | `1.0` | `5.0` | **`7.5`** | Increases box regression focus for precise eye/mouth boundary fitting. |
| **Class Weight** | `1.0` | `1.0` | **`1.25`** | Improves distinction between `open_eye` and `closed_eye` states. |
| **Resume Mode** | `False` | `False` | **`True`** | Resumes training from `mini_yolo_best.pth`. |

---

### Chapter 7: Training Progress & Validation Metrics History

#### 📈 Historical Metric Progress
The model is trained on a dataset containing 3 classes (`closed_eye`, `open_eye`, `yawning`).

*   **Initial Baseline (Before refactorings)**:
    *   *Best mAP@50*: `0.0649`
*   **New Refactored Training Run (Full 85 Epoch Progression)**:
    *   **Epoch 5**: `0.4284` mAP@50
    *   **Epoch 10**: `0.5067` mAP@50
    *   **Epoch 15**: `0.5817` mAP@50
    *   **Epoch 20**: `0.5926` mAP@50
    *   **Epoch 30**: `0.6120` mAP@50
    *   **Epoch 45**: `0.6469` mAP@50
    *   **Epoch 50**: `0.6712` mAP@50
    *   **Epoch 70**: **`0.6805`** mAP@50 (🥇 **Peak Validation Accuracy - Best Checkpoint**)
    *   **Epoch 85**: **`0.6805`** mAP@50 (🏆 **Final Completed Epoch**)

---

### Chapter 8: CPU Training Performance & Time Analysis

Due to hardware availability, training was run using the **CPU** rather than a GPU. Because of the size of the dataset, this introduces significant computational latency.

#### 1. Dataset Dimensions & Load Metrics
*   **Training Images**: 33,365
*   **Validation Images**: 5,477
*   **Batch Size**: 8
*   **Total Batches (Iterations) per Epoch**: 4,170 (calculated as $33,365 / 8$)

#### 2. Time Calculations
*   **Average Processing Speed**: $\sim 1.45 \text{ iterations (batches) per second}$
*   **Total Seconds per Epoch**: $\sim 2,875 \text{ seconds}$
*   **Total Minutes per Epoch**: $\sim 48.0 \text{ minutes}$ (roughly **$0.8\text{ hours}$**)
*   **Completed Stage 1 Duration (to Epoch 20)**: $\sim 960 \text{ minutes}$ (**$16.0\text{ hours}$**)
*   **Remaining Fine-Tuning Duration (Epochs 21 to 85, 65 Epochs)**: $\sim 3,120 \text{ minutes}$ (**$52.0\text{ hours}$**)
*   **Total Projected Duration for 85 Epochs**: $\sim 4,080 \text{ minutes}$ (**$68.0\text{ hours}$**)

---

### Chapter 9: Recommended Future Optimizations

To push the model's accuracy past `0.70` mAP@50 and improve compute speed, we recommend the following next steps:

1.  **Run with GPU (CUDA)**: Training on a CUDA-enabled GPU would increase batch iteration speed to $\sim 50\text{-}100\text{ iterations/second}$, cutting epoch training time down from **48 minutes** to **under 2 minutes**.
2.  **Add Model EMA (Exponential Moving Average)**: Keeping a moving average of weights during gradient updates smooths out validation fluctuations.
3.  **Adjust Image Size to 640**: Modern YOLO models are optimized for 640x640 resolution. Upgrading `IMG_SIZE` from 416 to 640 in `configs/config.py` will help resolve smaller object details.

---

### Chapter 10: Fine-Tuning Setup & Hyperparameter Adjustments

To achieve optimal fine-tuning performance without adding code complexity to the model architecture:

1.  **Checkpoint Resumption**: `RESUME` is enabled in `configs/config.py` pointing to `runs/train/mini_yolo_best.pth`.
2.  **Learning Rate Refinement**: `LEARNING_RATE` is adjusted to `5e-4` (half of initial rate) to refine features without disturbing established pre-trained weights. `train.py` dynamically updates optimizer parameter group learning rates upon loading checkpoints.
3.  **Loss Rebalancing**: `BOX_WEIGHT` is increased to `7.5` and `CLS_WEIGHT` to `1.25` to sharpen bounding box edges around subtle eye regions.
4.  **Code Simplicity Preserved**: Model architecture definitions (`src/models/yolo.py`, `backbone.py`, `head.py`, `neck.py`) remain untouched, ensuring the code stays lightweight, clean, and easy to study.

