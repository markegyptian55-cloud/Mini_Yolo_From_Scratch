# MiniYOLO Engineering Book

Record of the datasets, the model, and every training experiment run on this project.

**Rules this document follows.** Every number here is either (a) **[measured]** — produced by
a command run on this machine, with the source file named, or (b) **[from record]** — read out
of a file this session did not create. Where two surviving records disagree, the disagreement is
written down as a disagreement and **not** resolved by guessing. Where something cannot be
recovered, it says so instead of estimating. Nothing in this document is inferred and presented
as fact.

Last updated **2026-08-29**. Chapters 6-8 and Chapter 0 were added that day, after the
dataset audit and consolidation and the DFL head upgrade. **Chapter 0 corrects four
statements made earlier in this document.** Where a later chapter overturns an earlier
one, the earlier text is left standing with a pointer, not quietly edited -- a record
that rewrites itself is not a record.

---

## Table of Contents

- **Chapter 0** — Corrections: four things this document got wrong
- **Chapter 1** — The datasets and where they came from
- **Chapter 2** — Experiment 1: the v2 model, how it works, what happened
- **Chapter 3** — Experiment 2: why a second dataset was built, and did it help
- **Chapter 4** — Direct answers: fresh training vs fine-tuning, and the 29 MB checkpoint
- **Chapter 5** — What is still wrong, and what to do next
- **Chapter 6** — The dataset audit and the consolidation to one `dataset/`
- **Chapter 7** — Experiment 3: the DFL head (code complete, run not started)
- **Chapter 8** — Temporal driver monitoring: PERCLOS, blinks, microsleeps, yawns
- **Chapter 9** — The `checkpoints/` reorganisation and the two standing rules
- **Appendix A** — The v1 (`src/`) CPU model: an unresolved conflict in the record
- **Appendix B** — Original v1 engineering notes, preserved verbatim

---

## Chapter 0: Corrections — four things this document got wrong

Added 2026-08-29. Each item names where the wrong claim sits, what is actually true, and how
it was checked. None of these were caught by the user; all four surfaced during the dataset
audit in Chapter 6, three of them while trying to verify my own earlier work.

**C-1. A claimed 71% train/test leakage does not exist. Retracted in full.**
During the audit I found that 1,787 filename stems appear in more than one split and reported
this as catastrophic leakage. That was wrong. The stem before Roboflow's `.rf.<hash>` is a
shared numbering scheme, not an image identity: [measured] 5,173 of 8,179 stems map to more
than one `visual_group_id`. Concretely, the four `000007_jpg.rf.*` files are one man's face
plus three close-ups of a *different* person's eye — four distinct visual groups.
The authoritative identity is `visual_group_id` (perceptual near-duplicate clustering, in
`dataset/metadata/lineage/curated_dataset_manifest.csv`). [measured] **Zero visual groups span
two splits**, checked twice and re-checked by `validate_dataset.py`. True redundancy is 1.15×,
not 3.44×. The original curation was sound. This claim never reached a report or a metric.

**C-2. §3.1's rationale for experiment 2 is half wrong.**
§3.1 and §1.2 present the filtered images as "annotation contamination" — mislabels. A visual
audit of 30 flagged images shows the flagged set is roughly **half genuine mislabels** (an
`open_eye` box drawn over a whole face, including two faces wearing sunglasses) and roughly
**half legitimate eye close-up crops**, where a frame-filling box is *correct* because the
image genuinely is an eye. The removal still stands, but the honest justification is
**deployment relevance**, not error: a dashcam never sees a 640×640 single-eye crop, so
neither population belongs in this dataset. Two different reasons, one filter. §1.2's width
statistics are unaffected and remain correct as measured.

**C-3. The grayscale/IR train-vs-test mismatch was sampling noise.**
An early 250-image probe reported 11.2% grayscale/IR in train against 5.2% in test and I wrote
that up as a domain mismatch. [measured] The exhaustive count over all 18,447 images gives
**6.0% train vs 5.7% test**. The splits are well matched. There is no mismatch to fix.

**C-4. §5.3's "3.8% dark" figure is superseded, not wrong.**
It was measured on the pre-consolidation dataset. [measured] On today's `dataset/`:
859 of 18,447 images are `night_or_very_low_light` (4.7%) and 1,080 are grayscale/IR (5.9%).
The conclusion of §5.3 is unchanged and if anything better supported: low-light driving remains
untested.

One earlier error inside this document was already corrected in place on 2026-08-27: the
epoch-52 metrics in §2.3 had been read off a shifted CSV column. The corrected values are
P 0.514 / R 0.927 / mAP50 0.849 / mAP50-95 0.479 / fitness 0.516.

---

## Chapter 1: The datasets and where they came from

Three classes throughout, in this order: `closed_eye` (0), `open_eye` (1), `yawning` (2).

### 1.1 Dataset lineage

There have been three datasets in this project's life. Only the last two exist on this machine.

**Dataset-Main** — the original. **Not present on this machine.** [measured] The path
`C:\ssd projects\nano big\data` does not exist. Everything known about it comes from the header
comment inside `dataset/data.yaml`, which this session did not write. [from record] That header
states Dataset-Main is "UNTOUCHED, frozen, and remains the authoritative benchmark dataset", and
that its test split contained **5,589 images**.

**Dataset-Curated** — what `dataset/` currently holds. [from record] Per the same header, it was
derived from Dataset-Main by "near-duplicate collapse (clusters size>=3) and cross-split
near-duplicate leakage resolution", reducing the test split to **2,873 of Dataset-Main's 5,589
images**. The header carries an explicit warning that this folder's test split is a deduplicated
*subset*, so results on it are not comparable to experiments run against Dataset-Main.

[measured] Current contents of `dataset/`, counted directly:

| split | images | boxes |
|---|---|---|
| train | 22,215 | 30,424 |
| val | 3,082 | 4,206 |
| test | 2,873 | 3,981 |

Image counts from `ls`; box counts from the dataset scanner's own output during training and from
`checkpoints/Expi-1-imagez-384/REPORTS EXPI-1/evaluation.txt`.

**dataset_clean** — built during Experiment 2. Covered in Chapter 3.

### 1.2 The annotation-convention defect in Dataset-Curated

[measured] This was found by measuring the width distribution of every box in `dataset/`. The
distributions are **bimodal**: the same class name is used for boxes at two scales that differ by
3–8×.

| class | total boxes | width > 0.4 (face-scale) | width ≤ 0.4 (object-scale) |
|---|---|---|---|
| closed_eye | 17,556 | 23.7% | 76.3% |
| open_eye | 12,365 | 45.0% | 55.0% |
| yawning | 8,690 | 69.5% | 30.5% |

[measured] Comparing images that carry one eye box against images that carry two, within a single
class, the median box width is:

| class | 1-box images | 2-box images | ratio |
|---|---|---|---|
| closed_eye | 0.620 | 0.188 | 3.3× |
| open_eye | 0.778 | 0.098 | 8.0× |

Two annotation conventions are mixed together under the same labels:

- **object-level** — one small box per eye. Sources include filenames matching
  `closed_eye_#-jpg_face_#_jpg.*`, `dd_v#_closed-*`, `istockphoto-*`.
- **face-level** — one large box over the whole face, labelled with the eye state it depicts.
  Sources include filenames matching `s#_#_#_#_#_#_#_#_png.rf.*`.

For `closed_eye` and `open_eye` this is a genuine contradiction. For `yawning` it is not — a yawn
is inherently a face-scale event, and the large-box mode is the correct one for that class.

[measured] All images in both groups are 640×640, so the two conventions cannot be told apart by
image size — everything was resized by Roboflow.

---

## Chapter 2: Experiment 1 — the v2 model, how it works, what happened

Experiment folder `checkpoints/Expi-1-imagez-384/`.

### 2.1 How the model works

`src/v2/` — a custom detector, not Ultralytics. Read from source (`src/v2/models/`):

**Backbone** (`MiniDarknetV2`) — stem `Conv(3→16, stride 2)`, then four stages, each a
stride-2 `Conv` followed by a `C2f` block, with an `SPPF` at the end of stage 4. Scale `n` uses
widths `(16, 32, 64, 128, 256)` and depths `(1, 2, 2, 1)`. Three feature maps leave the backbone:
**P3 at stride 8, P4 at stride 16, P5 at stride 32**.

**Neck** (`MiniPANv2`) — a bidirectional feature pyramid. The FPN path upsamples P5 and
concatenates it into P4, then upsamples that into P3, carrying semantic context down to fine
resolution. The PAN path then runs stride-2 convolutions back up, concatenating at each level,
carrying spatial precision back up. Output is three fused maps at the same three strides.

**Head** (`MiniHeadV2`) — decoupled, anchor-free, and notable for three deliberate omissions:

- **No DFL.** `reg_max = 1`: the box branch emits **four raw scalars** per location (left, top,
  right, bottom distances in feature-cell units), trained with CIoU + L1. This is the weakest box
  regression option available, and Chapter 5 identifies it as the current accuracy ceiling.
- **No objectness branch.** The class score doubles as confidence (YOLOv8 convention onward).
- **Dual branch, one survives export.** A one-to-many branch trains alongside a one-to-one
  branch; only the one-to-one branch is kept at export, so **inference needs no NMS**
  (YOLOv10/YOLO26 style). The class branch uses depthwise-separable convolutions.

[measured] From the training banner: **2,501,882 parameters, 2.04 GFLOPs at 384², 10.01 MB fp32.**

### 2.2 Configuration

[from record] `checkpoints/Expi-1-imagez-384/REPORTS EXPI-1/args.yaml` and `hyp.yaml`:

scale `n` · imgsz 384 (multi-scale 320–512) · batch 64 · 300 epochs · optimizer `musgd` ·
seed 0 · patience 60 · AMP on · lr0 0.005 · lrf 0.05 · momentum 0.937 · weight_decay 0.0005 ·
warmup 3 epochs · EMA decay 0.9999 · loss weights box 7.5 / cls 0.7 / l1 1.0 · TAL topk 10,
alpha 0.5, beta 6.0, small-object TAL on · ProgLoss alpha 0.8 → 0.1.

Augmentation: mosaic 0.85 (closed for the final 20 epochs) · mixup 0.1 · scale 0.5 · degrees 10 ·
shear 2 · translate 0.1 · perspective 0.0005 · fliplr 0.5 · hsv 0.015/0.7/0.5 · grayscale 0.1 ·
blur 0.05 · random erasing 0.25.

### 2.3 What happened during the run

The run was **interrupted once, and it was not a crash.** [measured] Windows Event Log shows
`MoUsoCoreWorker.exe` initiating a restart at 21:45 on 2026-08-24 with reason
"Operating System: Service pack (Planned)", followed by `TrustedInstaller.exe` at 21:50 with
"Operating System: Upgrade (Planned)". Windows Update forced a reboot mid-training.

[measured] State at interruption, from `results.csv` epoch 52: **P 0.5140 · R 0.9275 ·
mAP50 0.8487 · mAP50-95 0.4794** (fitness 0.5164).

Training was resumed from `weights/last.pt` at epoch 53 and ran to completion at epoch 300.
[from record] The resumed leg took **12.12 h** for 248 epochs at 347 iterations/epoch. The wall
time of the first leg (epochs 1–52) was never timestamped and **cannot be recovered** — total
run time for experiment 1 is therefore not known exactly.

[measured] `results.csv` contains 300 epoch rows. Best checkpoint at **epoch 294**,
best_fitness **0.5730903592115826** (value as stored in `best.pt`).

### 2.4 Results

[from record] `checkpoints/Expi-1-imagez-384/REPORTS EXPI-1/evaluation.txt` — measured on Dataset-Curated:

**val split** (3,082 images / 4,206 boxes) · 1.11 ms/img (899 img/s)

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.644 | 0.950 | 0.907 | 0.536 |
| closed_eye | 0.581 | 0.935 | 0.888 | 0.439 |
| open_eye | 0.537 | 0.942 | 0.865 | 0.531 |
| yawning | 0.815 | 0.974 | 0.966 | 0.638 |

**test split** (2,873 images / 3,981 boxes) · 1.05 ms/img (956 img/s)

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.653 | 0.944 | 0.907 | 0.541 |
| closed_eye | 0.588 | 0.939 | 0.880 | 0.429 |
| open_eye | 0.579 | 0.925 | 0.881 | 0.543 |
| yawning | 0.793 | 0.968 | 0.961 | 0.650 |

[measured] Over the final 50 epochs the model gained **+0.0036 mAP50 and +0.0055 mAP50-95** — it
had converged. A longer schedule would not have helped.

**On the reported precision of 0.653.** This is precision measured at confidence 0.1, which is the
convention this codebase's metric code uses. It is not the precision you would deploy at.
[measured] From the F1-confidence curve, F1 peaks at about **0.84 at confidence ≈ 0.40**, and the
PR curve holds precision above 0.9 out to recall 0.75. The low headline precision number is a
reporting-threshold artifact, not a broken model.

### 2.5 Demo video

[from record] `checkpoints/Expi-1-imagez-384/REPORTS EXPI-1/demo_video/analysis.txt` — 2,639 frames of
`15-MaleGlasses.mp4` at conf 0.25: **12.23 ms/frame (81.8 FPS)**. Detections: open_eye 5,350
(86.78%), closed_eye 553 (8.97%), yawning 262 (4.25%).

---

## Chapter 3: Experiment 2 — why a second dataset was built, and did it help

Experiment folder `checkpoints/Expi-2-imagez-384/`.

### 3.1 Why

Experiment 1 produced mAP50 0.907 but mAP50-95 only 0.541 — a ratio of 0.60, where a healthy
detector sits above 0.70. Investigating that gap surfaced the annotation-convention defect
documented in §1.2: the box regressor was being asked to cover widths from 0.05 to 1.0 under the
same class name, so it could never be tight at high IoU.

**The hypothesis under test:** remove the contradiction and accuracy rises, most of all for
`open_eye`, the most contaminated class at 45.0%.

> ⚠️ **See Chapter 0, C-2.** The framing in this section — that the removed images are
> mislabels — is only about half true. Roughly half are legitimate eye close-up crops that
> are correctly annotated but irrelevant to a dashcam. The filter and its results below are
> unaffected; the *reason* is deployment relevance, not error.

### 3.2 What was built

`src/v2/tools/make_clean_dataset.py` → `dataset_clean/`. It drops every **image** that contains a
face-scale eye box (width > 0.4). It drops the whole image rather than the offending box, because
deleting the box alone would turn a real eye into unlabelled background and actively train the
model to miss it. Files are hardlinked, so the copy costs no additional disk.

[measured] Result:

| split | images kept | closed_eye | open_eye | yawning |
|---|---|---|---|---|
| train | 14,440 | 10,435 | 5,418 | 6,779 |
| val | 2,101 | 1,551 | 663 | 1,009 |
| test | 1,904 | 1,391 | 716 | 902 |

**Zero yawning boxes were removed** at any split — the filter targets eye classes only.
[measured] After filtering, face-scale eye boxes are **0.0%** of the eye boxes, down from 23.7%
and 45.0%.

### 3.3 What changed in the model

**Nothing.** Exactly one variable changed between experiments 1 and 2: the dataset. Scale `n`,
imgsz 384, batch 64, 300 epochs, `musgd`, seed 0, patience 60, AMP, and the entire `hyp.yaml`
were held identical. DFL was deliberately *not* introduced here, so that its effect can be
measured separately in a later experiment rather than being confounded with the dataset fix.

### 3.4 How the run went

[measured] Completed all 300 epochs in one session with no interruption. **6.09 h**, 225
iterations/epoch. Best checkpoint at **epoch 283**, best_fitness **0.5241072334190018**.
Over the final 50 epochs: **+0.0047 mAP50, +0.0024 mAP50-95** — converged again.

### 3.5 Results, and the comparison that is actually valid

Experiment 2's val and test splits are *subsets* of experiment 1's. **Comparing exp 1's 0.907 on
the mixed test split against exp 2's 0.900 on the clean test split is not a valid comparison** —
different images, different difficulty. To make a valid comparison, experiment 1's checkpoint was
re-evaluated on experiment 2's clean test split.

[measured] Both models, same clean test split (1,904 images / 3,009 boxes):

| metric | exp 1 | exp 2 | Δ |
|---|---|---|---|
| P | 0.642 | 0.664 | +0.022 |
| R | 0.926 | 0.940 | +0.014 |
| mAP50 | 0.870 | 0.900 | **+0.030** |
| mAP50-95 | 0.473 | 0.483 | +0.010 |

Per class, same split:

| class | exp 1 mAP50 | exp 2 mAP50 | Δ | exp 1 mAP50-95 | exp 2 mAP50-95 | Δ |
|---|---|---|---|---|---|---|
| closed_eye | 0.885 | 0.900 | +0.015 | 0.412 | 0.415 | +0.003 |
| open_eye | 0.763 | 0.836 | **+0.073** | 0.355 | 0.382 | +0.027 |
| yawning | 0.962 | 0.964 | +0.002 | 0.651 | 0.652 | +0.001 |

Experiment 2's own full evaluation is in `checkpoints/Expi-2-imagez-384/REPORTS EXPI-2/evaluation.txt`; on the
clean **val** split it scores P 0.657 / R 0.942 / mAP50 0.891 / mAP50-95 0.483.

### 3.6 Did it help? Partly — and the headline prediction was wrong

**What the result confirms.** The gain lands exactly where the diagnosis predicted. `open_eye`
was the most contaminated class and gained by far the most (+0.073 mAP50). `yawning` had zero
boxes removed and moved +0.002 — a control case behaving as a control should. Precision and
recall both rose. This is a controlled result, not noise.

**What the result refutes.** After Experiment 1 it was predicted that mAP50-95 would "move
substantially". **It did not.** It moved +0.010, and the mAP50/mAP50-95 ratio is essentially
unchanged (0.544 → 0.537). That prediction was wrong. The annotation-convention defect was
capping *detection and classification*, not *localisation*.

**Worth stating plainly:** in absolute terms the headline mAP50 went from 0.907 (exp 1, mixed
split) to 0.900 (exp 2, clean split). Those numbers are not comparable, but anyone reading the two
report folders side by side will see two numbers near 0.90 and reasonably ask what was gained. The
honest answer is: **+0.030 mAP50 on identical data, concentrated in one class, with the
localisation problem still unsolved.**

### 3.7 Demo video

[from record] `checkpoints/Expi-2-imagez-384/REPORTS EXPI-2/demo_video/analysis.txt` — same 2,639 frames:
**5.81 ms/frame (172.0 FPS)**. Detections: open_eye 4,958 (87.96%), closed_eye 373 (6.62%),
yawning 306 (5.43%).

**Do not read the speed change as a model improvement.** Experiment 1 measured 12.23 ms/frame on
an *identical* architecture, parameter count and input size. The difference is machine load at
measurement time — experiment 1's video was rendered while other applications were running. A
back-to-back re-benchmark on an idle machine is required before quoting any speedup.

---

## Chapter 4: Direct answers

### 4.1 Why was experiment 2 trained from scratch instead of fine-tuning exp 1's `best.pt`?

**It was a deliberate choice, made to keep the result interpretable.**

[measured] Experiment 2 started from random initialisation. The proof is in its own log: epoch 1
reported mAP50 = 0.0003. A fine-tune from experiment 1's weights would have started near 0.85.
No `--resume` flag and no pretrained weights were passed:

```
python src/v2/train.py --data dataset_clean/data.yaml --scale n --imgsz 384 \
    --batch 64 --epochs 300 --workers 6 --seed 0 --name v2_n384_clean
```

**The reasoning.** Experiment 2 exists to answer one question: *does removing the annotation
contradiction improve accuracy?* Fine-tuning from experiment 1 would have made that question
unanswerable, because experiment 1's weights were themselves fitted to the contaminated data —
including whatever the model learned about predicting big face-scale boxes. Any gain could then be
argued to come from the extra 300 epochs of training rather than from the data change. Starting
fresh means one variable moved, so the +0.030 is attributable.

**What it costs.** Fine-tuning would have been faster and might have reached a higher final
number. **That has not been tested.** It remains a genuine open option, and it is the natural
approach for the domain-gap problem in §5.2 — initialise from a trained checkpoint and fine-tune
on footage that resembles deployment.

### 4.2 Why is `best.pt` 29 MB when the model should be 5–10 MB?

**Because `best.pt` is a training checkpoint, not a deployment artifact. It contains three copies
of the network.** The model itself is exactly the size expected.

[measured] Decomposing `checkpoints/Expi-2-imagez-384/weights/best.pt` (30,690,602 bytes = 30.69 MB):

| component | size | what it is |
|---|---|---|
| `model` | 10.33 MB | raw training weights |
| `ema` | 10.33 MB | EMA weights — **this is the one used for inference** |
| `optimizer` | 10.08 MB | MuSGD momentum buffers, needed only to resume training |
| everything else | < 0.01 MB | epoch, fitness, hyp, args, class names |
| **total** | **30.75 MB** | |

Only the EMA copy is needed to run the model. The optimizer state exists purely so that
`--resume` works — which is exactly what saved experiment 1 after the Windows Update reboot.

[measured] Stripping down to a deployable model, step by step:

| artifact | size |
|---|---|
| EMA weights, fp32 | 10.33 MB |
| minus 418 stray `thop` profiling buffers (see §5.4) | 10.22 MB |
| minus the one-to-many training branch (120 tensors, 123,431 params) | 9.69 MB |
| **ONNX export, fp32** — `best_384.onnx` | **9.64 MB** |
| **ONNX export, fp16** — `--half` | **4.86 MB** |

[measured] The exported model reports **2,375,157 parameters, 2.00 GFLOPs at 384², 9.50 MB fp32**
after the training-only branch is discarded.

**So the target is already met.** 9.64 MB fp32 or 4.86 MB fp16 sits inside the 5–10 MB range, and
the model is a genuine YOLO26-nano-class network at 2.4 M parameters. Produce it with:

```
python -m src.v2.export --weights checkpoints/Expi-2-imagez-384/weights/best.pt --imgsz 384        # 9.64 MB
python -m src.v2.export --weights checkpoints/Expi-2-imagez-384/weights/best.pt --imgsz 384 --half # 4.86 MB
```

**One caveat, stated because it matters.** The training banner's "int8 ~2.50 MB" figure is an
arithmetic estimate (parameter count ÷ 4), **not a measured quantised export**. No int8 model has
been built or evaluated on this project. Do not quote 2.5 MB as an achieved result.

---

## Chapter 5: What is still wrong, and what to do next

### 5.1 Localisation is the remaining accuracy ceiling

mAP50 0.900 against mAP50-95 0.483 is a ratio of 0.54. Experiment 2 showed this is not caused by
the annotation conventions. The most likely remaining cause is the head: `reg_max = 1`, four raw
scalars, no DFL (§2.1). **Next experiment: set `reg_max` to 8 or 16, add the distribution focal
loss term, change nothing else.**

> **Status update (2026-08-29): this is now implemented.** `reg_max = 16` with a DFL loss term
> is the default head. See Chapter 7 for what was built, what it cost, and the two caveats
> that came with it. The experiment itself has **not been run**.

For context, and with the caveat that **this is a different dataset and not a valid direct
comparison**: published driver-fatigue work using an improved YOLOv8 (YOLO-FDCL) reports 94.2%
mAP50-95 on YAWDD. This project is at 48.3%. A gap that size is unlikely to be explained by
architecture alone and probably also reflects box tightness and consistency in the labels — but
**that has not been verified here**, and verifying it needs a hand-audit of a sample of boxes.

### 5.2 There is a measured train/deploy domain gap

[measured] Confidence distributions, experiment 2's model, conf threshold 0.25 — 600 sampled test
stills against every third frame of the demo video:

| source | detections | mean conf | median | > 0.7 | > 0.9 |
|---|---|---|---|---|---|
| test stills | 1,108 | 0.645 | 0.720 | 52.4% | 11.4% |
| demo video | 1,886 | 0.585 | 0.647 | 36.7% | **0.0%** |

Not one detection in 1,886 video frames exceeds 0.9 confidence, against 11.4% on stills. The model
is never confident on real footage.

[measured] Lighting is **not** the cause for this clip. Mean-grey brightness: training set p5 65.0
/ median 116.1 / p95 196.8; demo video p5 96.2 / median 102.7 / p95 110.8. The video sits well
inside the training range. The remaining differences are motion blur, video compression,
continuous capture versus curated stills, and the subject's glasses.

### 5.3 Night and low-light driving is untested

[measured] Only **3.8%** of training images are dark (mean grey < 60). For the variable-lighting
in-car deployment this project targets, that is the entire basis the model has. No result in this
document tests it, because the demo clip is daylight.

> **Superseded figure — see Chapter 0, C-4.** That 3.8% was measured on the pre-consolidation
> dataset. On today's `dataset/`: 859 of 18,447 images (4.7%) are `night_or_very_low_light`
> and 1,080 (5.9%) are grayscale/IR. The conclusion stands unchanged.

### 5.4 Bugs found and fixed this session

- `src/v2/val.py` — `load_state_dict` was strict, and every checkpoint carries 418 stray
  `total_ops` / `total_params` buffers left behind by `thop` profiling in `bench.py`. Loading any
  checkpoint raised `RuntimeError`. Fixed with `strict=False`; those keys are not weights.
- `src/v2/export.py` — the identical bug, which blocked ONNX export entirely. Same fix.
- `src/v2/report.py` — `plot_curves` indexed a name list with a `numpy.float32`, raising
  `TypeError`. Fixed with an `int` cast.

### 5.5 Latency headroom is the most underused asset here

[measured] 172 FPS on the demo clip against a 30 FPS requirement — roughly 5× headroom. `--scale s`
or `--imgsz 512` are affordable right now. Both should wait until §5.1 is resolved, so that a
larger model is not simply fitting label noise harder.

---

## Chapter 6: The dataset audit and the consolidation to one `dataset/`

Dated 2026-08-29. Goal: stop carrying three overlapping dataset directories and produce one
audited, validated, reproducible dataset that every future experiment is scored against.

### 6.1 What was there before

| directory | images | what it was |
|---|---|---|
| `dataset/` | 28,170 | Dataset-Curated. Experiment 1's data. Mixed annotation conventions. |
| `dataset_clean/` | 18,445 | The §3.2 filtered subset. Experiment 2's data. |
| `dataset_final_v1/` | 18,447 | Built during this audit. |

### 6.2 Choosing the threshold τ, on evidence rather than a convenient histogram

The filter rule is: drop the whole **image** if any eye-class box exceeds normalised width τ.
§3.2 used τ = 0.4 without justifying the number. `src/v2/tools/threshold_study.py` re-derived
it from three independent kinds of evidence and wrote `dataset/metadata/analysis/threshold_study/`.

[measured] **Statistical** — the two populations are cleanly separated, and not only by width:

| class | band | n | med width | med area | med aspect |
|---|---|---|---|---|---|
| closed_eye | above 0.40 | 4,155 | 0.703 | 0.4907 | **1.00** |
| closed_eye | below 0.40 | 13,401 | 0.184 | 0.0250 | 1.28 |
| open_eye | above 0.40 | 5,568 | 0.808 | 0.5452 | **1.00** |
| open_eye | below 0.40 | 6,797 | 0.097 | 0.0077 | 1.05 |

Median aspect ratio above the line is **exactly 1.00** for both classes — a square box, which
is what a frame-filling annotation on a square image looks like. Box *area* separates the two
populations by roughly **20×** (0.49–0.55 against 0.008–0.025), far more sharply than width
alone. This matters because the rule was originally defined on width only.

[measured] **Sensitivity** — all five candidates were measured, not just the chosen one:

| τ | images removed | images remaining | closed removed | open removed | yawning left |
|---|---|---|---|---|---|
| 0.25 | 11,925 | 16,243 | 7,601 | 5,855 | 8,690 |
| 0.30 | 10,705 | 17,463 | 5,436 | 5,776 | 8,690 |
| 0.35 | 10,183 | 17,985 | 4,650 | 5,688 | 8,690 |
| **0.40** | **9,723** | **18,445** | **4,179** | **5,568** | **8,690** |
| 0.45 | 9,221 | 18,947 | 3,823 | 5,399 | 8,690 |

**`yawning` loses exactly zero boxes at every τ**, which is the control this rule needed: a
yawn is legitimately face-scale, and no setting of τ touches it.

**Visual** — sample grids were rendered either side of the line. The 0.40–0.50 band is roughly
15 of 16 eye close-up crops; the 0.30–0.40 band is mostly correct eye-level boxes on visible
faces. That is where the line belongs.

### 6.3 What could not be automated, and was therefore not faked

The two populations above τ — legitimate eye crops and genuine face-scale mislabels — cannot be
told apart automatically on this machine. Three attempts failed:

- **Haar cascades** — 4% detection rate; misses rotated, dark and occluded faces.
- **Skin-tone fraction** — fails outright; the `session` source scores 0.000 because it is
  grayscale/IR, not because it is a crop.
- **mediapipe** — segfaults on a protobuf mismatch (`MessageFactory' object has no attribute
  'GetPrototype'`). No pretrained face detector is cached and the network is restricted.

Both populations are therefore removed together, for two stated reasons, and **no image was
given an invented label**. That decision is recorded rather than hidden because it is the one
place in this build where a better tool would have produced a better dataset.

### 6.4 The build, and the honest headline

`src/v2/tools/build_final_dataset.py`, deterministic (seed 0), hardlinks, never mutates its
source. [measured] From `dataset/metadata/build_info.json`:

```
images_inspected  = 28,170        tau  = 0.4        seed = 0
images_kept       = 18,447
removed_by_reason = {'C': 9,723}  -- 0 removed for corruption or invalid labels
test_files_locked = 3,808
```

Splits were **inherited unchanged**, not recomputed: the existing assignment was already
verified leakage-free (Chapter 0, C-1) and reshuffling would have destroyed a verified property
to gain nothing. Intra-split near-duplicates were **retained** — 4,880 images flagged
`intra_split_near_duplicate` in the manifest. They are redundancy, not leakage.

The result differs from `dataset_clean` by **exactly two images**: zero-box background negatives
(`yawn_new_1229_*`) that the older builder silently skipped and this one deliberately keeps,
because a model with P 0.66 benefits from negatives. So, stated plainly rather than dressed up:

> **The image selection is equivalent to `dataset_clean`; the improvement is the reproducible
> curation, validation, metadata, and documented rationale.**

That sentence is in `dataset/DATASET_REPORT.md` verbatim. This build is not a data improvement
and is not presented as one.

### 6.5 Validation

`src/v2/tools/validate_dataset.py` deliberately does **not** import the builder — a validator
that shares the builder's logic inherits the builder's bugs. Everything is re-derived from
files on disk, with the manifest used only as a cross-check target.

[measured] **Status: PASS. Baseline readiness: READY.** 14 gates pass — YAML consistency, class
mapping, image integrity (18,447 decoded, 0 corrupt), label integrity (28,864 boxes, 0 invalid),
split integrity (0 orphans), duplicate integrity, cross-split near-duplicate leakage = 0,
near-duplicate accounting, manifest consistency, annotation consistency (0 kept images exceed
τ), test set lock (3,808 SHA-256 hashes, 0 mismatched), reproducibility, deployment-domain
audit, dataset report present.

Two items are marked `[DOCUMENTED LIMITATION]` and deliberately **not** `[PASS]`:

1. **Subject/video-disjoint split cannot be established.** `session_id` is empty for all 50,654
   lineage rows; `subject_id` is unknown for 18,447 of 18,447 images. The same person may appear
   in train and test. The splits are near-duplicate-disjoint, which is a weaker guarantee.
   Every metric in this project should be read with that qualification.
2. **Glasses and head-pose coverage is unknown.** No source carries per-image metadata for
   either, so it is reported as unknown rather than estimated.

Six further limitations are enumerated in `dataset/DATASET_REPORT.md`.

### 6.6 The consolidation

Performed after validation passed, on explicit confirmation, never before.

- `dataset/VIDEO FOR TEST/` → moved into the surviving dataset first, so `report.py` and
  `hud.py` keep working.
- Lineage preserved: `curated_dataset_manifest.csv` (50,654 rows), `EDA/`,
  `duplicate_report.csv`, `quality_report.csv`, `class_statistics.csv`,
  `source_statistics.csv`, `dataset_summary.txt`, and the old `data.yaml` (as
  `source_data.yaml`) → `dataset/metadata/lineage/`. Without these the build would no longer
  be reproducible or auditable.
- Old `dataset/` and `dataset_clean/` deleted; `dataset_final_v1/` renamed to `dataset/`.

[measured] The three directories were hardlinked to one another, so deleting two only dropped
link counts — a probe file went from `nlink=3` to `nlink=1` with content intact, confirmed by
re-running the full 3,808-hash test-set lock afterwards with **0 mismatches**. The validator
was re-run against the renamed directory: still PASS, still READY.

### 6.7 Consequence for the numbers already in this document

Experiments 1 and 2 were scored on split directories that no longer exist. Today's `dataset/`
test split **is** experiment 2's test split (they differ only by two *train* images), so
**exp 2's numbers remain directly comparable** to anything scored on `dataset/` now.
**Experiment 1's do not** — it was scored on the larger contaminated split. To compare exp 1
against anything current, re-run `src/v2/val.py` against `dataset/data.yaml`. Its own §2.4
numbers stay in this document as the historical record of that run, correctly labelled.

---

## Chapter 7: Experiment 3 — the DFL head (code complete, run not started)

Dated 2026-08-29. **No training run has happened.** Everything below is either a code change
or a measurement of that code. No accuracy claim is made anywhere in this chapter.

### 7.1 The problem being attacked

mAP50 0.900 against mAP50-95 0.483 is a ratio of 0.54; healthy is above 0.70. Experiment 2
eliminated annotation convention as the cause (§3.6). The head was the remaining suspect: it
regressed **four raw scalars** per location (`reg_max = 1`) under CIoU + L1 — the weakest box
representation available. mAP50-95 averages IoU thresholds up to 0.95, so it is precisely the
metric a coarse box representation caps.

### 7.2 What was changed

- **`src/v2/models/head.py`** — new `DFL` module: each ltrb side is predicted as a softmax over
  `reg_max` integer bins and integrated back to a scalar by a **frozen 1×1 convolution whose
  weights are `[0, 1, ..., reg_max-1]`** — literally the expectation of the distribution. `cv2`
  now emits `4 * reg_max` channels instead of 4.
- **`src/v2/losses/loss.py`** — adds `L_dfl`: the linear interpolation of the cross-entropy
  against the two integer bins straddling each continuous target. A target of 7.3 asks for 70%
  of the mass on bin 7 and 30% on bin 8. CIoU supervises the decoded box; DFL supervises the
  shape of the distribution behind it.
- **`src/v2/cfg/hyp.yaml`** — new gain `dfl: 1.5` (the YOLOv8/v11 value; not tuned here).
- **`--reg-max`** is a training flag and its value is written into every checkpoint.

**CIoU and L1 are unchanged and both still present.** That is deliberate: it keeps experiment 3
to a single variable against experiment 2.

### 7.3 Backward compatibility, and why it was necessary

`reg_max = 1` still builds the original head. [measured] Parameter counts are identical to
before the change: 2,501,882 training / 2,375,157 exported. `val.py` and `export.py` default to
`reg_max = 1` when a checkpoint has no such key, because every pre-experiment-3 checkpoint was
trained with the scalar head. Defaulting to 16 there would have silently constructed the wrong
architecture and `strict=False` would have hidden the failure. [measured] Verified end to end:
experiment 2's `best.pt` loads through the new code and still detects (310 detections over a
150-frame clip, 2.07 per frame, 100% eye-detection coverage).

### 7.4 What it costs — measured, not estimated

[measured] ONNX exports at `--imgsz 384`, FP16, opset 13, all from real `export.py` runs:

| head | exported params | FP16 ONNX |
|---|---|---|
| `reg_max = 1` (exp 1 / exp 2) | 2,375,157 | **4.85 MB** |
| `reg_max = 8` | 2,376,593 | **4.86 MB** |
| `reg_max = 16` (exp 3 default) | 2,466,649 | **5.04 MB** |

**`reg_max = 16` lands 0.24 MB above the 4.8 MB edge budget.** It is not free and is not
reported as free. `reg_max = 8` buys DFL for +0.01 MB and is the in-budget alternative if that
ceiling is hard — `--reg-max 8` and nothing else changes.

The box trunk is widened to `2 * reg_max` rather than Ultralytics' `4 * reg_max`, specifically
to hold that line; the wider trunk measured roughly 139k more exported parameters. The DFL
integral exports as `Softmax` + `Conv`, both natively supported by ONNX Runtime, TensorRT and
OpenVINO — no custom operator.

### 7.5 Configuration verified

[measured] A 1-epoch run on `dataset/` (a config check, **not** an
experiment):

```
box=3.304  cls=5.984  l1=7.149  dfl=3.959  a=0.80  lr=5.00e-03
epoch 1  mAP50=0.0452  mAP50-95=0.0135  P=0.136  R=0.083
checkpoint reg_max = 16   |   results.csv gained a dfl_loss column
```

The DFL term is active and falling, validation runs, the checkpoint records `reg_max`. That is
all this proves.

### 7.6 Two caveats that must travel with this experiment

**1. L1 now starts 3.3× higher than it did.** [measured] `l1_loss` at initialisation is **7.15**
with the DFL head against **2.20** with the scalar head. The cause is structural: a uniform
distribution over 16 bins has expectation 7.5 cells, whereas the old head's bias started boxes
at 1 cell. With `l1: 1.0` that term is now the largest in the loss early in training and
competes with DFL for the same gradient. **Ultralytics drops L1 entirely when using DFL.**
Keeping it here is a deliberate trade — one variable per experiment — but if experiment 3
underperforms, `l1: 0.0` (call it experiment 3b) is the first thing to try, ahead of a larger
`reg_max`.

**2. This deliberately walks back a YOLO26 design decision.** YOLO26 removed DFL on purpose:
the 16-bin softmax hurts INT8 quantisation and is brittle in TFLite and NCNN compilers. This
project implements the rest of that recipe — MuSGD, STAL, ProgLoss — and is now knowingly out
of step with it on this one point. If experiment 3 does not clearly beat 0.483, the right
response is to revert, not to accumulate the portability cost for nothing.

### 7.7 How to run it

```powershell
conda activate AI-3.11
cd "C:\mini_yolo"
python -m src.v2.train --data dataset/data.yaml --scale n --imgsz 384 `
  --batch 64 --epochs 300 --workers 4 --seed 0 --reg-max 16 --name v2_n384_dfl
```

Directly comparable to experiment 2: same dataset, same splits, same `hyp.yaml`, same seed,
same everything except the head.

---

## Chapter 8: Temporal driver monitoring

Dated 2026-08-29. New module `src/v2/temporal.py`.

### 8.1 Why the detector is not enough

The detector is per-frame and stateless. Fatigue is not. A single closed-eye frame means
nothing; a closed-eye frame that is the 45th in a row means the driver is asleep. Everything
this project measured before this module was a per-frame class score, which cannot express
that difference. The previous HUD heuristic (`FatigueTracker`, a 45-frame blend of two class
fractions) is not a fatigue measure — it has no notion of event duration at all.

The temporal logic is kept out of both the model and the renderer so it can be tested without
a frame buffer, replayed offline, and reused by any front end.

### 8.2 The three signals

**PERCLOS** — the fraction of time the eyes are closed over a rolling 60 s window; the most
validated drowsiness proxy in the literature. Computed **only over frames where an eye was
actually detected**. Frames where the detector saw no eye go to a separate `coverage` figure
instead of silently biasing the score in either direction. This matters here specifically:
Chapter 5.2 documents that this model is never confident on real video, so dropped frames are
expected and must not be quietly counted as "eyes open".

**Blinks versus microsleeps** — the same visual event, separated by duration alone. 100–400 ms
is a natural blink. A closure at or past 1.5 s is a microsleep, and it fires **the moment the
threshold is crossed, not when the eyes reopen** — waiting for the end of the event to report
it would defeat the purpose. A lost face does **not** end a closure; only a confirmed
`open_eye` does, so the alarm stays latched if the detector drops the driver mid-event.

**Yawn frequency** — continuous duration plus occurrences per minute. A run must exceed 400 ms
to count, so a two-frame flicker is treated as detector noise rather than a yawn.

Alert ladder SAFE → WARNING → CRITICAL. An active microsleep outranks every windowed statistic,
because the windows are averages and the microsleep is happening now.

### 8.3 Tests

[measured] `python -m src.v2.tests.test_temporal` — **27 assertions, all passing**, over
synthetic 30 fps timelines so the expected frame indices are exact rather than approximate.
They cover: a 200 ms blink counted as a blink and not a microsleep; a 67 ms flicker rejected;
a 500 ms closure counted as neither; a 2 s closure firing exactly one microsleep at **frame 45**
of the closure (= 1.5 s); the alarm latching through a lost face and clearing only on a
confirmed open eye; PERCLOS excluding blind frames from its denominator while coverage drops
to 0.5; yawn counting and rate; and end-of-stream totals.

### 8.4 Rendering and reporting

`src/v2/hud.py` draws the telemetry panel (PERCLOS, blinks/min, yawns/min, live closure timer)
and a full-width alarm strip during a microsleep. `src/v2/report.py` writes
`video/analysis.txt` per clip: throughput, per-class detection share, the temporal readout, and
the frame counts at each alert level.

A regression was introduced and fixed in the same session: the nav panel is drawn *after* the
boxes and grew taller with the telemetry rows, which buried the labels of any detection in the
top-left corner — the exact complaint raised earlier about label legibility.
`draw_detection_boxes` now takes an `avoid` rectangle and relocates a buried label below the
box, then to the right of the panel.

### 8.5 The limitation that matters most

**Every threshold in this module is a conventional value from the driver-monitoring literature,
and not one of them has been validated on this project's data.** This dataset has no drowsiness
ground truth — there is no label anywhere in it that says a driver was drowsy — so nothing was
or could be tuned against one. PERCLOS 15%/30%, blink 100–400 ms, microsleep 1.5 s and yawn
400 ms are all inherited defaults.

The output is instrumentation, not a diagnosis, and must never be reported as a clinical
finding. That warning is repeated in the module docstring, in every generated
`video/analysis.txt`, and in `AGENTS.md`.

---

## Chapter 9: The `checkpoints/` reorganisation

Dated 2026-08-29. Working rule: **less files is better to understand.**

Before, one experiment was split across two trees — `runs/v2/<name>/` for weights and the
epoch CSV, `info/experiment N <name>/` for the report. Nothing tied them together except
memory, and neither name said what image size it was trained at.

Now every experiment is one self-contained folder with exactly two subdirectories:

```
checkpoints/Expi-<N>-imagez-<Size>/
  weights/            best.pt  last.pt  best.onnx
  REPORTS EXPI-<N>/   evaluation.txt  training_log.txt  training_log.csv
                      full_epoch_log.txt  args.yaml  hyp.yaml
                      plots/  demo_video/
```

[measured] `checkpoints/Expi-1-imagez-384/` 21 files / 90 MB;
`checkpoints/Expi-2-imagez-384/` 22 files / 90 MB (it also carries
`BASELINE_for_comparison.txt`, the cross-evaluation that makes exp 1 and exp 2
comparable on one split); `checkpoints/Expi-3-imagez-384/` created empty in the same
shape. `runs/` was deleted after migration. Experiment 1 had no ONNX export, so one was
generated during the migration so the two finished experiments have identical structure
(2,375,157 params, 9.50 MB fp32).

Three things were **moved rather than deleted**, because deleting them would have broken
citations in documents that are themselves the record:

- `analysis/` → `dataset/metadata/analysis/`. It holds the τ study and the 3-way dataset
  comparison, cited twice in `dataset/DATASET_REPORT.md` and once in Chapter 6 above. All
  ten references across six files were repointed, and the two tools that write there had
  their output constants updated so a re-run lands in the new location.
- `info/experiment 1 baseline/historical_v1_cpu_baseline/` → `info/historical_v1_cpu_baseline/`.
  v1 is not a v2 experiment and did not belong nested inside one.
- Each run's `args.yaml` / `hyp.yaml` → into the report folder. They are the only surviving
  statement of what configuration produced those numbers.

One file was deleted outright: each run's `results.png`, which is regenerated from
`training_log.csv` by `src/v2/utils/plots.plot_results`.

The layout is enforced in code, not by convention. `train.py` defaults to
`--project checkpoints`; `--exist-ok` lets a run land in a pre-created folder instead of
forking to `<name>2`, while still refusing to start if `weights/best.pt` is already there.
When a `REPORTS EXPI-*` folder exists in the target, the trainer writes `training_log.csv`,
`args.yaml`, `hyp.yaml` and `plots/01_training_curves.png` straight into it.
[measured] Verified with a 1-epoch run into the pre-created Expi-3 folder: six files, all
in the right place, nothing loose in the experiment root, no forked directory. That test
run was then deleted so the Expi-3 target is empty.

### 9.1 Configuration and timing are now captured, not remembered

Added the same day, before experiment 3 started. Three gaps in the record were closed:

**Augmentations were never in any report.** `hyp.yaml` holds sixteen augmentation
parameters — mosaic 0.85, close_mosaic 20, mixup 0.10, scale 0.5, degrees 10.0, shear 2.0,
translate 0.1, perspective 0.0005, fliplr 0.5, hsv_h/s/v, gray 0.10, blur 0.05, erasing
0.25 — and until now a reader of an experiment report could not see any of them without
guessing that the live `src/v2/cfg/hyp.yaml` still matched what that run used. It will not,
the moment anything is tuned. Both `args.yaml` and `hyp.yaml` are now snapshotted into
`REPORTS EXPI-<N>/` at startup, before the first batch.

**Per-epoch time was never recorded.** `training_log.csv` now carries `epoch_seconds`,
`train_seconds`, `val_seconds` and `elapsed_hours`. Train and validation are split on
purpose: an augmentation or hyperparameter change moves the training portion, while
validation is near-fixed overhead, and a single total conceals which one moved.

**Total wall clock lived only in scrollback.** `training_summary.txt` is written at the end
of every run with the total, epochs completed, mean / fastest / slowest / first / last
epoch, best fitness, final validation metrics, and the full configuration and augmentation
block.

[measured] Verified with a 2-epoch run into the pre-created Expi-3 folder:

| epoch | total | train | val | mAP50 |
|---|---|---|---|---|
| 1 | 125.2 s | 109.3 s | 15.9 s | 0.0090 |
| 2 | 86.6 s | 76.1 s | 10.5 s | 0.1019 |

The 38.6 s spread between two identically configured epochs is itself the argument for
recording this: epoch 1 carries first-touch disk reads and CUDA warm-up, so any single-epoch
timing estimate would have been wrong by 45%. Seven files were produced, all in the Rule 1
positions, nothing loose. The run was then deleted so the Expi-3 target is empty.

One asymmetry is recorded rather than papered over: **experiments 1 and 2 have no per-epoch
timings at all**, because the instrumentation did not exist when they ran. Experiment 1's
"12.12 h" and experiment 2's "6.09 h" are process wall clock read from console output, and
experiment 1's figure additionally spans a Windows Update reboot. Those two runs cannot be
compared to experiment 3 at epoch granularity, only at the coarse total. For that reason a
resumed run now reports **two** totals — this process's wall clock, and the sum of the
per-epoch timings recovered from the CSV, which is the honest one.

Two root folders were deleted outright on the same day, on explicit confirmation:
`env/` (47 MB — an in-project virtualenv with no torch in it; the project runs on the conda
env `AI-3.11`) and `graphify-out/` (2.1 MB — a knowledge-graph run from 2026-08-24 that
predated most of the current code, and is regenerable). [measured] Neither is imported
anywhere in `src/` or `configs/`, and the active interpreter was confirmed to be the conda
one, not `env/`, before deleting.

`configs/` was deliberately **left in place**. It is v1's configuration and 10 files under
`src/` still do `from configs import config`. `src/v2/` never reads it. Removing it would
mean editing frozen v1 code for tidiness alone, which is not a trade worth making — so the
project root has five directories, not four, and the reason is written down here rather
than left to be rediscovered.

Both standing rules are written at the top of `AGENTS.md`. Rule 2 is why this chapter
exists.

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

