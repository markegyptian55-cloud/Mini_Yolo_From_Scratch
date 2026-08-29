# AGENTS.md — handoff notes for this project

Read this before touching anything. It says what exists, what was just built, what's
broken/fixed, and what's left. Written 2026-08-24; **substantially updated 2026-08-29**
after the dataset consolidation, the DFL head upgrade, and the move to the unified
`checkpoints/` layout.

---

# THE TWO STANDING RULES

These are not suggestions. Everything below assumes both hold.

## Rule 1 — Minimal Directory Standard

**Less files is better to understand.** Every training run writes its weights and its
report strictly inside one self-contained folder:

```
checkpoints/Expi-<N>-imagez-<Size>/
  weights/                 best.pt   last.pt   best.onnx
  REPORTS EXPI-<N>/        evaluation.txt          full test-set metrics, per class
                           training_log.txt        hypothesis, variable changed, result
                           training_log.csv        raw per-epoch numbers + timing
                           training_summary.txt    wall clock, config, augmentations
                           full_epoch_log.txt      raw per-epoch numbers, text form
                           args.yaml  hyp.yaml     the run's exact configuration
                           plots/                  the 10 standard figures
                           demo_video/             annotated clip + analysis.txt
```

**Exactly two subdirectories per experiment. No third one. No files loose in the
experiment root.** There is no `runs/`, no `outputs/`, and no per-experiment folder under
`info/` — those all existed once and were consolidated away on 2026-08-29.

The code enforces this rather than relying on memory:

- `train.py` defaults to `--project checkpoints`. Pass `--name Expi-<N>-imagez-<Size>`.
- `train.py --exist-ok` writes into a **pre-created** experiment folder instead of forking
  to `<name>2`. It still refuses to run if `weights/best.pt` is already there, so a
  finished experiment cannot be silently overwritten.
- When a `REPORTS EXPI-*` folder exists in the target, the trainer writes
  `training_log.csv`, `args.yaml`, `hyp.yaml` and `plots/01_training_curves.png` **into
  it** automatically. Nothing needs moving afterwards.
- `report.py --out` must be `"checkpoints/Expi-<N>-imagez-<Size>/REPORTS EXPI-<N>"`.
- `export.py --name best` writes `best.onnx` beside the weights instead of
  `best_<imgsz>.onnx`.

### Rule 1a — configuration and timing are captured automatically

Nothing below is optional or manual. The trainer writes all of it.

**Hyperparameters and augmentations.** `args.yaml` (every CLI argument, including `scale`,
`imgsz`, `batch`, `optimizer`, `seed`, `reg_max`) and `hyp.yaml` (every loss gain and every
augmentation — `mosaic`, `close_mosaic`, `mixup`, `scale`, `degrees`, `shear`, `translate`,
`perspective`, `fliplr`, `flipud`, `hsv_h/s/v`, `gray`, `blur`, `erasing`, plus the
multi-scale window) are written into `REPORTS EXPI-<N>/` at startup, before the first
batch. They are a snapshot of that run, not a pointer at `src/v2/cfg/hyp.yaml`, which will
drift.

**Per-epoch timing.** `training_log.csv` carries four timing columns beside the metrics:

| column | meaning |
|---|---|
| `epoch_seconds` | wall clock for the whole epoch |
| `train_seconds` | training portion only |
| `val_seconds` | validation portion only |
| `elapsed_hours` | cumulative since the run began |

Train and val are split deliberately: a hyperparameter or augmentation change moves
`train_seconds`, while `val_seconds` is near-fixed overhead. A total alone hides which one
grew. The console line reports the same figures live, with an ETA computed from the mean of
the last five epochs so a `close_mosaic` slowdown is visible rather than averaged away.

**Total wall clock.** `training_summary.txt` is written when the run finishes: total
duration, epochs completed, mean / fastest / slowest / first / last epoch, best fitness,
the final validation metrics, and the configuration and augmentation block repeated in
full so the run can be read without opening three files.

⚠️ **A resumed run reports two totals and they differ.** `this process` is the wall clock
of the current process; `sum of epoch timings` is recovered from `training_log.csv` on
`--resume` and is the honest total across the interruption. Experiment 1 was interrupted by
a Windows Update reboot, so this case is not hypothetical — quote the epoch-timing sum, and
if the run stopped early the summary is stamped `[INTERRUPTED / EARLY STOP]`.

Root directories are `dataset/`, `src/`, `checkpoints/`, `info/` — plus `configs/`, which
stays only because 10 files in `src/` (v1) do `from configs import config`. It is v1's, it
is frozen, and `src/v2/` never reads it. Folding it into `src/` means editing those 10
frozen files; that has not been done. Do not add any other root directory without a reason
that survives being written down here.

## Rule 2 — Mandatory logging in `info/book.md`

**An experiment is not finished when training ends. It is finished when it is written up.**
No run counts as complete until `info/book.md` carries, for that run:

1. the **hypothesis** — the single variable being changed and what it is predicted to do,
2. the **final metrics** — P, R, mAP50, mAP50-95, per class and overall, with the split
   they were measured on named,
3. the **conclusion** — whether the hypothesis held, and specifically **what it refuted**
   if it failed,
4. the **cost** — total wall clock and mean epoch time, taken from
   `training_summary.txt`. An accuracy gain that tripled training time is a different
   result from one that did not, and the record has to say which.

`book.md`'s own rules apply: every number is tagged `[measured]` or `[from record]`, a
disagreement between two records is written down as a disagreement rather than resolved by
guessing, and a claim later found wrong is corrected in **Chapter 0** with a pointer left
at the original text — never silently edited. A result that only exists in
`checkpoints/` is not documented; `checkpoints/` holds the evidence, `book.md` holds the
argument.

---

**Full record with every verified number: `info/book.md`.** That document distinguishes
measured values from values read out of older files, and flags what could not be recovered.
Do not quote a number about this project that isn't in there or in this file.

## What this project is

Driver-monitoring object detector: 3 classes (`closed_eye`, `open_eye`, `yawning`),
custom YOLO-family model, built from scratch (not Ultralytics). Two model generations
live side by side:

- **`src/`** — v1, the original CPU model. **Its weights no longer exist anywhere on this
  machine** (verified by a whole-drive search). Two surviving records of its training
  disagree with each other — see `info/book.md` Appendix A. Treat v1 as unciteable: no
  benchmark, no comparison, frozen code only. `configs/config.py` belongs to v1 and is
  **not read by `src/v2/`** — v2's hyperparameters live in `src/v2/cfg/hyp.yaml`.
- **`src/v2/`** — the current model. Full README at `src/v2/README.md` — read that before
  making any change in `src/v2/`, it has the full rationale.

## Where everything lives now

The whole project is at **`C:\mini_yolo`**. All paths below are relative to it.

⚠️ **The project was moved here from `C:\ssd projects\mini_yolo` on 2026-08-29.** Files
written *before* the move still record the old absolute path — `evaluation.txt`,
`full_epoch_log.txt`, and the `weights:` line inside each report. Those were correct when
written and are left as historical record. Don't "fix" them; regenerating a report just to
change a path would overwrite measured numbers with a fresh run.

## Dataset — there is now exactly ONE

**`dataset/`** is the single canonical dataset. It is the validated build formerly called
`dataset_final_v1`, promoted into place on 2026-08-29.

| split | images | boxes | closed_eye | open_eye | yawning |
|---|---|---|---|---|---|
| train | 14,442 | 22,632 | 10,435 | 5,418 | 6,779 |
| val | 2,101 | 3,223 | 1,551 | 663 | 1,009 |
| test | 1,904 | 3,009 | 1,391 | 716 | 902 |
| **total** | **18,447** | **28,864** | **13,377** | **6,797** | **8,690** |

```
dataset/
  data.yaml              relative paths only, no `path:` key -- survives moves
  DATASET_REPORT.md      full provenance, 8 documented limitations, the tau rationale
  images/{train,val,test}   labels/{train,val,test}
  VIDEO FOR TEST/15-MaleGlasses.mp4
  metadata/              build_info.json, dataset_manifest.csv, source_manifest.csv,
                         split_manifest.csv, quality_report.csv, class_statistics.csv,
                         test_set_lock.sha256   (3,808 hashes, re-verified after the move)
  metadata/lineage/      the pre-consolidation provenance: curated_dataset_manifest.csv
                         (50,654 rows), EDA/, duplicate_report.csv, quality_report.csv,
                         class_statistics.csv, source_statistics.csv, dataset_summary.txt,
                         source_data.yaml
```

Re-validate any time with:

```powershell
python -m src.v2.tools.validate_dataset --data dataset --full-hash
```

Last run: **PASS**, 14 gates green, 2 `[DOCUMENTED LIMITATION]`, baseline readiness READY.

### What was deleted on 2026-08-29, and what replaced it

- `dataset/` (old, 28,170 images, contaminated — experiment 1's data) — **deleted**
- `dataset_clean/` (18,445 images — experiment 2's data) — **deleted**
- `dataset_final_v1/` — **renamed to `dataset/`**

The image *content* survives: the three directories were hardlinked to each other, so
deleting two of them only dropped link counts. The lineage CSVs were moved into
`dataset/metadata/lineage/` before the delete, so the build is still traceable.

⚠️ **Consequence for old numbers.** Experiments 1 and 2 were scored on val/test splits that
no longer exist as directories. The current `dataset/` test split is exp 2's test split plus
nothing (they differ by 2 zero-box background negatives in *train*), so **exp 2's numbers are
comparable to anything scored on `dataset/` today; exp 1's are not.** Exp 1 was scored on the
larger contaminated split. To compare exp 1 against anything, re-run
`src/v2/val.py --weights checkpoints/Expi-1-imagez-384/weights/best.pt --data dataset/data.yaml`.

## Training runs — both complete

**Experiment 1** — `checkpoints/Expi-1-imagez-384/` (was `runs/v2/v2_n384` +
`info/experiment 1 baseline`).
300 epochs on the old contaminated dataset. Interrupted once at epoch 52 by a **Windows
Update forced reboot** (not a crash — Event Log confirms `MoUsoCoreWorker.exe` /
`TrustedInstaller.exe` planned restarts), resumed from `last.pt`, finished. Best epoch 294.
Its own (mixed) test split: **P 0.653 · R 0.944 · mAP50 0.907 · mAP50-95 0.541**.

**Experiment 2** — `checkpoints/Expi-2-imagez-384/` (was `runs/v2/v2_n384_clean` +
`info/experiment 2 clean-labels`).
300 epochs on what is now `dataset/`, 6.09 h, no interruption. Best epoch 283.
**Exactly one variable changed from experiment 1: the dataset.** Trained **from scratch, not
fine-tuned** from exp 1 (epoch 1 mAP50 = 0.0003 proves it); reasoning in `info/book.md` §4.1.
The exp-1-vs-exp-2 cross-evaluation on a shared split is
`checkpoints/Expi-2-imagez-384/REPORTS EXPI-2/BASELINE_for_comparison.txt`.

Both models on the **same** clean test split:

| | exp 1 | exp 2 |
|---|---|---|
| mAP50 | 0.870 | **0.900** |
| mAP50-95 | 0.473 | **0.483** |

The gain is concentrated in `open_eye` (+0.073 mAP50, the most contaminated class);
`yawning` (nothing removed) moved +0.002, behaving correctly as a control.
**mAP50-95 barely moved** — the earlier prediction that it would rise substantially was
wrong. Localisation is still capped. That is what experiment 3 attacks.

## Experiment 3 — DFL head. Code is in, run is not.

mAP50 0.900 with mAP50-95 0.483 is a ratio of 0.54; healthy is above 0.70. Experiment 2
proved annotation convention was not the cause. The head was the remaining suspect: it
regressed **4 raw scalars** per location (`reg_max = 1`) under CIoU + L1, the weakest box
representation available.

**As of 2026-08-29 the head does Distribution Focal Loss, `reg_max = 16`, by default.**

- `src/v2/models/head.py` — new `DFL` module (frozen 1×1 conv = the expectation of a
  softmax over 16 bins). `cv2` now emits `4 * reg_max` channels. `reg_max=1` still builds
  the old head: parameter counts are identical to before the change (2,501,882 train /
  2,375,157 export), and exp 2's `best.pt` loads and detects through it unchanged.
- `src/v2/losses/loss.py` — adds `L_dfl`, gain `dfl: 1.5` in `hyp.yaml`. CIoU and L1 are
  unchanged and still present, so exp 3 changes **one** thing versus exp 2.
- `--reg-max` is a training flag; the value is written into every checkpoint.

### Run it

`checkpoints/Expi-3-imagez-384/` is **already created and empty**, in the Rule 1 shape.
The three commands below fill it in and need no cleanup afterwards.

```powershell
conda activate AI-3.11
cd "C:\mini_yolo"

# 1. train  ->  checkpoints/Expi-3-imagez-384/weights/ + REPORTS EXPI-3/{training_log.csv,
#               args.yaml, hyp.yaml, plots/01_training_curves.png}
python -m src.v2.train --data dataset/data.yaml --scale n --imgsz 384 `
  --batch 64 --epochs 300 --workers 4 --seed 0 --reg-max 16 `
  --project checkpoints --name Expi-3-imagez-384 --exist-ok

# 2. report ->  REPORTS EXPI-3/{evaluation.txt, full_epoch_log.txt, plots/, demo_video/}
python -m src.v2.report --weights "checkpoints/Expi-3-imagez-384/weights/best.pt" `
  --data dataset/data.yaml --video "dataset/VIDEO FOR TEST/15-MaleGlasses.mp4" `
  --out "checkpoints/Expi-3-imagez-384/REPORTS EXPI-3"

# 3. export ->  checkpoints/Expi-3-imagez-384/weights/best.onnx
python -m src.v2.export --weights "checkpoints/Expi-3-imagez-384/weights/best.pt" `
  --imgsz 384 --name best
```

Comparable to experiment 2 directly — same dataset, same splits, same hyp, same seed.
Do **not** stack other changes into this run.
**Then apply Rule 2**: write the hypothesis, the metrics and the conclusion into
`info/book.md`. Chapter 7 is already there with the hypothesis and the two caveats; it
needs the result section appended once the run finishes.

### Two honest caveats about experiment 3

1. **`l1_loss` starts ~3.3× higher than it did with the scalar head** (measured: 7.15 vs
   2.20 at initialisation). A uniform distribution over 16 bins has expectation 7.5 cells,
   whereas the old head's bias started boxes at 1 cell. With `l1: 1.0` that term is now the
   largest in the loss early on and competes with DFL for the same gradient. Ultralytics
   drops L1 entirely when using DFL. **Keeping L1 is deliberate** — it keeps exp 3 to one
   variable — but if exp 3 underperforms, `l1: 0.0` is the first thing to try (call it
   exp 3b), not a bigger `reg_max`.
2. **This walks back a YOLO26 design decision.** YOLO26 removed DFL on purpose: the 16-bin
   softmax hurts INT8 quantisation and is brittle in TFLite/NCNN compilers. This project
   implements the rest of the YOLO26 recipe (MuSGD, STAL, ProgLoss) and is now deliberately
   out of step with it on this one point. If exp 3 does not clearly beat 0.483, revert —
   don't accumulate the portability cost for nothing.

## Known open problems

1. **Localisation ceiling.** Experiment 3 (above) is the current attempt. Unresolved until
   it runs.
2. **Train/deploy domain gap — measured.** On 600 test stills, 11.4% of detections exceed
   0.9 confidence. On the demo video, **0 of 1,886 do**. Median confidence 0.720 vs 0.647.
   Lighting is *not* the cause for that clip (its brightness sits inside the training
   range). Motion blur, compression, and continuous capture are the likely causes.
   Untested fix: fine-tune from a trained checkpoint on labelled real dashcam frames.
3. **Night driving untested.** `dataset/metadata/` reports 859 `night_or_very_low_light`
   images of 18,447 (4.7%) and 1,080 grayscale/IR (5.9%). Nothing measured so far covers
   the low-light case the project targets.
4. **Label tightness unaudited.** Published fatigue work reports ~0.94 mAP50-95 on other
   datasets vs 0.483 here. Different data, not a valid comparison — but before spending
   more experiments, hand-audit ~100 boxes for tightness. If the labels are loose, no
   architecture change will lift mAP50-95.
5. **Not subject-disjoint, and cannot be made so.** `session_id` is empty for all 50,654
   lineage rows. The same person may appear in train and test. Splits are
   near-duplicate-disjoint (0 `visual_group_id` spans a split, verified twice), which is
   not the same guarantee. Qualify every metric accordingly.

## Temporal driver monitoring — `src/v2/temporal.py`

The detector is per-frame and stateless; fatigue is not. `DriverStateMonitor` is the state
machine in between, deliberately separate from both the model and the renderer.

- **PERCLOS** over a rolling 60 s window, computed **only over frames where an eye was
  detected**. Frames with no visible eye go to a separate `coverage` figure instead of
  silently biasing the score.
- **Blink vs microsleep** by duration alone: 100–400 ms is a blink; ≥ 1.5 s is a microsleep,
  and it fires **the moment the threshold is crossed**, not when the eyes reopen. A lost
  face does not end a closure — only a confirmed `open_eye` does, so the alarm stays latched
  if the detector drops the driver mid-event.
- **Yawn** continuous duration + occurrences per minute; a run must exceed 400 ms to count.

Alert ladder SAFE → WARNING → CRITICAL; an active microsleep outranks every windowed
statistic. `src/v2/hud.py` renders it (nav panel + full-width alarm strip) and
`src/v2/report.py` writes `video/analysis.txt` per clip.

⚠️ **Thresholds are the conventional DMS literature values and are NOT validated here.**
This dataset has no drowsiness ground truth, so nothing in that file was tuned against a
label. Report its output as instrumentation, never as a diagnosis.

Tests: `python -m src.v2.tests.test_temporal` — 27 assertions over synthetic 30 fps
timelines, all passing. Run it after touching that file.

## Deployment / model size — measured, not estimated

`best.pt` is ~30 MB because it holds three copies of the network: `model` + `ema` +
`optimizer`. Only `ema` is needed for inference; the optimizer state exists so `--resume`
works, which is what rescued experiment 1.

Measured ONNX exports at `--imgsz 384`, FP16, opset 13 (all from real `export.py` runs):

| head | export params | FP16 ONNX |
|---|---|---|
| `reg_max=1` (exp 1 / exp 2) | 2,375,157 | **4.85 MB** |
| `reg_max=8` | 2,376,593 | **4.86 MB** |
| `reg_max=16` (exp 3 default) | 2,466,649 | **5.04 MB** |

**`reg_max=16` costs 0.19 MB over the DFL-free head and lands 0.24 MB above the 4.8 MB
edge budget.** `reg_max=8` buys DFL for +0.01 MB and is the in-budget alternative if that
ceiling is hard — set `--reg-max 8`, nothing else changes. The box trunk is widened to
`2 * reg_max`, not Ultralytics' `4 * reg_max`, precisely to hold this line; the wider
trunk measured ~139k more exported params.

```powershell
python -m src.v2.export --weights checkpoints\Expi-<N>-imagez-384\weights\best.pt --imgsz 384 --name best
python -m src.v2.export --weights checkpoints\Expi-<N>-imagez-384\weights\best.pt --imgsz 384 --name best --half --device cuda:0
```

⚠️ `--half` needs `--device cuda:0`; half-precision convs are not implemented on CPU.
⚠️ The training banner prints an "int8 ~x MB" figure. That is arithmetic (params ÷ 4),
**not a measured quantised export**. No int8 model has been built or evaluated. Don't quote it.
The DFL integral exports as `Softmax` + `Conv`, both natively supported by ONNX Runtime,
TensorRT and OpenVINO — no custom op. NCNN/TFLite still need `--opset 12 --raw-head`.

## Bugs fixed — do not reintroduce

- **`src/v2/val.py` and `src/v2/export.py`** loaded state dicts with `strict=True`. Every
  checkpoint carries **418 stray `total_ops` / `total_params` buffers** left behind by
  `thop` profiling in `bench.py`, so both raised `RuntimeError` on any real checkpoint —
  export was completely blocked. Both now use `strict=False`. Those keys are not weights.
  The clean fix would be to stop `bench.py` writing them into the model in the first place.
- **`src/v2/report.py`** — `plot_curves` indexed a name list with a `numpy.float32`. Fixed
  with an `int` cast.
- **Checkpoints without a `reg_max` key** are pre-experiment-3 and must be rebuilt with
  `reg_max=1`. `val.py` and `export.py` both default to 1 for exactly this reason. Defaulting
  to 16 there would silently construct the wrong architecture and load almost nothing
  (`strict=False` would hide it). Verified: exp 2's `best.pt` still loads and still detects.
- **HUD labels vanishing under the nav panel.** The panel is drawn after the boxes and grew
  taller with the telemetry rows. `draw_detection_boxes` now takes an `avoid` rect and moves
  a buried label below the box, then right of the panel.

## Reporting convention

The folder shape is **Rule 1** at the top of this file. `demo_video/analysis.txt` carries
throughput, per-class detection share, the temporal readout (PERCLOS, blinks, yawns,
microsleeps) and alert-level frame counts.

Everything except `training_log.txt` is generated in one pass by `src/v2/report.py`;
`training_log.txt` is the written argument and is authored by hand alongside the
`book.md` entry Rule 2 requires. **Do not hand-write metric files** — regenerate
so the numbers come from the checkpoint. Video HUD is `src/v2/hud.py`; the temporal logic it
displays is `src/v2/temporal.py`.

`training_log.csv` (formerly `results.csv`) gained a **`dfl_loss`** column and the four
timing columns (`epoch_seconds`, `train_seconds`, `val_seconds`, `elapsed_hours`) at
experiment 3. **Experiments 1 and 2 have none of these** — their runs predate the change and
their timings were never recorded per epoch. Anything parsing that file must tolerate the
absence, and no per-epoch timing may be quoted for exp 1 or exp 2 because none was measured.

## Historical tools — guarded, not runnable

These document how the dataset was derived and are kept for the record. All four now refuse
to run against the consolidated `dataset/` (they check for `metadata/build_info.json`) or
read the lineage CSVs from their new home:

- `src/v2/tools/make_clean_dataset.py` — built the old `dataset_clean/`
- `src/v2/tools/compare_datasets.py` — the 3-way comparison → `dataset/metadata/analysis/dataset_comparison.md`
- `src/v2/tools/threshold_study.py` — chose τ = 0.40 → `dataset/metadata/analysis/threshold_study/`
- `src/v2/tools/build_final_dataset.py` — built what is now `dataset/`

`src/v2/tools/validate_dataset.py` is **not** historical. Run it whenever the dataset is
touched.

Their outputs moved with the 2026-08-29 cleanup: the τ study and the 3-way comparison now
live in `dataset/metadata/analysis/`, beside the lineage they were derived from, so the
project root stays at four directories. `dataset/DATASET_REPORT.md` cites them from there.

## Infra notes

- **Use the conda env: `conda activate AI-3.11`** (torch 2.6.0+cu124, RTX 2000 Ada 16 GB).
  There is no in-project virtualenv — the old `env/` folder had no torch and was deleted on
  2026-08-29. Don't create another one inside the project (Rule 1).
- RAM is tight (34 GB). `src/v2/data/build.py` auto-caps `--workers` by free RAM and
  validation stays in-process (`--val-workers 0`). Don't raise `--workers` past the auto-cap
  without checking free RAM — that caused worker-death crashes previously.
- **Windows Update killed experiment 1 mid-run.** Every epoch writes `last.pt`, so recovery
  is `--resume "checkpoints/Expi-<N>-imagez-<Size>/weights/last.pt"`. Consider deferring updates before a long
  run; that is a system setting and was deliberately left for the user to change.

## What is in `info/`

- `book.md` — the engineering record. Rule 2 makes this mandatory, not optional.
- `study.md` — the beginner-facing architecture walkthrough.
- `historical_v1_cpu_baseline/` — the v1 CPU model's surviving report and figures, moved
  out of the experiment-1 folder on 2026-08-29 because v1 is not a v2 experiment. See
  `book.md` Appendix A for the unresolved conflict in that record.

## graphify knowledge graph — deleted

`graphify-out/` held a run from 2026-08-24 (615 nodes, 1090 edges, 45 communities). It was
deleted on 2026-08-29: it predated `report.py`, `hud.py`, `temporal.py`, the DFL head, the
dataset consolidation and the `checkpoints/` reorganisation, so it described a project that
no longer exists. It is regenerable — run `/graphify --update` if a graph is wanted again,
and note that its output directory would be a fifth root folder (see Rule 1).

## Things NOT to do

- Don't retrain v1 (`src/`) — frozen, weights gone, not comparable to v2.
- Don't compare an experiment-1 score against anything scored on today's `dataset/`. Exp 1's
  test split was the larger contaminated one. Re-evaluate before comparing.
- Don't stack multiple changes into one experiment.
- Don't add a P2 detection head or attention blocks to `src/v2` — deliberately excluded,
  see `src/v2/README.md` §3.
- Don't raise `--workers` past the auto-cap without checking free RAM.
- Don't re-run the historical dataset tools against `dataset/`; the guards will stop you,
  and overriding them would shadow a validated build.
- Don't quote a PERCLOS or microsleep number as a clinical finding. See the warning above.
- Don't write experiment output anywhere but `checkpoints/Expi-<N>-imagez-<Size>/`. No
  `runs/`, no `outputs/`, no new folder under `info/`. That is Rule 1.
- Don't call an experiment done before it is in `book.md`. That is Rule 2.
- Don't put a third subdirectory inside an experiment folder, or leave loose files in its
  root — the trainer already routes everything into `weights/` and `REPORTS EXPI-<N>/`.
