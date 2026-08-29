# MiniYOLO-v2 (MiniYOLO-E) — edge-first driver-monitoring detector

Built from scratch. Anchor-free, **DFL-free**, **NMS-free**, dual-head, trained with
task-aligned assignment + STAL + ProgLoss + MuSGD.

v1 (`src/`) is untouched so you can A/B against the 0.5817 mAP@50 baseline.

---

## 1. Run the training

> **The dataset must live on the SSD.** `D:` is a spinning HDD (HGST HUS722T1TALA604).
> Mosaic pulls 4 random images per sample, so a batch of 64 is ~256 random reads; on the
> HDD that produced 8.9 s iterations with the GPU idle. A copy is already staged at
> `C:\ssd projects\BaSuny_mini_yolo_data` (1.4 GB) with its `data.yaml` fixed up.
> `D:\...\mini_yolo\dataset` is untouched.
>
> **RAM is tight (34 GB total, ~11 GB free with your usual apps open) and the pagefile
> sits on the `D:` HDD.** Each DataLoader worker on Windows is a full process that
> re-imports torch (~1.4 GB RSS); enough of them plus a browser can push the system
> into HDD-backed swap, which stalls allocations until a worker dies
> (`DataLoader worker exited unexpectedly` / `cv2.error: Insufficient memory`). The
> trainer now auto-caps `--workers` by free RAM and keeps validation in-process
> (`--val-workers 0`) so it doesn't hold extra processes alive for the whole run. If you
> still hit this: lower `--workers`, close memory-heavy apps, or move the pagefile to
> the `C:` SSD (System Properties → Advanced → Performance → Advanced → Virtual memory).

Open a new terminal, activate the conda env that actually has torch, and run:

```powershell
conda activate AI-3.11
cd "D:\project\Driver project\BaSuny\mini_yolo"

python -m src.v2.train `
  --data "C:\ssd projects\BaSuny_mini_yolo_data\data.yaml" `
  --scale n `
  --imgsz 384 `
  --epochs 300 `
  --batch 64 `
  --workers 12 `
  --optimizer musgd `
  --name v2_n384
```

Note the local `env\` folder has **no torch installed** — use the conda env.

Outputs land in `runs/v2/v2_n384/`:
`weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png`, `hyp.yaml`, `args.yaml`.

Resume after an interruption:

```powershell
python -m src.v2.train --resume runs/v2/v2_n384/weights/last.pt --name v2_n384 --epochs 300
```

Evaluate, export, run live:

```powershell
python -m src.v2.val     --weights runs/v2/v2_n384/weights/best.pt --split test
python -m src.v2.val     --weights runs/v2/v2_n384/weights/best.pt --split test --no-e2e   # NMS path, for comparison
python -m src.v2.export  --weights runs/v2/v2_n384/weights/best.pt --imgsz 384 --opset 13              # ORT / TensorRT / OpenVINO
python -m src.v2.export  --weights runs/v2/v2_n384/weights/best.pt --imgsz 384 --opset 12 --raw-head    # NCNN / TFLite
python -m src.v2.predict --weights runs/v2/v2_n384/weights/best.pt --source 0 --show        # webcam
```

### Export modes

| flag | output | graph ops | runtimes |
| --- | --- | --- | --- |
| default | `(1, 300, 6)` `[x,y,w,h,conf,cls]` | adds `TopK`, `GatherElements`, `Mod` | ONNX Runtime, TensorRT, OpenVINO |
| `--raw-head` | `(1, 4+nc, A)` xywh in px + sigmoid class scores | `Conv MaxPool Resize Sigmoid Add Mul Sub Div Concat Slice Split Reshape` only | **NCNN, TFLite**, plus all of the above |

With `--raw-head` the host does the selection: `score = max over classes`, keep
`score > conf`. Still **no NMS** — the one-to-one head is already non-redundant.
Measured on ONNX Runtime CPU at 384: **4.6 ms/image** either way.

---

## 2. Why the v1 model plateaued at 0.58

| v1 | Problem |
| --- | --- |
| `yolo_loss.py` assigned every GT to the centre cell **+ 4 neighbours at all 3 strides** | ~15 positives per object with no scale filtering and no quality weighting. A stride-32 cell and a stride-8 cell were treated as equally good matches for a 40 px eye. This alone caps mAP. |
| Objectness branch + focal loss on hard 1.0 class targets | Obsolete since YOLOv8. Focal loss fights the assigner instead of helping it. |
| `exp(tw) * stride` box regression | Unbounded and unstable; needs `clamp(-5, 5)` to survive, which kills gradients. |
| Triple Python loop over batch × GT × scale in the loss | The loss was a measurable share of step time. |
| No EMA, no warmup, no mosaic, no multi-scale, `cudnn.deterministic=True`, `NUM_WORKERS=0` | ~30% throughput lost to determinism, dataloader starved the GPU. |
| Head = 4 full-width 3×3 convs per scale | Most parameters sat in the head, not the feature extractor. |
| ImageNet mean/std normalisation with no pretrained weights | Pointless, and it complicates INT8 quantisation. |

## 3. What v2 does instead

### Architecture
```
Input 384×384
  MiniDarknetV2   Conv-s2 stem → 4 CSP stages (C2f) → SPPF        → P3/8 P4/16 P5/32
  MiniPANv2       PAN-FPN, YOLOv8 layout (no 1×1 reduction convs)  → N3 N4 N5
  DualDetect      box branch : 2× Conv3×3 → Conv1×1 → 4 scalars (ltrb, reg_max=1)
                  cls branch : 2× depthwise-separable → Conv1×1 → nc logits
                  ×2 (one-to-many + one-to-one), one-to-one survives export
```

| Trick | Source | Why it is here |
| --- | --- | --- |
| **DFL-free regression** (`reg_max=1`, 4 raw scalars) | YOLO26 | DFL's 16-bin softmax + matmul is brittle across ONNX/TFLite/NCNN compilers and blocks clean INT8. Replaced by direct ltrb + an L1 term. |
| **NMS-free one-to-one head** | YOLOv10 → YOLO26 | Fixed `(B, 300, 6)` output, no suppression pass. Removes 1–4 ms/frame and the messiest part of export. |
| **Dual assignment** (o2m topk=10, o2o topk=1) | YOLOv10 | The o2m branch supplies dense supervision during training; only the o2o branch is deployed, so inference cost is unchanged. |
| **ProgLoss** `L = α·L_o2m + (1−α)·L_o2o`, α: 0.8 → 0.1 linear | YOLO26 | Early training needs the recall of dense matching; late training must optimise the head that actually ships. Unlike YOLOv10, the o2o branch is **not** detached from the trunk — otherwise the hand-over could not reach the backbone. |
| **TAL** `align = cls^0.5 · CIoU^6.0`, IoU-normalised soft targets | TOOD / YOLOv8 | Replaces v1's static geometric assignment. This is the single biggest expected gain. |
| **STAL** — GT sides < 8 px inflated to 16 px for candidate selection only | YOLO26 | 2.2% of your boxes are under 16 px and can end up with zero anchors after discretisation, contributing no gradient at all. Regression still targets the true box. |
| **MuSGD** (Muon-orthogonalised direction blended into SGD) | YOLO26 | Measured on a 64-image overfit probe at 60 epochs: mAP50-95 **0.805** vs AdamW 0.712 vs SGD 0.684. |
| **Depthwise-separable cls head** | YOLO11 | Head is now ~5% of parameters instead of dominating them. |
| **EMA** (decay 0.9999, τ=2000) | YOLOv5+ | Was the top recommendation in your own Train-1 report and was never implemented. |
| No attention, no DFL, no exotic ops | deployment constraint | Graph is Conv / BN / SiLU / MaxPool / Concat / Upsample-nearest / topk only — TensorRT, OpenVINO, NCNN and TFLite all take it. |

### Deliberately **not** included
* **P2 head (stride 4).** Standard advice for small objects, wrong for this dataset: mean box is 41% of frame width, median `closed_eye` is 141×114 px, median `yawning` is 466×383 px. A P2 branch would add ~11% FLOPs for objects that do not exist here.
* **Area attention / C2PSA (YOLOv12, YOLO11).** Real accuracy gains, but attention blocks quantise badly to INT8 and are the first thing to break in NCNN/TFLite conversion. You chose mobile as a deploy target.
* **Copy-paste augmentation.** Needs segmentation masks; box-level copy-paste on faces produces garbage.

---

## 4. Model sizes (measured, `--imgsz 384`)

| scale | deploy params | GFLOPs | FP32 | INT8 (est.) |
| --- | --- | --- | --- | --- |
| `p` | 0.59 M | 0.49 | 2.4 MB | 0.6 MB |
| `t` | 1.34 M | 1.10 | 5.4 MB | 1.3 MB |
| **`n`** (default) | **2.38 M** | **2.00** | **9.5 MB** | **2.4 MB** |
| `s` | 6.67 M | 5.53 | 26.7 MB | 6.7 MB |

Reference: YOLO26n is 2.4 M params / 5.4 GFLOPs at 640. Same parameter budget, 2.7× less
compute, because 384 is the right resolution for this data.

Train `n` first. If you need a smaller model afterwards, retrain `t` — do not prune `n`.

---

## 5. Hyperparameters — `src/v2/cfg/hyp.yaml`

Anchored on the published YOLO26n COCO recipe, adjusted for 22.2 k images / 3 classes /
large objects / a car cabin.

**Optimisation.** `lr0=0.005`, `lrf=0.05`, `momentum=0.937`, `weight_decay=0.0005`,
3 warm-up epochs, cosine decay, nominal batch 64.
YOLO26n used `lr0=0.0054`, `lrf=0.0495`, `momentum=0.947` at batch 128 — same regime.
`--optimizer adamw` automatically swaps in `lr0=0.001`.

**Loss gains.** `box=7.5` (CIoU), `cls=0.7` (BCE on soft TAL targets), `l1=1.0`.
YOLO26n publishes `5.63 / 0.56 / 9.04`, but its distance term is on *normalised*
distances while ours is in feature-cell units, so the gain is not transferable — 1.0
puts the L1 term at roughly the same magnitude as the CIoU term at convergence here.

**Augmentation** (the domain-specific part):

| setting | value | reason |
| --- | --- | --- |
| `mosaic` | 0.85 | main source of scale diversity; also creates truncated faces, which is realistic for a dash camera |
| `close_mosaic` | 20 | last 20 epochs train on clean images — always worth 1–2 mAP |
| `mixup` | 0.10 | low; blended faces are unnatural |
| `degrees` | **10.0** | head tilt is a real signal in a cabin. COCO recipes use ~0–1 |
| `hsv_v` | **0.5** | cabin lighting swings between tunnel, direct sun and night |
| `gray` | **0.10** | driver-monitoring cameras are frequently IR/monochrome |
| `blur` | 0.05 | motion blur and defocus dominate live video artefacts |
| `erasing` | 0.25 | hand over the mouth, sunglasses, steering wheel |
| `scale` | 0.5 | ±50% |
| `fliplr` / `flipud` | 0.5 / 0.0 | never flip a face upside down |
| multi-scale | 320–512 | trains one set of weights that holds up across deploy resolutions |

---

## 6. Expected behaviour

* ~347 iterations/epoch at batch 64.
* The first 3 epochs are warm-up; mAP stays near zero — that is normal.
* `results.csv` logs `loss_o2m` and `loss_o2o` separately. Early on `loss_o2o` is higher;
  they should converge toward each other as α anneals. If `loss_o2o` diverges upward
  while `loss_o2m` falls, lower `prog_alpha_final`.
* Validation runs through the **NMS-free** head, so reported mAP is the deployed number,
  not an NMS-inflated one. `--no-e2e` shows the NMS path for comparison; a gap under
  ~1 mAP is expected and healthy.
* Early stopping: `--patience 60` (`0` disables).

## 7. If it plateaus — what to change, in order

1. **`--scale s` at the same settings.** 6.7 M params / 5.5 GFLOPs. If `s` is not clearly
   better than `n`, the ceiling is the data, not the model — stop tuning and go label.
2. **`--imgsz 512`.** Recovers the small tail (p10 box area = 45 px). Costs ~1.8×.
3. **`box: 10.0`** in `hyp.yaml` if `mAP50` is high but `mAP50-95` lags — that gap is pure
   localisation, and the CIoU gain is the lever.
4. **`tal_topk: 13`** if recall is low, `tal_topk: 7` if precision is low.
5. **`prog_alpha_final: 0.05`** if the NMS-free number trails the `--no-e2e` number by
   more than ~1.5 mAP — the one-to-one head needs a larger share of late training.
6. **`degrees: 15`, `erasing: 0.35`** if train loss keeps dropping while val mAP flattens.
7. **Teacher → student distillation.** Train `s`, then distil into `n` by adding a KL term
   on the one-to-one class logits plus an L1 on the ltrb distances of the teacher's
   positives. This is the standard last 1–2 mAP for a nano model and is the single
   biggest remaining lever once the recipe above is exhausted. Not implemented here —
   it needs its own training script and doubles the compute budget.

Do **not** add a P2 head, and do not add attention blocks: see §3 for why both are wrong
for this dataset and this deployment target.

## 8. File map

```
src/v2/
  cfg/hyp.yaml          all hyperparameters
  models/blocks.py      Conv / DWConv / Bottleneck / C2f / SPPF  (+ BN fusion)
  models/backbone.py    MiniDarknetV2
  models/neck.py        MiniPANv2
  models/head.py        DualDetect  (o2m + o2o, DFL-free, NMS-free postprocess)
  models/yolo.py        assembly, scales p/t/n/s, fuse(), export_ready()
  losses/tal.py         TaskAlignedAssigner + STAL
  losses/loss.py        CIoU + BCE(soft) + L1, ProgLoss blending
  data/augment.py       letterbox, perspective, hsv, gray, blur, cutout, mixup
  data/dataset.py       label cache, mosaic, item pipeline
  data/build.py         dataloaders
  engine/trainer.py     warmup, cosine, EMA, AMP, multi-scale, close_mosaic, ckpt
  engine/validator.py   mAP50 / mAP50-95 + latency
  utils/                boxes, metrics, ema, optim (Muon/MuSGD), plots, general
  train.py val.py export.py predict.py
```

## 9. Sources

- [YOLO26 docs](https://docs.ultralytics.com/models/yolo26/) ·
  [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/) ·
  [Ultralytics YOLO26 paper](https://arxiv.org/html/2606.03748v1) (STAL surrogate-box and ProgLoss formulas) ·
  [YOLO26 benchmarking paper](https://arxiv.org/html/2509.25164v3)
- [ProgLoss / STAL / MuSGD explainer](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd)
- [YOLOv10 — consistent dual assignments](https://docs.ultralytics.com/models/yolov10/)
- [Muon optimizer](https://kellerjordan.github.io/posts/muon/)
- [YOLOv9 PGI](https://docs.ultralytics.com/models/yolov9/) (evaluated, not adopted — see §3)
