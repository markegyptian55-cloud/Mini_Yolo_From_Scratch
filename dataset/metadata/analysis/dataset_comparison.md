# Dataset comparison: `dataset/` vs `dataset_clean/` vs `candidate_final`

Candidate rule: drop image if any eye-class box width > **tau = 0.4** (chosen in `analysis/threshold_study/`).

Domain figures are sampled (see `n` column), not exhaustive.

### dataset/ (Experiment 1)

- images: **28170**  |  boxes: **38611**
- split: train=22215, val=3082, test=2873
- boxes per class: closed_eye=17556, open_eye=12365, yawning=8690
- images containing class: closed_eye=11490, open_eye=9193, yawning=8664

| class | boxes | med width | p10 w | p90 w | med aspect |
|---|---|---|---|---|---|
| closed_eye | 17556 | 0.220 | 0.069 | 0.750 | 1.18 |
| open_eye | 12365 | 0.175 | 0.053 | 0.980 | 1.02 |
| yawning | 8690 | 0.729 | 0.159 | 1.000 | 1.13 |

- source distribution: bare_numeric=14429, unmatched=6897, yawn_new=3531, dd_v1=931, session=916, istockphoto=862, img=358, Yimg=246

- near-duplicate groups: 24424, intra-split pairs: 3746 (7492 images), **cross-split groups: 0**

| split | n | grayscale/IR % | dark(<60) % | brightness mean | p5 | p95 | blur median |
|---|---|---|---|---|---|---|---|
| train | 400 | 10.2 | 8.0 | 114 | 52 | 196 | 127 |
| val | 400 | 5.8 | 9.8 | 109 | 48 | 183 | 126 |
| test | 400 | 4.0 | 7.0 | 118 | 54 | 196 | 123 |

### dataset_clean/ (Experiment 2)

- images: **18445**  |  boxes: **28864**
- split: train=14440, val=2101, test=1904
- boxes per class: closed_eye=13377, open_eye=6797, yawning=8690
- images containing class: closed_eye=7335, open_eye=3625, yawning=8664

| class | boxes | med width | p10 w | p90 w | med aspect |
|---|---|---|---|---|---|
| closed_eye | 13377 | 0.184 | 0.050 | 0.283 | 1.28 |
| open_eye | 6797 | 0.097 | 0.044 | 0.169 | 1.05 |
| yawning | 8690 | 0.729 | 0.159 | 1.000 | 1.13 |

- source distribution: unmatched=6804, bare_numeric=5976, yawn_new=3529, istockphoto=862, dd_v1=593, img=358, Yimg=246, session=77

- near-duplicate groups: 16006, intra-split pairs: 2439 (4878 images), **cross-split groups: 0**

| split | n | grayscale/IR % | dark(<60) % | brightness mean | p5 | p95 | blur median |
|---|---|---|---|---|---|---|---|
| train | 400 | 5.8 | 6.0 | 120 | 57 | 199 | 132 |
| val | 400 | 4.8 | 6.2 | 120 | 52 | 197 | 172 |
| test | 400 | 5.8 | 4.2 | 125 | 65 | 199 | 140 |

### candidate_final (proposed)

- images: **18447**  |  boxes: **28864**
- split: train=14442, val=2101, test=1904
- boxes per class: closed_eye=13377, open_eye=6797, yawning=8690
- images containing class: closed_eye=7335, open_eye=3625, yawning=8664

| class | boxes | med width | p10 w | p90 w | med aspect |
|---|---|---|---|---|---|
| closed_eye | 13377 | 0.184 | 0.050 | 0.283 | 1.28 |
| open_eye | 6797 | 0.097 | 0.044 | 0.169 | 1.05 |
| yawning | 8690 | 0.729 | 0.159 | 1.000 | 1.13 |

- source distribution: unmatched=6804, bare_numeric=5976, yawn_new=3531, istockphoto=862, dd_v1=593, img=358, Yimg=246, session=77

- near-duplicate groups: 16007, intra-split pairs: 2440 (4880 images), **cross-split groups: 0**

| split | n | grayscale/IR % | dark(<60) % | brightness mean | p5 | p95 | blur median |
|---|---|---|---|---|---|---|---|
| train | 400 | 5.8 | 4.2 | 123 | 66 | 204 | 149 |
| val | 400 | 4.8 | 6.2 | 120 | 52 | 197 | 172 |
| test | 400 | 5.8 | 4.2 | 125 | 65 | 199 | 140 |

## Equivalence check: candidate_final vs dataset_clean

- identical image selection: **False**
- in candidate only: 2  |  in dataset_clean only: 0

## Decision table

| category | KEEP / REMOVE | reason | evidence |
|---|---|---|---|
| Eye box width <= 0.4 on a face/scene | KEEP | deployment case: dashcam sees faces at scene scale | band grid 0.30-0.40 is mostly correct eye-level boxes |
| Eye box width > 0.4, eye close-up crop | REMOVE | not deployment-relevant; a dashcam never frames a single eye | band grid 0.40-0.50 is ~15/16 eye crops |
| Eye box width > 0.4, whole-face box | REMOVE | genuine mislabel (incl. faces in sunglasses tagged open_eye) | visual audit of 30 flagged images |
| yawning boxes (any scale) | KEEP | face-scale is the correct convention for this class | 0 yawning boxes removed at every tau |
| Intra-split near-duplicate pairs | KEEP | redundancy, not leakage; preserve information first | 0 cross-split groups, verified twice |
| Existing split assignment | KEEP | already leakage-free; reshuffling would destroy a verified property | visual_group_id spans 0 splits |
| Grayscale/IR images | KEEP | real low-light/IR diversity, scarce and valuable | train 11.2% vs test 5.2% (documented mismatch) |
