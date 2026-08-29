# Threshold (tau) study - eye-box scale rule

Source: `dataset/` - 28168 images, 38611 boxes.

The rule drops an IMAGE when any eye box width > tau. Both populations above the
line (legitimate eye close-up crops, and face-scale mislabels) are removed: neither
is deployment-relevant for a dashcam, and they cannot be separated automatically.

## Eye-box width histogram (the valley test)

| width bin | closed_eye | open_eye |
|---|---|---|
| 0.00-0.05 | 1322 | 993 |
| 0.05-0.10 | 1847 | 2478 |
| 0.10-0.15 | 2348 | 2193 |
| 0.15-0.20 | 1930 | 760 |
| 0.20-0.25 | 3237 | 95 |
| 0.25-0.30 | 1771 | 70 |
| 0.30-0.35 | 583 | 85 |
| 0.35-0.40 | 349 | 118 |
| 0.40-0.45 | 340 | 160 |
| 0.45-0.50 | 327 | 275 |
| 0.50-0.60 | 736 | 802 |
| 0.60-0.70 | 665 | 682 |
| 0.70-0.80 | 576 | 782 |
| 0.80-0.90 | 348 | 1015 |
| 0.90-1.01 | 1177 | 1857 |

## Multi-signal profile of eye boxes

| class | band | n | med w | med h | med area | med aspect | p10 w | p90 w |
|---|---|---|---|---|---|---|---|---|
| closed_eye | above 0.40 | 4155 | 0.703 | 0.716 | 0.4907 | 1.00 | 0.463 | 1.000 |
| closed_eye | below 0.40 | 13401 | 0.184 | 0.141 | 0.0250 | 1.28 | 0.050 | 0.283 |
| open_eye | above 0.40 | 5568 | 0.808 | 0.799 | 0.5452 | 1.00 | 0.518 | 1.000 |
| open_eye | below 0.40 | 6797 | 0.097 | 0.081 | 0.0077 | 1.05 | 0.044 | 0.169 |

## Candidate thresholds

| tau | imgs removed | imgs remaining | closed removed | open removed | closed left | open left | yawning left |
|---|---|---|---|---|---|---|---|
| 0.25 | 11925 | 16243 | 7601 | 5855 | 9955 | 6510 | 8690 |
| 0.30 | 10705 | 17463 | 5436 | 5776 | 12120 | 6589 | 8690 |
| 0.35 | 10183 | 17985 | 4650 | 5688 | 12906 | 6677 | 8690 |
| 0.40 | 9723 | 18445 | 4179 | 5568 | 13377 | 6797 | 8690 |
| 0.45 | 9221 | 18947 | 3823 | 5399 | 13733 | 6966 | 8690 |

### Source distribution of REMAINING images, per tau

| tau | Yimg | bare_numeric | dd_v1 | img | istockphoto | session | unmatched | yawn_new |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 246 | 5469 | 573 | 358 | 862 | 5 | 5201 | 3529 |
| 0.30 | 246 | 5571 | 575 | 358 | 862 | 18 | 6304 | 3529 |
| 0.35 | 246 | 5704 | 579 | 358 | 862 | 42 | 6665 | 3529 |
| 0.40 | 246 | 5976 | 593 | 358 | 862 | 77 | 6804 | 3529 |
| 0.45 | 246 | 6389 | 616 | 358 | 862 | 109 | 6838 | 3529 |

### Per-split effect, per tau

| tau | train keep | val keep | test keep |
|---|---|---|---|
| 0.25 | 12726 | 1844 | 1673 |
| 0.30 | 13679 | 1985 | 1799 |
| 0.35 | 14084 | 2044 | 1857 |
| 0.40 | 14440 | 2101 | 1904 |
| 0.45 | 14836 | 2159 | 1952 |
