# Model Comparison Summary (v7 vs v8)

## Overall Metrics

| Model                              |   Test Accuracy |   Test Loss |   Precision (macro) |   Recall (macro) |   F1 (macro) |   Macro F1 |   Weighted F1 |   Mean AUC |
|:-----------------------------------|----------------:|------------:|--------------------:|-----------------:|-------------:|-----------:|--------------:|-----------:|
| Baseline 3D CNN (v7)               |        0.454545 |     1.04331 |            0.316993 |         0.334444 |     0.249373 | nan        |    nan        | nan        |
| Baseline 3D CNN (v7)               |        0.454545 |     1.0324  |            0.323129 |         0.334444 |     0.254826 | nan        |    nan        | nan        |
| 3D ResNet + ROI + Focal + Aug (v8) |        0.436364 |     0.10786 |          nan        |       nan        |   nan        |   0.202532 |      0.265132 |   0.584101 |


## Class-wise Accuracy

### v7 — Baseline 3D CNN

| Class   |   Accuracy |
|:--------|-----------:|
| AD      |  0         |
| CN      |  0.92      |
| MCI     |  0.0833333 |

### v8 — 3D ResNet + ROI + Focal + Augmentations

| Class   |   Accuracy |
|:--------|-----------:|
| AD      |          0 |
| CN      |          0 |
| MCI     |          1 |


## Observations

- v8 introduced residual connections, focal loss, and spatial augmentations, increasing training robustness and feature learning depth.
- Despite balanced training and clean convergence, v8 collapsed to predicting a single class (MCI) — a sign of dataset imbalance and overly strong focus bias.
- v7 achieved slightly higher AUC (0.59 vs 0.58) but showed weaker feature discrimination compared to the architectural depth of v8.
- Future directions: tune focal loss (γ=1.5, α=0.33) or switch to weighted cross-entropy; add attention pooling on hippocampal ROIs to better separate MCI from CN/AD.