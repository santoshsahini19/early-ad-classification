# 3D ResNet + ROI + Focal Loss + Augmentations (Model v8)

## Overall Performance Summary

| Model                         |   Test Accuracy |   Test Loss |   Macro F1 |   Weighted F1 |   Mean AUC |
|:------------------------------|----------------:|------------:|-----------:|--------------:|-----------:|
| 3D ResNet + ROI + Focal + Aug |        0.436364 |     0.10786 |   0.202532 |      0.265132 |   0.584101 |


## Class-wise Accuracy Breakdown

| Class   |   Accuracy |
|:--------|-----------:|
| AD      |          0 |
| CN      |          0 |
| MCI     |          1 |


## Notes

- Model architecture: Residual 3D CNN with ROI cropping (80³), Focal Loss, and spatial augmentations (Cutout + Rotation + Flips).
- Focused on hippocampal-centered ROI for improved diagnostic sensitivity.
- Augmentations increase robustness to scan variability and motion artifacts.
- Focal Loss (γ=2.0, α=0.25) rebalances class gradients to better learn AD and MCI categories.
- Early stopping applied (patience=10) with checkpoint saving at lowest validation loss.
- Accuracy, F1, and AUC computed on held-out ADNI test set (n ≈ 55).