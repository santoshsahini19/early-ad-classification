# 3D CNN Model Performance Summary

## Model Comparison

| Model             |   Test Accuracy |   Test Loss |   Precision (macro) |   Recall (macro) |   F1 (macro) |
|:------------------|----------------:|------------:|--------------------:|-----------------:|-------------:|
| Base 3D CNN       |        0.454545 |     1.04331 |            0.316993 |         0.334444 |     0.249373 |
| Fine-tuned 3D CNN |        0.454545 |     1.0324  |            0.323129 |         0.334444 |     0.254826 |


## Class-wise Accuracy Breakdown

| Class   |   Accuracy |
|:--------|-----------:|
| AD      |  0         |
| CN      |  0.92      |
| MCI     |  0.0833333 |


## Notes

- Training and fine-tuning conducted on ADNI dataset subset (CN vs MCI vs AD).
- Fine-tuned model reloaded from best checkpoint and optimized with smaller learning rate (1e-5).
- Each model trained using early stopping (patience=10) to prevent overfitting.
- Input volumes normalized and resized to 96×96×96 voxels.
- Accuracy metrics computed on the held-out test set (n ≈ 102 scans).