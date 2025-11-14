# Model v10 – 3D ResNet + ROI + Class Weights

**Task:** 3-class classification of Alzheimer's status from structural MRI.

**Classes:** AD, CN, MCI (multi-class softmax)


## Data

- Total labeled scans: 639
- Train / Val / Test (scans): 511 / 64 / 64

Class distribution in full labeled set:
- AD: 72 scans
- CN: 296 scans
- MCI: 271 scans

## Model & Training

- Architecture: Small 3D ResNet with three residual stages (32 → 64 → 128 filters)
- Input ROI: 80×80×80 center crop, intensity normalized to [0, 1]
- Augmentations: 3D flips, 90° rotations (XY/YZ/XZ planes), mild Gaussian noise
- Loss: Categorical cross-entropy with class weights (to rebalance AD vs CN/MCI)
- Optimizer: Adam (lr=1e-4 with ReduceLROnPlateau)
- Callbacks: Early stopping (patience=10, restore best), ModelCheckpoint on val_loss

## Test Performance

- **Test accuracy:** 0.500
- **Test loss:** 0.965894
- **Macro ROC-AUC:** 0.664

### Per-class precision / recall / F1

- **AD** → Precision: 0.167, Recall: 0.125, F1: 0.143 (Support: 8)
- **CN** → Precision: 0.520, Recall: 0.897, F1: 0.658 (Support: 29)
- **MCI** → Precision: 0.625, Recall: 0.185, F1: 0.286 (Support: 27)

### Confusion Matrix (rows = true, columns = predicted)

| True \ Pred | AD | CN | MCI |
|---|---|---|---|
| AD | 1 | 6 | 1 |
| CN | 1 | 26 | 2 |
| MCI | 4 | 18 | 5 |

### Saved Artifacts

- Training curves: `v10_training_curves.png`
- Confusion matrix: `v10_confusion_matrix.png`
- ROC curves: `v10_roc_curves.png`
- Classification report (CSV): `v10_classification_report.csv`
- Performance summary (CSV): `v10_performance_summary.csv`
- Best weights: `v10_best_weights.keras`