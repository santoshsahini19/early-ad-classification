# DeepSurv and Hybrid Conversion Model Report

## Survival Model (DeepSurv-style, any AD conversion)
- Features used (2): baseline_dx_num, n_visits
- Training epochs: 25 (with early stopping)
- C-index (train): **0.645**
- C-index (test): **0.599**
- Loss curves: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/loss_plot.png`
- Learning-rate curve: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/lr_plot.png`
- Risk distribution: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/risk_distribution.png`
- Kaplan–Meier by risk tertiles: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/km_by_risk_group.png`
- Risk scores CSV: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/subject_risk_scores.csv`
- C-index summary: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/cindex_summary.txt`

## SHAP Explainability
- SHAP bar plot: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/shap_feature_importance.png`
- SHAP beeswarm: `/Users/addieclark/Desktop/Harrisburg/Fall 2025/ANLY 715 - Applied Multivariate Data Analysis/Course Project/early-ad-classification/AC_Code/survival_model_v1_tracking/shap_beeswarm.png`

## Hybrid Conversion Model (Tabular + CNN AD-likeness)
- ROC-AUC (24m conversion): **1.000**
- Features: baseline_dx_num, n_visits plus `AD_likeness`
- Class balance (0=stable, 1=converted):
y_conv24
0    20
1     6
