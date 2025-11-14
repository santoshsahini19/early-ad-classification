"""
evaluate.py — Version 2
Evaluate a trained model on the TEST SET.
Produces:
  - classification report JSON
  - confusion matrix PNG
  - ROC curves PNG
  - accuracy JSON
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize

from dataset import get_test_loader
from models import Baseline3DCNN, Deep3DCNN, ResNet3D18


# Load model from config
def load_model(run_dir, device):
    # Load config.json
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model_name = cfg["model_name"]

    if model_name == "Baseline3DCNN":
        model = Baseline3DCNN(num_classes=3)
    elif model_name == "Deep3DCNN":
        model = Deep3DCNN(num_classes=3)
    elif model_name == "resnet3d18":
        model = ResNet3D18(num_classes=3)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load best weights
    weight_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    return model, cfg


def predict(model, loader, device):
    softmax = torch.nn.Softmax(dim=1)

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            prob = softmax(outputs)
            preds = torch.argmax(prob, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion_matrix(y_true, y_pred, out_path):
    labels = ["AD", "MCI", "CN"]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true, y_prob, out_path):
    labels = ["AD", "MCI", "CN"]
    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    plt.figure(figsize=(7,6))
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc_val:.2f})")

    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    #RUN_DIR = "runs/deep3d_v2"   # <--- need to change based on current version
    RUN_DIR = "checkpoints"
    EXPERIMENT_DIR = "checkpoints"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(EXPERIMENT_DIR, device)

    test_loader = get_test_loader()

    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # Save report
    report = classification_report(
        y_true, y_pred, target_names=["AD","MCI","CN"], output_dict=True
    )

    # create directory if not exists
    os.makedirs(os.path.join(RUN_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(RUN_DIR, "metrics/classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Save accuracy
    acc = accuracy_score(y_true, y_pred)
    with open(os.path.join(RUN_DIR, "metrics/accuracy.json"), "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

    # Plots
    os.makedirs(os.path.join(RUN_DIR, "metrics"), exist_ok=True)

    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RUN_DIR, "metrics/confusion_matrix.png")
    )

    plot_roc(
        y_true, y_prob,
        os.path.join(RUN_DIR, "metrics/roc_curves.png")
    )

    print("\nEvaluation complete!")
    print(f"Metrics saved in: {RUN_DIR}/metrics/")


if __name__ == "__main__":
    main()
