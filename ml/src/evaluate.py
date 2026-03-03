"""
evaluate.py — Comprehensive evaluation of all trained models.

Prints accuracy, precision, recall, F1, AUC-ROC and generates plots.

Usage:
  python evaluate.py --splits_dir ../data/splits --model_dir ../models --plots_dir ../plots
"""

import argparse
import os

import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def plot_roc_curves(models_probs: dict, y_test: np.ndarray, output_dir: str):
    plt.figure(figsize=(8, 6))
    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()
    print("Saved roc_curves.png")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--splits_dir", default="../data/splits")
    parser.add_argument("--model_dir", default="../models")
    parser.add_argument("--plots_dir", default="../plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    y_test = np.load(os.path.join(args.splits_dir, "y_test.npy"))

    models_probs = {}

    # Load XGBoost probabilities
    xgb_path = os.path.join(args.splits_dir, "xgb_test_probs.npy")
    if os.path.exists(xgb_path):
        models_probs["XGBoost"] = np.load(xgb_path)

    # Load MLP probabilities
    mlp_path = os.path.join(args.splits_dir, "mlp_test_probs.npy")
    if os.path.exists(mlp_path):
        models_probs["MLP"] = np.load(mlp_path)

    # Load ensemble meta-learner
    meta_path = os.path.join(args.model_dir, "meta_learner.joblib")
    if os.path.exists(meta_path) and "XGBoost" in models_probs and "MLP" in models_probs:
        meta = joblib.load(meta_path)
        X_meta = np.column_stack([models_probs["XGBoost"], models_probs["MLP"]])
        models_probs["Ensemble"] = meta.predict_proba(X_meta)[:, 1]

    if not models_probs:
        print("No model outputs found. Run training scripts first.")
        return

    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    for name, probs in models_probs.items():
        y_pred = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))
        print(f"AUC-ROC: {auc:.4f} | AP: {ap:.4f}")

        plot_confusion_matrix(y_test, y_pred, name, args.plots_dir)

    plot_roc_curves(models_probs, y_test, args.plots_dir)

    print("\nEvaluation complete. Plots saved to", args.plots_dir)


if __name__ == "__main__":
    main()
