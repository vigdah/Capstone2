"""
ensemble.py — Stack XGBoost + MLP outputs using a Logistic Regression meta-learner.

Reads the saved probability outputs from train_xgboost.py and train_mlp.py,
trains a meta-learner, and saves it for use in export_onnx.py.

Usage:
  python ensemble.py --splits_dir ../data/splits --model_dir ../models
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def build_meta_features(xgb_probs: np.ndarray, mlp_probs: np.ndarray) -> np.ndarray:
    """Stack XGBoost and MLP probability outputs as meta-features."""
    return np.column_stack([xgb_probs, mlp_probs])


def main():
    parser = argparse.ArgumentParser(description="Train stacking ensemble meta-learner")
    parser.add_argument("--splits_dir", default="../data/splits")
    parser.add_argument("--model_dir", default="../models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading base model probability outputs...")
    xgb_train = np.load(os.path.join(args.splits_dir, "xgb_train_probs.npy"))
    xgb_test = np.load(os.path.join(args.splits_dir, "xgb_test_probs.npy"))
    mlp_train = np.load(os.path.join(args.splits_dir, "mlp_train_probs.npy"))
    mlp_test = np.load(os.path.join(args.splits_dir, "mlp_test_probs.npy"))
    y_train = np.load(os.path.join(args.splits_dir, "y_train.npy"))
    y_test = np.load(os.path.join(args.splits_dir, "y_test.npy"))

    # Build meta-feature matrices
    X_meta_train = build_meta_features(xgb_train, mlp_train)
    X_meta_test = build_meta_features(xgb_test, mlp_test)

    print(f"Meta-feature shape: train={X_meta_train.shape}, test={X_meta_test.shape}")

    # Train meta-learner
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_learner.fit(X_meta_train, y_train)

    # Evaluate
    y_pred = meta_learner.predict(X_meta_test)
    y_prob = meta_learner.predict_proba(X_meta_test)[:, 1]

    print("\n=== Ensemble Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Compare base models
    print("\n--- Base Model Comparison ---")
    for name, probs in [("XGBoost", xgb_test), ("MLP", mlp_test)]:
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        print(f"{name}: AUC={auc:.4f}")

    # Save meta-learner
    path = os.path.join(args.model_dir, "meta_learner.joblib")
    joblib.dump(meta_learner, path)
    print(f"\nMeta-learner saved to {path}")

    # Save ensemble weights for simple weighted average alternative
    # (weights derived from individual AUC scores)
    xgb_auc = roc_auc_score(y_test, xgb_test)
    mlp_auc = roc_auc_score(y_test, mlp_test)
    total = xgb_auc + mlp_auc
    weights = {"xgboost": xgb_auc / total, "mlp": mlp_auc / total}
    import json
    with open(os.path.join(args.model_dir, "ensemble_weights.json"), "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Ensemble weights: {weights}")


if __name__ == "__main__":
    main()
