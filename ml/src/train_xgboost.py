"""
train_xgboost.py — Train XGBoost classifier on the preprocessed feature splits.

Usage:
  python train_xgboost.py --splits_dir ../data/splits --model_dir ../models
"""

import argparse
import os

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost malware classifier")
    parser.add_argument("--splits_dir", default="../data/splits")
    parser.add_argument("--model_dir", default="../models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading splits...")
    X_train = np.load(os.path.join(args.splits_dir, "X_train.npy"))
    X_test = np.load(os.path.join(args.splits_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.splits_dir, "y_train.npy"))
    y_test = np.load(os.path.join(args.splits_dir, "y_test.npy"))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Handle class imbalance
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost Evaluation ===")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Also save with joblib for ensemble use
    joblib.dump(model, os.path.join(args.model_dir, "xgboost_model.joblib"))

    # Save OOB predictions for stacking
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    np.save(os.path.join(args.splits_dir, "xgb_train_probs.npy"), train_probs)
    np.save(os.path.join(args.splits_dir, "xgb_test_probs.npy"), test_probs)
    print("Saved XGBoost probability outputs for stacking.")


if __name__ == "__main__":
    main()
