"""
features.py — Feature engineering for the malware detection model.

Reads processed.csv (from ingest.py), engineers features,
and outputs train/test splits as numpy arrays.

No scaling applied — XGBoost is tree-based and does not require
feature normalization. Raw values are saved directly.

Usage:
  python features.py --input ../data/processed.csv --output_dir ../data/splits
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COL = "is_malicious"
DROP_COLS = ["Label", "is_malicious"]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_split(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows.")

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int32)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Class balance: benign={int((y==0).sum())}, malicious={int((y==1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Feature engineering and train/test split")
    parser.add_argument("--input", default="../data/processed.csv")
    parser.add_argument("--output_dir", default="../data/splits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols = load_and_split(args.input)

    # Save raw (unscaled) splits — XGBoost does not require normalization
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    # Save feature column names (used by export_onnx.py to determine input_dim)
    with open(os.path.join(args.output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nSaved splits to {args.output_dir}/")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
