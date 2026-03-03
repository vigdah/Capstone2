"""
features.py — Feature engineering for the malware detection model.

Reads processed.csv (from ingest.py), engineers features,
and outputs train/test splits as numpy arrays.

Usage:
  python features.py --input ../data/processed.csv --output_dir ../data/splits
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def main():
    parser = argparse.ArgumentParser(description="Feature engineering and train/test split")
    parser.add_argument("--input", default="../data/processed.csv")
    parser.add_argument("--output_dir", default="../data/splits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_cols = load_and_split(args.input)

    # Fit scaler on training data only
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save splits
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(args.output_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    # Save scaler for inference-time preprocessing
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    # Save feature column names for documentation
    with open(os.path.join(args.output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nSaved splits and scaler to {args.output_dir}/")
    print(f"  X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
    print(f"  Feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
