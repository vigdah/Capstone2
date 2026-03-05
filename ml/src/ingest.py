"""
ingest.py — Load and clean the CICAndMal2017 dataset (or any CSV-based Android malware dataset).

Expected input: CSV files with network-traffic-based features per app per time window.
Expected columns (CICAndMal2017 format):
  - Flow features: Bwd Packet Length Max, Flow Bytes/s, Flow Packets/s, etc.
  - Label column: "Label" with values like "Benign", "Adware", "Ransomware", etc.

Usage:
  python ingest.py --data_dir ../data/raw --output ../data/processed.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


LABEL_COL = "Label"
BENIGN_LABELS = {"Benign", "benign", "BENIGN"}

# 9 CICAndMal2017 features that can be computed from Android NetworkStatsManager data.
# Order matches the feature vector produced by FeatureExtractor.kt.
FEATURE_COLUMNS = [
    "Total Fwd Packets",           # → packets_sent
    "Total Backward Packets",      # → packets_received
    "Total Length of Fwd Packets", # → bytes_sent
    "Total Length of Bwd Packets", # → bytes_received
    "Flow Bytes/s",                # → (bytes_sent + bytes_received) / window_sec
    "Flow Packets/s",              # → (packets_sent + packets_received) / window_sec
    "Fwd Packet Length Mean",      # → bytes_sent / max(packets_sent, 1)
    "Bwd Packet Length Mean",      # → bytes_received / max(packets_received, 1)
    "Average Packet Size",         # → (bytes_sent + bytes_received) / max(total_pkts, 1)
]


def load_csvs(data_dir: str) -> pd.DataFrame:
    """Load all CSV files from data_dir (recursively) and concatenate them."""
    csv_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        print("Please download CICAndMal2017 dataset and place CSV files in ml/data/raw/")
        print("Dataset: https://www.unb.ca/cic/datasets/andmal2017.html")
        sys.exit(1)

    dfs = []
    for path in csv_files:
        print(f"  Loading {os.path.basename(path)}...")
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(csv_files)} files.")
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the dataset."""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop rows where label is missing
    df = df.dropna(subset=[LABEL_COL])

    # Binarize label: 0 = benign, 1 = malicious
    df["is_malicious"] = df[LABEL_COL].apply(
        lambda x: 0 if str(x).strip() in BENIGN_LABELS else 1
    )

    # Keep only feature columns that exist in the dataframe
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        # Fallback: use all numeric columns except label
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in available_features if c != "is_malicious"]
        print(f"Warning: Using fallback feature set ({len(available_features)} features)")

    keep_cols = available_features + ["is_malicious", LABEL_COL]
    df = df[keep_cols].copy()

    # Replace inf with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {rows_before - len(df):,} rows with NaN/Inf.")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print(f"\nClass distribution:")
    print(df["is_malicious"].value_counts())
    print(f"\nLabel breakdown:")
    print(df[LABEL_COL].value_counts())

    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest and clean malware dataset")
    parser.add_argument("--data_dir", default="../data/raw", help="Directory with CSV files")
    parser.add_argument("--output", default="../data/processed.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    df = load_csvs(args.data_dir)

    print("Cleaning data...")
    df = clean(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} clean rows to {args.output}")


if __name__ == "__main__":
    main()
