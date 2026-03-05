"""
run_pipeline.py — Convenience script to run the full ML pipeline end-to-end.

Steps: ingest → features → train_xgboost → export_onnx

Usage:
  python run_pipeline.py --data_dir ../data/raw --android_assets ../../android/app/src/main/assets/
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full ML pipeline")
    parser.add_argument("--data_dir", default="../data/raw")
    parser.add_argument("--android_assets", default="../../android/app/src/main/assets/")
    args = parser.parse_args()

    steps = [
        [sys.executable, "ingest.py", "--data_dir", args.data_dir],
        [sys.executable, "features.py"],
        [sys.executable, "train_xgboost.py"],
        [sys.executable, "export_onnx.py", "--android_assets", args.android_assets],
    ]

    for step in steps:
        run(step)

    print("\nPipeline complete! model.onnx is ready for Android integration.")


if __name__ == "__main__":
    main()
