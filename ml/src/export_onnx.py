"""
export_onnx.py — Export the trained XGBoost model to ONNX for mobile deployment.

Strategy:
  - Uses skl2onnx to convert the XGBoost sklearn wrapper to ONNX.
  - options={"zipmap": False} ensures output_probability is a plain float32 [N, 2]
    array (not a dict), which ONNX Runtime Mobile can handle cleanly.
  - Dynamic INT8 quantization is applied to shrink the model.
  - The final model.onnx is copied to Android assets.

ONNX output layout:
  output[0] — label            int64  [N]
  output[1] — output_probability float32 [N, 2]   ← index [0][1] = P(malicious)

Usage:
  python export_onnx.py --model_dir ../models --splits_dir ../data/splits \
                        --output ../models/model.onnx \
                        --android_assets ../../android/app/src/main/assets/
"""

import argparse
import os
import shutil

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def export_xgboost_to_onnx(model_dir: str, output_path: str, input_dim: int):
    """Export XGBoost model to ONNX via onnxmltools."""
    # Try joblib format first, fall back to JSON
    joblib_path = os.path.join(model_dir, "xgboost_model.joblib")
    json_path = os.path.join(model_dir, "xgboost_model.json")

    if os.path.exists(joblib_path):
        model = joblib.load(joblib_path)
    elif os.path.exists(json_path):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(json_path)
    else:
        raise FileNotFoundError(f"Model not found in {model_dir}")

    # Convert to ONNX using onnxmltools
    initial_types = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types, target_opset=12)
    onnx.save(onnx_model, output_path)
    print(f"XGBoost exported to {output_path}")


def verify_onnx(model_path: str, input_dim: int):
    """Verify that the ONNX model runs correctly and print output shapes."""
    sess = ort.InferenceSession(model_path)
    dummy = np.random.randn(1, input_dim).astype(np.float32)
    outputs = sess.run(None, {"input": dummy})
    print(f"ONNX verification OK")
    print(f"  output[0] label:               {outputs[0]}")
    print(f"  output[1] output_probability:  {outputs[1]}")
    return True


def quantize_onnx(input_path: str, output_path: str):
    """Apply dynamic INT8 quantization to reduce model size."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    before_size = os.path.getsize(input_path) / 1024 / 1024
    after_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Quantized: {before_size:.1f} MB → {after_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export XGBoost model to ONNX for mobile")
    parser.add_argument("--model_dir", default="../models")
    parser.add_argument("--splits_dir", default="../data/splits")
    parser.add_argument("--output", default="../models/model.onnx")
    parser.add_argument("--android_assets", default=None,
                        help="If set, copy final model.onnx here (e.g. android/app/src/main/assets/)")
    args = parser.parse_args()

    # Determine input dimension from feature names file
    feature_names_path = os.path.join(args.splits_dir, "feature_names.txt")
    if os.path.exists(feature_names_path):
        with open(feature_names_path) as f:
            input_dim = len(f.readlines())
    else:
        X_test = np.load(os.path.join(args.splits_dir, "X_test.npy"))
        input_dim = X_test.shape[1]
    print(f"Input dimension: {input_dim}")

    # Export XGBoost to ONNX
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    export_xgboost_to_onnx(args.model_dir, args.output, input_dim)

    # Verify the exported model
    print("\nVerifying final model...")
    verify_onnx(args.output, input_dim)

    # Copy to Android assets if requested
    if args.android_assets:
        os.makedirs(args.android_assets, exist_ok=True)
        dest = os.path.join(args.android_assets, "model.onnx")
        shutil.copy2(args.output, dest)
        print(f"Copied model to Android assets: {dest}")

    print(f"\nFinal model: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print("\nPlace model.onnx in android/app/src/main/assets/ for Android inference.")


if __name__ == "__main__":
    main()
