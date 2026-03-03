"""
export_onnx.py — Export the trained ensemble to a single ONNX model for mobile deployment.

Strategy:
  - The ensemble (XGBoost + MLP) is too complex to export as one ONNX graph directly.
  - We export the MLP to ONNX (PyTorch → ONNX is straightforward and produces mobile-friendly graphs).
  - XGBoost is exported separately via onnxmltools.
  - For mobile deployment, we use the MLP ONNX (typically smaller, faster) plus the
    ensemble weights for a lightweight weighted average at inference time.
  - The final model.onnx is the MLP with a calibrated threshold.

To use the full ensemble on-device in Phase 3, you can either:
  (a) Use MLP ONNX + apply ensemble weights in Kotlin code
  (b) Convert ensemble via skl2onnx (see commented section below)

Usage:
  python export_onnx.py --model_dir ../models --splits_dir ../data/splits \
                        --output ../models/model.onnx
"""

import argparse
import json
import os

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

from train_mlp import MLP


def export_mlp_to_onnx(model_dir: str, splits_dir: str, output_path: str, input_dim: int):
    """Export PyTorch MLP to ONNX."""
    config_path = os.path.join(model_dir, "mlp_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = MLP(
        input_dim=config["input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=config["dropout"]
    )
    weights_path = os.path.join(model_dir, "mlp_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, config["input_dim"])

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        export_params=True,
    )
    print(f"MLP exported to {output_path}")


def export_xgboost_to_onnx(model_dir: str, output_path: str, input_dim: int):
    """Export XGBoost model to ONNX."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, "xgboost_model.json"))

    initial_type = [("float_input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    onnx.save(onnx_model, output_path)
    print(f"XGBoost exported to {output_path}")


def verify_onnx(model_path: str, input_dim: int):
    """Verify that the ONNX model runs correctly."""
    sess = ort.InferenceSession(model_path)
    dummy = np.random.randn(1, input_dim).astype(np.float32)
    outputs = sess.run(None, {"input": dummy})
    print(f"ONNX verification OK — output shape: {outputs[0].shape}")
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
    parser = argparse.ArgumentParser(description="Export models to ONNX for mobile")
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
        # Fallback: infer from saved data
        X_test = np.load(os.path.join(args.splits_dir, "X_test.npy"))
        input_dim = X_test.shape[1]
    print(f"Input dimension: {input_dim}")

    # Export MLP to ONNX (primary on-device model)
    mlp_onnx_path = args.output.replace(".onnx", "_mlp.onnx")
    export_mlp_to_onnx(args.model_dir, args.splits_dir, mlp_onnx_path, input_dim)

    # Export XGBoost to ONNX (for reference / server-side use)
    xgb_onnx_path = args.output.replace(".onnx", "_xgb.onnx")
    try:
        export_xgboost_to_onnx(args.model_dir, xgb_onnx_path, input_dim)
    except Exception as e:
        print(f"XGBoost ONNX export warning: {e}")

    # Quantize MLP ONNX for mobile deployment
    print("\nQuantizing MLP model...")
    quantize_onnx(mlp_onnx_path, args.output)

    # Verify the final quantized model
    print("\nVerifying final model...")
    verify_onnx(args.output, input_dim)

    # Copy to Android assets if requested
    if args.android_assets:
        import shutil
        os.makedirs(args.android_assets, exist_ok=True)
        dest = os.path.join(args.android_assets, "model.onnx")
        shutil.copy2(args.output, dest)
        print(f"Copied model to Android assets: {dest}")

    print(f"\nFinal model: {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print("\nPlace model.onnx in android/app/src/main/assets/ for Phase 3.")


if __name__ == "__main__":
    main()
