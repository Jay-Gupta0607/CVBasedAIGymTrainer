#!/usr/bin/env python3
"""
LightGBM Form Scorer to ONNX Conversion Script

Converts trained LightGBM model to ONNX format for fast inference.

Requirements:
- LightGBM model file (form_scorer.txt or .pkl)
- skl2onnx
- onnxruntime

Usage:
    python convert_lightgbm_to_onnx.py \
        --model_path ./models/form_scorer.txt \
        --output_dir ./models \
        --n_features 156  # 33 landmarks * 3 coords * 2 (user + trainer) or 33*3 for no-trainer
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def convert_lightgbm_to_onnx(model_path: str, output_dir: str, n_features: int, model_name: str = "form_scorer") -> bool:
    """Convert LightGBM model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"Converting LightGBM model to ONNX...")
    print(f"{'='*60}")

    try:
        import lightgbm as lgb
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install lightgbm skl2onnx onnx onnxruntime")
        return False

    # Load LightGBM model
    print(f"Loading model from: {model_path}")
    if model_path.endswith(".txt"):
        model = lgb.Booster(model_file=model_path)
    elif model_path.endswith(".pkl") or model_path.endswith(".joblib"):
        import joblib
        model = joblib.load(model_path)
    else:
        print(f"❌ Unsupported model format: {model_path}")
        return False

    # Define input type (batch_size, n_features)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    # Convert to ONNX
    print(f"Converting with {n_features} input features...")
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)

    # Save ONNX model
    output_path = Path(output_dir) / f"{model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ ONNX model saved to: {output_path}")

    # Verify the model
    verify_onnx_model(str(output_path))

    return True


def verify_onnx_model(onnx_path: str) -> bool:
    """Verify ONNX model can be loaded and run inference."""
    print(f"\n{'='*60}")
    print("Verifying ONNX model...")
    print(f"{'='*60}")

    try:
        import onnx
        import onnxruntime as ort

        # Check with onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX model structure valid")
        print(f"   IR version: {model.ir_version}")
        print(f"   Opset: {model.opset_import[0].version}")
        print(f"   Inputs: {[i.name for i in model.graph.input]}")
        print(f"   Outputs: {[o.name for o in model.graph.output]}")

        # Test inference with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        print(f"✅ ONNX Runtime session created")
        print(f"   Providers: {session.get_providers()}")

        # Create dummy input
        input_name = session.get_inputs()[0].name
        n_features = session.get_inputs()[0].shape[1]
        dummy_input = np.random.randn(1, n_features).astype(np.float32)

        outputs = session.run(None, {input_name: dummy_input})
        print(f"✅ Inference test passed")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Output value: {outputs[0][0]}")

        return True
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_model(output_dir: str, n_features: int = 156, n_samples: int = 1000) -> str:
    """Create a sample LightGBM model for testing (if no trained model exists)."""
    print(f"\n{'='*60}")
    print("Creating sample LightGBM model for testing...")
    print(f"{'='*60}")

    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return None

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create labels based on some synthetic rules
    y = (X[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'n_estimators': 100,
    }

    model = lgb.train(params, train_data, num_boost_round=100)

    # Save model
    output_path = Path(output_dir) / "form_scorer.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    print(f"✅ Sample model saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="LightGBM to ONNX Converter")
    parser.add_argument("--model_path", type=str, help="Path to LightGBM model (.txt, .pkl, .joblib)")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--n_features", type=int, default=156,
                        help="Number of input features (33 landmarks * 3 coords * 2 = 198 for with-trainer, 99 for no-trainer)")
    parser.add_argument("--model_name", type=str, default="form_scorer", help="Output model name")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create a sample model if no model_path provided")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.create_sample or not args.model_path:
        if not args.model_path:
            print("No model path provided, creating sample model...")
        model_path = create_sample_model(str(output_dir), args.n_features)
        if not model_path:
            return 1
        args.model_path = model_path

    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1

    success = convert_lightgbm_to_onnx(args.model_path, str(output_dir), args.n_features, args.model_name)

    if success:
        print(f"\n{'='*60}")
        print("Conversion complete!")
        print(f"Output: {output_dir / f'{args.model_name}.onnx'}")
        print(f"{'='*60}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())