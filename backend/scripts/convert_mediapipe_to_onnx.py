#!/usr/bin/env python3
"""
MediaPipe Pose Landmarker to ONNX Conversion Script

Converts MediaPipe BlazePose model to ONNX format, then optionally to TensorRT engine.

Requirements:
- MediaPipe model file (pose_landmarker_full.task) from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- TensorFlow 2.x
- tf2onnx
- onnxruntime
- tensorrt (for TensorRT conversion, requires GPU)

Usage:
    python convert_mediapipe_to_onnx.py \
        --model_path pose_landmarker_full.task \
        --output_dir ./models \
        --convert_trt

Or step by step:
    # 1. Export to SavedModel
    python convert_mediapipe_to_onnx.py --step export_savedmodel --model_path pose_landmarker_full.task --output_dir ./models

    # 2. Convert to ONNX
    python convert_mediapipe_to_onnx.py --step onnx --saved_model_dir ./models/saved_model --output_dir ./models

    # 3. Optimize with ONNX Runtime
    python convert_mediapipe_to_onnx.py --step ort --onnx_path ./models/pose_landmarker.onnx --output_dir ./models

    # 4. Convert to TensorRT (requires GPU)
    python convert_mediapipe_to_onnx.py --step trt --onnx_path ./models/pose_landmarker.onnx --output_dir ./models --fp16
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def export_savedmodel(model_path: str, output_dir: str) -> bool:
    """Export MediaPipe model to TensorFlow SavedModel format."""
    cmd = [
        sys.executable, "-m", "mediapipe.model_maker.pose_landmarker.export_saved_model",
        f"--model_path={model_path}",
        f"--export_dir={output_dir}/saved_model"
    ]
    return run_command(cmd, "Export MediaPipe to SavedModel")


def convert_to_onnx(saved_model_dir: str, output_dir: str, opset: int = 17) -> bool:
    """Convert SavedModel to ONNX format."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        f"--saved-model={saved_model_dir}",
        f"--output={output_dir}/pose_landmarker.onnx",
        f"--opset={opset}"
    ]
    return run_command(cmd, "Convert SavedModel to ONNX")


def optimize_with_ort(onnx_path: str, output_dir: str) -> bool:
    """Optimize ONNX model with ONNX Runtime."""
    cmd = [
        sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
        onnx_path,
        f"{output_dir}/pose_landmarker.ort"
    ]
    return run_command(cmd, "Optimize ONNX with ONNX Runtime")


def convert_to_tensorrt(onnx_path: str, output_dir: str, fp16: bool = True, workspace: int = 4096) -> bool:
    """Convert ONNX to TensorRT engine (requires GPU)."""
    engine_path = f"{output_dir}/pose_landmarker{'_fp16' if fp16 else '_fp32'}.engine"
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
    ]
    if fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--fp32")

    return run_command(cmd, "Convert ONNX to TensorRT")


def verify_onnx_model(onnx_path: str) -> bool:
    """Verify ONNX model can be loaded."""
    print(f"\n{'='*60}")
    print("Verifying ONNX model...")
    print(f"{'='*60}")

    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX model verified: {onnx_path}")
        print(f"   IR version: {model.ir_version}")
        print(f"   Opset: {model.opset_import[0].version}")
        print(f"   Producer: {model.producer_name}")
        return True
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False


def verify_ort_model(ort_path: str) -> bool:
    """Verify ONNX Runtime can load the model."""
    print(f"\n{'='*60}")
    print("Verifying ONNX Runtime model...")
    print(f"{'='*60}")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ort_path)
        print(f"✅ ONNX Runtime model loaded: {ort_path}")
        print(f"   Providers: {session.get_providers()}")
        print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"   Outputs: {[o.name for o in session.get_outputs()]}")
        return True
    except Exception as e:
        print(f"❌ ONNX Runtime verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose to ONNX/TensorRT Converter")
    parser.add_argument("--model_path", type=str, help="Path to MediaPipe .task model file")
    parser.add_argument("--saved_model_dir", type=str, help="Path to SavedModel directory")
    parser.add_argument("--onnx_path", type=str, help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--step", type=str, choices=["export_savedmodel", "onnx", "ort", "trt", "all"],
                        default="all", help="Conversion step to run")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for TensorRT")
    parser.add_argument("--workspace", type=int, default=4096, help="TensorRT workspace size (MB)")
    parser.add_argument("--convert_trt", action="store_true", help="Also convert to TensorRT")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    if args.step in ["export_savedmodel", "all"]:
        if not args.model_path:
            print("❌ --model_path required for export_savedmodel step")
            return 1
        success = export_savedmodel(args.model_path, str(output_dir))
        if not success:
            return 1

    if args.step in ["onnx", "all"]:
        saved_model_dir = args.saved_model_dir or str(output_dir / "saved_model")
        success = convert_to_onnx(saved_model_dir, str(output_dir), args.opset)
        if not success:
            return 1
        # Verify
        verify_onnx_model(str(output_dir / "pose_landmarker.onnx"))

    if args.step in ["ort", "all"]:
        onnx_path = args.onnx_path or str(output_dir / "pose_landmarker.onnx")
        success = optimize_with_ort(onnx_path, str(output_dir))
        if not success:
            return 1
        # Verify
        verify_ort_model(str(output_dir / "pose_landmarker.ort"))

    if args.step in ["trt", "all"] or args.convert_trt:
        onnx_path = args.onnx_path or str(output_dir / "pose_landmarker.onnx")
        success = convert_to_tensorrt(onnx_path, str(output_dir), args.fp16, args.workspace)
        if not success:
            print("⚠️  TensorRT conversion failed (may need GPU). Continuing...")
            # Don't return 1 - TensorRT is optional

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())