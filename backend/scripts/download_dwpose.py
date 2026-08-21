"""Download and convert DWPose model to ONNX format.

DWPose: https://github.com/IDEA-Research/DWPose
HuggingFace: https://huggingface.co/yzd-v/DWPose

This script downloads the model and converts it to ONNX for use with ONNX Runtime.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add DWPose to path
DWPOSE_REPO = "https://github.com/IDEA-Research/DWPose.git"
DWPOSE_DIR = Path(__file__).parent.parent / "dwpose_repo"


def clone_dwpose_repo():
    """Clone DWPose repository if not present."""
    import subprocess
    if not DWPOSE_DIR.exists():
        logger.info("Cloning DWPose repository...")
        subprocess.run(["git", "clone", DWPOSE_REPO, str(DWPOSE_DIR)], check=True)
    else:
        logger.info("DWPose repository already exists")


def download_model_weights(output_dir: Path):
    """Download DWPose model weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the main model
    model_path = hf_hub_download(
        repo_id="yzd-v/DWPose",
        filename="dwpose-l-384.pth",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    logger.info(f"Downloaded model to {model_path}")
    return Path(model_path)


def convert_to_onnx(model_path: Path, output_path: Path, input_size: tuple = (288, 384)):
    """Convert DWPose PyTorch model to ONNX."""
    sys.path.insert(0, str(DWPOSE_DIR))

    from dwpose.models import build_model
    from dwpose.utils import Config

    # Load config
    config_path = DWPOSE_DIR / "configs" / "dwpose" / "dwpose-l-384.py"
    if not config_path.exists():
        # Try alternative config locations
        config_files = list(DWPOSE_DIR.rglob("*dwpose*384*.py"))
        if config_files:
            config_path = config_files[0]
        else:
            raise FileNotFoundError("DWPose config not found")

    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None  # We'll load weights manually

    # Build model
    model = build_model(cfg.model)
    model.eval()

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    logger.info("Model weights loaded")

    # Create dummy input
    dummy_input = torch.randn(1, 3, *input_size)

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["keypoints", "scores"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "keypoints": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        verbose=False,
    )

    logger.info(f"ONNX model exported to {output_path}")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
    outputs = sess.run(None, inputs)
    logger.info(f"ONNX verification: outputs shapes = {[o.shape for o in outputs]}")


def optimize_onnx(onnx_path: Path, output_path: Path):
    """Optimize ONNX model using ONNX Runtime."""
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.onnx_model import OnnxModel

    model = OnnxModel(str(onnx_path))
    model.optimize()
    model.save_model_to_file(str(output_path))
    logger.info(f"Optimized ONNX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and convert DWPose to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/models",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[288, 384],
        help="Input size (H W)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading weights (use existing)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX model",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "dwpose.onnx"
    optimized_path = output_dir / "dwpose_optimized.onnx"

    # Clone repo
    clone_dwpose_repo()

    # Download weights
    if not args.skip_download:
        model_path = download_model_weights(output_dir)
    else:
        model_path = output_dir / "dwpose-l-384.pth"
        if not model_path.exists():
            logger.error(f"Model weights not found at {model_path}")
            sys.exit(1)

    # Convert to ONNX
    convert_to_onnx(model_path, onnx_path, tuple(args.input_size))

    # Optimize if requested
    if args.optimize:
        optimize_onnx(onnx_path, optimized_path)
        logger.info(f"Done! Optimized model: {optimized_path}")
    else:
        logger.info(f"Done! Model: {onnx_path}")


if __name__ == "__main__":
    main()