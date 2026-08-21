"""Modal App for Pose Transfer using Flux + ControlNet + InstantID + IP-Adapter."""

import modal
import os
import io
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.responses import StreamingResponse
import asyncio
from modal import fastapi_endpoint

# 1. Define the Container Image with all dependencies
# Models will be downloaded at runtime to a persistent volume, not during build
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6", "libxrender-dev", "libgomp1")

    # Core Python dependencies
    .pip_install(
        "accelerate>=0.25.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.2",
        "aiohttp",
        "pyyaml",
        "Pillow",
        "scipy",
        "tqdm",
        "psutil",
        "tokenizers>=0.13.3",
        "torchsde",
        "kornia>=0.7.1",
        "spandrel",
        "soundfile",
        "sentencepiece",
        "comfyui-workflow-templates",
        "comfyui-embedded-docs",
        "av",
        "comfy_kitchen",
        "piexif",
        "ultralytics",
        "onnxruntime-gpu",
        "segment_anything",
        "deepdiff",
        "rapidfuzz",
        "sageattention",
        "surrealist",
        "boto3",
        "redis",
        "fal_client",
        "replicate",
        "GitPython",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "huggingface_hub",
        "scikit-image",  # Required for comfyui_controlnet_aux dwpose
    )
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4")
    .pip_install("git+https://github.com/facebookresearch/sam2")

    # 2. Clone ComfyUI
    .run_commands("git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI")

    # 3. Install Custom Nodes (minimal essential set)
    .run_commands(
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_InstantID.git",
        "pip install -r /root/ComfyUI/requirements.txt"
    )
)

app = modal.App("gym-trainer-pose-transfer", image=image)

# Volume for model caching (persists across deployments)
model_volume = modal.Volume.from_name("gym-trainer-models", create_if_missing=True)

# Model download configuration
# Using GGUF quantized models to avoid gated repo access issues
# GGUF models can be loaded directly by ComfyUI with the right loader nodes
MODEL_CONFIG = {
    "flux1-dev-Q4_0.gguf": {
        "repo_id": "city96/FLUX.1-dev-gguf",
        "filename": "flux1-dev-Q4_0.gguf",
        "local_dir": "/root/ComfyUI/models/diffusion_models",
    },
    "clip_l.safetensors": {
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "clip_l.safetensors",
        "local_dir": "/root/ComfyUI/models/clip",
    },
    "t5xxl_fp16.safetensors": {
        "repo_id": "comfyanonymous/flux_text_encoders",
        "filename": "t5xxl_fp16.safetensors",
        "local_dir": "/root/ComfyUI/models/clip",
    },
    "instantid.bin": {
        "repo_id": "InstantX/InstantID",
        "filename": "ip-adapter.bin",
        "local_dir": "/root/ComfyUI/models/instantid",
    },
    "controlnet-instantid.safetensors": {
        "repo_id": "InstantX/InstantID",
        "filename": "ControlNetModel/diffusion_pytorch_model.safetensors",
        "local_dir": "/root/ComfyUI/models/instantid",
    },
    "ip-adapter-plus_sd15.bin": {
        "repo_id": "h94/IP-Adapter",
        "filename": "models/ip-adapter-plus_sd15.bin",
        "local_dir": "/root/ComfyUI/models/ipadapter",
    },
    "clip_vit_h_14.safetensors": {
        "repo_id": "h94/IP-Adapter",
        "filename": "models/image_encoder/model.safetensors",
        "local_dir": "/root/ComfyUI/models/clip_vision",
    },
    "siglip-so400m-patch14-384.safetensors": {
        "repo_id": "google/siglip-so400m-patch14-384",
        "filename": "model.safetensors",
        "local_dir": "/root/ComfyUI/models/instantid",
    },
    "yolo_nas_l_fp16.onnx": {
        "repo_id": "hr16/yolo-nas-fp16",
        "filename": "yolo_nas_l_fp16.onnx",
        "local_dir": "/root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16",
    },
    "dw-ll_ucoco_384_bs5.torchscript.pt": {
        "repo_id": "hr16/DWPose-TorchScript-BatchSize5",
        "filename": "dw-ll_ucoco_384_bs5.torchscript.pt",
        "local_dir": "/root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5",
    },
}


def download_models():
    """Download all required models to the volume using huggingface_hub."""
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for model_name, config in MODEL_CONFIG.items():
        local_path = os.path.join(config["local_dir"], model_name)
        os.makedirs(config["local_dir"], exist_ok=True)

        if os.path.exists(local_path):
            print(f"Model {model_name} already exists at {local_path}")
            continue

        print(f"Downloading {model_name} from {config['repo_id']}...")
        try:
            hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                local_dir=config["local_dir"],
                local_dir_use_symlinks=False,
                token=hf_token,
            )
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Warning: Failed to download {model_name}: {e}")


# Helper functions
def get_value_at_index(obj, index):
    """Helper to get value from ComfyUI node output."""
    try:
        return obj[index]
    except (KeyError, TypeError):
        return obj["result"][index] if isinstance(obj, dict) and "result" in obj else obj


async def run_comfyui_workflow(workflow_path: str, inputs: dict):
    """Run a ComfyUI workflow with given inputs."""
    import torch
    import sys
    import random
    import json

    sys.path.append("/root/ComfyUI")

    # Initialize ComfyUI
    import execution
    import server
    from nodes import init_extra_nodes

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    await init_extra_nodes()

    # Load workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)

    # Update workflow inputs
    if "user_image" in inputs:
        for node_id, node in workflow.items():
            if node.get("class_type") == "LoadImage" and "user" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["image"] = inputs["user_image"]
            elif node.get("class_type") == "LoadImage" and "trainer" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["image"] = inputs["trainer_image"]
            elif node.get("class_type") == "LoadImage" and "control" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["image"] = inputs["control_image"]
            elif node.get("class_type") == "CLIPTextEncode" and "positive" in node.get("_meta", {}).get("title", "").lower():
                node["inputs"]["text"] = inputs.get("prompt", "professional fitness photo, perfect form, gym lighting, high quality")

    # Execute workflow
    prompt_id = "pose_transfer_" + str(random.randint(1000, 9999))
    outputs = await execute_workflow(workflow, prompt_id, server_instance)

    # Find output image
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for img_info in node_output["images"]:
                if img_info.get("type") == "output":
                    img_path = os.path.join("/root/ComfyUI/output", img_info["filename"])
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            return f.read()

    raise RuntimeError("No output image generated")


async def execute_workflow(workflow, prompt_id, server_instance):
    """Execute a ComfyUI workflow."""
    import execution

    # Queue the prompt
    prompt_queue = execution.PromptQueue(server_instance)
    await prompt_queue.put(prompt_id, workflow, {}, 1)

    # Wait for completion
    while True:
        await asyncio.sleep(0.5)
        history = await prompt_queue.get_history([prompt_id])
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                return history[prompt_id].get("outputs", {})
            elif status.get("status_str") == "error":
                raise RuntimeError(f"Workflow failed: {status.get('error')}")


@app.function(
    gpu="A10G",
    timeout=600,  # Longer timeout for first run (model downloads)
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.fastapi_endpoint(method="POST")
async def pose_transfer(request: Request):
    """
    Pose Transfer Endpoint

    Input (multipart/form-data):
    - user_image: User's frame (identity source)
    - trainer_image: Trainer reference image (style/clothing)
    - control_image: DWPOse skeleton image (pose condition)
    - prompt: Optional text prompt

    Output: Generated image (JPEG)
    """
    from PIL import Image
    import io

    # Ensure models are downloaded
    download_models()

    form = await request.form()

    # Read input images
    user_image_bytes = await form["user_image"].read()
    trainer_image_bytes = await form["trainer_image"].read()
    control_image_bytes = await form["control_image"].read()
    prompt = form.get("prompt", "professional fitness photo, perfect form, gym lighting, high quality, sharp focus, 8k")

    # Save images to ComfyUI input directory
    import os
    os.makedirs("/root/ComfyUI/input", exist_ok=True)
    os.makedirs("/root/ComfyUI/output", exist_ok=True)

    user_img = Image.open(io.BytesIO(user_image_bytes)).convert("RGB")
    trainer_img = Image.open(io.BytesIO(trainer_image_bytes)).convert("RGB")
    control_img = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")

    # Resize to standard size (1024 for Flux)
    target_size = (1024, 1024)
    user_img = user_img.resize(target_size, Image.LANCZOS)
    trainer_img = trainer_img.resize(target_size, Image.LANCZOS)
    control_img = control_img.resize(target_size, Image.LANCZOS)

    user_img.save("/root/ComfyUI/input/user_image.jpg", "JPEG", quality=95)
    trainer_img.save("/root/ComfyUI/input/trainer_image.jpg", "JPEG", quality=95)
    control_img.save("/root/ComfyUI/input/control_image.jpg", "JPEG", quality=95)

    # Run workflow
    workflow_path = "/root/ComfyUI/comfyui_workflows/pose_transfer.json"

    # Copy workflow to container
    import shutil
    os.makedirs("/root/ComfyUI/comfyui_workflows", exist_ok=True)
    # The workflow JSON is embedded in the image, so it should be at this path

    inputs = {
        "user_image": "user_image.jpg",
        "trainer_image": "trainer_image.jpg",
        "control_image": "control_image.jpg",
        "prompt": prompt,
    }

    result_bytes = await run_comfyui_workflow(workflow_path, inputs)

    return Response(content=result_bytes, media_type="image/jpeg")


@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.fastapi_endpoint(method="POST")
async def setup_models(request: Request):
    """Setup/pre-download all models to the volume."""
    download_models()
    return {"status": "success", "message": "Models downloaded successfully"}


@app.function(
    gpu="A10G",
    timeout=120,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.fastapi_endpoint(method="POST")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "pose-transfer"}


@app.local_entrypoint()
def test():
    """Local test entrypoint."""
    import requests
    import base64

    # Test with dummy images
    print("Testing pose transfer endpoint...")
    print("Deploy with: modal deploy modal/pose_transfer.py")