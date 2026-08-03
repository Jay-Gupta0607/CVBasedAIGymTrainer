import os
import io
import time
import shutil
import modal
import numpy as np
import cv2
import torch
from PIL import Image

# Import FastAPI types
from fastapi import Response, UploadFile, File, Form

# --- Constants ---
CHECKPOINT_DIR = "/root/checkpoints"
DWPOSE_CACHE_DIR = f"{CHECKPOINT_DIR}/dwpose"
FLUX_CACHE_DIR = "/cache"

# --- 1. Define the Download Function ---
def download_models():
    from huggingface_hub import snapshot_download
    import os
    
    # A. Setup Directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DWPOSE_CACHE_DIR, exist_ok=True)
    os.makedirs(FLUX_CACHE_DIR, exist_ok=True)
    
    # B. Download Florence-2
    print("⬇️ Downloading Florence-2...")
    snapshot_download(repo_id="microsoft/Florence-2-large", local_dir=f"{CHECKPOINT_DIR}/florence2")

    # C. Download SAM 2
    print("⬇️ Downloading SAM 2 Checkpoint...")
    os.system(f"wget -P {CHECKPOINT_DIR} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")

    # D. Download DWPose Models
    print("⬇️ Downloading DWPose Models (ONNX)...")
    os.system(f"wget -q -O {DWPOSE_CACHE_DIR}/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx")
    os.system(f"wget -q -O {DWPOSE_CACHE_DIR}/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx")

    # E. Download Flux & Nunchaku Adapters
    print("⬇️ Downloading Flux & Nunchaku Models...")
    # Note: Ensure you have access/login for black-forest-labs if needed, or use public mirrors
    snapshot_download(repo_id="mit-han-lab/nunchaku-t5", local_dir=f"{FLUX_CACHE_DIR}/nunchaku-t5")
    snapshot_download(repo_id="nunchaku-tech/nunchaku-flux.1-kontext-dev", local_dir=f"{FLUX_CACHE_DIR}/nunchaku-flux")
    snapshot_download(repo_id="black-forest-labs/FLUX.1-Kontext-dev", local_dir=f"{FLUX_CACHE_DIR}/flux-kontext")
    
    # Download LoRA
    os.system(f"wget -P {FLUX_CACHE_DIR} https://huggingface.co/thedeoxen/refcontrol-flux-kontext-reference-pose-lora/resolve/main/refcontrol_pose.safetensors")

# --- 2. Define the Container Image ---
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .pip_install(
        # Core & Torch
        "torch", "torchvision", "accelerate", "numpy",
        
        # Computer Vision (Florence/SAM/DWPose)
        "transformers==4.46.3", 
        "opencv-python", 
        "pillow", 
        "huggingface_hub",
        "timm", 
        "einops", 
        "dwpose",            
        "onnxruntime-gpu",   
        "mediapipe",         
        "protobuf",          
        "scipy",
        "scikit-image",
        "matplotlib",
        
        # Web
        "fastapi",
        "python-multipart",
        
        # Generative (Flux/Diffusers/Nunchaku)
        "diffusers",
        "nunchaku" # Ensure this is the correct pip package name for Nunchaku
    )
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("unified-flux-control-pipeline", image=image)

# --- 3. The Main Unified Class ---
@app.cls(
    gpu="A100", # Using A100 is recommended for Flux + Detection stack. A10G might OOM.
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1200, # Increased timeout for generation
    max_containers=1
)
class UnifiedGenerator:
    @modal.enter()
    def load_models(self):
        """Loads ALL models into GPU memory."""
        print("🚀 Starting Model Loading Process...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- 1. Load Preprocessing Models (Florence, SAM, DWPose) ---
        self._load_preprocessing_models()
        
        # --- 2. Load Generative Models (Flux + Nunchaku) ---
        self._load_generative_models()

        print("✅ All Models Loaded Successfully.")

    def _load_preprocessing_models(self):
        from transformers import AutoProcessor, AutoModelForCausalLM
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import dwpose
        from dwpose import DwposeDetector

        print("   🔹 Loading Florence-2, SAM2, and DWPose...")
        
        # Fix DWPose paths
        dwpose_install_path = os.path.dirname(dwpose.__file__)
        target_dir = os.path.join(dwpose_install_path, "ckpts/yzd-v/DWPose")
        os.makedirs(target_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(target_dir, "yolox_l.onnx")):
            shutil.copy(f"{DWPOSE_CACHE_DIR}/yolox_l.onnx", os.path.join(target_dir, "yolox_l.onnx"))
        if not os.path.exists(os.path.join(target_dir, "dw-ll_ucoco_384.onnx")):
            shutil.copy(f"{DWPOSE_CACHE_DIR}/dw-ll_ucoco_384.onnx", os.path.join(target_dir, "dw-ll_ucoco_384.onnx"))

        # Florence-2
        florence_path = f"{CHECKPOINT_DIR}/florence2"
        self.florence_processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

        # SAM 2
        sam2_ckpt = f"{CHECKPOINT_DIR}/sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 
        self.sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # DWPose
        self.dwpose_model = DwposeDetector.from_pretrained_default()
        if hasattr(self.dwpose_model, 'to'):
             self.dwpose_model.to(self.device)

    def _load_generative_models(self):
        from diffusers import FluxKontextPipeline
        from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
        from nunchaku.utils import get_precision

        print("   🔹 Loading Flux and Nunchaku...")
        self.num_inference_steps = 30
        precision = get_precision() # Auto-detect precision (likely int4 or nf4)
        
        # Load Transformer (Quantized)
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"{FLUX_CACHE_DIR}/nunchaku-flux/svdq-{precision}_r32-flux.1-kontext-dev.safetensors"
        )
        
        # Load LoRA Adapter
        transformer.update_lora_params(
            f"{FLUX_CACHE_DIR}/refcontrol_pose.safetensors"
        )
        
        # Load Text Encoder (Quantized)
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            f"{FLUX_CACHE_DIR}/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        )
        
        # Load Pipeline
        self.flux_pipe = FluxKontextPipeline.from_pretrained(
            f"{FLUX_CACHE_DIR}/flux-kontext",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        apply_cache_on_pipe(self.flux_pipe, residual_diff_threshold=0.12)

    def _get_control_image(self, image_input, text_prompt):
        """
        Generates the 'Control Image' (Skeleton + Masked Edges) 
        using Florence, SAM2, and DWPose.
        """
        # Ensure image is PIL
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            image = image_input.convert("RGB")
            
        w, h = image.size
        
        # 1. Florence Detection
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        prompt = task_prompt + text_prompt
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.inference_mode():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        bboxes = parsed_answer.get(task_prompt, {}).get('bboxes', [])
        
        # 2. SAM2 Segmentation (if object found)
        thick_edges = np.zeros((h, w), dtype=np.uint8)
        if bboxes:
            box_prompt = np.array(bboxes[0])
            self.sam2_predictor.set_image(np.array(image))
            masks, _, _ = self.sam2_predictor.predict(box=box_prompt[None, :], multimask_output=False)
            
            binary_mask_img = (masks[0] > 0).astype(np.uint8) * 255
            edges = cv2.Canny(binary_mask_img, 100, 200)
            kernel = np.ones((2, 2), np.uint8)
            thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 3. DWPose Skeleton
        out_img, _, _ = self.dwpose_model(
            image, include_hand=True, include_face=False, include_body=True,
            image_and_json=True, detect_resolution=512
        )
        if not isinstance(out_img, Image.Image):
             out_img = Image.fromarray(out_img)
        skeleton_image = out_img.resize((w, h), Image.LANCZOS)
        
        # 4. Compose (Skeleton + Edges)
        base_np = np.array(skeleton_image.convert("RGB"))
        edge_mask = thick_edges > 100
        base_np[edge_mask] = [255, 255, 255] # White edges on top of skeleton
        
        return Image.fromarray(base_np)

    @modal.method()
    def process_and_generate(self, image_bytes, prompt_flux, prompt_detection):
        """
        Full Pipeline: Image -> Control Map -> Flux Generation
        """
        print(f"🎨 Processing request: Detect='{prompt_detection}', Gen='{prompt_flux}'")
        
        # Load Source Image
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Step A: Create Control Image (Skeleton + Object Edges)
        print("   Step A: Generating Control Image...")
        control_image = self._get_control_image(original_image, prompt_detection)
        
        # Step B: Prepare Flux Input (Side-by-Side: Original | Control)
        # Resize to standard size for better flux performance
        target_w, target_h = 768, 1024 
        
        # Helper to resize and pad
        def resize_pad(img, tw, th):
            w, h = img.size
            scale = min(tw/w, th/h)
            nw, nh = int(w*scale), int(h*scale)
            img_resized = img.resize((nw, nh), Image.LANCZOS)
            new_img = Image.new("RGB", (tw, th), (0,0,0))
            new_img.paste(img_resized, ((tw-nw)//2, (th-nh)//2))
            return new_img

        user_resized = resize_pad(original_image, target_w, target_h)
        control_resized = resize_pad(control_image, target_w, target_h)
        
        # Concatenate for RefControl (User Left, Control Right)
        flux_input = Image.new("RGB", (target_w * 2, target_h))
        flux_input.paste(user_resized, (0, 0))
        flux_input.paste(control_resized, (target_w, 0))
        
        # Step C: Flux Inference
        print("   Step C: Flux Inference...")
        out = self.flux_pipe(
            prompt="refcontrolpose " + prompt_flux,
            image=flux_input,
            num_inference_steps=self.num_inference_steps,
        ).images[0]
        
        # Return bytes
        byte_arr = io.BytesIO()
        out.save(byte_arr, format='JPEG')
        return byte_arr.getvalue()

    @modal.web_endpoint(method="POST")
    async def web_inference(
        self, 
        image: UploadFile = File(...), 
        prompt: str = Form(...),
        object_to_detect: str = Form(default="rod")
    ):
        """
        Web Endpoint: Upload Image -> Get Flux Result
        """
        image_bytes = await image.read()
        
        result_bytes = self.process_and_generate.remote(image_bytes, prompt, object_to_detect)
        
        return Response(content=result_bytes, media_type="image/jpeg")

# --- 4. Local Entrypoint ---
@app.local_entrypoint()
def main(filepath: str = "bench.jpeg", prompt: str = "a man holding a glowing lightsaber", obj: str = "rod"):
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return

    print(f"🚀 Reading '{filepath}'...")
    with open(filepath, "rb") as f:
        image_bytes = f.read()

    pipeline = UnifiedGenerator()
    try:
        result_bytes = pipeline.process_and_generate.remote(image_bytes, prompt, obj)
    except Exception as e:
        print(f"❌ Remote execution failed: {e}")
        return

    output_path = "flux_output.jpg"
    with open(output_path, "wb") as f:
        f.write(result_bytes)
    
    print(f"✅ Success! Result saved to: {output_path}")