import os
import io
import shutil
import modal

# --- Constants ---
CHECKPOINT_DIR = "/root/checkpoints"
DWPOSE_CACHE_DIR = f"{CHECKPOINT_DIR}/dwpose"

# --- 1. Define the Download Function ---
def download_models():
    from huggingface_hub import snapshot_download
    import os
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DWPOSE_CACHE_DIR, exist_ok=True)
    
    print("⬇️ Downloading Florence-2...")
    snapshot_download(repo_id="microsoft/Florence-2-large", local_dir=f"{CHECKPOINT_DIR}/florence2")

    print("⬇️ Downloading SAM 2 Checkpoint...")
    os.system(f"wget -P {CHECKPOINT_DIR} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")

    print("⬇️ Downloading DWPose Models (ONNX)...")
    # Download to our persistent cache folder
    os.system(f"wget -q -O {DWPOSE_CACHE_DIR}/yolox_l.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx")
    os.system(f"wget -q -O {DWPOSE_CACHE_DIR}/dw-ll_ucoco_384.onnx https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx")

# --- 2. Define the Container Image ---
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision",
        "transformers==4.46.3", 
        "accelerate",
        "opencv-python", 
        "pillow", 
        "huggingface_hub",
        "timm", 
        "einops", 
        "numpy",
        "dwpose",            
        "onnxruntime-gpu",   
        "mediapipe",         
        "protobuf",          
        "scipy",
        "scikit-image",
        "matplotlib",
    )
    .pip_install("git+https://github.com/facebookresearch/sam2.git")
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("maskedbar-pipeline-full", image=image)

# --- 3. The Main Inference Class ---
@app.cls(
    gpu="A10G", 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    max_containers=1  # FIXED: Renamed from concurrency_limit
)
class ImagePipeline:
    @modal.enter()
    def load_models(self):
        """Loads models into GPU memory when the container starts."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Specific Imports for DWPose
        import dwpose
        from dwpose import DwposeDetector

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading models on {self.device}...")

        # --- A. FIX DWPOSE PATHS ---
        # The dwpose library hardcodes paths inside site-packages. 
        # We manually copy our cached models to where dwpose expects them.
        
        # 1. Find where dwpose is installed
        dwpose_install_path = os.path.dirname(dwpose.__file__)
        target_dir = os.path.join(dwpose_install_path, "ckpts/yzd-v/DWPose")
        
        # 2. Create the directory structure
        os.makedirs(target_dir, exist_ok=True)
        
        # 3. Copy/Link files from our Cache to the Target
        # (Using copy is safer than symlink in some container contexts)
        print(f"🛠️ Injecting DWPose models into: {target_dir}")
        
        source_yolo = f"{DWPOSE_CACHE_DIR}/yolox_l.onnx"
        target_yolo = os.path.join(target_dir, "yolox_l.onnx")
        if not os.path.exists(target_yolo):
            shutil.copy(source_yolo, target_yolo)
            
        source_dw = f"{DWPOSE_CACHE_DIR}/dw-ll_ucoco_384.onnx"
        target_dw = os.path.join(target_dir, "dw-ll_ucoco_384.onnx")
        if not os.path.exists(target_dw):
            shutil.copy(source_dw, target_dw)

        # --- B. Load Florence-2 ---
        florence_path = f"{CHECKPOINT_DIR}/florence2"
        self.florence_processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_path, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device).eval()

        # --- C. Load SAM 2 ---
        sam2_ckpt = f"{CHECKPOINT_DIR}/sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" 
        self.sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # --- D. Load DWPose ---
        # Now it will find the files in the local site-packages folder
        self.dwpose_model = DwposeDetector.from_pretrained_default()
        if hasattr(self.dwpose_model, 'to'):
             self.dwpose_model.to(self.device)

        print("✅ All Models Loaded Successfully.")

    @modal.method()
    def process_image(self, image_bytes, text_prompt):
        import torch
        import numpy as np
        import cv2
        from PIL import Image

        # Load Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        
        # --- STEP 1: Florence (Detection) ---
        print(f"   Step 1: Detection for '{text_prompt}'...")
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
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(w, h)
        )
        
        bboxes = parsed_answer.get(task_prompt, {}).get('bboxes', [])
        
        if not bboxes:
            print("   ⚠️ No object detected. Returning original.")
            byte_arr = io.BytesIO()
            image.save(byte_arr, format='PNG')
            return byte_arr.getvalue()
            
        box_prompt = np.array(bboxes[0])
        print(f"   🎯 Detected Box: {box_prompt}")

        # --- STEP 2: SAM 2 (Segmentation) ---
        print(f"   Step 2: Segmentation...")
        self.sam2_predictor.set_image(np.array(image))
        masks, _, _ = self.sam2_predictor.predict(
            box=box_prompt[None, :],
            multimask_output=False
        )
        final_mask = masks[0]
        
        # Mask Processing
        binary_mask_img = (final_mask > 0).astype(np.uint8) * 255
        edges = cv2.Canny(binary_mask_img, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # --- STEP 3: DWPose (Skeleton) ---
        print(f"   Step 3: Generating DWPose Skeleton...")
        
        out_img, keypoints, source = self.dwpose_model(
            image,
            include_hand=True,
            include_face=False,
            include_body=True,
            image_and_json=True,
            detect_resolution=512
        )
        
        if not isinstance(out_img, Image.Image):
             out_img = Image.fromarray(out_img)

        skeleton_image = out_img.resize((w, h), Image.LANCZOS)
        
        # --- STEP 4: Final Overlay ---
        print(f"   Step 4: Composing Result...")
        
        base_np = np.array(skeleton_image.convert("RGB"))
        edge_mask = thick_edges > 100
        base_np[edge_mask] = [255, 255, 255]
        
        final_result = Image.fromarray(base_np)

        byte_arr = io.BytesIO()
        final_result.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

# --- 4. Local Entrypoint ---
@app.local_entrypoint()
def main(filepath: str = "bench.jpeg", prompt: str = "rod"):
    if not os.path.exists(filepath):
        print(f"❌ Error: File '{filepath}' not found.")
        return

    print(f"🚀 Reading '{filepath}' and sending to Modal...")
    with open(filepath, "rb") as f:
        image_bytes = f.read()

    pipeline = ImagePipeline()
    try:
        result_bytes = pipeline.process_image.remote(image_bytes, prompt)
    except Exception as e:
        print(f"❌ Remote execution failed: {e}")
        return

    output_path = "modal_output_dwpose.png"
    with open(output_path, "wb") as f:
        f.write(result_bytes)
    
    print(f"✅ Success! Result saved to: {output_path}")