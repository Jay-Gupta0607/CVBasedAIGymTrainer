#for modal

import modal
import os
import sys
import shutil
import random
from typing import List

# 1. Define the Container Image
# We replicate your Colab environment here
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "ffmpeg")
    
    # Install Python Dependencies from your setup file
    .pip_install(
        "accelerate", "transformers>=4.28.1", "safetensors>=0.4.2", "aiohttp", 
        "pyyaml", "Pillow", "scipy", "tqdm", "psutil", "tokenizers>=0.13.3", 
        "torchsde", "kornia>=0.7.1", "spandrel", "soundfile", "sentencepiece",
        "comfyui-workflow-templates", "comfyui-embedded-docs", "av", 
        "comfy_kitchen", "piexif", "ultralytics", "onnxruntime-gpu", 
        "segment_anything", "deepdiff", "rapidfuzz", "sageattention", 
        "surrealist", "boto3", "redis", "fal_client", "replicate", "GitPython"
    )
    .pip_install("torch", "torchvision", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy==1.26.4") # Force numpy version as per your script
    .pip_install("git+https://github.com/facebookresearch/sam2") # SAM2 Dependency

    # 2. Clone ComfyUI
    .run_commands("git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI")
    
    # 3. Install Custom Nodes (Consolidated list from your file)
    .run_commands(
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/romandev-codex/ComfyUI-Downloader",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/lrzjason/Comfyui-QwenEditUtils.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/chrisgoringe/cg-use-everywhere.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/rgthree/rgthree-comfy.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes.git",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/jags111/efficiency-nodes-comfyui", # Using jags111 fork as listed last
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/mcmonkeyprojects/sd-dynamic-thresholding",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/zwaigani/ComfyUI-LoRA-stacker",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/traugdor/ComfyUI-quadMoons-nodes",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/alexopus/ComfyUI-Image-Saver",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/WASasquatch/was-node-suite-comfyui",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Smirnov75/ComfyUI-mxToolkit",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/crystian/ComfyUI-Crystools",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/yolain/ComfyUI-Easy-Use",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/AIExplorer25/ComfyUI_AutoDownloadModels",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-Florence2",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/kijai/ComfyUI-segment-anything-2",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_essentials",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/un-seen/comfyui-tensorops",
        "cd /root/ComfyUI/custom_nodes && git clone https://github.com/MicheleGuidi/ComfyUI-Contextual-SAM2",
        "pip install -r /root/ComfyUI/requirements.txt"
    )

    # 4. Download Models (Pre-cache them to make the app fast)
    .run_commands(
        # FLUX GGUF
        "wget -c -P /root/ComfyUI/models/diffusion_models https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_0.gguf",
        
        # FLUX VAE
        "wget -c -P /root/ComfyUI/models/vae https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/vae/diffusion_pytorch_model.safetensors",
        
        # Florence 2
        "mkdir -p /root/ComfyUI/models/LLM",
        "wget -O /root/ComfyUI/models/LLM/florence2.safetensors https://huggingface.co/microsoft/Florence-2-large/resolve/main/model.safetensors?download=true",
        
        # SAM 2 Models
        "mkdir -p /root/ComfyUI/models/sam2",
        "wget -O /root/ComfyUI/models/sam2/sam2_hiera_base_plus.safetensors https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "wget -O /root/ComfyUI/models/sam2/sam2.1_hiera_small.safetensors https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt",
        "wget -O /root/ComfyUI/models/sam2/sam2.1_hiera_base_plus.safetensors https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt",
        
        # ControlNet / Pose Models (Found in your logs)
        "mkdir -p /root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16",
        "wget -O /root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/yolo-nas-fp16/yolo_nas_l_fp16.onnx https://huggingface.co/hr16/yolo-nas-fp16/resolve/main/yolo_nas_l_fp16.onnx",
        "mkdir -p /root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5",
        "wget -O /root/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt"
    )
)

app = modal.App("my-comfy-workflow", image=image)

# Helper functions for the script
def get_value_at_index(obj, index):
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

@app.function(gpu="T4", timeout=1200) # Increased timeout for first run
def run_workflow(user_prompt: str, image_bytes: bytes):
    import torch
    import sys
    import random
    
    # 1. Setup ComfyUI paths
    sys.path.append("/root/ComfyUI")
    from nodes import NODE_CLASS_MAPPINGS, LoadImage, SaveImage
    
    # 2. Handle Input Image
    # Save the bytes to a temp file so LoadImage can read it
    input_path = "/root/ComfyUI/input/temp_input.jpg"
    with open(input_path, "wb") as f:
        f.write(image_bytes)

    # 3. Initialization
    # We must initialize custom nodes manually
    import asyncio
    import execution
    import server
    from nodes import init_extra_nodes
    
    # Setup server context (Mocking the server)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())

    # 4. The Workflow Logic (Adapted from your converted_workflowLatest.py)
    with torch.inference_mode():
        # --- LOAD IMAGE ---
        loadimage = LoadImage()
        # "temp_input.jpg" is automatically found in the input folder
        loadimage_330 = loadimage.load_image(image="temp_input.jpg") 

        # --- TEXT PROMPTS ---
        text_prompt_jps = NODE_CLASS_MAPPINGS["Text Prompt (JPS)"]()
        # DYNAMIC PROMPT HERE:
        text_prompt_jps_303 = text_prompt_jps.text_prompt(text=user_prompt) 

        # --- FLORENCE 2 ---
        downloadandloadflorence2model = NODE_CLASS_MAPPINGS["DownloadAndLoadFlorence2Model"]()
        downloadandloadflorence2model_88 = downloadandloadflorence2model.loadmodel(
            model="microsoft/Florence-2-large", precision="fp16", attention="sdpa", convert_to_safetensors=False
        )

        florence2run = NODE_CLASS_MAPPINGS["Florence2Run"]()
        florence2run_87 = florence2run.encode(
            text_input=get_value_at_index(text_prompt_jps_303, 0),
            task="caption_to_phrase_grounding",
            fill_mask=False, keep_model_loaded=True, max_new_tokens=1024, num_beams=1,
            do_sample=True, output_mask_select="", seed=random.randint(1, 2**64),
            image=get_value_at_index(loadimage_330, 0),
            florence2_model=get_value_at_index(downloadandloadflorence2model_88, 0),
        )

        # --- SAM 2 LOAD ---
        downloadandloadsam2model = NODE_CLASS_MAPPINGS["DownloadAndLoadSAM2Model"]()
        downloadandloadsam2model_270 = downloadandloadsam2model.loadmodel(
            model="sam2_hiera_base_plus.safetensors", segmentor="single_image", device="cuda", precision="fp32"
        )

        # --- PROCESSING LOOP (Simplified to Single Run for API) ---
        # Note: I removed the 'for q in range(10)' loop because usually APIs generate 1 image at a time.
        # If you need batch processing, we can add it back.
        
        florence2tocoordinates = NODE_CLASS_MAPPINGS["Florence2toCoordinates"]()
        florence2tocoordinates_255 = florence2tocoordinates.segment(
            index="", batch=True, data=get_value_at_index(florence2run_87, 3)
        )

        splitbboxes = NODE_CLASS_MAPPINGS["SplitBboxes"]()
        splitbboxes_373 = splitbboxes.splitbbox(
            index=1, bboxes=get_value_at_index(florence2tocoordinates_255, 1)
        )

        sam2contextsegmentation = NODE_CLASS_MAPPINGS["Sam2ContextSegmentation"]()
        sam2contextsegmentation_325 = sam2contextsegmentation.segment(
            context_scale=2.0, force_square_context=True, limit_tile_size=False, max_tile_size=1024,
            mask_filter_mode="disabled", min_mask_area=70, min_mask_area_percent=0.02,
            fill_individual_masks=False, close_mask_gaps=0, dilate_masks=0, keep_model_loaded=True,
            mask_opacity=0.5, individual_objects=False,
            sam2_model=get_value_at_index(downloadandloadsam2model_270, 0),
            image=get_value_at_index(loadimage_330, 0),
            bboxes=get_value_at_index(splitbboxes_373, 0),
        )

        # --- MASK TO IMAGE & EDGES ---
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        masktoimage_421 = masktoimage.EXECUTE_NORMALIZED(
            mask=get_value_at_index(sam2contextsegmentation_325, 0)
        )

        image_edge_detection_filter = NODE_CLASS_MAPPINGS["Image Edge Detection Filter"]()
        image_edge_detection_filter_423 = image_edge_detection_filter.image_edges(
            mode="normal", image=get_value_at_index(masktoimage_421, 0)
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_410 = imagetomask.EXECUTE_NORMALIZED(
            channel="red", image=get_value_at_index(image_edge_detection_filter_423, 0)
        )

        mask_dilate_region = NODE_CLASS_MAPPINGS["Mask Dilate Region"]()
        mask_dilate_region_411 = mask_dilate_region.dilate_region(
            iterations=3, masks=get_value_at_index(imagetomask_410, 0)
        )

        # --- FIX: MANUAL GET IMAGE SIZE (Replacing broken node) ---
        temp_img = get_value_at_index(loadimage_330, 0)
        # [Width, Height]
        getimagesize_428 = [temp_img.shape[2], temp_img.shape[1]] 

        # --- POSE ESTIMATION ---
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        dwpreprocessor_427 = dwpreprocessor.estimate_pose(
            detect_hand="enable", detect_body="enable", detect_face="enable", resolution=1024,
            bbox_detector="yolo_nas_l_fp16.onnx", pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
            scale_stick_for_xinsr_cn="disable", image=get_value_at_index(loadimage_330, 0),
        )

        # --- RESIZE (With unique_id FIX) ---
        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        imageresizekjv2_426 = imageresizekjv2.resize(
            width=get_value_at_index(getimagesize_428, 0),
            height=get_value_at_index(getimagesize_428, 1),
            upscale_method="nearest-exact", keep_proportion="pad_edge", pad_color="0, 0, 0",
            crop_position="top", divisible_by=2, device="gpu",
            image=get_value_at_index(dwpreprocessor_427, 0),
            unique_id=0 # <--- Fixed required argument
        )

        # --- SAVING THE FINAL OUTPUT ---
        # We use standard Comfy SaveImage, but we need to know where it puts it
        saveimage = SaveImage()
        saveimage_439 = saveimage.save_images(
            filename_prefix="modal_output",
            images=get_value_at_index(imageresizekjv2_426, 0),
        )

        # 5. Retrieve and Return Image
        # ComfyUI saves to /root/ComfyUI/output
        output_dir = "/root/ComfyUI/output"
        files = os.listdir(output_dir)
        # Sort by modification time to get the one we just made
        files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        latest_file = os.path.join(output_dir, files[-1])
        
        with open(latest_file, "rb") as f:
            return f.read()

@app.local_entrypoint()
def main():
    # Read a local image to send to the server
    image_filename = "test_input.jpg" # Make sure this file exists on your laptop!
    
    if not os.path.exists(image_filename):
        print(f"Please create a dummy image named '{image_filename}' to test the script.")
        return

    with open(image_filename, "rb") as f:
        img_bytes = f.read()

    print("Sending request to Modal...")
    output_bytes = run_workflow.remote("rod", img_bytes)
    
    with open("final_output.png", "wb") as f:
        f.write(output_bytes)
    print("Success! Saved final_output.png")