import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import LoadImage, NODE_CLASS_MAPPINGS, SaveImage


def main():
    import_custom_nodes()
    with torch.inference_mode():
        text_prompt_jps = NODE_CLASS_MAPPINGS["Text Prompt (JPS)"]()
        text_prompt_jps_303 = text_prompt_jps.text_prompt(text="rod")

        loadimage = LoadImage()
        loadimage_330 = loadimage.load_image(image="bench-press.jpg")

        downloadandloadflorence2model = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadFlorence2Model"
        ]()
        downloadandloadflorence2model_88 = downloadandloadflorence2model.loadmodel(
            model="microsoft/Florence-2-large",
            precision="fp16",
            attention="sdpa",
            convert_to_safetensors=False,
        )

        florence2run = NODE_CLASS_MAPPINGS["Florence2Run"]()
        florence2run_87 = florence2run.encode(
            text_input=get_value_at_index(text_prompt_jps_303, 0),
            task="caption_to_phrase_grounding",
            fill_mask=False,
            keep_model_loaded=True,
            max_new_tokens=1024,
            num_beams=1,
            do_sample=True,
            output_mask_select="",
            seed=random.randint(1, 2**64),
            image=get_value_at_index(loadimage_330, 0),
            florence2_model=get_value_at_index(downloadandloadflorence2model_88, 0),
        )

        downloadandloadsam2model = NODE_CLASS_MAPPINGS["DownloadAndLoadSAM2Model"]()
        downloadandloadsam2model_270 = downloadandloadsam2model.loadmodel(
            model="sam2_hiera_base_plus.safetensors",
            segmentor="single_image",
            device="cuda",
            precision="fp32",
        )

        text_prompt_jps_319 = text_prompt_jps.text_prompt(text="Florence")

        text_prompt_jps_385 = text_prompt_jps.text_prompt(text="ObjectMask")

        text_prompt_jps_400 = text_prompt_jps.text_prompt(text="CannyImage\n")

        text_prompt_jps_419 = text_prompt_jps.text_prompt(text="Object_Skeleton_added")

        text_prompt_jps_437 = text_prompt_jps.text_prompt(text="SkeletonImage\n")

        florence2tocoordinates = NODE_CLASS_MAPPINGS["Florence2toCoordinates"]()
        cr_text_concatenate = NODE_CLASS_MAPPINGS["CR Text Concatenate"]()
        saveimage = SaveImage()
        splitbboxes = NODE_CLASS_MAPPINGS["SplitBboxes"]()
        sam2contextsegmentation = NODE_CLASS_MAPPINGS["Sam2ContextSegmentation"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        image_edge_detection_filter = NODE_CLASS_MAPPINGS[
            "Image Edge Detection Filter"
        ]()
        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        mask_dilate_region = NODE_CLASS_MAPPINGS["Mask Dilate Region"]()
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        cannyedgepreprocessor = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()

        for q in range(10):
            florence2tocoordinates_255 = florence2tocoordinates.segment(
                index="", batch=True, data=get_value_at_index(florence2run_87, 3)
            )

            cr_text_concatenate_318 = cr_text_concatenate.concat_text(
                separator="_", text2=get_value_at_index(text_prompt_jps_319, 0)
            )

            saveimage_320 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_concatenate_318, 0),
                images=get_value_at_index(florence2run_87, 0),
            )

            splitbboxes_373 = splitbboxes.splitbbox(
                index=1, bboxes=get_value_at_index(florence2tocoordinates_255, 1)
            )

            sam2contextsegmentation_325 = sam2contextsegmentation.segment(
                context_scale=2.0000000000000004,
                force_square_context=True,
                limit_tile_size=False,
                max_tile_size=1024,
                mask_filter_mode="disabled",
                min_mask_area=70,
                min_mask_area_percent=0.020000000000000004,
                fill_individual_masks=False,
                close_mask_gaps=0,
                dilate_masks=0,
                keep_model_loaded=True,
                mask_opacity=0.5,
                individual_objects=False,
                sam2_model=get_value_at_index(downloadandloadsam2model_270, 0),
                image=get_value_at_index(loadimage_330, 0),
                bboxes=get_value_at_index(splitbboxes_373, 0),
            )

            masktoimage_365 = masktoimage.EXECUTE_NORMALIZED(
                mask=get_value_at_index(sam2contextsegmentation_325, 0)
            )

            cr_text_concatenate_386 = cr_text_concatenate.concat_text(
                separator="_", text1=get_value_at_index(text_prompt_jps_385, 0)
            )

            saveimage_387 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_concatenate_386, 0),
                images=get_value_at_index(masktoimage_365, 0),
            )

            cr_text_concatenate_401 = cr_text_concatenate.concat_text(
                separator="_", text1=get_value_at_index(text_prompt_jps_400, 0)
            )

            masktoimage_421 = masktoimage.EXECUTE_NORMALIZED(
                mask=get_value_at_index(sam2contextsegmentation_325, 0)
            )

            image_edge_detection_filter_423 = image_edge_detection_filter.image_edges(
                mode="normal", image=get_value_at_index(masktoimage_421, 0)
            )

            saveimage_402 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_concatenate_401, 0),
                images=get_value_at_index(image_edge_detection_filter_423, 0),
            )

            imagetomask_410 = imagetomask.EXECUTE_NORMALIZED(
                channel="red",
                image=get_value_at_index(image_edge_detection_filter_423, 0),
            )

            mask_dilate_region_411 = mask_dilate_region.dilate_region(
                iterations=3, masks=get_value_at_index(imagetomask_410, 0)
            )

            masktoimage_412 = masktoimage.EXECUTE_NORMALIZED(
                mask=get_value_at_index(mask_dilate_region_411, 0)
            )

            getimagesize_428 = getimagesize.EXECUTE_NORMALIZED(
                image=get_value_at_index(loadimage_330, 0),
                # unique_id=7595546873847430331,
            )

            dwpreprocessor_427 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=1024,
                bbox_detector="yolo_nas_l_fp16.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                scale_stick_for_xinsr_cn="disable",
                image=get_value_at_index(loadimage_330, 0),
            )

            imageresizekjv2_426 = imageresizekjv2.resize(
                width=get_value_at_index(getimagesize_428, 0),
                height=get_value_at_index(getimagesize_428, 1),
                upscale_method="nearest-exact",
                keep_proportion="pad_edge",
                pad_color="0, 0, 0",
                crop_position="top",
                divisible_by=2,
                device="gpu",
                image=get_value_at_index(dwpreprocessor_427, 0),
                unique_id=15132394062155144486,
            )

            imagecompositemasked_415 = imagecompositemasked.EXECUTE_NORMALIZED(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(imageresizekjv2_426, 0),
                source=get_value_at_index(masktoimage_412, 0),
                mask=get_value_at_index(mask_dilate_region_411, 0),
            )

            cr_text_concatenate_418 = cr_text_concatenate.concat_text(
                separator="_", text2=get_value_at_index(text_prompt_jps_419, 0)
            )

            saveimage_417 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_concatenate_418, 0),
                images=get_value_at_index(imagecompositemasked_415, 0),
            )

            cannyedgepreprocessor_424 = cannyedgepreprocessor.execute(
                low_threshold=100,
                high_threshold=255,
                resolution=512,
                image=get_value_at_index(masktoimage_421, 0),
            )

            cr_text_concatenate_438 = cr_text_concatenate.concat_text(
                separator="_", text2=get_value_at_index(text_prompt_jps_437, 0)
            )

            saveimage_439 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_concatenate_438, 0),
                images=get_value_at_index(imageresizekjv2_426, 0),
            )


if __name__ == "__main__":
    main()
