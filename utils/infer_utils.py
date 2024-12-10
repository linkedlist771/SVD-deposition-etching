from typing import List
import re
from PIL import Image, ImageDraw
import cv2
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
import os
import numpy as np
import torch

from loguru import logger
from tqdm import tqdm


def natural_sort_key(s):
    """按照自然数顺序排序文件名"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def generate_images_for_validation_folders(
    pipeline,
    validation_images_folders: List[str],
    output_dir: str,
    num_frames: int,
    fps=7,
    height=576,
    width=1024,
    decode_chunk_size=8,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    device="cuda",
    mixed_precision="fp16",
):
    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    assert len(validation_images_folders) > 0, "No validation folders provided"
    val_parent_dir = os.path.dirname(validation_images_folders[0])
    # Outer progress bar for folders

    for folder_path in tqdm(validation_images_folders, desc="Processing folders"):
        base_dir = os.path.basename(folder_path)
        _output_dir = os.path.join(output_dir, base_dir)
        os.makedirs(_output_dir, exist_ok=True)
        logger.info(f"Processing folder: {folder_path}")
        logger.info(f"Output subdirectory: {_output_dir}")

        original_images = os.listdir(folder_path)
        original_images.sort(key=natural_sort_key)
        logger.info(f"Found {len(original_images)} images in folder")

        with torch.autocast(
            device,
            enabled=mixed_precision == "fp16",
        ):
            # Inner progress bar for images within each folder
            for i in tqdm(
                range(0, len(original_images), num_frames), #
                desc="Processing images",
                leave=False,
            ):
                original_image_paths = original_images[i]
                idx = int(os.path.splitext(original_image_paths)[0]) # 原始的图片的里面的idx是
                # 1.png 2.png ... 这样的
                suffix = os.path.splitext(original_image_paths)[1]

                logger.debug(f"Processing image {original_image_paths}")
                image_full_path = os.path.join(folder_path, original_image_paths)
                input_image = load_image(image_full_path).resize((width, height))
                try:
                    # Generate video frames
                    video_frames = pipeline(
                        input_image,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        decode_chunk_size=decode_chunk_size,
                        motion_bucket_id=motion_bucket_id,
                        fps=fps,
                        noise_aug_strength=noise_aug_strength,
                    ).frames[0]

                    # Convert frames to numpy arrays
                    video_frames = [np.array(frame) for frame in video_frames]

                    # Save the generated frames
                    for frame_idx, video_frame in enumerate(video_frames):
                        output_image_path = os.path.join(
                            _output_dir, f"{idx + frame_idx + 1}{suffix}"  # generating the next frames.....
                        ) # 所以保存的时候是后面的12帧。
                        cv2.imwrite(
                            output_image_path,
                            cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR),
                        )

                    logger.debug(
                        f"Successfully generated and saved {len(video_frames)} frames for image {original_image_paths}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing image {original_image_paths}: {str(e)}"
                    )
                    continue

        logger.info(f"Completed processing folder: {folder_path}")

    logger.success("Completed processing all validation folders")
