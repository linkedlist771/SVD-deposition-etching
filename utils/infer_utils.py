from typing import List
import re
from PIL import Image, ImageDraw
from altair.examples.scatter_linked_table import origin
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
import os
import numpy as np
import torch


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
        mixed_precision="fp16"
):
    os.makedirs(output_dir, exist_ok=True)
    # 首先判断这个文件夹是否存咋， 如果不存在则创建。
    # 然后创建和validation_images_folders 一样的子文件夹架构
    # 获取验证文件夹的共同基础路径
    assert len(validation_images_folders) > 0
    common_prefix = os.path.commonpath(validation_images_folders)

    # 创建对应的子文件夹结构
    for folder_path in validation_images_folders:
        # 获取相对于共同基础路径的相对路径
        relative_path = os.path.relpath(folder_path, common_prefix)

        # 构建输出子目录路径
        output_subdir = os.path.join(output_dir, relative_path)

        # 创建子目录
        os.makedirs(output_subdir, exist_ok=True)

        original_images = os.listdir(folder_path)
        original_images.sort(key=natural_sort_key)
        # set sep as the number of frames

        with torch.autocast(
                str(accelerator.device).replace(":0", ""),
                enabled=accelerator.mixed_precision == "fp16",
        ):
            for i in range(0, len(original_images), num_frames):
                original_image_paths =  original_images[i]
                idx = int(os.path.splitext(original_image_paths)[0])
                suffix = os.path.splitext(original_image_paths)[1]
                input_image = load_image(original_image_paths).resize((width, height))
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
                # save the following frames to the output_subdir with the name of the
                for video_frame in video_frames:
                    output_image_path = os.path.join(output_subdir, f"{idx}{suffix}")

                    idx += 1




