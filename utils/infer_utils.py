from PIL import Image, ImageDraw
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
import os
import numpy as np
import torch

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


def generate_video_from_image(
        pipeline,
        image_path,
        save_dir,
        global_step,
        fps=7,
        duration_sec=4,
        height=576,
        width=1024,
        decode_chunk_size=8,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        device="cuda",
        mixed_precision="fp16"
):
    """
    Generate videos from input images using a StableVideoDiffusionPipeline.

    Args:
        pipeline: StableVideoDiffusionPipeline instance
        image_path (str): Path to the input image file
        save_dir (str): Base directory to save outputs
        global_step (int): Current training step number
        fps (int, optional): Frames per second. Defaults to 7.
        duration_sec (int, optional): Duration of output video in seconds. Defaults to 4.
        height (int, optional): Output video height. Defaults to 576.
        width (int, optional): Output video width. Defaults to 1024.
        decode_chunk_size (int, optional): Chunk size for decoding. Defaults to 8.
        motion_bucket_id (int, optional): Motion bucket ID. Defaults to 127.
        noise_aug_strength (float, optional): Noise augmentation strength. Defaults to 0.02.
        device (str, optional): Device to run inference on. Defaults to "cuda".
        mixed_precision (str, optional): Mixed precision mode. Defaults to "fp16".
    """
    # Create step-specific directory
    step_dir = os.path.join(save_dir, f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    # Calculate number of frames based on fps and duration
    num_frames = fps * duration_sec

    # Get original image filename without extension
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    with torch.autocast(
            device,
            enabled=mixed_precision == "fp16",
    ):
        try:
            # Load and process image
            input_image = load_image(image_path).resize((width, height))

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

            # Define output path using original image name
            out_file = os.path.join(step_dir, f"{img_name}.mp4")

            # Convert frames to numpy arrays
            video_frames = [np.array(frame) for frame in video_frames]

            # Export to video file
            export_to_gif(video_frames, out_file, fps)

            print(f"Successfully generated video for {img_name} at step {global_step}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    return out_file
