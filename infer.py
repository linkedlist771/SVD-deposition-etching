import argparse
import os
import torch
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from utils.infer_utils import generate_images_for_validation_folders

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run inference with Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        required=True,
        type=str,
        help="Base directory containing folders of input images."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained model or a model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained model or a model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames to generate per sample."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of the generated frames."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Height of the generated frames."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The directory where the outputs will be saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A random seed for reproducible generation."
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.99,
        help="Ratio of training data to total data",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Optionally set the seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Load the UNet model
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.unet_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )

    # Load the stable video diffusion pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )

    # Move pipeline to GPU
    pipe.to("cuda:0")

    # Set up dataset splitting
    split_ratio = args.split_ratio
    folders = os.listdir(args.base_folder)
    folders = [os.path.join(args.base_folder, f) for f in folders if os.path.isdir(os.path.join(args.base_folder, f))]
    n_train = int(len(folders) * split_ratio)
    validation_images_folders = folders[n_train:]

    # Run inference on validation set
    generate_images_for_validation_folders(
        pipeline=pipe,
        validation_images_folders=validation_images_folders,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )