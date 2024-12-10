#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# ...
# Script modified to remove accelerate usage.

import argparse
import random
import logging
import math
import os
import re
from os.path import split

import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, RandomSampler, DataLoader
import transformers
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    UNetSpatioTemporalConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from loguru import logger

check_min_version("0.24.0.dev0")

def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

class DummyDataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        split="train",
        split_ratio=0.8,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
    ):
        self.num_samples = num_samples
        self.base_folder = base_folder
        folders = os.listdir(self.base_folder)

        n_train = int(len(folders) * split_ratio)
        if split == "train":
            self.folders = folders[:n_train]
        else:
            self.folders = folders[n_train:]

        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        self.available_samples = 0
        for folder in self.folders:
            folder_path = os.path.join(self.base_folder, folder)
            frames = os.listdir(folder_path)
            if len(frames) >= self.sample_frames:
                self.available_samples += len(frames) - self.sample_frames + 1

    def __len__(self):
        return self.available_samples

    def natural_sort_key(self, s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

    def __getitem__(self, idx):
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = os.listdir(folder_path)
        frames.sort(key=self.natural_sort_key)

        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames."
            )

        start_idx = random.randint(0, len(frames) - self.sample_frames)
        selected_frames = frames[start_idx : start_idx + self.sample_frames]

        pixel_values = torch.empty(
            (self.sample_frames, self.channels, self.height, self.width)
        )

        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(folder_path, frame_name)
            with Image.open(frame_path) as img:
                img = img.convert("RGB")

                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                img_normalized = img_tensor / 127.5 - 1

                if self.channels == 3:
                    img_normalized = img_normalized.permute(2, 0, 1)
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(dim=2, keepdim=True)

                pixel_values[i] = img_normalized
        return {"pixel_values": pixel_values}

def _compute_padding(kernel_size):
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1
    )

    out = output.view(b, c, h, w)
    return out

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (
        torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
        - window_size // 2
    ).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])
    return out

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners
    )
    return output

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def export_to_gif(frames, output_gif_path, fps):
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

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents

def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Video Diffusion.")
    parser.add_argument("--base_folder",required=True,type=str,)
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None,required=True)
    parser.add_argument("--revision",type=str,default=None,required=False)
    parser.add_argument("--num_frames",type=int,default=25,)
    parser.add_argument("--width",type=int,default=1024,)
    parser.add_argument("--height",type=int,default=576,)
    parser.add_argument("--num_validation_images",type=int,default=1,)
    parser.add_argument("--validation_steps",type=int,default=500,)
    parser.add_argument("--output_dir",type=str,default="./outputs",)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size",type=int,default=1,)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps",type=int,default=None,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    parser.add_argument("--gradient_checkpointing",action="store_true",)
    parser.add_argument("--learning_rate",type=float,default=1e-4,)
    parser.add_argument("--scale_lr",action="store_true",default=False,)
    parser.add_argument("--lr_scheduler",type=str,default="constant",)
    parser.add_argument("--lr_warmup_steps",type=int,default=500,)
    parser.add_argument("--conditioning_dropout_prob",type=float,default=0.1,)
    parser.add_argument("--use_8bit_adam",action="store_true",)
    parser.add_argument("--allow_tf32",action="store_true",)
    parser.add_argument("--use_ema", action="store_true",)
    parser.add_argument("--non_ema_revision",type=str,default=None,required=False,)
    parser.add_argument("--num_workers",type=int,default=8,)
    parser.add_argument("--adam_beta1",type=float,default=0.9,)
    parser.add_argument("--adam_beta2",type=float,default=0.999,)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2,)
    parser.add_argument("--adam_epsilon",type=float,default=1e-08,)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,)
    parser.add_argument("--push_to_hub",action="store_true",)
    parser.add_argument("--hub_token",type=str,default=None,)
    parser.add_argument("--hub_model_id",type=str,default=None,)
    parser.add_argument("--logging_dir",type=str,default="logs",)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--report_to",type=str,default="tensorboard",)
    parser.add_argument("--local_rank",type=int,default=-1,)
    parser.add_argument("--checkpointing_steps",type=int,default=500,)
    parser.add_argument("--checkpoints_total_limit",type=int,default=2,)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,)
    parser.add_argument("--enable_xformers_memory_efficient_attention",action="store_true",)
    parser.add_argument("--pretrain_unet",type=str,default=None,)
    parser.add_argument("--split_ratio",type=float,default=0.99,)

    args = parser.parse_args()
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    return args

def download_image(url):
    original_image = (
        lambda image_url_or_path: (
            load_image(image_url_or_path)
            if urlparse(image_url_or_path).scheme
            else PIL.Image.open(image_url_or_path).convert("RGB")
        )
    )(url)
    return original_image

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Install wandb for wandb logging.")
        import wandb

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name,
            exist_ok=True,
            token=args.hub_token,
        ).repo_id

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="feature_extractor",
        revision=args.revision,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant="fp16",
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        (args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet),
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder = image_encoder.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            model_cls=UNetSpatioTemporalConditionModel,
            model_config=unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 has known issues.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # scale lr
    if args.scale_lr:
        # single device only
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.per_gpu_batch_size
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    for name, param in unet.named_parameters():
        if "temporal_transformer_block" in name:
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if True:
        rec_txt1 = open("params_freeze.txt", "w")
        rec_txt2 = open("params_train.txt", "w")
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f"{name}\n")
            else:
                rec_txt2.write(f"{name}\n")
        rec_txt1.close()
        rec_txt2.close()

    global_batch_size = args.per_gpu_batch_size
    train_dataset = DummyDataset(
        args.base_folder,
        split_ratio=args.split_ratio,
        width=args.width,
        height=args.height,
        sample_frames=args.num_frames,
    )
    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    val_dataset = DummyDataset(
        args.base_folder,
        split="val",
        split_ratio=args.split_ratio,
        width=args.width,
        height=args.height,
        sample_frames=args.num_frames,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError("Mismatch in time embedding dimension.")
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    total_batch_size = (
        args.per_gpu_batch_size
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (with accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0
    best_val_loss = float("inf")
    best_val_outputs = None

    # Resume from checkpoint if needed
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist.")
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            full_path = os.path.join(args.output_dir, path)
            checkpoint = torch.load(os.path.join(full_path, "training_state.pt"))
            unet.load_state_dict(torch.load(os.path.join(full_path, "unet.pt")))
            optimizer.load_state_dict(torch.load(os.path.join(full_path, "optimizer.pt")))
            lr_scheduler.load_state_dict(torch.load(os.path.join(full_path, "scheduler.pt")))
            if args.use_ema:
                ema_unet.load_state_dict(torch.load(os.path.join(full_path, "ema_unet.pt")))

            global_step = checkpoint["global_step"]
            first_epoch = checkpoint["epoch"]
            resume_step = checkpoint["resume_step"]

    # Move unet to device
    unet.to(device, dtype=weight_dtype)

    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            pixel_values = batch["pixel_values"].to(weight_dtype).to(device)
            conditional_pixel_values = pixel_values[:, 0:1, :, :, :]

            latents = tensor_to_vae_latent(pixel_values, vae)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            cond_sigmas = rand_log_normal(
                shape=[bsz,],
                loc=-3.0,
                scale=0.5,
                device=device
            )
            noise_aug_strength = cond_sigmas[0]
            cond_sigmas = cond_sigmas[:, None, None, None, None]
            conditional_pixel_values = (
                torch.randn_like(conditional_pixel_values) * cond_sigmas
                + conditional_pixel_values
            )
            conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
            conditional_latents = conditional_latents / vae.config.scaling_factor

            sigmas = rand_log_normal(
                shape=[bsz,],
                loc=0.7,
                scale=1.6,
                device=device
            )
            sigmas = sigmas[:, None, None, None, None]
            noisy_latents = latents + noise * sigmas
            timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas.reshape(-1)]).to(device)

            inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
            encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].float())
            added_time_ids = _get_add_time_ids(
                7, 127, noise_aug_strength, encoder_hidden_states.dtype, bsz
            ).to(device)

            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(bsz, device=device)
                prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                null_conditioning = torch.zeros_like(encoder_hidden_states)
                encoder_hidden_states = torch.where(
                    prompt_mask,
                    null_conditioning.unsqueeze(1),
                    encoder_hidden_states.unsqueeze(1),
                )

                image_mask_dtype = conditional_latents.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(bsz, 1, 1, 1)
                conditional_latents = image_mask * conditional_latents

            conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
            inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

            target = latents
            model_pred = unet(inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample

            c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
            c_skip = 1 / (sigmas**2 + 1)
            denoised_latents = model_pred * c_out + c_skip * noisy_latents
            weighing = (1 + sigmas**2) * (sigmas**-2.0)

            loss = torch.mean(
                (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                dim=1,
            ).mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if args.use_ema:
                ema_unet.step(unet.parameters())

            global_step += 1
            progress_bar.update(1)
            train_loss += loss.item() / args.gradient_accumulation_steps
            if global_step % args.gradient_accumulation_steps == 0:
                print(f"Step: {global_step}, Train Loss: {train_loss}")
                train_loss = 0.0

            # Checkpoints
            if global_step % args.checkpointing_steps == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(
                            f"removing checkpoints: {', '.join(removing_checkpoints)}"
                        )
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                torch.save(unet.state_dict(), os.path.join(save_path, "unet.pt"))
                torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                if args.use_ema:
                    torch.save(ema_unet.state_dict(), os.path.join(save_path, "ema_unet.pt"))
                torch.save({"global_step": global_step, "epoch": epoch, "resume_step": step},
                           os.path.join(save_path, "training_state.pt"))
                logger.info(f"Saved state to {save_path}")

            # Validation
            if (global_step % args.validation_steps == 0) or (global_step == 1):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} videos."
                )

                unet.eval()
                val_loss = 0
                val_outputs_current = []
                for val_step, val_batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        pixel_values = val_batch["pixel_values"].to(weight_dtype).to(device)
                        conditional_pixel_values = pixel_values[:, 0:1, :, :, :]

                        latents = tensor_to_vae_latent(pixel_values, vae)
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        cond_sigmas = rand_log_normal(
                            shape=[bsz,], loc=-3.0, scale=0.5, device=device
                        )
                        noise_aug_strength = cond_sigmas[0]
                        cond_sigmas = cond_sigmas[:, None, None, None, None]
                        conditional_pixel_values = (
                            torch.randn_like(conditional_pixel_values) * cond_sigmas
                            + conditional_pixel_values
                        )
                        conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                        conditional_latents = conditional_latents / vae.config.scaling_factor

                        sigmas = rand_log_normal(
                            shape=[bsz,], loc=0.7, scale=1.6, device=device
                        )
                        sigmas = sigmas[:, None, None, None, None]
                        noisy_latents = latents + noise * sigmas
                        timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas.reshape(-1)]).to(device)

                        inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                        encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].float())
                        added_time_ids = _get_add_time_ids(
                            7, 127, noise_aug_strength, encoder_hidden_states.dtype, bsz
                        ).to(device)

                        if args.conditioning_dropout_prob is not None:
                            random_p = torch.rand(bsz, device=device)
                            prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                            null_conditioning = torch.zeros_like(encoder_hidden_states)
                            encoder_hidden_states = torch.where(
                                prompt_mask,
                                null_conditioning.unsqueeze(1),
                                encoder_hidden_states.unsqueeze(1),
                            )

                            image_mask_dtype = conditional_latents.dtype
                            image_mask = 1 - (
                                (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                                * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                            )
                            image_mask = image_mask.reshape(bsz, 1, 1, 1)
                            conditional_latents = image_mask * conditional_latents

                        conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                        inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)

                        target = latents
                        model_pred = unet(inp_noisy_latents, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample

                        c_out = -sigmas / ((sigmas**2 + 1) ** 0.5)
                        c_skip = 1 / (sigmas**2 + 1)
                        denoised_latents = model_pred * c_out + c_skip * noisy_latents
                        weighing = (1 + sigmas**2) * (sigmas**-2.0)

                        loss = torch.mean(
                            (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                            dim=1,
                        ).mean()

                        val_outputs_current.append(
                            {
                                "inputs": conditional_pixel_values.cpu().numpy(),
                                "preds": denoised_latents.cpu().numpy(),
                                "trues": target.cpu().numpy(),
                            }
                        )

                        val_loss += loss.item()

                val_loss /= len(val_dataloader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_outputs = val_outputs_current
                print(f"Step: {global_step}, Val Loss: {val_loss}")

                if args.use_ema:
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unet,
                    image_encoder=image_encoder,
                    vae=vae,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(device)
                pipeline.set_progress_bar_config(disable=True)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=(args.mixed_precision=="fp16")):
                    for val_img_idx in range(args.num_validation_images):
                        num_frames = args.num_frames
                        video_frames = pipeline(
                            load_image("demo.png").resize((args.width, args.height)),
                            height=args.height,
                            width=args.width,
                            num_frames=num_frames,
                            decode_chunk_size=8,
                            motion_bucket_id=127,
                            fps=7,
                            noise_aug_strength=0.02,
                        ).frames[0]

                        val_save_dir = os.path.join(args.output_dir, "validation_images")
                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)
                        out_file = os.path.join(
                            val_save_dir,
                            f"step_{global_step}_val_img_{val_img_idx}.mp4",
                        )
                        for i in range(num_frames):
                            img = video_frames[i]
                            video_frames[i] = np.array(img)
                        export_to_gif(video_frames, out_file, 8)

                if args.use_ema:
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    unet = unet
    if args.use_ema:
        ema_unet.copy_to(unet.parameters())

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
    )
    pipeline.save_pretrained(args.output_dir)

    if args.push_to_hub:
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

    if best_val_outputs is not None:
        results_all = {}
        for k in best_val_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in best_val_outputs], axis=0)

        mae = np.mean(np.abs(results_all["preds"] - results_all["trues"]))
        mse = np.mean((results_all["preds"] - results_all["trues"]) ** 2)
        results_all["metrics"] = np.array([mae, mse])

        save_dir = os.path.join(args.output_dir, "saved")
        os.makedirs(save_dir, exist_ok=True)

        for np_data in ["metrics", "inputs", "trues", "preds"]:
            np.save(os.path.join(save_dir, f"{np_data}.npy"), results_all[np_data])

        print(f"Best validation MAE: {mae:.4f}, MSE: {mse:.4f}")

if __name__ == "__main__":
    main()