import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import cv2
import os
os.environ["WANDB_DISABLED"] = "true"

class ImageVideoDataset(Dataset):
    def __init__(self, image_path, video_path):
        self.image = Image.open(image_path).convert("RGB")
        self.video_frames = self.load_video_frames(video_path)

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        return {
            "input_images": self.image,
            "target_frames": self.video_frames[idx]
        }

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
            success, frame = cap.read()
        cap.release()
        return frames

class VideoDataCollator:
    def __call__(self, examples):
        input_images = [example["input_images"] for example in examples]
        target_frames = [example["target_frames"] for example in examples]
        return {
            "input_images": input_images,
            "target_frames": target_frames
        }

# Load pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Get model components
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler
image_encoder = pipe.image_encoder
image_processor = pipe.image_processor  # Use the image_processor

class VideoDiffusionModel(torch.nn.Module):
    def __init__(self, vae, unet, scheduler, image_encoder, image_processor):
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet
        self.scheduler = scheduler
        self.image_encoder = image_encoder.eval()
        self.image_processor = image_processor

        # Freeze VAE and image encoder parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_images, target_frames):
        device = self.unet.device

        # Process input images using image_processor
        input_images = self.image_processor(images=input_images, return_tensors="pt").pixel_values.to(device)

        # Get encoder hidden states
        encoder_hidden_states = self.image_encoder(input_images).last_hidden_state

        # Process target frames
        target_frames = self.image_processor(images=target_frames, return_tensors="pt").pixel_values.to(device)
        target_frames = (target_frames - 0.5) * 2  # Normalize to [-1, 1]

        # Encode target frames
        with torch.no_grad():
            latents = self.vae.encode(target_frames).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calculate loss
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
        return {"loss": loss}

# Initialize model
model = VideoDiffusionModel(vae, unet, scheduler, image_encoder, image_processor)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = ImageVideoDataset("assets/rocket.png", "generated.mp4")

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    report_to=[],  # Disable all reporting tools, including wandb
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=VideoDataCollator(),
)

# Start training
trainer.train()
