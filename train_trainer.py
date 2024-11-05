import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import cv2

class ImageVideoDataset(Dataset):
    def __init__(self, image_path, video_path):
        self.image = Image.open(image_path).convert("RGB")
        self.image = self.image.resize((1024, 576))
        self.video_frames = self.load_video_frames(video_path)

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        return {"input_image": self.image, "target_frame": frame}

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((1024, 576))
            frames.append(pil_image)
            success, frame = cap.read()
        cap.release()
        return frames

class VideoDataCollator:
    def __call__(self, examples):
        input_images = [example["input_image"] for example in examples]
        target_frames = [example["target_frame"] for example in examples]
        return {"input_images": input_images, "target_frames": target_frames}

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
# Instead of image_processor, we'll use the video_processor
video_processor = pipe.video_processor

class VideoDiffusionModel(torch.nn.Module):
    def __init__(self, vae, unet, scheduler, image_encoder, video_processor):
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet
        self.scheduler = scheduler
        self.image_encoder = image_encoder.eval()
        self.video_processor = video_processor

        # Freeze VAE and image encoder parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_images, target_frames):
        device = self.unet.device

        # Process input images using video_processor
        input_images = torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 
                                  for img in input_images]).to(device)
        input_images = (input_images - 0.5) * 2  # Normalize to [-1, 1]

        # Get encoder hidden states
        encoder_hidden_states = self.image_encoder(input_images).last_hidden_state

        # Process target frames
        target_frames = [torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in target_frames]
        target_frames = torch.stack(target_frames).to(device)
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
model = VideoDiffusionModel(vae, unet, scheduler, image_encoder, video_processor)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = ImageVideoDataset("assests/rocket.png", "generated.mp4")

# Training arguments
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
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
