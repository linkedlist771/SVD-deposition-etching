import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import cv2

# 自定义数据集类，用于加载图像和视频帧
class ImageVideoDataset(Dataset):
    def __init__(self, image_path, video_path):
        # 加载并预处理图像
        self.image = Image.open(image_path).convert("RGB")
        self.image = self.image.resize((1024, 576))
        
        # 加载并预处理视频帧
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

# 定义数据整理器
class VideoDataCollator:
    def __call__(self, examples):
        input_images = [example["input_image"] for example in examples]
        target_frames = [example["target_frame"] for example in examples]
        return {"input_images": input_images, "target_frames": target_frames}

# 加载预训练的StableVideoDiffusionPipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
print(dir(pipe))
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 获取模型组件
vae = pipe.vae
unet = pipe.unet
scheduler = pipe.scheduler
image_encoder = pipe.image_encoder
image_processor = pipe.image_processor

# 自定义模型类
class VideoDiffusionModel(torch.nn.Module):
    def __init__(self, vae, unet, scheduler, image_encoder, image_processor):
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet
        self.scheduler = scheduler
        self.image_encoder = image_encoder.eval()
        self.image_processor = image_processor

        # 冻结VAE和图像编码器的参数
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_images, target_frames):
        device = self.unet.device

        # 处理输入图像
        input_images = self.image_processor(input_images, return_tensors="pt").pixel_values.to(device)
        encoder_hidden_states = self.image_encoder(input_images).last_hidden_state

        # 处理目标帧
        target_frames = [torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in target_frames]
        target_frames = torch.stack(target_frames).to(device)
        target_frames = (target_frames - 0.5) * 2  # 归一化到 [-1, 1]

        # 编码目标帧为潜在表示
        with torch.no_grad():
            latents = self.vae.encode(target_frames).latent_dist.sample() * 0.18215

        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # 预测噪声
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 计算损失
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
        return {"loss": loss}

# 实例化模型
model = VideoDiffusionModel(vae, unet, scheduler, image_encoder, image_processor)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = ImageVideoDataset("assests/rocket.png ", "generated.mp4")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
)

# 实例化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=VideoDataCollator(),
)

# 开始训练
trainer.train()
