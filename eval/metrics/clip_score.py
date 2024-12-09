from configs.model_configs import CLIP_MODEL_DIR_PATH
from configs.path_configs import MODELS_DIR
from eval.constants import IMAGE_EXTENSIONS
from eval.metrics.base_metrics import BaseMetrics
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Set, List, Tuple
from tqdm import tqdm
from loguru import logger
import torch
from eval.utils.file_utils import get_image_files
import os

os.environ["TORCH_HOME"] = MODELS_DIR.as_posix()
torch.hub.set_dir(MODELS_DIR.as_posix())
import clip

device: str = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(
    "ViT-B/32", device=device, download_root=CLIP_MODEL_DIR_PATH.as_posix()
)
logit_scale = model.logit_scale.exp()


class ClipScoreMetric(BaseMetrics):

    @torch.no_grad()
    def _calculate_clip_score(self, img1_path: Path, img2_path: Path) -> float:
        """计算两张图片的Clip Score值"""
        # 读取并转换图像为numpy数组
        image1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
        image2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
        # normalize features
        image1_features = image1_features / image1_features.norm(
            dim=1, keepdim=True
        ).to(torch.float32)
        image2_features = image2_features / image2_features.norm(
            dim=1, keepdim=True
        ).to(torch.float32)
        score = logit_scale * (image1_features * image2_features).sum()

        return score

    def calculate(self) -> float:
        # 获取两个目录下的所有图片文件
        real_images = get_image_files(self.real_dataset_dir_path)
        gen_images = get_image_files(self.generated_dataset_dir_path)

        # 找到两个目录中文件名相同的图片
        common_names = set(real_images.keys()) & set(gen_images.keys())

        if not common_names:
            raise ValueError("No matching image pairs found in the directories")

        logger.info(f"Found {len(common_names)} matching image pairs")

        total_clip_score = 0.0
        for name in tqdm(common_names, desc="Calculating Clip Score"):
            clip_score = self._calculate_clip_score(real_images[name], gen_images[name])
            total_clip_score += clip_score

        avg_clip_score = total_clip_score / len(common_names)
        return float(avg_clip_score)
