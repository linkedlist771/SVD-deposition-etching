from configs.path_configs import MODELS_DIR
from eval.metrics.base_metrics import BaseMetrics
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import torch
import lpips
import os
from eval.utils.file_utils import get_image_files
os.environ['TORCH_HOME'] = MODELS_DIR.as_posix()
torch.hub.set_dir(MODELS_DIR.as_posix())


class LPIPSMetric(BaseMetrics):
    def _preprocess_image(self, image_path: Path) -> torch.Tensor:
        """预处理图像为LPIPS所需格式"""
        # 读取图像
        img = Image.open(image_path).convert("RGB")
        # 转换为numpy数组并归一化到[-1,1]
        img = np.array(img).astype(np.float32) / 127.5 - 1
        # 转换为torch tensor并调整维度
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return img.to(device)

    def _calculate_lpips(self, img1_path: Path, img2_path: Path) -> float:
        """计算两张图片的LPIPS值"""
        # 初始化LPIPS模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = lpips.LPIPS(pretrained=True, net="alex", verbose=False
                              ).eval().to(device)

        # 预处理图像
        img1 = self._preprocess_image(img1_path)
        img2 = self._preprocess_image(img2_path)

        # 计算LPIPS距离
        with torch.no_grad():
            lpips_value = loss_fn(img1, img2)

        return float(lpips_value.item())

    def calculate(self) -> float:
        # 获取两个目录下的所有图片文件
        real_images = get_image_files(self.real_dataset_dir_path)
        gen_images = get_image_files(self.generated_dataset_dir_path)

        # 找到两个目录中文件名相同的图片
        common_names = set(real_images.keys()) & set(gen_images.keys())

        if not common_names:
            raise ValueError("No matching image pairs found in the directories")

        logger.info(f"Found {len(common_names)} matching image pairs")

        # 计算所有匹配图片对的LPIPS
        total_lpips = 0.0
        # 使用tqdm显示LPIPS计算进度
        for name in tqdm(common_names, desc="Calculating LPIPS"):
            lpips_value = self._calculate_lpips(real_images[name], gen_images[name])
            total_lpips += lpips_value

        # 返回平均LPIPS
        avg_lpips = total_lpips / len(common_names)
        return float(avg_lpips)
