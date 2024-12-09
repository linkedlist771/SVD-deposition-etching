from eval.constants import IMAGE_EXTENSIONS
from eval.metrics.base_metrics import BaseMetrics
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Set, List, Tuple
from tqdm import tqdm
from loguru import logger


class PSNRMetric(BaseMetrics):

    def _get_image_files(self, directory: Path) -> Dict[str, Path]:
        """获取目录下所有图片文件，返回{文件名: 文件路径}的字典"""
        image_files = {}
        # 使用tqdm显示文件搜索进度
        for ext in tqdm(IMAGE_EXTENSIONS, desc="Searching image extensions"):
            for file_path in directory.glob(f"*.{ext}"):
                # 获取不带扩展名的文件名作为key
                image_files[file_path.stem] = file_path
        return image_files

    def _calculate_psnr(self, img1_path: Path, img2_path: Path) -> float:
        """计算两张图片的PSNR值"""
        # 读取并转换图像为numpy数组
        img1 = np.array(Image.open(img1_path)).astype(np.float32)
        img2 = np.array(Image.open(img2_path)).astype(np.float32)

        # 计算MSE
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')

        # 假设像素最大值为255
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        return psnr

    def calculate(self) -> float:
        # 获取两个目录下的所有图片文件
        real_images = self._get_image_files(self.real_dataset_dir_path)
        gen_images = self._get_image_files(self.generated_dataset_dir_path)

        # 找到两个目录中文件名相同的图片
        common_names = set(real_images.keys()) & set(gen_images.keys())

        if not common_names:
            raise ValueError("No matching image pairs found in the directories")

        logger.info(f"Found {len(common_names)} matching image pairs")

        # 计算所有匹配图片对的PSNR
        total_psnr = 0.0
        # 使用tqdm显示PSNR计算进度
        for name in tqdm(common_names, desc="Calculating PSNR"):
            psnr = self._calculate_psnr(real_images[name], gen_images[name])
            total_psnr += psnr

        # 返回平均PSNR
        avg_psnr = total_psnr / len(common_names)
        return float(avg_psnr)
