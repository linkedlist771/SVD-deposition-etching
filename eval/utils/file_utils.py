from pathlib import Path
from typing import Dict
from tqdm import tqdm

from eval.constants import IMAGE_EXTENSIONS


def get_image_files(directory: Path) -> Dict[str, Path]:
    """获取目录下所有图片文件，返回{文件名: 文件路径}的字典"""
    image_files = {}
    # 使用tqdm显示文件搜索进度
    for ext in tqdm(IMAGE_EXTENSIONS, desc="Searching image extensions"):
        for file_path in directory.glob(f"*.{ext}"):
            # 获取不带扩展名的文件名作为key
            image_files[file_path.stem] = file_path
    return image_files
