from typing import List
import re
import os

def natural_sort_key(s):
    """按照自然数顺序排序文件名"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]

# def create_validation_dir(validation_images_folders: List[str],  sampled_length: int, output_dir: str):
#
#     for folder in validation_images_folders:
#         # Sort the images in the folder
#         images = os.listdir(folder)
#         images.sort(key=natural_sort_key)
#         images = images[: -sampled_length]
#
#     raise NotImplementedError
