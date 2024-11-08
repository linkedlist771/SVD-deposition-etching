import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

from configs.path_configs import DATA_DIR


class DepositionDataToSvd(object):
    def __init__(self, source_data_dir: Path, target_data_dir: Path):
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
        self.target_data_dir.mkdir(parents=True, exist_ok=True)

    def process_all_npz(self):
        # Get all npz files from source directory
        npz_files = list(self.source_data_dir.glob("*.npz"))

        for npz_path in tqdm(npz_files, desc="Processing NPZ files"):
            # Create folder with same name as npz file (without extension)
            folder_name = npz_path.stem
            save_dir = self.target_data_dir / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # Load and process images
            npz = np.load(npz_path)
            images = npz["images"]  # Shape: (156, 384, 512)

            # Save each image as PNG
            for idx in range(images.shape[0]):
                img = images[idx]
                # Normalize to 0-255 range if needed
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                # Save image
                save_path = save_dir / f"{idx + 1}.png"
                cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    source_path = Path(
        r"C:\Users\23174\Desktop\GitHub Project\ic-device-llm\resources\deposition\letter_box_384x512_processed_data"
    )
    target_path = DATA_DIR

    processor = DepositionDataToSvd(source_path, target_path)
    processor.process_all_npz()
