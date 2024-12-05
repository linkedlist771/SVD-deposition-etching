import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

from utils.image_utils import remove_watermark
from configs.path_configs import DATA_DIR


class ShaoJiDataToSvd(object):
    def __init__(self, source_data_dir: Path, target_data_dir: Path):
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
        self.target_data_dir.mkdir(parents=True, exist_ok=True)

    def process_single_video(self, video_path: Path):
        # Create folder with same name as video file (without extension)
        folder_name = video_path.stem
        save_dir = self.target_data_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(str(video_path))

        # Get total frame count for tqdm
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create progress bar
        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Remove watermark from frame
                frame_processed = remove_watermark(frame)

                # Save processed frame
                save_path = save_dir / f"{frame_count + 1}.png"
                cv2.imwrite(str(save_path), frame_processed)

                frame_count += 1
                pbar.update(1)

        cap.release()

    def process_all_videos(self):
        # Get all video files from source directory (assuming mp4 format)
        video_files = list(self.source_data_dir.glob("*.mp4"))

        for video_path in tqdm(video_files, desc="Processing videos"):
            self.process_single_video(video_path)


if __name__ == "__main__":
    source_path = DATA_DIR / "shaoji-video"
    target_path = DATA_DIR / "shaoji-data"
    target_path.mkdir(exist_ok=True)
    processor = ShaoJiDataToSvd(source_path, target_path)
    # Process single video for testing
    test_video = next(source_path.glob("*.mp4"))  # Get first video file
    processor.process_single_video(test_video)
