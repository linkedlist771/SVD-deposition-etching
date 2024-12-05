import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import time

from utils.image_utils import remove_watermark
from configs.path_configs import DATA_DIR


class ShaoJiDataToSvd(object):
    def __init__(self, source_data_dir: Path, target_data_dir: Path, sample_interval: int = 1):
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
        self.sample_interval = sample_interval  # 每隔多少帧取一帧
        self.target_data_dir.mkdir(parents=True, exist_ok=True)

    def process_single_video(self, video_path: Path):
        creation_time = video_path.stat().st_mtime
        folder_name = time.strftime('%m-%d-%H-%M-%S', time.localtime(creation_time))
        save_dir = self.target_data_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算实际需要处理的帧数
        processed_frames = total_frames // self.sample_interval

        with tqdm(total=processed_frames, desc=f"Processing {video_path.name}") as pbar:
            frame_count = 0
            saved_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 只处理符合采样间隔的帧
                if frame_count % self.sample_interval == 0:
                    save_path = save_dir / f"{saved_count + 1}.png"
                    cv2.imwrite(str(save_path), frame)
                    saved_count += 1
                    pbar.update(1)

                frame_count += 1

        cap.release()

    def process_all_videos(self):
        video_files = list(self.source_data_dir.glob("*.mp4"))
        for video_path in tqdm(video_files, desc="Processing videos"):
            self.process_single_video(video_path)


if __name__ == "__main__":
    source_path = DATA_DIR / "shaoji-video"
    target_path = DATA_DIR / "shaoji-data"
    target_path.mkdir(exist_ok=True)

    # 设置采样间隔，例如设置为30表示每秒采样一帧（假设视频为30fps）
    sample_interval = 1

    processor = ShaoJiDataToSvd(source_path, target_path, sample_interval)
    # Process single video for testing
    test_video = next(source_path.glob("*.mp4"))  # Get first video file
    processor.process_single_video(test_video)
