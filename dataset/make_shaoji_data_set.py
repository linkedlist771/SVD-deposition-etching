import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import time
import multiprocessing
from multiprocessing import Pool

from utils.image_utils import remove_watermark
from configs.path_configs import DATA_DIR


class ShaoJiDataToSvd(object):
    def __init__(self, source_data_dir: Path, target_data_dir: Path, sample_interval: int = 1):
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir
        self.sample_interval = sample_interval
        self.target_data_dir.mkdir(parents=True, exist_ok=True)

    def process_single_video(self, video_path: Path):
        creation_time = video_path.stat().st_mtime
        folder_name = time.strftime('%m-%d-%H-%M-%S', time.localtime(creation_time))
        save_dir = self.target_data_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = total_frames // self.sample_interval

        with tqdm(total=processed_frames, desc=f"Processing {video_path.name}") as pbar:
            frame_count = 0
            saved_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.sample_interval == 0:
                    save_path = save_dir / f"{saved_count + 1}.png"
                    cv2.imwrite(str(save_path), frame)
                    saved_count += 1
                    pbar.update(1)

                frame_count += 1

        cap.release()
        return f"Completed processing {video_path.name}"

    @staticmethod
    def _process_video_wrapper(args):
        """Wrapper function for multiprocessing"""
        self_instance, video_path = args
        return self_instance.process_single_video(video_path)

    def process_all_videos(self, num_processes=None):
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        video_files = list(self.source_data_dir.glob("*.mp4"))
        with Pool(processes=num_processes) as pool:
            args = [(self, video_path) for video_path in video_files]
            results = list(tqdm(
                pool.imap(self._process_video_wrapper, args),
                total=len(video_files),
                desc="Processing videos"
            ))
        return results



if __name__ == "__main__":
    source_path = DATA_DIR / "shaoji-video"
    target_path = DATA_DIR / "shaoji-data"
    target_path.mkdir(exist_ok=True)

    sample_interval = 10
    processor = ShaoJiDataToSvd(source_path, target_path, sample_interval)

    # Using multiprocessing to process all videos
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    results = processor.process_all_videos(num_processes=num_processes)

    # Alternatively, for testing a single video:
    # test_video = next(source_path.glob("*.mp4"))
    # processor.process_single_video(test_video)
