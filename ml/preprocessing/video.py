"""Video preprocessing and frame sampling pipeline for TruthLens AI."""

import os
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from ml.preprocessing.image import ImagePreprocessor


class VideoPreprocessor:
    """Samples uniform frames from video files and preprocesses them for visual encoders."""

    def __init__(self, sample_frames: int = 8, image_size: int = 224):
        self.sample_frames = sample_frames
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)

    def extract_frames(self, video_path: Union[str, Path]) -> Tuple[List[Image.Image], List[int]]:
        """Extract uniformly spaced frames from video.

        Returns:
            frames: List of PIL Image objects
            frame_indices: Indices of sampled frames in the original video
        """
        video_path_str = str(video_path)
        if not os.path.exists(video_path_str):
            raise FileNotFoundError(f"Video file not found: {video_path_str}")

        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path_str}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video with 0 frames: {video_path_str}")

        # Choose uniform indices across video length
        if total_frames <= self.sample_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num=self.sample_frames, dtype=int).tolist()

        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                frames.append(pil_frame)

        cap.release()

        if not frames:
            raise ValueError(f"Failed to read any frames from video: {video_path_str}")

        return frames, indices[:len(frames)]

    def preprocess_video(self, video_path: Union[str, Path]) -> Tuple[torch.Tensor, List[Image.Image], List[int]]:
        """Extract sampled frames and convert to batch tensor [N, 3, H, W]."""
        pil_frames, indices = self.extract_frames(video_path)
        tensors = [self.image_preprocessor.preprocess(frame) for frame in pil_frames]
        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor, pil_frames, indices
