"""Tests for text, image, and video preprocessing modules."""

from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import cv2
import torch
import pytest

from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from ml.preprocessing.video import VideoPreprocessor


def test_text_preprocessing_tokenization():
    prep = TextPreprocessor(max_length=32)
    sample_text = "Breaking: AI deepfake detected in official broadcast!"
    out = prep.tokenize(sample_text)

    assert "input_ids" in out
    assert "attention_mask" in out
    assert out["input_ids"].shape == (1, 32)
    assert out["attention_mask"].shape == (1, 32)

    tokens = prep.decode_tokens(out["input_ids"])
    assert "[CLS]" in tokens or "[cls]" in [t.lower() for t in tokens]
    assert len(tokens) == 32


def test_image_preprocessing():
    prep = ImagePreprocessor(image_size=224)
    img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    tensor = prep.preprocess(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    # Check normalization: values should not all be between 0 and 1 because of ImageNet norm
    assert tensor.dtype == torch.float32


def test_video_frame_sampling():
    # Create temporary dummy video with 15 frames
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_video_path = tmp_file.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video_path, fourcc, 10.0, (128, 128))
        for i in range(15):
            frame = np.full((128, 128, 3), i * 15, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        video_prep = VideoPreprocessor(sample_frames=6, image_size=224)
        frames, indices = video_prep.extract_frames(tmp_video_path)

        assert len(frames) == 6
        assert len(indices) == 6
        assert all(isinstance(f, Image.Image) for f in frames)

        batch_tensor, _, _ = video_prep.preprocess_video(tmp_video_path)
        assert batch_tensor.shape == (6, 3, 224, 224)

    finally:
        if Path(tmp_video_path).exists():
            Path(tmp_video_path).unlink()
