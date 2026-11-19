"""Comprehensive test suite for the corrected TruthLens AI video processing pipeline."""

import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.preprocessing.video import VideoPreprocessor
from ml.models.vision_encoder import VisionEncoder
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.inference.predictor import TruthLensPredictor

client = TestClient(app)


def create_synthetic_test_video(num_frames: int = 16, width: int = 128, height: int = 128) -> str:
    """Helper to generate a temporary multi-frame video with distinct visual features per frame."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        # Create non-uniform frame patterns (e.g. progressing gradients and shapes)
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 10 + i * 2, (255 - i * 10, i * 12, 100), -1)
        writer.write(frame)
    writer.release()
    return video_path


# ==============================================================================
# Test 1 — Frame Sampling
# ==============================================================================


def test_frame_sampling():
    """Verify that VideoPreprocessor extracts the expected 8 uniform frames from a video."""
    video_path = create_synthetic_test_video(num_frames=20)
    try:
        prep = VideoPreprocessor(sample_frames=8, image_size=224)
        frames, indices = prep.extract_frames(video_path)

        assert len(frames) == 8
        assert len(indices) == 8
        assert all(isinstance(f, Image.Image) for f in frames)
        assert indices == sorted(indices)
        # Frames should span beginning to end of video
        assert indices[0] == 0
        assert indices[-1] == 19
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# ==============================================================================
# Test 2 — Frame-Level Feature Extraction & Temporal Pooling
# ==============================================================================


def test_frame_feature_extraction_and_pooling():
    """Verify that each sampled frame reaches ResNet18 individually and produces 8 feature vectors."""
    encoder = VisionEncoder(projection_dim=256, pretrained=False)
    # 8 individual sampled frames [8, 3, 224, 224]
    frames_tensor = torch.randn(8, 3, 224, 224)

    out = encoder.encode_video_frames(frames_tensor)

    # 1. Individual frame features (8 x 512)
    assert "frame_features" in out
    assert out["frame_features"].shape == (8, 512)

    # 2. Individual projected frame embeddings (8 x 256)
    assert "projected_frames" in out
    assert out["projected_frames"].shape == (8, 256)

    # 3. Temporally mean-pooled feature vector (1 x 512)
    assert "pooled_features" in out
    assert out["pooled_features"].shape == (1, 512)

    # 4. Temporally mean-pooled video embedding (1 x 256)
    assert "projected_pooled" in out
    assert out["projected_pooled"].shape == (1, 256)

    # 5. Temporally pooled spatial sequence for cross-attention (1 x 49 x 256)
    assert "projected_sequence" in out
    assert out["projected_sequence"].shape == (1, 49, 256)


# ==============================================================================
# Test 3 — Mathematical Feature Pooling Correctness
# ==============================================================================


def test_feature_pooling_matches_mathematical_mean():
    """Verify that projected_pooled is the exact mathematical mean across frame feature vectors."""
    encoder = VisionEncoder(projection_dim=256, pretrained=False)
    frames_tensor = torch.randn(8, 3, 224, 224)

    out = encoder.encode_video_frames(frames_tensor)
    expected_mean = torch.mean(out["projected_frames"], dim=0, keepdim=True)

    assert torch.allclose(out["projected_pooled"], expected_mean, atol=1e-6)


# ==============================================================================
# Test 4 — Regression Test for the Old Bug (No Raw-Pixel Averaging)
# ==============================================================================


def test_regression_no_raw_pixel_averaging():
    """Verify that feature-level pooling produces different, richer representations than the old bug.

    The old bug did:
        image_tensor = frames_tensor.mean(dim=0, keepdim=True)
        feat = encoder(image_tensor)

    The correct architecture does:
        feat_frames = encoder(frames_tensor)
        feat = feat_frames.mean(dim=0, keepdim=True)

    Due to non-linearities (ReLU, BatchNorm, Conv filters), E[f(X)] != f(E[X]).
    We verify they are mathematically distinct.
    """
    encoder = VisionEncoder(projection_dim=256, pretrained=False)
    encoder.eval()

    # Generate 8 distinct non-identical frames
    frames_tensor = torch.zeros(8, 3, 224, 224)
    for i in range(8):
        frames_tensor[i] = torch.randn(3, 224, 224) * (i + 1)

    with torch.no_grad():
        # Correct pipeline
        correct_out = encoder.encode_video_frames(frames_tensor)
        correct_embedding = correct_out["projected_pooled"]  # [1, 256]

        # Old buggy pipeline (raw RGB pixel averaging)
        blurry_averaged_pixels = frames_tensor.mean(dim=0, keepdim=True)  # [1, 3, 224, 224]
        old_bug_out = encoder(blurry_averaged_pixels)
        old_bug_embedding = old_bug_out["projected_pooled"]  # [1, 256]

    # Must NOT be identical — proves that true frame-level feature pooling is active
    diff = torch.norm(correct_embedding - old_bug_embedding).item()
    assert diff > 0.05, f"Expected distinct representations, but got difference {diff}"


# ==============================================================================
# Test 5 — Video-Only Forward Pass
# ==============================================================================


def test_video_only_forward_pass():
    """Verify that MultimodalVerificationModel accepts video_frames alone and outputs valid logits."""
    model = MultimodalVerificationModel(projection_dim=256, pretrained_vision=False)
    video_frames = torch.randn(8, 3, 224, 224)

    out = model(video_frames=video_frames)
    assert out["logits"].shape == (1, 2)
    assert "video" in out["modalities"]
    assert out["fused_features"].shape == (1, 256)
    assert out["video_frame_features"].shape == (8, 256)


# ==============================================================================
# Test 6 — Text + Video Multimodal Forward Pass (Cross-Attention)
# ==============================================================================


def test_text_plus_video_multimodal_forward_pass():
    """Verify that Text + Video fuses cleanly through Bidirectional Cross-Attention."""
    model = MultimodalVerificationModel(
        projection_dim=256,
        fusion_type="cross_attention",
        pretrained_vision=False,
    )
    input_ids = torch.randint(0, 1000, (1, 16))
    attention_mask = torch.ones(1, 16)
    video_frames = torch.randn(8, 3, 224, 224)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        video_frames=video_frames,
    )
    assert out["logits"].shape == (1, 2)
    assert set(out["modalities"]) == {"text", "video"}
    assert "cross_attention" in out
    assert out["cross_attention"] is not None
    assert "text_to_vision_attn" in out["cross_attention"]
    assert "vision_to_text_attn" in out["cross_attention"]


# ==============================================================================
# Test 7 — End-to-End Inference with TruthLensPredictor
# ==============================================================================


def test_video_inference_end_to_end_with_predictor():
    """Verify complete predictor inference flow on a real video file."""
    video_path = create_synthetic_test_video(num_frames=12)
    try:
        predictor = TruthLensPredictor.get_instance()
        report = predictor.predict_multimodal(video_path=video_path)

        assert report.prediction in ["REAL", "FAKE", "UNCERTAIN"]
        assert 0.0 <= report.confidence <= 1.0
        assert "video" in report.modalities_used
        assert report.explanation is not None

        # Verify video evidence contains feature pooling metadata
        video_ev = next((e for e in report.evidence if e.modality == "video"), None)
        assert video_ev is not None
        assert video_ev.details is not None
        assert video_ev.details.get("sampled_frames_count") == 8
        assert video_ev.details.get("temporal_pooling") == "feature_mean_pooling"
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# ==============================================================================
# Test 8 — API Text + Video Multimodal Endpoint
# ==============================================================================


def test_api_text_plus_video_multimodal_endpoint():
    """Verify POST /verify/multimodal accepting text AND video file."""
    video_path = create_synthetic_test_video(num_frames=10)
    try:
        with open(video_path, "rb") as f:
            response = client.post(
                "/verify/multimodal",
                data={
                    "text": "Breaking broadcast claims major policy reform.",
                    "case_name": "Multimodal Text+Video API Test",
                },
                files={"video": ("sample_clip.mp4", f, "video/mp4")},
            )

        assert response.status_code == 200
        data = response.json()
        assert set(data["modalities_used"]) == {"text", "video"}
        assert data["id"] is not None
        assert data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]

        # Confirm saved in SQLite history
        get_res = client.get(f"/verifications/{data['id']}")
        assert get_res.status_code == 200
        assert get_res.json()["case_name"] == "Multimodal Text+Video API Test"
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
