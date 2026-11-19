"""Tests for Audio Preprocessing, Projection, Model Forward, Gradients, and Persistence (Phase 3)."""

import os
import io
import tempfile
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import SessionLocal, init_db
from backend.app.services.history_service import HistoryService
from ml.preprocessing.audio import AudioPreprocessor
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.inference.predictor import TruthLensPredictor
from ml.datasets.dataset import TruthLensMultimodalDataset
from ml.training.train_multimodal import verify_audio_gradient_flow

client = TestClient(app)


def generate_temp_wav(duration_sec: float = 1.0, sample_rate: int = 16000, freq: float = 440.0) -> str:
    """Generates a synthetic sinusoidal WAV audio file for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate audio with fundamental tone and a second harmonic
    signal = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * (freq * 2) * t)
    data = (signal * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wavfile.write(f.name, sample_rate, data)
        return f.name


# Test 1: Audio Preprocessing -> [128]
def test_audio_preprocessing_128d():
    wav_path = generate_temp_wav(duration_sec=1.0)
    try:
        preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=64)
        out = preprocessor.extract_features(wav_path)

        assert "feature_tensor" in out
        feat = out["feature_tensor"]
        assert isinstance(feat, torch.Tensor)
        assert feat.shape == (128,), f"Expected shape (128,), got {feat.shape}"
        assert out["duration_sec"] > 0.0
        assert 0.0 <= out["high_freq_ratio"] <= 1.0

        # Batch extraction check -> [B, 128]
        batch_feat = preprocessor.batch_extract_features([wav_path, wav_path, wav_path])
        assert batch_feat.shape == (3, 128), f"Expected shape (3, 128), got {batch_feat.shape}"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Test 2: Audio Projection -> [B, 256]
def test_audio_projection_dimension():
    model = MultimodalVerificationModel(projection_dim=256)
    audio_proj = model.audio_projection

    single_feat = torch.randn(1, 128)
    out_single = audio_proj(single_feat)
    assert out_single.shape == (1, 256), f"Expected (1, 256), got {out_single.shape}"

    batch_feat = torch.randn(4, 128)
    out_batch = audio_proj(batch_feat)
    assert out_batch.shape == (4, 256), f"Expected (4, 256), got {out_batch.shape}"


# Test 3: Audio Model Forward Pass
def test_audio_model_forward_pass():
    model = MultimodalVerificationModel(fusion_type="cross_attention")
    model.eval()

    dummy_audio = torch.randn(3, 128)
    with torch.no_grad():
        out = model(audio_features=dummy_audio)

    assert "logits" in out
    assert out["logits"].shape == (3, 2), f"Expected logits [3, 2], got {out['logits'].shape}"
    assert out["modalities"] == ["audio"]
    assert out["audio_features"].shape == (3, 256)


# Test 4: Audio Parameters Receive Gradients (Backpropagation Verification)
def test_audio_parameters_receive_gradients():
    model = MultimodalVerificationModel(fusion_type="cross_attention")
    device = torch.device("cpu")
    grad_norms = verify_audio_gradient_flow(model, device)

    assert len(grad_norms) > 0, "No audio parameters received gradients!"
    # Verify both audio_projection and audio_classifier received gradients
    proj_grads = [v for k, v in grad_norms.items() if "audio_projection" in k]
    clf_grads = [v for k, v in grad_norms.items() if "audio_classifier" in k]

    assert len(proj_grads) > 0, "audio_projection parameters received no gradients"
    assert len(clf_grads) > 0, "audio_classifier parameters received no gradients"
    assert all(g > 0.0 for g in proj_grads), f"Non-positive projection gradients: {proj_grads}"
    assert all(g > 0.0 for g in clf_grads), f"Non-positive classifier gradients: {clf_grads}"


# Test 5: Audio-Only Inference
def test_audio_only_inference():
    wav_path = generate_temp_wav(duration_sec=1.2)
    try:
        predictor = TruthLensPredictor()
        report = predictor.predict_multimodal(audio_path=wav_path)

        assert report.prediction in ["REAL", "FAKE", "UNCERTAIN"]
        assert 0.0 <= report.confidence <= 1.0
        assert "audio" in report.modalities_used
        assert len(report.evidence) > 0
        audio_evidence = [e for e in report.evidence if e.modality == "audio"]
        assert len(audio_evidence) > 0, "Expected audio evidence item in report"
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Test 6: Text + Audio Inference (Bimodal with Cross-Attention)
def test_text_plus_audio_inference():
    wav_path = generate_temp_wav(duration_sec=1.0)
    try:
        predictor = TruthLensPredictor()
        report = predictor.predict_multimodal(
            text="Official press briefing regarding macroeconomic indicators.",
            audio_path=wav_path,
        )

        assert report.prediction in ["REAL", "FAKE", "UNCERTAIN"]
        assert "text" in report.modalities_used
        assert "audio" in report.modalities_used
        assert len(report.evidence) >= 2
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Test 7: Multimodal Inference with All Available Modalities (Tri-Modal & Quad-Modal)
def test_multimodal_inference_all_modalities():
    wav_path = generate_temp_wav(duration_sec=0.8)
    # Generate a temporary image
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name
    img = Image.new("RGB", (224, 224), color=(60, 80, 120))
    img.save(img_path)

    try:
        predictor = TruthLensPredictor()
        # Tri-modal: Text + Image + Audio
        report = predictor.predict_multimodal(
            text="Scientific broadcast on solar astronomy.",
            image_source=img_path,
            audio_path=wav_path,
        )
        assert report.prediction in ["REAL", "FAKE", "UNCERTAIN"]
        assert "text" in report.modalities_used
        assert "visual" in report.modalities_used
        assert "audio" in report.modalities_used
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(img_path):
            os.remove(img_path)


# Test 8: Audio Persistence in SQLite and FastAPI Endpoint
def test_audio_persistence_in_sqlite():
    init_db()
    wav_path = generate_temp_wav(duration_sec=0.6)
    try:
        # 1. Direct Predictor + HistoryService save
        predictor = TruthLensPredictor()
        report = predictor.predict_multimodal(audio_path=wav_path)

        history_service = HistoryService()
        record = history_service.save_verification(
            report=report,
            case_name="Test Audio Clip Persistence",
            audio_ref=wav_path,
        )

        assert record.id is not None
        retrieved = history_service.get_verification(record.id)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.prediction == report.prediction
        assert retrieved.audio_reference == wav_path

        # 2. API Endpoint POST /verify/audio with SQLite persistence
        with open(wav_path, "rb") as f:
            resp = client.post("/verify/audio", files={"audio": ("test_tone.wav", f, "audio/wav")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
        assert "audio" in data["modalities_used"]
        assert data["id"] is not None

        # Verify API created persistent database record
        api_record = history_service.get_verification(data["id"])
        assert api_record is not None
        assert api_record.prediction == data["prediction"]
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# Test 9: Dataset Audio Loading and Safe Missing Fallback
def test_dataset_audio_loading_and_fallback():
    metadata_csv = Path(__file__).resolve().parent.parent / "ml" / "data" / "metadata" / "dataset.csv"
    dataset = TruthLensMultimodalDataset(metadata_csv=metadata_csv, split="train")

    assert len(dataset) > 0
    item = dataset[0]

    # Verify existing keys are preserved
    assert "id" in item
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "image" in item
    assert "label" in item
    assert "raw_text" in item

    # Verify audio key and neutral zero-tensor fallback
    assert "audio" in item
    assert "has_audio" in item
    assert item["audio"].shape == (128,)
    assert isinstance(item["audio"], torch.Tensor)
    # In current dataset, audio files are not present so has_audio is 0.0
    assert item["has_audio"].item() == 0.0
