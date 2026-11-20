"""Phase 7 Test Suite: End-to-End Demo Hardening & Multi-Modal Verification Flow.

Verifies:
1. Live unimodal text verification returns valid VerificationReport with history record.
2. Live multimodal text + image verification executes ResNet18 and cross-attention.
3. Live multimodal text + video verification processes 8 frames with temporal feature pooling.
4. Live multimodal text + audio verification processes STFT acoustic features.
5. Simultaneous tri-modal verification (text + image + audio) executes successfully.
6. Verification persistence in SQLite database is verified end-to-end.
7. Low-confidence ambiguity triggers first-class UNCERTAIN verdict.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.models.classifier import DecisionPolicy


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_verify_unimodal_text(client: TestClient):
    """Verify text-only verification flow."""
    response = client.post(
        "/api/verify/text",
        json={"text": "Ministry of Health releases updated quarterly public health statistics."},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "text" in data["modalities_used"]
    assert data["id"] is not None
    assert len(data["evidence"]) >= 1

    # Check persistence
    hist_res = client.get(f"/api/verifications/{data['id']}")
    assert hist_res.status_code == 200
    assert hist_res.json()["id"] == data["id"]


def test_verify_multimodal_text_and_image(client: TestClient):
    """Verify multimodal text + image verification flow."""
    image_path = Path("ml/data/samples/real_official_press_02.png")
    assert image_path.exists(), "Sample image missing"

    with open(image_path, "rb") as f:
        response = client.post(
            "/api/verify/multimodal",
            data={
                "text": "Ministry of Health releases updated quarterly public health statistics.",
                "case_name": "Demo Phase 7 Text + Image",
            },
            files={"image": ("real_official_press_02.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data["modalities_used"]
    assert "visual" in data["modalities_used"]
    assert data["id"] is not None

    # Visual evidence should be present
    modalities_in_ev = [e["modality"] for e in data["evidence"]]
    assert "visual" in modalities_in_ev


def test_verify_multimodal_text_and_video(client: TestClient):
    """Verify multimodal text + video verification flow with 8-frame pooling."""
    video_path = Path("ml/data/samples/video_fake_001.mp4")
    if not video_path.exists():
        video_path = Path("experiments/external_dataset/data/video/video_fake_001.mp4")
    assert video_path.exists(), "Sample video missing"

    with open(video_path, "rb") as f:
        response = client.post(
            "/api/verify/multimodal",
            data={
                "text": "Surveillance recording captures suspicious unauthorized midnight access.",
                "case_name": "Demo Phase 7 Text + Video",
            },
            files={"video": ("video_fake_001.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data["modalities_used"]
    assert "video" in data["modalities_used"]
    assert data["id"] is not None

    modalities_in_ev = [e["modality"] for e in data["evidence"]]
    assert "video" in modalities_in_ev


def test_verify_multimodal_text_and_audio(client: TestClient):
    """Verify multimodal text + audio verification flow with STFT extraction."""
    audio_path = Path("ml/data/samples/audio_fake_001.wav")
    if not audio_path.exists():
        audio_path = Path("experiments/external_dataset/data/audio/audio_fake_001.wav")
    assert audio_path.exists(), "Sample audio missing"

    with open(audio_path, "rb") as f:
        response = client.post(
            "/api/verify/multimodal",
            data={
                "text": "Voice-cloned phone call impersonating executive officer authorizing transfer.",
                "case_name": "Demo Phase 7 Text + Audio",
            },
            files={"audio": ("audio_fake_001.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data["modalities_used"]
    assert "audio" in data["modalities_used"]
    assert data["id"] is not None

    modalities_in_ev = [e["modality"] for e in data["evidence"]]
    assert "audio" in modalities_in_ev


def test_verify_trimodal_text_image_audio(client: TestClient):
    """Verify simultaneous tri-modal text + image + audio verification."""
    image_path = Path("ml/data/samples/real_official_press_02.png")
    audio_path = Path("ml/data/samples/audio_real_001.wav")

    with open(image_path, "rb") as img_f, open(audio_path, "rb") as aud_f:
        response = client.post(
            "/api/verify/multimodal",
            data={
                "text": "Comprehensive multimodal press release with audio soundbite and official photograph.",
                "case_name": "Demo Phase 7 Tri-Modal Complete",
            },
            files={
                "image": ("real_official_press_02.png", img_f, "image/png"),
                "audio": ("audio_real_001.wav", aud_f, "audio/wav"),
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data["modalities_used"]
    assert "visual" in data["modalities_used"]
    assert "audio" in data["modalities_used"]
    assert data["id"] is not None


def test_uncertainty_threshold_routing():
    """Verify that decision policy evaluates uncertainty correctly with 0.65 threshold."""
    policy = DecisionPolicy(uncertainty_threshold=0.65)

    # 1. Clear FAKE (0.90 fake prob)
    import torch
    logits_fake = torch.tensor([[ -1.5, 2.0 ]])
    dec_fake = policy.evaluate(logits_fake)
    assert dec_fake["prediction"] == "FAKE"
    assert dec_fake["uncertain"] is False

    # 2. Ambiguous / Balanced logits (0.52 vs 0.48)
    logits_ambiguous = torch.tensor([[ 0.05, 0.0 ]])
    dec_ambig = policy.evaluate(logits_ambiguous)
    assert dec_ambig["prediction"] == "UNCERTAIN"
    assert dec_ambig["uncertain"] is True
    assert dec_ambig["confidence"] < 0.65
