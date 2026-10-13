"""Integration tests for FastAPI endpoints."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "TruthLens AI"
    assert "device" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "text_backbone" in data
    assert "vision_backbone" in data
    assert "fusion_architecture" in data


def test_dataset_stats_endpoint():
    response = client.get("/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["initialized", "uninitialized"]
    assert "total_samples" in data


def test_verify_text_endpoint():
    response = client.post("/verify/text", json={"text": "Unverified leaked audio suggests political scandal"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "evidence" in data
    assert "explanation" in data


def test_verify_image_endpoint():
    # Create test image in memory
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/verify/image", files={"image": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "visual" in data["modalities_used"]


def test_verify_multimodal_endpoint():
    img = Image.new("RGB", (100, 100), color=(120, 50, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/verify/multimodal",
        data={"text": "Official satellite confirmation"},
        files={"image": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data["modalities_used"]) == {"text", "visual"}
    assert "explanation" in data


def test_verify_invalid_file_extension():
    fake_txt = io.BytesIO(b"not an image or video")
    response = client.post("/verify/image", files={"image": ("script.sh", fake_txt, "text/plain")})
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
