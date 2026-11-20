"""Phase 7 Test Suite: Evaluation Hub Backend Endpoints.

Verifies:
1. /api/evaluation/overview returns valid schema with production configuration metadata.
2. /api/evaluation/in-domain returns genuine test split metrics.
3. /api/evaluation/generalization returns real-world benchmark evaluations (Fakeddit, Audio, Video).
4. /api/evaluation/generalization explicitly marks FakeAVCeleb as gated/restricted.
5. /api/evaluation/semantic-mismatch returns condition breakdown and comparative evaluations.
6. /api/evaluation/calibration returns ECE metrics and temperature parameters.
7. /api/evaluation/ablation returns full Config A-F comparison matrix.
8. /api/evaluation/all returns aggregated payload for frontend hub initialization.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_evaluation_overview(client: TestClient):
    """Test /api/evaluation/overview endpoint."""
    response = client.get("/api/evaluation/overview")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["selected_model"] == "Config A"
    assert "metrics" in data
    assert data["metrics"]["in_domain_accuracy"] == 100.0
    assert "total_automated_tests" in data["metrics"]


def test_evaluation_in_domain(client: TestClient):
    """Test /api/evaluation/in-domain endpoint."""
    response = client.get("/api/evaluation/in-domain")
    assert response.status_code == 200
    data = response.json()

    assert "dataset" in data
    assert "baselines" in data
    assert "multimodal_cross_attention" in data
    mca = data["multimodal_cross_attention"]
    assert mca["metrics"]["accuracy"] == 1.0
    assert mca["metrics"]["f1"] == 1.0


def test_evaluation_generalization(client: TestClient):
    """Test /api/evaluation/generalization endpoint."""
    response = client.get("/api/evaluation/generalization")
    assert response.status_code == 200
    data = response.json()

    # Ensure fakeavceleb status is explicitly marked restricted/gated
    assert "fakeavceleb_status" in data
    status = data["fakeavceleb_status"]
    assert "Gated" in status["status"] or "Restricted" in status["status"]

    assert "fakeddit" in data
    assert "audio" in data
    assert "video" in data


def test_evaluation_semantic_mismatch(client: TestClient):
    """Test /api/evaluation/semantic-mismatch endpoint."""
    response = client.get("/api/evaluation/semantic-mismatch")
    assert response.status_code == 200
    data = response.json()

    assert "conditions_info" in data
    assert len(data["conditions_info"]) == 4
    assert "phase6_contrastive_runs" in data
    assert "best_lambda" in data


def test_evaluation_calibration(client: TestClient):
    """Test /api/evaluation/calibration endpoint."""
    response = client.get("/api/evaluation/calibration")
    assert response.status_code == 200
    data = response.json()

    assert "optimal_temperature" in data
    assert "uncertainty_threshold" in data
    assert data["uncertainty_threshold"] == 0.65
    assert "suite_results" in data


def test_evaluation_ablation(client: TestClient):
    """Test /api/evaluation/ablation endpoint."""
    response = client.get("/api/evaluation/ablation")
    assert response.status_code == 200
    data = response.json()

    assert "configurations" in data
    assert "selection_audit" in data
    audit = data["selection_audit"]
    assert audit.get("selected_configuration") == "Config A"


def test_evaluation_all(client: TestClient):
    """Test /api/evaluation/all aggregate endpoint."""
    response = client.get("/api/evaluation/all")
    assert response.status_code == 200
    data = response.json()

    assert "overview" in data
    assert "in_domain" in data
    assert "generalization" in data
    assert "semantic_mismatch" in data
    assert "calibration" in data
    assert "ablation" in data
    assert data["overview"]["selected_model"] == "Config A"
