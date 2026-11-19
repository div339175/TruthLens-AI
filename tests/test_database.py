"""Tests for SQLite database persistence, HistoryService, and verification history APIs."""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.session import SessionLocal, init_db
from backend.app.database.models import VerificationRecord
from backend.app.services.history_service import HistoryService
from backend.app.schemas.verification import VerificationReport, ModalityEvidence

client = TestClient(app)


def create_dummy_report(id_val="test_id_001", prediction="FAKE", confidence=0.88):
    return VerificationReport(
        id=id_val,
        case_name="Test Case Sample",
        prediction=prediction,
        confidence=confidence,
        real_probability=1.0 - confidence,
        fake_probability=confidence,
        uncertain=False,
        modalities_used=["text", "visual"],
        manipulation_type="Potential visual manipulation",
        evidence=[
            ModalityEvidence(
                modality="text",
                type="model-derived",
                description="Test text evidence.",
            )
        ],
        explanation="Test explanation.",
        model_version="TruthLens Test v1.0",
        fusion_type="cross_attention",
        device_used="cpu",
        execution_mode="real",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def test_database_init():
    """Verify tables exist and SessionLocal binds properly."""
    init_db()
    db = SessionLocal()
    try:
        count = db.query(VerificationRecord).count()
        assert count >= 0
    finally:
        db.close()


def test_history_service_save_and_retrieve():
    """Test saving a VerificationReport and retrieving it by ID."""
    service = HistoryService()
    report = create_dummy_report(id_val="save_test_101", prediction="REAL", confidence=0.92)

    saved = service.save_verification(
        report=report,
        case_name="Hospital Opening Fact Check",
        text_snippet="Hospital opens new wing today.",
    )

    assert saved.id == "save_test_101"
    assert saved.case_name == "Hospital Opening Fact Check"
    assert saved.prediction == "REAL"
    assert saved.confidence == 0.92

    # Retrieve record
    retrieved = service.get_verification("save_test_101")
    assert retrieved is not None
    assert retrieved.id == "save_test_101"
    assert retrieved.prediction == "REAL"

    # Verify report JSON roundtrip
    full_dict = retrieved.to_full_report_dict()
    assert full_dict["id"] == "save_test_101"
    assert full_dict["confidence"] == 0.92
    assert len(full_dict["evidence"]) == 1


def test_history_service_list_ordering():
    """Test that list_verifications returns records in descending order (newest first)."""
    import time
    service = HistoryService()
    id_1 = f"order_001_{time.time()}"
    r1 = create_dummy_report(id_val=id_1)
    service.save_verification(r1, case_name="First Case")
    time.sleep(0.01)
    id_2 = f"order_002_{time.time()}"
    r2 = create_dummy_report(id_val=id_2)
    service.save_verification(r2, case_name="Second Case")

    records = service.list_verifications(limit=50)
    record_ids = [r.id for r in records]
    assert id_1 in record_ids
    assert id_2 in record_ids
    # id_2 must appear earlier in the list than id_1
    assert record_ids.index(id_2) < record_ids.index(id_1)


def test_get_verifications_api():
    """Test GET /verifications endpoint."""
    response = client.get("/verifications")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["total"] >= 1

    item = data["items"][0]
    assert "id" in item
    assert "case_name" in item
    assert "prediction" in item
    assert "confidence" in item
    assert "modalities_used" in item


def test_get_verification_by_id_api():
    """Test GET /verifications/{id} endpoint."""
    service = HistoryService()
    report = create_dummy_report(id_val="api_fetch_999", prediction="FAKE", confidence=0.85)
    service.save_verification(report, case_name="API Fetch Test")

    response = client.get("/verifications/api_fetch_999")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "api_fetch_999"
    assert data["prediction"] == "FAKE"
    assert data["confidence"] == 0.85
    assert "evidence" in data
    assert "explanation" in data


def test_get_nonexistent_verification_404():
    """Test that querying a nonexistent ID returns HTTP 404."""
    response = client.get("/verifications/nonexistent_xyz_id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_post_verification_save_api():
    """Test POST /verifications endpoint for explicit saving."""
    report = create_dummy_report(id_val="explicit_save_001")
    payload = {
        "case_name": "Manually Saved Verification",
        "report": report.model_dump(),
    }

    response = client.post("/verifications", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "explicit_save_001"
    assert data["case_name"] == "Manually Saved Verification"

    # Verify it can be retrieved
    get_res = client.get("/verifications/explicit_save_001")
    assert get_res.status_code == 200
    assert get_res.json()["case_name"] == "Manually Saved Verification"


def test_delete_verification_api():
    """Test DELETE /verifications/{id} endpoint."""
    service = HistoryService()
    report = create_dummy_report(id_val="delete_target_001")
    service.save_verification(report, case_name="To Be Deleted")

    del_res = client.delete("/verifications/delete_target_001")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Confirm it is gone
    get_res = client.get("/verifications/delete_target_001")
    assert get_res.status_code == 404
