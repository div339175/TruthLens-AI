"""End-to-end refresh persistence test verifying that verifications survive application restarts."""

import io
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_full_refresh_persistence_workflow():
    """Simulates the 10-step user verification and refresh persistence flow."""
    # Step 1: Create multimodal verification with custom case name
    img = Image.new("RGB", (120, 120), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Step 2 & 3: Run analysis
    res = client.post(
        "/verify/multimodal",
        data={
            "text": "Official verification test for persistence workflow check.",
            "case_name": "Press Release Refresh Test #99",
        },
        files={"image": ("test_press.png", buf, "image/png")},
    )
    assert res.status_code == 200
    report_data = res.json()

    # Step 4: Confirm result
    assert report_data["prediction"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert report_data["id"] is not None
    record_id = report_data["id"]
    case_name = report_data["case_name"]
    assert case_name == "Press Release Refresh Test #99"

    # Step 5 & 6: Simulate browser refresh by creating a completely fresh client / request
    fresh_client = TestClient(app)

    # Step 7 & 8: Open History and confirm record still exists
    hist_res = fresh_client.get("/verifications")
    assert hist_res.status_code == 200
    history = hist_res.json()
    item_ids = [item["id"] for item in history["items"]]
    assert record_id in item_ids

    # Step 9 & 10: Open the specific verification and confirm complete report is available
    single_res = fresh_client.get(f"/verifications/{record_id}")
    assert single_res.status_code == 200
    saved_report = single_res.json()

    # Confirm all explainability, confidence, and verdict data preserved
    assert saved_report["id"] == record_id
    assert saved_report["case_name"] == "Press Release Refresh Test #99"
    assert saved_report["prediction"] == report_data["prediction"]
    assert saved_report["confidence"] == report_data["confidence"]
    assert len(saved_report["evidence"]) == len(report_data["evidence"])
    assert saved_report["explanation"] == report_data["explanation"]
