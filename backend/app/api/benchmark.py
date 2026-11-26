"""Fakeddit External Benchmark API router."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.config import settings

router = APIRouter(prefix="/benchmark/fakeddit", tags=["External Benchmark"])

BASE_DIR = settings.BASE_DIR if hasattr(settings, "BASE_DIR") else Path(__file__).resolve().parent.parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_JSON = REPORTS_DIR / "fakeddit_external_results.json"
FAKEDDIT_RAW_DIR = BASE_DIR / "data" / "raw" / "fakeddit"


def get_cached_results() -> Dict[str, Any]:
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read benchmark results: {e}")
    # If not yet executed, run it dynamically
    from experiments.fakeddit_external import run_fakeddit_evaluation
    return run_fakeddit_evaluation()


@router.get("/results")
def get_benchmark_results() -> Dict[str, Any]:
    """Retrieve full Fakeddit external benchmark results, metrics, and summary."""
    data = get_cached_results()
    # Return metrics summary and top-level counts
    return {
        "status": "success",
        "dataset": data.get("dataset", "Fakeddit"),
        "evaluation_samples": data.get("evaluation_samples", 60),
        "training_source": data.get("training_source", "TruthLens course dataset"),
        "training_samples": data.get("training_samples", 320),
        "fakeddit_used_for_training": data.get("fakeddit_used_for_training", False),
        "fakeddit_used_for_external_evaluation": data.get("fakeddit_used_for_external_evaluation", True),
        "class_distribution": data.get("class_distribution", {"real": 30, "fake": 30}),
        "accuracy": data.get("accuracy", 0.0),
        "resolved_accuracy": data.get("resolved_accuracy", 0.0),
        "precision": data.get("precision", 0.0),
        "recall": data.get("recall", 0.0),
        "f1_score": data.get("f1_score", 0.0),
        "counts": data.get("counts", {}),
        "rates": data.get("rates", {}),
        "confusion_matrix": data.get("confusion_matrix", {}),
    }


@router.post("/run")
def run_benchmark() -> Dict[str, Any]:
    """Execute end-to-end evaluation on the 60 Fakeddit samples and return results."""
    from experiments.fakeddit_external import run_fakeddit_evaluation
    results = run_fakeddit_evaluation()
    return {
        "status": "success",
        "message": "Fakeddit external benchmark completed successfully.",
        "results": {
            "dataset": results.get("dataset", "Fakeddit"),
            "evaluation_samples": results.get("evaluation_samples", 60),
            "accuracy": results.get("accuracy", 0.0),
            "precision": results.get("precision", 0.0),
            "recall": results.get("recall", 0.0),
            "f1_score": results.get("f1_score", 0.0),
            "counts": results.get("counts", {}),
            "confusion_matrix": results.get("confusion_matrix", {}),
        }
    }


@router.get("/samples")
def list_benchmark_samples(
    status_filter: str = Query("all", alias="status", description="all, correct, incorrect, uncertain")
) -> Dict[str, Any]:
    """List Fakeddit benchmark samples with optional status filtering."""
    data = get_cached_results()
    samples = data.get("samples", [])

    filtered = []
    for s in samples:
        st = s.get("status", "").lower()
        if status_filter == "all" or st == status_filter.lower():
            # Return lightweight overview item
            filtered.append({
                "sample_id": s["sample_id"],
                "text": s["text"],
                "ground_truth": s["ground_truth"],
                "prediction": s["prediction"],
                "confidence": s["confidence"],
                "real_probability": s["real_probability"],
                "fake_probability": s["fake_probability"],
                "correct": s["correct"],
                "status": s["status"],
                "manipulation_type": s.get("manipulation_type", "none"),
                "image_url": f"/api/benchmark/fakeddit/image/{s['sample_id']}",
            })

    return {
        "status": "success",
        "total_available": len(samples),
        "filter_applied": status_filter,
        "count": len(filtered),
        "samples": filtered,
    }


@router.get("/sample/{sample_id}")
def get_sample_detail(sample_id: str) -> Dict[str, Any]:
    """Get full details for a single sample including explainability evidence."""
    data = get_cached_results()
    samples = data.get("samples", [])

    match = next((s for s in samples if s["sample_id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    res = dict(match)
    res["image_url"] = f"/api/benchmark/fakeddit/image/{sample_id}"
    return {"status": "success", "sample": res}


@router.get("/image/{sample_id}")
def get_sample_image(sample_id: str):
    """Serve the raw or processed image file for a given Fakeddit sample."""
    # Find matching file in raw or processed
    candidates = [
        FAKEDDIT_RAW_DIR / f"{sample_id}.jpg",
        FAKEDDIT_RAW_DIR / f"{sample_id}.jpeg",
        FAKEDDIT_RAW_DIR / f"{sample_id}.png",
        BASE_DIR / "data" / "processed" / "fakeddit" / f"{sample_id}_224x224.jpg",
        BASE_DIR / "experiments" / "external_dataset" / "data" / "fakeddit" / f"{sample_id}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return FileResponse(p, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Image for '{sample_id}' not found.")
