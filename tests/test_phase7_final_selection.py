"""Phase 7 Test Suite: Final Model Selection & Production Checkpoint Alias.

Verifies:
1. Production model alias 'models/production/selected_model.pt' exists and is non-empty.
2. Production configuration JSON contains complete architecture metadata and calibrated thresholds.
3. Model selection matrix CSV records all 6 evaluated configurations (A through F).
4. TruthLensPredictor cleanly loads the selected production model alias without errors.
"""

import json
from pathlib import Path
import pytest
import torch

from backend.app.config import settings
from ml.inference.predictor import TruthLensPredictor
from ml.models.multimodal_model import MultimodalVerificationModel


def test_production_model_alias_exists():
    """Verify that models/production/selected_model.pt exists and is a valid checkpoint."""
    prod_path = Path("models/production/selected_model.pt")
    assert prod_path.exists(), f"Production model alias not found at {prod_path}"
    assert prod_path.is_file(), f"{prod_path} is not a file"
    file_size_mb = prod_path.stat().st_size / (1024 * 1024)
    assert file_size_mb > 50, f"Production checkpoint is unexpectedly small: {file_size_mb:.2f} MB"


def test_production_config_json():
    """Verify that models/production/production_config.json is well-formed with required keys."""
    config_path = Path("models/production/production_config.json")
    assert config_path.exists(), f"Production config not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["selected_configuration"] == "Config A"
    assert data["source_checkpoint"] == "models/multimodal_cross_attention.pt"
    assert data["production_checkpoint_alias"] == "models/production/selected_model.pt"
    assert data["temperature"] == 1.0
    assert data["uncertainty_threshold"] == 0.65

    assert "metrics_summary" in data
    metrics = data["metrics_summary"]
    assert metrics["in_domain_accuracy"] == 100.0
    assert metrics["composite_score"] > 8.0

    assert "all_candidates" in data
    assert len(data["all_candidates"]) == 6


def test_model_selection_matrix_csv():
    """Verify that the model selection matrix CSV records all 6 configurations."""
    csv_path = Path("experiments/phase7/model_selection_matrix.csv")
    assert csv_path.exists(), f"Model selection matrix not found at {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0]
    assert "code" in header and "composite_score" in header
    # 1 header + 6 configs (A to F) = 7 rows minimum
    assert len(lines) >= 7
    config_names = [line.split(",")[0] for line in lines[1:]]
    for expected in ["Config A", "Config B", "Config C", "Config D", "Config E", "Config F"]:
        assert any(expected in c for c in config_names), f"Missing {expected} in matrix"


def test_predictor_loads_production_checkpoint():
    """Verify that TruthLensPredictor loads models/production/selected_model.pt."""
    predictor = TruthLensPredictor()
    assert predictor.is_checkpoint_loaded, "Predictor did not load a trained checkpoint"
    assert predictor.model is not None, "Model was not initialized in predictor"
    assert isinstance(predictor.model, MultimodalVerificationModel)
    assert predictor.temperature == settings.CALIBRATION_TEMPERATURE
