"""Automated test suite for Phase 6 Probability Calibration and Decision Policy."""

import pytest
import json
import torch
import torch.nn.functional as F
from pathlib import Path

from ml.models.classifier import DecisionPolicy
from experiments.phase6.calibration import (
    TemperatureScaler,
    compute_calibration_metrics,
    evaluate_policy_on_logits,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE6_MODELS = BASE_DIR / "models" / "phase6"


class TestPhase6TemperatureScaler:
    """Tests TemperatureScaler module and optimization logic."""

    def test_temperature_scaler_scaling(self):
        """Verifies that logits are divided by temperature."""
        scaler = TemperatureScaler()
        with torch.no_grad():
            scaler.temperature.fill_(2.0)

        logits = torch.tensor([[2.0, 4.0], [-1.0, 1.0]])
        scaled = scaler(logits)
        expected = torch.tensor([[1.0, 2.0], [-0.5, 0.5]])
        assert torch.allclose(scaled, expected, atol=1e-5)

    def test_temperature_scaler_fitting(self):
        """Verifies that fit routine optimizes temperature on synthetic logits."""
        # Overconfident logits on random labels
        torch.manual_seed(42)
        logits = torch.randn(100, 2) * 5.0  # high confidence
        labels = torch.randint(0, 2, (100,))

        scaler = TemperatureScaler()
        t_opt = scaler.fit(logits, labels)

        assert t_opt > 0.0, "Optimal temperature must be strictly positive"
        assert isinstance(t_opt, float)

    def test_calibration_parameters_artifact_exists(self):
        """Ensures models/phase6/calibration_params.json is saved and valid."""
        params_file = PHASE6_MODELS / "calibration_params.json"
        assert params_file.exists(), f"Expected artifact at {params_file}"

        with open(params_file) as f:
            data = json.load(f)

        assert "optimal_temperature" in data
        assert "uncertainty_threshold" in data
        assert data["optimal_temperature"] > 0.0
        assert data["uncertainty_threshold"] == 0.65


class TestPhase6CalibrationMetrics:
    """Tests ECE, MCE, and Brier Score computation."""

    def test_perfect_calibration_produces_low_ece(self):
        """Uniform perfectly calibrated probabilities should yield low ECE."""
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

        assert "ece" in metrics
        assert "brier_score" in metrics
        assert metrics["ece"] < 0.15
        assert metrics["brier_score"] == 0.25

    def test_brier_score_bounds(self):
        """Brier score must be between 0.0 and 1.0."""
        y_true = [1, 0, 1]
        y_prob = [0.9, 0.1, 0.8]
        metrics = compute_calibration_metrics(y_true, y_prob)

        assert 0.0 <= metrics["brier_score"] <= 1.0


class TestPhase6DecisionPolicyRouting:
    """Verifies REAL / FAKE / UNCERTAIN routing under calibrated probabilities."""

    def test_uncertainty_routing_under_scaled_logits(self):
        """Verifies that higher temperature scales logits into the UNCERTAIN threshold range."""
        policy = DecisionPolicy(uncertainty_threshold=0.65)
        # Raw logit produces p ~ 0.73 (decisive under 0.65)
        raw_logit = torch.tensor([0.0, 1.0])
        res_raw = policy.evaluate(raw_logit)
        assert res_raw["prediction"] == "FAKE"
        assert not res_raw["uncertain"]

        # Scaled logit with T=2.0 produces p ~ 0.62 (uncertain under 0.65)
        scaled_logit = raw_logit / 2.0
        res_scaled = policy.evaluate(scaled_logit)
        assert res_scaled["prediction"] == "UNCERTAIN"
        assert res_scaled["uncertain"]
