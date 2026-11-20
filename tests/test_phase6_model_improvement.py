"""Automated test suite for Phase 6 Model Improvement and Domain Adaptation."""

import pytest
import csv
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

from ml.models.text_encoder import TextEncoder
from ml.models.multimodal_model import MultimodalVerificationModel

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE6_MODELS = BASE_DIR / "models" / "phase6"
PHASE6_DOCS = BASE_DIR / "docs"
PHASE6_FIGURES = BASE_DIR / "reports" / "figures"
PHASE6_RESULTS = BASE_DIR / "experiments" / "phase6" / "results"


class TestPhase6DomainAdaptedText:
    """Verifies that domain-adapted text model exists and runs inference correctly."""

    def test_domain_adapted_checkpoint_exists(self):
        """Ensures models/phase6/text_domain_adapted.pt exists."""
        ckpt = PHASE6_MODELS / "text_domain_adapted.pt"
        assert ckpt.exists(), f"Expected checkpoint at {ckpt}"
        data = torch.load(ckpt, map_location="cpu")
        assert "encoder_state" in data
        assert "classifier_state" in data

    def test_domain_adapted_text_encoder_forward(self):
        """Verifies forward pass through adapted text encoder."""
        ckpt = PHASE6_MODELS / "text_domain_adapted.pt"
        data = torch.load(ckpt, map_location="cpu")

        encoder = TextEncoder(model_name="distilbert-base-uncased", projection_dim=256)
        encoder.load_state_dict(data["encoder_state"], strict=False)
        encoder.eval()

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        sample_text = "Breaking: Viral social media claim debunks ongoing crisis reports."
        inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            out = encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        assert "projected_pooled" in out
        assert "projected_sequence" in out
        assert out["projected_pooled"].shape == (1, 256)


class TestPhase6DeliverablesAndReports:
    """Verifies that all Phase 6 documentation and figures exist and are non-empty."""

    def test_phase6_docs_exist(self):
        """Verifies required markdown documentation files exist."""
        required_docs = [
            "PHASE6_IMPLEMENTATION_PLAN.md",
            "PHASE6_TEXT_DOMAIN_ADAPTATION.md",
            "PHASE6_CONTRASTIVE_ALIGNMENT.md",
            "PHASE6_CALIBRATION.md",
            "PHASE6_ABLATION.md",
        ]
        for doc in required_docs:
            p = PHASE6_DOCS / doc
            assert p.exists(), f"Missing required doc: {p}"
            assert p.stat().st_size > 500, f"Doc {p} is suspiciously small"

    def test_phase6_figures_exist(self):
        """Verifies all 7 required publication figures exist in reports/figures/."""
        required_figures = [
            "phase6_domain_shift_comparison.png",
            "phase6_semantic_mismatch_comparison.png",
            "phase6_calibration_curve.png",
            "phase6_confidence_distribution.png",
            "phase6_roc_comparison.png",
            "phase6_ablation_comparison.png",
            "phase6_attention_comparison.png",
        ]
        for fig in required_figures:
            p = PHASE6_FIGURES / fig
            assert p.exists(), f"Missing figure: {p}"
            assert p.stat().st_size > 10_000, f"Figure {p} is empty or invalid"

    def test_phase6_results_csv_and_json_exist(self):
        """Verifies ablation results CSV and JSON files exist."""
        csv_file = PHASE6_RESULTS / "ablation_study_results.csv"
        json_file = PHASE6_RESULTS / "ablation_study_results.json"
        assert csv_file.exists(), f"Missing CSV: {csv_file}"
        assert json_file.exists(), f"Missing JSON: {json_file}"
