"""Evaluation Hub API endpoints serving verified experimental artifacts."""

import json
import csv
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from backend.app.config import settings

router = APIRouter(prefix="/evaluation", tags=["Evaluation Hub"])

BASE_DIR = settings.MODELS_DIR.parent
REPORTS_DIR = BASE_DIR / "experiments" / "reports"
PHASE6_RESULTS_DIR = BASE_DIR / "experiments" / "phase6" / "results"
PHASE7_DIR = BASE_DIR / "experiments" / "phase7"
PROD_DIR = BASE_DIR / "models" / "production"


@router.get("/overview")
def get_evaluation_overview() -> Dict[str, Any]:
    """Returns high-level project benchmarks and selected production model metadata."""
    prod_config_file = PROD_DIR / "production_config.json"
    prod_info = {}
    if prod_config_file.exists():
        with open(prod_config_file) as f:
            prod_info = json.load(f)

    return {
        "status": "success",
        "selected_model": prod_info.get("selected_configuration", "Config A"),
        "selected_model_name": prod_info.get("configuration_name", "Baseline Multimodal Cross-Attention"),
        "checkpoint": prod_info.get("source_checkpoint", "models/multimodal_cross_attention.pt"),
        "temperature": prod_info.get("temperature", 1.0),
        "uncertainty_threshold": 0.65,
        "metrics": {
            "in_domain_accuracy": 100.0,
            "in_domain_f1": 1.0000,
            "in_domain_source": "Phase 4 Master Evaluation (Course Test Split, N=48)",
            "fakeddit_multimodal_accuracy": 46.7,
            "fakeddit_text_adapted_accuracy": 56.7,
            "fakeddit_source": "Phase 5 Generalization & Phase 6 Domain Adaptation (N=60)",
            "semantic_mismatch_accuracy": 60.0,
            "semantic_mismatch_c2_accuracy": 100.0,
            "semantic_mismatch_source": "Phase 6 Contrastive Cross-Modal Alignment (N=60)",
            "audio_roc_auc": 0.8300,
            "audio_source": "Deepfake Audio v4 Benchmark (N=60)",
            "video_roc_auc": 0.8200,
            "video_source": "Generative Video Sora Benchmark (N=60)",
            "calibrated_ece": 0.2256,
            "calibrated_ece_source": "Phase 6 Temperature Scaling (Fakeddit Benchmark)",
            "total_automated_tests": 79,
        },
    }


@router.get("/in-domain")
def get_indomain_evaluation() -> Dict[str, Any]:
    """Returns Phase 4 In-Domain benchmark results for unimodal and multimodal baselines."""
    master_file = REPORTS_DIR / "master_evaluation_results.json"
    if not master_file.exists():
        raise HTTPException(status_code=404, detail="In-domain evaluation results not found.")

    with open(master_file) as f:
        data = json.load(f)

    return {
        "dataset": "Course In-Domain News Benchmark (N=48 Test Samples)",
        "baselines": data.get("baselines", {}),
        "fusion_comparison": data.get("fusion_comparison", {}),
        "ablation_study": data.get("ablation_study", {}),
        "multimodal_cross_attention": data.get("fusion_comparison", {}).get("cross_attention", {}),
    }


@router.get("/generalization")
def get_generalization_evaluation() -> Dict[str, Any]:
    """Returns Phase 5 real-world external generalization benchmark results."""
    gen_file = REPORTS_DIR / "generalization_results.json"
    if not gen_file.exists():
        raise HTTPException(status_code=404, detail="Generalization results not found.")

    with open(gen_file) as f:
        data = json.load(f)

    # Add explicit notice for FakeAVCeleb restricted access
    data["fakeavceleb_status"] = {
        "dataset_name": "FakeAVCeleb (Full Video-Audio)",
        "status": "Not Available / Restricted Gated Dataset",
        "description": "Dataset requires institutional academic approval from KAIST. Surrogated in TruthLens AI via Deepfake Audio v4 (ElevenLabs/Hume) and Generative Video Sora benchmarks.",
    }

    return data


@router.get("/semantic-mismatch")
def get_semantic_mismatch_evaluation() -> Dict[str, Any]:
    """Returns controlled 4-condition Semantic Mismatch evaluation results."""
    # Compare Phase 5 baseline vs Phase 6 contrastive results
    p5_file = REPORTS_DIR / "semantic_mismatch_results.json"
    p6_file = PHASE6_RESULTS_DIR / "contrastive_grid_results.json"

    p5_data = {}
    if p5_file.exists():
        with open(p5_file) as f:
            p5_data = json.load(f)

    p6_data = {}
    if p6_file.exists():
        with open(p6_file) as f:
            p6_data = json.load(f)

    return {
        "conditions_info": [
            {"id": "condition_1", "name": "Condition 1: Congruent Real", "description": "Authentic image + Authentic caption", "target": "REAL"},
            {"id": "condition_2", "name": "Condition 2: Semantic Mismatch Fake", "description": "Authentic image + Fabricated crisis claim", "target": "FAKE"},
            {"id": "condition_3", "name": "Condition 3: Swapped Context", "description": "Authentic caption + Swapped unrelated image", "target": "FAKE"},
            {"id": "condition_4", "name": "Condition 4: Unrelated Multimodal", "description": "Authentic image + Unrelated everyday caption", "target": "FAKE"},
        ],
        "phase5_baseline": p5_data.get("conditions", {}),
        "phase6_contrastive_runs": p6_data.get("runs", []),
        "best_lambda": p6_data.get("best_lambda", 0.05),
    }


@router.get("/calibration")
def get_calibration_evaluation() -> Dict[str, Any]:
    """Returns Phase 6 probability calibration and temperature scaling results."""
    params_file = BASE_DIR / "models" / "phase6" / "calibration_params.json"
    csv_file = PHASE6_RESULTS_DIR / "calibration_results.csv"

    if not params_file.exists():
        raise HTTPException(status_code=404, detail="Calibration parameters not found.")

    with open(params_file) as f:
        params = json.load(f)

    csv_rows = []
    if csv_file.exists():
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for r in reader:
                csv_rows.append(r)

    return {
        "optimal_temperature": params.get("optimal_temperature", 1.4628),
        "uncertainty_threshold": params.get("uncertainty_threshold", 0.65),
        "suite_results": params.get("suite_results", {}),
        "comparison_table": csv_rows,
    }


@router.get("/ablation")
def get_ablation_evaluation() -> Dict[str, Any]:
    """Returns complete 6-configuration ablation study results across all 5 benchmark suites."""
    ablation_file = PHASE6_RESULTS_DIR / "ablation_study_results.json"
    if not ablation_file.exists():
        raise HTTPException(status_code=404, detail="Ablation results not found.")

    with open(ablation_file) as f:
        data = json.load(f)

    prod_file = PROD_DIR / "production_config.json"
    prod_data = {}
    if prod_file.exists():
        with open(prod_file) as f:
            prod_data = json.load(f)

    return {
        "configurations": data,
        "selection_audit": prod_data,
    }


@router.get("/all")
def get_all_evaluation_data() -> Dict[str, Any]:
    """Bundles all verified evaluation metrics into a single payload for the frontend."""
    return {
        "overview": get_evaluation_overview(),
        "in_domain": get_indomain_evaluation(),
        "generalization": get_generalization_evaluation(),
        "semantic_mismatch": get_semantic_mismatch_evaluation(),
        "calibration": get_calibration_evaluation(),
        "ablation": get_ablation_evaluation(),
    }
