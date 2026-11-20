"""Real-World Benchmark Expansion and Cross-Dataset Generalization (Phase 5).

Evaluates in-domain trained TruthLens models on external real-world benchmarks:
1. EXP_FAKEDDIT_TEXT_001: Unimodal Text on Fakeddit (N=60)
2. EXP_FAKEDDIT_VISION_001: Unimodal Vision on Fakeddit (N=60)
3. EXP_FAKEDDIT_CONCAT_001: Multimodal Concat on Fakeddit (N=60)
4. EXP_FAKEDDIT_CA_001: Multimodal Cross-Attention on Fakeddit (N=60)
5. EXP_AUDIO_GEN_001: Audio-Only on Deepfake Audio Detection v4 (N=40)
6. EXP_VIDEO_GEN_001: Video-Only on Generative Video Deepfake Benchmark (N=20)
7. EXP_FAKEAVCELEB_001: Marked NOT AVAILABLE (Gated / Google Form required)

Measures zero-shot domain-shift degradation against Phase 4 in-domain benchmarks.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.external_dataset.dataset import TruthLensExternalDataset
from ml.models.multimodal_model import MultimodalVerificationModel
from experiments.metrics import compute_comprehensive_metrics

REPORTS_DIR = BASE_DIR / "experiments" / "reports"
MODELS_DIR = BASE_DIR / "models"
BASELINES_DIR = BASE_DIR / "experiments" / "baseline"
MANIFEST_PATH = BASE_DIR / "experiments" / "external_dataset" / "external_dataset_manifest.csv"

# Phase 4 In-Domain Baselines for Delta Comparison
PHASE4_IN_DOMAIN_METRICS = {
    "text": {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0},
    "vision": {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0},
    "multimodal_concat": {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0},
    "multimodal_ca": {"accuracy": 1.0, "f1": 1.0, "roc_auc": 1.0},
}


def evaluate_fakeddit_experiments(device: torch.device) -> Dict[str, Any]:
    """Evaluates unimodal and multimodal TruthLens models on Fakeddit benchmark."""
    torch.manual_seed(42)
    dataset = TruthLensExternalDataset(MANIFEST_PATH, source_dataset="fakeddit")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"\n[INFO] Evaluating on Fakeddit External Benchmark (N={len(dataset)})...")

    results = {}

    # 1. Text-Only Baseline
    text_model = MultimodalVerificationModel(fusion_type="concat", projection_dim=256).to(device)
    text_ckpt = BASELINES_DIR / "text_baseline.pt"
    if text_ckpt.exists():
        data = torch.load(text_ckpt, map_location=device)
        text_model.text_encoder.load_state_dict(data["encoder_state"], strict=False)
        text_model.text_classifier.load_state_dict(data["classifier_state"], strict=False)
    text_model.eval()

    y_true, y_pred, y_prob, failures_text = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true.extend(batch["label"].tolist())
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = text_model(input_ids=ids, attention_mask=mask)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob.extend(probs)
            y_pred.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures_text.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "text": batch["raw_text"][i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics_text = compute_comprehensive_metrics(y_true, y_pred, y_prob)
    results["fakeddit_text"] = {
        "experiment_id": "EXP_FAKEDDIT_TEXT_001",
        "model_name": "TruthLens DistilBERT Unimodal Baseline",
        "benchmark": "Fakeddit (External)",
        "modality": "text",
        "sample_count": len(y_true),
        "status": "EVALUATED",
        "metrics": metrics_text,
        "domain_shift": {
            "in_domain_f1": PHASE4_IN_DOMAIN_METRICS["text"]["f1"],
            "external_f1": metrics_text["f1"],
            "delta_f1": metrics_text["f1"] - PHASE4_IN_DOMAIN_METRICS["text"]["f1"],
            "in_domain_acc": PHASE4_IN_DOMAIN_METRICS["text"]["accuracy"],
            "external_acc": metrics_text["accuracy"],
            "delta_acc": metrics_text["accuracy"] - PHASE4_IN_DOMAIN_METRICS["text"]["accuracy"],
        },
        "failures": failures_text,
    }

    # 2. Vision-Only Baseline
    vision_model = MultimodalVerificationModel(fusion_type="concat", projection_dim=256).to(device)
    vision_ckpt = BASELINES_DIR / "vision_baseline.pt"
    if vision_ckpt.exists():
        data = torch.load(vision_ckpt, map_location=device)
        vision_model.vision_encoder.load_state_dict(data["encoder_state"], strict=False)
        vision_model.vision_classifier.load_state_dict(data["classifier_state"], strict=False)
    vision_model.eval()

    y_true_v, y_pred_v, y_prob_v, failures_vision = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true_v.extend(batch["label"].tolist())
            imgs = batch["image"].to(device)
            out = vision_model(images=imgs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob_v.extend(probs)
            y_pred_v.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures_vision.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics_vision = compute_comprehensive_metrics(y_true_v, y_pred_v, y_prob_v)
    results["fakeddit_vision"] = {
        "experiment_id": "EXP_FAKEDDIT_VISION_001",
        "model_name": "TruthLens ResNet18 Unimodal Baseline",
        "benchmark": "Fakeddit (External)",
        "modality": "visual",
        "sample_count": len(y_true_v),
        "status": "EVALUATED",
        "metrics": metrics_vision,
        "domain_shift": {
            "in_domain_f1": PHASE4_IN_DOMAIN_METRICS["vision"]["f1"],
            "external_f1": metrics_vision["f1"],
            "delta_f1": metrics_vision["f1"] - PHASE4_IN_DOMAIN_METRICS["vision"]["f1"],
            "in_domain_acc": PHASE4_IN_DOMAIN_METRICS["vision"]["accuracy"],
            "external_acc": metrics_vision["accuracy"],
            "delta_acc": metrics_vision["accuracy"] - PHASE4_IN_DOMAIN_METRICS["vision"]["accuracy"],
        },
        "failures": failures_vision,
    }

    # 3. Multimodal Concat
    concat_model = MultimodalVerificationModel(fusion_type="concat", projection_dim=256).to(device)
    concat_ckpt = MODELS_DIR / "multimodal_concat.pt"
    if concat_ckpt.exists():
        concat_model.load_state_dict(torch.load(concat_ckpt, map_location=device), strict=False)
    concat_model.eval()

    y_true_c, y_pred_c, y_prob_c, failures_concat = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true_c.extend(batch["label"].tolist())
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            imgs = batch["image"].to(device)
            out = concat_model(input_ids=ids, attention_mask=mask, images=imgs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob_c.extend(probs)
            y_pred_c.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures_concat.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "text": batch["raw_text"][i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics_concat = compute_comprehensive_metrics(y_true_c, y_pred_c, y_prob_c)
    results["fakeddit_concat"] = {
        "experiment_id": "EXP_FAKEDDIT_CONCAT_001",
        "model_name": "TruthLens Concatenation Fusion",
        "benchmark": "Fakeddit (External)",
        "modality": "text+visual",
        "sample_count": len(y_true_c),
        "status": "EVALUATED",
        "metrics": metrics_concat,
        "domain_shift": {
            "in_domain_f1": PHASE4_IN_DOMAIN_METRICS["multimodal_concat"]["f1"],
            "external_f1": metrics_concat["f1"],
            "delta_f1": metrics_concat["f1"] - PHASE4_IN_DOMAIN_METRICS["multimodal_concat"]["f1"],
            "in_domain_acc": PHASE4_IN_DOMAIN_METRICS["multimodal_concat"]["accuracy"],
            "external_acc": metrics_concat["accuracy"],
            "delta_acc": metrics_concat["accuracy"] - PHASE4_IN_DOMAIN_METRICS["multimodal_concat"]["accuracy"],
        },
        "failures": failures_concat,
    }

    # 4. Multimodal Cross-Attention
    ca_model = MultimodalVerificationModel(fusion_type="cross_attention", projection_dim=256).to(device)
    ca_ckpt = MODELS_DIR / "multimodal_cross_attention.pt"
    if ca_ckpt.exists():
        ca_model.load_state_dict(torch.load(ca_ckpt, map_location=device), strict=False)
    ca_model.eval()

    y_true_ca, y_pred_ca, y_prob_ca, failures_ca = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true_ca.extend(batch["label"].tolist())
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            imgs = batch["image"].to(device)
            out = ca_model(input_ids=ids, attention_mask=mask, images=imgs)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob_ca.extend(probs)
            y_pred_ca.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures_ca.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "text": batch["raw_text"][i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics_ca = compute_comprehensive_metrics(y_true_ca, y_pred_ca, y_prob_ca)
    results["fakeddit_cross_attention"] = {
        "experiment_id": "EXP_FAKEDDIT_CA_001",
        "model_name": "TruthLens Bidirectional Cross-Attention",
        "benchmark": "Fakeddit (External)",
        "modality": "text+visual",
        "sample_count": len(y_true_ca),
        "status": "EVALUATED",
        "metrics": metrics_ca,
        "domain_shift": {
            "in_domain_f1": PHASE4_IN_DOMAIN_METRICS["multimodal_ca"]["f1"],
            "external_f1": metrics_ca["f1"],
            "delta_f1": metrics_ca["f1"] - PHASE4_IN_DOMAIN_METRICS["multimodal_ca"]["f1"],
            "in_domain_acc": PHASE4_IN_DOMAIN_METRICS["multimodal_ca"]["accuracy"],
            "external_acc": metrics_ca["accuracy"],
            "delta_acc": metrics_ca["accuracy"] - PHASE4_IN_DOMAIN_METRICS["multimodal_ca"]["accuracy"],
        },
        "failures": failures_ca,
    }

    return results


def evaluate_audio_benchmark(device: torch.device) -> Dict[str, Any]:
    """Evaluates TruthLens Audio branch on Deepfake Audio Detection external benchmark."""
    torch.manual_seed(42)
    dataset = TruthLensExternalDataset(MANIFEST_PATH, source_dataset="deepfake_audio_v4")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    print(f"\n[INFO] Evaluating on Audio Deepfake External Benchmark (N={len(dataset)})...")

    model = MultimodalVerificationModel(fusion_type="cross_attention", projection_dim=256).to(device)
    ca_ckpt = MODELS_DIR / "multimodal_cross_attention.pt"
    if ca_ckpt.exists():
        model.load_state_dict(torch.load(ca_ckpt, map_location=device), strict=False)
    model.eval()

    y_true, y_pred, y_prob, failures = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true.extend(batch["label"].tolist())
            auds = batch["audio"].to(device)
            out = model(audio_features=auds)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob.extend(probs)
            y_pred.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "audio_path": batch["audio_path"][i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)

    return {
        "experiment_id": "EXP_AUDIO_GEN_001",
        "model_name": "TruthLens Audio Projection Branch",
        "benchmark": "Deepfake Audio Detection (v4, ElevenLabs/Hume vs Human)",
        "modality": "audio",
        "sample_count": len(y_true),
        "status": "EVALUATED",
        "metrics": metrics,
        "failures": failures,
    }


def evaluate_video_benchmark(device: torch.device) -> Dict[str, Any]:
    """Evaluates TruthLens Video temporal pooling on Generative Video benchmark."""
    torch.manual_seed(42)
    dataset = TruthLensExternalDataset(MANIFEST_PATH, source_dataset="generative_video_benchmark")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\n[INFO] Evaluating on Video Deepfake External Benchmark (N={len(dataset)})...")

    model = MultimodalVerificationModel(fusion_type="cross_attention", projection_dim=256).to(device)
    ca_ckpt = MODELS_DIR / "multimodal_cross_attention.pt"
    if ca_ckpt.exists():
        model.load_state_dict(torch.load(ca_ckpt, map_location=device), strict=False)
    model.eval()

    y_true, y_pred, y_prob, failures = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            y_true.extend(batch["label"].tolist())
            vid_frames = batch["video"].to(device)  # [1, 8, 3, 224, 224]
            # Squeeze batch dimension for video_frames [8, 3, 224, 224]
            out = model(video_frames=vid_frames.squeeze(0))
            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            y_prob.extend(probs)
            y_pred.extend(preds)

            for i in range(len(preds)):
                if preds[i] != batch["label"][i].item():
                    failures.append({
                        "sample_id": batch["sample_id"][i],
                        "true_label": batch["label"][i].item(),
                        "pred_label": preds[i],
                        "confidence": probs[i] if preds[i] == 1 else 1 - probs[i],
                        "video_path": batch["video_path"][i],
                        "failure_type": "false_positive" if preds[i] == 1 else "false_negative",
                    })

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob)

    return {
        "experiment_id": "EXP_VIDEO_GEN_001",
        "model_name": "TruthLens Video Temporal Feature Pooling",
        "benchmark": "Generative Video Benchmark (Sora/Veo vs Real)",
        "modality": "video",
        "sample_count": len(y_true),
        "status": "EVALUATED",
        "metrics": metrics,
        "failures": failures,
    }


def get_fakeavceleb_status() -> Dict[str, Any]:
    """Records official feasibility status of FakeAVCeleb per scientific integrity rule."""
    return {
        "experiment_id": "EXP_FAKEAVCELEB_001",
        "model_name": "TruthLens Tri-modal Architecture",
        "benchmark": "FakeAVCeleb (Audio-Video Deepfakes)",
        "modality": "video+audio",
        "status": "NOT AVAILABLE",
        "reason": (
            "Official FakeAVCeleb requires signed academic Google Request Form approval. "
            "Public HuggingFace mirrors contain metadata only (raw video bytes: None). "
            "In compliance with TruthLens Scientific Integrity Guidelines, this benchmark is "
            "transparently documented as restricted/unavailable rather than fabricating access."
        ),
    }


def run_all_generalization_evaluations(device_str: str = "auto") -> Dict[str, Any]:
    """Executes the complete Phase 5 generalization evaluation suite."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print("=" * 70)
    print("TRUTHLENS AI — PHASE 5 BENCHMARK EXPANSION & GENERALIZATION")
    print(f"Device: {device}")
    print("=" * 70)

    fakeddit_results = evaluate_fakeddit_experiments(device)
    audio_results = evaluate_audio_benchmark(device)
    video_results = evaluate_video_benchmark(device)
    fakeavceleb_results = get_fakeavceleb_status()

    consolidated = {
        "fakeddit": fakeddit_results,
        "audio": audio_results,
        "video": video_results,
        "fakeavceleb": fakeavceleb_results,
    }

    # Save JSON Report
    json_path = REPORTS_DIR / "generalization_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    # Save Comparison CSV
    csv_path = REPORTS_DIR / "generalization_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment ID", "Model", "Benchmark", "Modality", "Sample Count", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Delta Acc (vs In-Domain)", "Delta F1 (vs In-Domain)"])

        for k, v in fakeddit_results.items():
            m = v["metrics"]
            ds = v["domain_shift"]
            writer.writerow([
                v["experiment_id"],
                v["model_name"],
                v["benchmark"],
                v["modality"],
                v["sample_count"],
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m.get('roc_auc', 0.0):.4f}",
                f"{ds['delta_acc']:+.4f}",
                f"{ds['delta_f1']:+.4f}",
            ])

        # Audio row
        am = audio_results["metrics"]
        writer.writerow([
            audio_results["experiment_id"],
            audio_results["model_name"],
            audio_results["benchmark"],
            audio_results["modality"],
            audio_results["sample_count"],
            f"{am['accuracy']:.4f}",
            f"{am['precision']:.4f}",
            f"{am['recall']:.4f}",
            f"{am['f1']:.4f}",
            f"{am.get('roc_auc', 0.0):.4f}",
            "N/A (New Modality)",
            "N/A (New Modality)",
        ])

        # Video row
        vm = video_results["metrics"]
        writer.writerow([
            video_results["experiment_id"],
            video_results["model_name"],
            video_results["benchmark"],
            video_results["modality"],
            video_results["sample_count"],
            f"{vm['accuracy']:.4f}",
            f"{vm['precision']:.4f}",
            f"{vm['recall']:.4f}",
            f"{vm['f1']:.4f}",
            f"{vm.get('roc_auc', 0.0):.4f}",
            "N/A (New Modality)",
            "N/A (New Modality)",
        ])

    print("\n" + "=" * 70)
    print("GENERALIZATION EVALUATION COMPLETE")
    print(f"Results saved to: {json_path}")
    print(f"Comparison CSV saved to: {csv_path}")
    print("=" * 70)

    return consolidated


if __name__ == "__main__":
    run_all_generalization_evaluations()
