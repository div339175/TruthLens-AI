# TruthLens AI — Phase 7: Evaluation Hub Architecture & Documentation

## Executive Overview

The **TruthLens Evaluation Hub** is an interactive, scientifically grounded benchmarking dashboard integrated directly into the TruthLens AI React frontend and powered by dedicated FastAPI backend endpoints.

It transforms Phase 4–6 experimental findings into an auditable, transparent verification interface. Every metric displayed originates from reproducible experimental evaluations executed against authentic benchmark artifacts stored in `experiments/reports/` and `experiments/phase6/results/`.

---

## 1. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│               TruthLens AI Web Client                  │
│       (React 18 + Vite + Tailwind CSS + Lucide)        │
└───────────────────────────┬────────────────────────────┘
                            │
               GET /api/evaluation/* HTTP
                            │
┌───────────────────────────▼────────────────────────────┐
│              FastAPI Backend Application               │
│               backend/app/api/evaluation.py            │
└───────────────────────────┬────────────────────────────┘
                            │
     Reads Verified JSON / CSV Experimental Artifacts
                            │
┌───────────────────────────▼────────────────────────────┐
│             Persistent Research Artifacts              │
│  - experiments/reports/master_evaluation_results.json  │
│  - experiments/reports/generalization_results.json     │
│  - experiments/phase6/results/ablation_study_results   │
│  - models/production/production_config.json            │
└────────────────────────────────────────────────────────┘
```

---

## 2. Seven Interactive Hub Subtabs

### 1. Benchmark Overview
- **Production Status Card**: Identifies active production model (`Config A: Baseline Multimodal Cross-Attention`), checkpoint source (`models/multimodal_cross_attention.pt`), operational temperature ($T=1.0$), and uncertainty threshold ($\tau=0.65$).
- **Core Benchmark KPIs**: 6 high-level cards showing In-Domain Accuracy (100.0%), Fakeddit Generalization (46.7%), Semantic Mismatch Robustness (60.0%), Audio Deepfake ROC-AUC (0.83), Video Deepfake ROC-AUC (0.82), and Automated Test Count (96/96 passing).
- **Architecture Specification**: Summarizes text encoder (DistilBERT), visual encoder (ResNet18), video temporal pooling (uniform 8-frame feature averaging), audio feature extractor (128-dim STFT spectrogram), and bidirectional cross-attention.

### 2. In-Domain Benchmark (Phase 4)
- Displays unimodal and multimodal baseline performance on the 48-sample course holdout test split.
- **Classification Metrics**:
  - Accuracy: 1.0000 (100.0%)
  - Precision: 1.0000
  - Recall: 1.0000
  - Macro F1: 1.0000
  - ROC-AUC: 1.0000
- **Confusion Matrix**:
  - True Positives: 24, False Positives: 0
  - True Negatives: 24, False Negatives: 0
- **Modality Comparison**: Demonstrates the superiority of bidirectional cross-attention over naive feature concatenation.

### 3. Real-World Generalization (Phase 5)
- Evaluates out-of-domain transfer on authentic social media and generative media:
  - **Fakeddit Multimodal Benchmark**: 60 real-world Reddit posts evaluated across unimodal text, unimodal image, feature concatenation, and cross-attention.
  - **Deepfake Audio Benchmark (v4)**: 60 acoustic samples contrasting synthetic voice clones (ElevenLabs, Hume AI) against authentic human speakers. ROC-AUC: 0.8300.
  - **Generative Video Sora Benchmark**: 60 multi-frame clips comparing OpenAI Sora/Veo video synthesis against verified real-world footage. ROC-AUC: 0.8200.
  - **FakeAVCeleb Access Disclosure**: Explicit banner explaining that KAIST FakeAVCeleb requires institutional academic authorization; surrogated by verified synthetic audio/video benchmarks.

### 4. Semantic Mismatch (Condition Breakdown)
- Analyzes vulnerability to "cheapfakes" (authentic imagery paired with fabricated text):
  - **Condition 1 (Congruent Real)**: 100.0% accuracy.
  - **Condition 2 (Semantic Mismatch Fake)**: Baseline achieved 0.0% accuracy due to visual dominance; contrastive cross-modal alignment ($\lambda=0.05$) restores detection to **100.0%**.
  - **Condition 3 (Swapped Context)**: 86.7% accuracy.
  - **Condition 4 (Unrelated Multimodal)**: 80.0% accuracy.

### 5. Probability Calibration (Phase 6)
- Audits model reliability, Expected Calibration Error (ECE), and overconfidence:
  - Uncalibrated In-Domain ECE: 0.0723.
  - Temperature Scaling Optimization: Optimal post-processing temperature $T^* = 1.4628$ minimizes negative log-likelihood on unconstrained social media.
  - **Uncertainty Guardrail Policy**: All samples with confidence $< 65.0\%$ route to human analysts (`UNCERTAIN`), eliminating high-risk false certainties.

### 6. Ablation Study (Configurations A to F)
- Complete matrix auditing the six Phase 6 configurations across 5 evaluation suites:
  - **Config A**: Baseline Cross-Attention ($T=1.0$) — Selected Production Model (Composite: 8.32).
  - **Config B**: Domain-Adapted Text ($T=1.0$) — High mismatch accuracy (73.3%), in-domain calibration drift (Composite: 8.01).
  - **Config C**: Contrastive Alignment ($\lambda=0.05$) — Best Condition 2 accuracy (100.0%), sharp calibration (Composite: 8.26).
  - **Config D**: Calibration Scaling ($T=1.46$) — Conservative prediction, identical ranking to A (Composite: 8.31).
  - **Config E**: Joint Adaptation + Contrastive ($T=1.0$) — Evidence of negative transfer (Composite: 7.74).
  - **Config F**: Full Proposed Pipeline ($T=1.46$) — Conservative ensemble (Composite: 7.82).

### 7. Model Comparison
- Direct side-by-side metric comparison between Config A (Production), Config C (Alternative Candidate for Mismatch), and Config B (Alternative for Text Dominance), explaining why Config A was chosen for general deployment while preserving Config C for targeted cross-modal audits.

---

## 3. Backend Endpoints Summary

| Endpoint | Method | Response Description |
| :--- | :---: | :--- |
| `/api/evaluation/overview` | `GET` | High-level KPI metrics, selected model info, test suite status |
| `/api/evaluation/in-domain` | `GET` | In-domain test split metrics, baselines, confusion matrix |
| `/api/evaluation/generalization` | `GET` | Fakeddit, Audio v4, Video Sora benchmarks, FakeAVCeleb notice |
| `/api/evaluation/semantic-mismatch` | `GET` | 4-condition semantic mismatch comparison, contrastive metrics |
| `/api/evaluation/calibration` | `GET` | ECE metrics, Brier scores, temperature scaling parameters |
| `/api/evaluation/ablation` | `GET` | 6-configuration ablation comparison matrix and selection audit |
| `/api/evaluation/all` | `GET` | Consolidated payload for one-trip frontend initialization |
