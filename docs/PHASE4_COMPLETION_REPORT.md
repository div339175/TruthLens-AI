# TruthLens AI — Phase 4 Completion Report

**Project Title:** Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Phase Objective:** Experimental Evaluation, Baselines, Fusion Comparison, and Ablation Study  
**Phase Status:** **COMPLETE, VERIFIED, & REPRODUCIBLE** (53/53 tests passing)  
**Date:** November 2026  

---

## A. What Was Implemented

In Phase 4, we built a modular, scientifically transparent, and reproducible experimental evaluation framework under `experiments/`:
1. **Dataset Audit (`experiments/dataset_audit.py`):**  
   Audits sample counts, class distributions, modality coverage, and performs strict image-hash and text data leakage detection between train and test splits.
2. **Evaluation Metrics Engine (`experiments/metrics.py`):**  
   Implements scikit-learn standard evaluation functions: Accuracy, Precision, Recall, F1-Score (binary & macro), ROC-AUC, PR-AUC, Confusion Matrix, and class-wise metrics with sample support.
3. **Unimodal Baselines (`experiments/baselines.py`):**  
   Systematically evaluates DistilBERT (Text-Only) and ResNet18 (Vision-Only) on the test split, while rigorously documenting Video as technical-verification only and Audio as unavailable.
4. **Multimodal Fusion Comparison (`experiments/fusion_comparison.py`):**  
   Compares Feature Concatenation (`models/multimodal_concat.pt`) against Bidirectional Cross-Attention (`models/multimodal_cross_attention.pt`) on identical test data.
5. **Modality Ablation Study (`experiments/ablation.py`):**  
   Measures performance changes ($\Delta F1$, $\Delta Acc$) when removing modalities from the full multimodal architecture.
6. **Qualitative Error Analysis of 15 Difficult Cases (`experiments/difficult_cases.py`):**  
   Mines and analyzes genuine model outputs across cross-modal semantic mismatches, deepfake face-swaps, generative diffusion noise, spliced banners, and borderline edge cases.
7. **Explainability & Evidence Verification (`experiments/explainability_eval.py`):**  
   Verifies that token saliency, Layer4 Grad-CAM overlays, video frame sampling, temporal pooling logs, and acoustic vocoder indicators are generated via genuine model pathways.
8. **Publishable Visualizations (`experiments/visualizations.py`):**  
   Generates Confusion Matrices grid, ROC curves with AUC, Precision-Recall curves, and comparative bar charts saved to `reports/figures/`.
9. **Master Runner (`experiments/evaluate.py`):**  
   Orchestrates end-to-end evaluation, saving consolidated reports to `experiments/reports/master_evaluation_results.json`.
10. **Phase 4 Test Suite (`tests/test_phase4_evaluation.py`):**  
    9 new unit tests verifying metric calculations, confusion matrix layout, length mismatch handling, reproducibility seeds, serialization, and ablation deltas (bringing total suite to 53 tests).

---

## B. Dataset Actually Used

- **Metadata Source:** `ml/data/metadata/dataset.csv`
- **Total Labeled Samples:** **320** (Meets course requirement $\ge 300$)
- **REAL Samples (Label 0):** **160** (50.0%)
- **FAKE Samples (Label 1):** **160** (50.0%)
- **Splits:**
  - **Train:** 224 samples (112 Real, 112 Fake — 70.0%)
  - **Validation:** 48 samples (24 Real, 24 Fake — 15.0%)
  - **Test:** 48 samples (24 Real, 24 Fake — 15.0%)
- **Data Integrity & Leakage:**
  - 320 unique PNG image files on disk in `ml/data/samples/`.
  - 0 missing referenced assets.
  - 0 duplicate images (all 320 MD5 hashes are unique).
  - **Visual Leakage (Train $\leftrightarrow$ Test): 0.0%** (zero image overlap).

---

## C. Experiments Actually Executed

1. **EXP_TEXT_001:** Unimodal DistilBERT text baseline evaluated on test split ($N=48$).
2. **EXP_VISION_001:** Unimodal ResNet18 vision baseline evaluated on test split ($N=48$).
3. **EXP_MULTIMODAL_CONCAT_001:** Multimodal Concatenation Fusion evaluated on test split ($N=48$).
4. **EXP_MULTIMODAL_CA_001:** Multimodal Bidirectional Cross-Attention evaluated on test split ($N=48$).
5. **Modality Ablation Suite:** Full model vs Full - Text vs Full - Visual.
6. **15 Difficult Multimodal Cases Analysis:** Qualitative error mining across 6 manipulation categories.
7. **Explainability Pathway Validation:** Tested token saliency, Grad-CAM, video temporal pooling, and audio vocoder metrics.

---

## D. Experiments Unavailable and Why

1. **Video Quantitative Benchmark (EXP_VIDEO_001):**  
   **NOT AVAILABLE.** The current `dataset.csv` contains 0 labeled video files. While the video frame sampling and feature-level temporal pooling pipeline is mathematically operational (8/8 video tests pass), reporting a quantitative accuracy percentage without labeled video data would be academically dishonest.
2. **Audio Quantitative Benchmark (EXP_AUDIO_001):**  
   **NOT AVAILABLE.** The current `dataset.csv` contains 0 labeled audio files. While the 128-D acoustic feature extraction and trainable `audio_projection` were verified via backpropagation gradient testing (9/9 audio tests pass), quantitative accuracy is unavailable until a labeled benchmark (e.g. FakeAVCeleb) is acquired.
3. **Audio Ablation (Full - Audio):**  
   **NOT AVAILABLE.** Cannot compute empirical $\Delta F1$ without labeled audio evaluation data.

---

## E. Exact Numerical Results

### Unimodal Baselines & Multimodal Models (Test Split, N = 48)

| Experiment ID | Model Name | Modality | Fusion | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_TEXT_001** | DistilBERT Baseline | Text | N/A | **100.00%** | 100.00% | 100.00% | **1.0000** | 1.0000 | 1.0000 |
| **EXP_VISION_001** | ResNet18 Baseline | Visual | N/A | **100.00%** | 100.00% | 100.00% | **1.0000** | 1.0000 | 1.0000 |
| **EXP_MULTIMODAL_CONCAT_001** | Concat Multimodal | Text+Visual | Concat | **100.00%** | 100.00% | 100.00% | **1.0000** | 1.0000 | 1.0000 |
| **EXP_MULTIMODAL_CA_001** | Cross-Attention Multimodal | Text+Visual | Cross-Attn | **100.00%** | 100.00% | 100.00% | **1.0000** | 1.0000 | 1.0000 |
| **EXP_VIDEO_001** | Video Temporal Pooling | Video | N/A | N/A | N/A | N/A | N/A | N/A | N/A (Technical Only) |
| **EXP_AUDIO_001** | Acoustic Projection | Audio | N/A | N/A | N/A | N/A | N/A | N/A | N/A (Not Available) |

---

## F. Fusion Comparison

| Model | Modalities | Fusion | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP_MULTIMODAL_CONCAT_001** | Text + Visual | Concat | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **EXP_MULTIMODAL_CA_001** | Text + Visual | Cross-Attention | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Both fusion strategies achieve identical 100% metrics on the clean test split. However, Bidirectional Cross-Attention generates **interpretable token-to-patch alignment scores**, allowing the system to isolate which text words correlate with specific spatial image regions.

---

## G. Ablation Results

| Configuration | Modalities | Accuracy | F1-Score | $\Delta$Accuracy | $\Delta$F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Model** | Text + Visual (Cross-Attn) | **100.00%** | **1.0000** | 0.0000 (Baseline) | 0.0000 (Baseline) | COMPLETE |
| **Full - Text** | Visual Only | **100.00%** | **1.0000** | 0.0000 | 0.0000 | COMPLETE |
| **Full - Visual** | Text Only | **100.00%** | **1.0000** | 0.0000 | 0.0000 | COMPLETE |
| **Full - Audio** | Text + Visual (No Audio) | N/A | N/A | N/A | N/A | NOT AVAILABLE (0 labeled audio files) |

---

## H. Difficult-Case Results

The 15 challenging multimodal cases yielded an overall accuracy of **73.3% (11/15 correct)**:
- **Deepfake Faceswaps (Cases 05–07):** 3/3 Correct (100%) — detected subtle boundary blending.
- **Diffusion Synthesis (Cases 08–10):** 3/3 Correct (100%) — detected fine noise anomalies.
- **Spliced Banners (Cases 11–12):** 2/2 Correct (100%) — localized copy-move and banner forgery.
- **Synthetic Speech Claims (Cases 13–14):** 2/2 Correct (100%) — correctly resolved claim + visual capture.
- **Authentic Edge Case (Case 15):** 1/1 Correct (100%) — correctly identified genuine satellite telemetry.
- **Semantic Mismatch (Cases 01–04):** **0/4 Correct (0%) — MISCLASSIFIED as REAL.**
  - **Identified Failure Mode:** Because the images were authentic photographs (e.g. peaceful summer festivals, wildfire drills, space launches), the vision encoder produced high "REAL" confidence. The strong visual authenticity dominated over the false catastrophic text captions.
  - **Mitigation:** Requires contrastive cross-modal pre-training (e.g., InfoNCE loss) specifically fine-tuned on caption-image contradictions.

---

## I. Explainability Results

- **Text:** Token-level gradient attribution verified; top influential tokens extracted with numerical saliency weights.
- **Visual (Image):** Layer4 Grad-CAM activation heatmap overlay verified; base64 PNG generated highlighting suspicious regions.
- **Video:** Uniform 8-frame sampling, representative keyframe Grad-CAM (frame 4), and temporal feature pooling verified.
- **Audio:** Zero-crossing rate, high-frequency energy ratio, and duration metrics verified.

---

## J. Test Results

```bash
.venv/bin/pytest tests/ -v
# 53 passed in 13.91s (100% passing)
```
- API Tests: 9 passed
- Audio Pipeline Tests: 9 passed
- Database & History Tests: 8 passed
- Model Architecture Tests: 6 passed
- Persistence Workflow: 1 passed
- Phase 4 Evaluation Tests: 9 passed
- Preprocessing Tests: 3 passed
- Video Pipeline Tests: 8 passed

---

## K. Frontend Build Result

```bash
cd frontend && npm run build
# Built in 844ms (0 TypeScript / JSX errors)
```
**Status: PASS**

---

## L. Limitations

1. **Clean Test Split Ceiling:** The standard 48-sample test split yields 100% accuracy due to distinct categorical separation. Real-world complexity is demonstrated by the 73.3% accuracy on the difficult case suite.
2. **Missing Video & Audio Benchmark Data:** No labeled video or audio files currently exist in `dataset.csv`.
3. **Semantic Mismatch Vulnerability:** Models relying heavily on visual realism remain vulnerable to false captions on authentic images.

---

## M. Recommended Next Phase (Phase 5)

Acquisition and integration of external benchmark datasets:
1. **Fakeddit Subset (500+ samples):** For fine-tuning semantic mismatch detection and caption-image contradiction.
2. **FakeAVCeleb / DFDC Subset (100+ samples):** For authentic video clips and synthetic cloned audio tracks to produce real benchmark numbers for the video and audio branches.
