# TruthLens AI — Phase 6 Implementation Plan
## Model Improvement, Cross-Modal Alignment & Probability Calibration

**Project Title:** TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Phase Objective:** Improve Real-World Robustness and Reliability  
**Date:** November 2026  
**Status:** **PLAN APPROVED & READY FOR IMPLEMENTATION**  

---

## 1. Existing Architecture Summary

- **Text Encoder (`ml/models/text_encoder.py`):** DistilBERT backbone returning 768-D sequence and pooled features, projected via 2-layer MLP to a 256-D shared embedding space.
- **Vision Encoder (`ml/models/vision_encoder.py`):** ResNet18 convolutional stem extracting $7 \times 7$ feature maps, pooled and projected to the same 256-D embedding space. Supports static images and uniform 8-frame video sequences with temporal mean pooling.
- **Audio Branch (`ml/preprocessing/audio.py`, `ml/models/multimodal_model.py`):** 128-D acoustic features projected to 256-D space via `audio_projection`.
- **Fusion Mechanisms (`ml/models/fusion.py`, `ml/models/cross_attention.py`):**
  - Feature Concatenation (`models/multimodal_concat.pt`): 512-D concatenated features projected to 256-D.
  - Bidirectional Cross-Attention (`models/multimodal_cross_attention.pt`): 4-head cross-attention ($Q_v, K_t, V_t$ and $Q_t, K_v, V_v$) with residual LayerNorm.
- **Classification Head (`ml/models/classifier.py`):** 2-layer MLP (256 $\to$ 128 $\to$ 2) producing binary logits (REAL=0, FAKE=1).
- **Decision Policy (`ml/models/classifier.py`):** Softmax probabilities with calibrated `UNCERTAINTY_THRESHOLD = 0.65`.

---

## 2. Existing Checkpoints to Protect

The following checkpoints must be preserved without overwriting:
- `models/multimodal_cross_attention.pt` (Phase 1–5 primary checkpoint)
- `models/multimodal_concat.pt` (Phase 1–5 baseline checkpoint)
- `models/multimodal_baseline.pt` (Phase 1–5 initial baseline)
- `experiments/baseline/text_baseline.pt` (Phase 4 unimodal text)
- `experiments/baseline/vision_baseline.pt` (Phase 4 unimodal vision)

**Phase 6 Checkpoint Directory:** All new checkpoints will be saved strictly under `models/phase6/`:
- `models/phase6/text_domain_adapted.pt`
- `models/phase6/multimodal_contrastive_lambda0.pt`
- `models/phase6/multimodal_contrastive_lambda05.pt`
- `models/phase6/multimodal_contrastive_lambda1.pt`
- `models/phase6/multimodal_contrastive_lambda2.pt`
- `models/phase6/multimodal_calibrated.pt`
- `models/phase6/multimodal_phase6_final.pt`

---

## 3. Existing Training & External Benchmark Data

- **Course In-Domain Dataset (`ml/data/metadata/dataset.csv`):**
  - Train: 224 samples (112 Real, 112 Fake)
  - Val: 48 samples (24 Real, 24 Fake)
  - Test: 48 samples (24 Real, 24 Fake)
- **External Evaluation Manifest (`experiments/external_dataset/external_dataset_manifest.csv`):**
  - Fakeddit Test: 60 samples (30 Real, 30 Fake)
  - Audio Deepfakes: 40 samples (20 Real, 20 Fake)
  - Video Deepfakes: 20 samples (10 Real, 10 Fake)
  - Semantic Mismatch Suite: 60 samples (4 conditions $\times$ 15 samples)
- **Phase 6 Training & Validation Sets (Disjoint from Test):**
  - Ingestion from Fakeddit `train_balanced.csv` ($N=300$: 150 Real, 150 Fake) and `val_balanced.csv` ($N=100$: 50 Real, 50 Fake).
  - Explicit assertion: 0 overlap with the 60 Fakeddit test samples.

---

## 4. Current Weaknesses Identified in Phase 5

1. **Text Domain Shift:** DistilBERT accuracy plummeted from 100% on in-domain formal news to 33.33% on Fakeddit social media text ($F1 = 0.3333$, $\Delta F1 = -0.6667$).
2. **Cross-Modal Semantic Mismatch Vulnerability:** In Condition 2 (authentic photograph + false catastrophic claim), the model predicted FAKE in only 93.3% of cases on a crude threshold, but in Condition 1 (congruent real), accuracy was only 6.7% due to visual dominance (93.3% visual dominance across mismatch suites).
3. **Probability Miscalibration:** Continuous probabilities produce strong AUCs (0.81 in audio, 0.82 in video), but fixed 0.50 / 0.65 thresholds lead to extreme precision/recall imbalances and uncalibrated uncertainty routing.

---

## 5. Proposed Phase 6 Methodology

### Part A: Domain-Adaptive Text Fine-Tuning
1. Construct disjoint domain-adaptation training set ($N=300$: 150 Real, 150 Fake from Fakeddit `train_balanced.csv`) and validation set ($N=100$: 50 Real, 50 Fake from `val_balanced.csv`).
2. Train DistilBERT with joint in-domain rehearsal (224 course samples) to prevent catastrophic forgetting.
3. Train with AdamW, learning rate $3 \times 10^{-5}$, cosine decay, weight decay $0.01$, for 5 epochs with early stopping.
4. Save to `models/phase6/text_domain_adapted.pt`.
5. Evaluate on both in-domain test split and out-of-domain Fakeddit test split.

### Part B: Contrastive Cross-Modal Alignment
1. Implement symmetric InfoNCE loss (`CrossModalContrastiveLoss` in `ml/training/losses.py`):
   $$\mathcal{L}_{NCE} = -\frac{1}{2B} \sum_{i=1}^B \left[ \log \frac{\exp(\mathbf{t}_i \cdot \mathbf{v}_i / \tau)}{\sum_j \exp(\mathbf{t}_i \cdot \mathbf{v}_j / \tau)} + \log \frac{\exp(\mathbf{v}_i \cdot \mathbf{t}_i / \tau)}{\sum_j \exp(\mathbf{v}_i \cdot \mathbf{t}_j / \tau)} \right]$$
2. In-batch negative sampling where $i \neq j$, augmented with hard negative pairs from the semantic mismatch pool.
3. Multi-task loss: $\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{NCE}$.
4. Evaluate $\lambda \in \{0.0, 0.05, 0.1, 0.2\}$ where $\lambda = 0.0$ serves as unaligned baseline.
5. Save models to `models/phase6/multimodal_contrastive_*.pt`.
6. Re-evaluate on the 4-condition semantic mismatch suite ($N=60$).

### Part C: Probability Calibration
1. Implement Temperature Scaling on held-out validation sets (course val $N=48$ + Fakeddit val $N=100$):
   $$\hat{P}(Y=1 | \mathbf{z}) = \sigma\left(\frac{z_1 - z_0}{T}\right)$$
   where $T > 0$ is optimized via L-BFGS to minimize negative log-likelihood (NLL).
2. Measure Expected Calibration Error (ECE) and Brier Score before and after calibration:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
3. Re-evaluate the REAL / FAKE / UNCERTAIN decision policy: quantify reduction in false high-confidence errors and assess improvement in uncertainty routing.

---

## 6. Files That Will Be Modified and Created

### Files to Create:
- `docs/PHASE6_IMPLEMENTATION_PLAN.md` (This document)
- `docs/PHASE6_TEXT_DOMAIN_ADAPTATION.md`
- `docs/PHASE6_CONTRASTIVE_ALIGNMENT.md`
- `docs/PHASE6_CALIBRATION.md`
- `docs/PHASE6_ABLATION.md`
- `docs/PHASE6_COMPLETION_REPORT.md`
- `experiments/phase6/data/` (disjoint training/validation manifests)
- `experiments/phase6/train_text_domain_adaptation.py`
- `experiments/phase6/train_contrastive_multimodal.py`
- `experiments/phase6/calibration.py`
- `experiments/phase6/ablation_study.py`
- `experiments/phase6/evaluate_all.py`
- `experiments/phase6/visualizations_phase6.py`
- `tests/test_phase6_model_improvement.py`
- `tests/test_phase6_calibration.py`
- `tests/test_phase6_contrastive.py`

### Files to Modify:
- `PROJECT_STATUS.md` (document Phase 6 progress and test counts)

### Files to Leave Untouched:
- All Phase 1–5 checkpoints in `models/` and `experiments/baseline/`
- All Phase 1–5 datasets and metadata in `ml/data/` and `experiments/external_dataset/`
- All Phase 1–5 tests in `tests/test_*.py`

---

## 7. Expected Experiments & Success Criteria

| Experiment ID | Focus | Target Benchmark | Primary Metrics | Success Criterion |
| :--- | :--- | :--- | :--- | :--- |
| `EXP_PHASE6_TEXT_001` | Domain-Adaptive DistilBERT | Fakeddit Text Test ($N=60$) | F1, Accuracy | $\Delta F1 > +0.10$ vs Phase 5 (0.3333) |
| `EXP_PHASE6_CONTRASTIVE_001` | InfoNCE Alignment Grid ($\lambda \in \{0, .05, .1, .2\}$) | Semantic Mismatch Suite ($N=60$) | Visual Dominance %, Mismatch Acc | Reduced visual dominance in Cond 1 & 2 |
| `EXP_PHASE6_CALIB_001` | Temperature Scaling | Combined Validation Set | ECE, Brier Score, NLL | Lower ECE (improved calibration) |
| `EXP_PHASE6_ABLATION_001` | Full Ablation (A through F) | Multi-Benchmark Suite | Accuracy, F1, ECE | Quantify individual & joint impact |

---

## 8. Risks and Limitations

1. **Catastrophic Forgetting:** Fine-tuning DistilBERT exclusively on Reddit posts may degrade performance on formal news. *Mitigation:* Train on a mixed batch of 50% in-domain course text and 50% Fakeddit text.
2. **Contrastive Batch Size:** Standard InfoNCE benefits from large batches; with batch size 8–16, negative diversity is limited. *Mitigation:* Hard-negative pair sampling and symmetric loss.
3. **Calibration Preserves Ranking:** Temperature scaling is monotonic; it changes probabilities and ECE, but preserves ROC-AUC. Accuracy changes only if the decision threshold was uncalibrated.
