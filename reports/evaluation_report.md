# TruthLens AI — Public Dataset Evaluation Report

**Benchmark Dataset:** r/Fakeddit (Multimodal Misinformation Benchmark, LREC 2020)  
**Evaluation Split:** Held-out Test Set ($N=9$)  
**Model Architecture:** DistilBERT + ResNet18 (Cross-Attention Fusion)  
**Generated On:** Automated Execution via `scripts/train_and_eval_public.py`  

---

## 1. Executive Summary & Test Performance

The TruthLens AI cross-attention model was evaluated on the independent, held-out test split of the public Fakeddit benchmark ($N=9$, balanced REAL vs. FAKE).

| Metric | Measured Value | Percentage / Score | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.2222** | **22.2%** | Overall fraction of correct predictions |
| **Precision (FAKE)** | **0.0000** | **0.0%** | True Fakes divided by all predicted Fakes |
| **Recall (FAKE)** | **0.0000** | **0.0%** | Proportion of actual Fakes correctly detected |
| **F1-Score** | **0.0000** | **0.0000** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.0500** | **0.0500** | Area under the Receiver Operating Characteristic |
| **False Positive Rate (FPR)** | **0.6000** | **60.0%** | Authentic samples falsely flagged as fake |
| **False Negative Rate (FNR)** | **1.0000** | **100.0%** | Deceptive samples missed by the verifier |

---

## 2. Confusion Matrix Analysis

```text
                  Predicted REAL (0)    Predicted FAKE (1)
True REAL (0)             2 (TN)                3 (FP)
True FAKE (1)             4 (FN)                0 (TP)
```

- **True Negatives (TN):** 2 authentic social media posts correctly verified as REAL.
- **True Positives (TP):** 0 manipulated/misleading posts correctly flagged as FAKE.
- **False Positives (FP):** 3 benign posts flagged as fake (FPR = 0.600).
- **False Negatives (FN):** 4 deceptive posts missed (FNR = 1.000).

Visual Confusion Matrix available at: `reports/confusion_matrix.png`  
Visual ROC Curve available at: `reports/roc_curve.png`  

---

## 3. Methodological Discussion: Why Accuracy Alone Is Misleading

In real-world content moderation, **accuracy alone is inherently deceptive**:
1. **Asymmetric Error Costs:** A False Positive (flagging legitimate journalism as fake) censors truthful civic speech and degrades user trust. Conversely, a False Negative (allowing a deepfake to circulate) exposes the public to digital harm.
2. **Class Imbalance in the Wild:** If a platform experiences 95% authentic content and 5% misinformation, a naive "predict everything REAL" baseline achieves **95% accuracy** while detecting exactly **0%** of deepfakes (Recall = 0.0).
3. **F1 and ROC-AUC Standard:** Evaluating F1-Score (0.0000) and ROC-AUC (0.0500) provides a robust, threshold-independent measure of discriminative capacity that cannot be inflated by class distribution shifts.

---

## 4. Per-Sample Test Set Verification Log

| Sample ID | Text Snippet | Ground Truth | Prediction | Fake Prob | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fakeddit_real_021` | this alcoholic bottle has a magnifying glass so you can see ... | **REAL** | FAKE | 0.7153 | INCORRECT |
| `fakeddit_fake_012` | the new rock landscaping next to my house | **FAKE** | REAL | 0.1799 | INCORRECT |
| `fakeddit_fake_028` | smokestacks high as a kite | **FAKE** | REAL | 0.3463 | INCORRECT |
| `fakeddit_real_027` | former world chess champion vishy anand welcomed at the delh... | **REAL** | REAL | 0.2827 | CORRECT |
| `fakeddit_fake_010` | four letters fuck love | **FAKE** | REAL | 0.2679 | INCORRECT |
| `fakeddit_real_030` | google rewarded over to the guy who bought googlecom for and... | **REAL** | REAL | 0.4330 | CORRECT |
| `fakeddit_real_015` | japanese police arrest yearold ninja thief suspect | **REAL** | FAKE | 0.5074 | INCORRECT |
| `fakeddit_real_007` | moss growing with artificial lights on roman ruins from the ... | **REAL** | FAKE | 0.6217 | INCORRECT |
| `fakeddit_fake_029` | silence the giant motel | **FAKE** | REAL | 0.1958 | INCORRECT |

---

## 5. Experimental Hyperparameters

- **Base Models:** `distilbert-base-uncased` (Text) + `resnet18` (Vision)
- **Multimodal Fusion:** Bidirectional Cross-Attention (Query-Key-Value projection with residual normalization)
- **Training Epochs:** 5
- **Learning Rate:** 2e-05 (AdamW optimizer)
- **Batch Size:** 8
- **Random Seed:** 42
