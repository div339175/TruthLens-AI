# TruthLens AI — Public Dataset Class Balance & Modality Report

**Benchmark Dataset:** r/Fakeddit (Multimodal Text-Image Benchmark, LREC 2020)  
**Evaluation Scope:** Public Academic Generalization  
**Generated On:** Automated Execution via `scripts/analyze_dataset.py`  

---

## 1. Executive Summary

```text
Dataset Summary
-------------------------
Total Samples: 60

REAL:
30 (50.0%)

FAKE:
30 (50.0%)

Imbalance Ratio:
1.00
```

---

## 2. Class Distribution Analysis

| Class | Label ID | Sample Count | Percentage | Class Balance Status |
| :--- | :---: | :---: | :---: | :--- |
| **REAL** | `0` | **30** | **50.0%** | Perfect parity balance ($1:1$) |
| **FAKE** | `1` | **30** | **50.0%** | Perfect parity balance ($1:1$) |
| **Total** | -- | **60** | **100.0%** | Baseline evaluation standard |

The benchmark split exhibits an **Imbalance Ratio of 1.00**, indicating an ideally balanced evaluation distribution with zero majority-class bias. Accuracy, precision, recall, and F1-score evaluate on equal footings without requiring synthetic re-weighting.

---

## 3. Fine-Grained Manipulation Typologies

Fakeddit partitions fake multimodal content into distinct deceptive mechanisms:

| Manipulation Category | Sample Count | Proportion | Pedagogical Role |
| :--- | :---: | :---: | :--- |
| `imposter_content` | **8** | 13.3% | Synthetic or misattributed manipulation |
| `authentic` | **30** | 50.0% | Authentic reference standard |
| `satire_parody` | **10** | 16.7% | Synthetic or misattributed manipulation |
| `misleading_content` | **8** | 13.3% | Synthetic or misattributed manipulation |
| `manipulated_content` | **4** | 6.7% | Synthetic or misattributed manipulation |

---

## 4. Modality & Dimensional Profile

- **Modalities Present:** `text + image` (60 samples, 100%)
- **Audio / Video Samples:** 0 native in Fakeddit (reserved for dedicated acoustic/temporal suites)
- **Visual Spatial Resolution:**
  - *Original Aspect Ratios:* Mean 314.8 x 297.2 px (Min: 216x167, Max: 320x569)
  - *Processed Standardized Resolution:* $224 \times 224$ px (RGB 3-channel, normalized via ImageNet distributions)
- **Textual Claim Characteristics:**
  - *Average Claim Length:* 55.4 characters (9.4 words)
  - *Character Span:* Min 12 to Max 215 characters
