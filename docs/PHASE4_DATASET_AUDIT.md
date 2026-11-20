# TruthLens AI — Phase 4 Dataset Audit Report

**Audit Status:** COMPLETE & VERIFIED  
**Target CSV:** `/Users/divakarmaurya/nlp_project/ml/data/metadata/dataset.csv`  

## 1. Summary Statistics

| Metric | Value | Proportion | Notes |
| :--- | :--- | :--- | :--- |
| **Total Samples** | 320 | 100.0% | Meets university requirement ($\ge 300$) |
| **REAL Samples (Label 0)** | 160 | 50.0% | Perfectly balanced |
| **FAKE Samples (Label 1)** | 160 | 50.0% | Perfectly balanced |
| **Train Split** | 224 | 70.0% | Stratified |
| **Validation Split** | 48 | 15.0% | Stratified |
| **Test Split** | 48 | 15.0% | Stratified |

## 2. Class Distribution per Split

| Split | Total | REAL (0) | FAKE (1) | Balance Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 224 | 112 | 112 | 50.0% / 50.0% |
| **Val** | 48 | 24 | 24 | 50.0% / 50.0% |
| **Test** | 48 | 24 | 24 | 50.0% / 50.0% |

## 3. Modality Availability Audit

| Modality | Available Rows | Missing Files | Coverage Status |
| :--- | :--- | :--- | :--- |
| **Text Claims** | 320 / 320 | 0 | 100.0% Complete |
| **Image Files** | 320 / 320 | 0 | 100.0% Complete (320 unique files) |
| **Video Files** | 0 / 320 | 0 | 0.0% (No labeled video tracks) |
| **Audio Files** | 0 / 320 | 0 | 0.0% (No labeled audio tracks) |

## 4. Data Leakage & Integrity Check

- **Image Duplication (MD5)**: 0 duplicate images detected across the entire 320-sample corpus.
- **Visual Leakage (Train $\leftrightarrow$ Test)**: **0.0% overlap** (no training image appears in test/val sets).
- **Text Overlap**: 2 generic boilerplate category prompts appear across splits.
- **Missing Referenced Assets**: 0 missing referenced image files on disk.

## 5. Experiment Feasibility Matrix

| Experiment Candidate | Status | Reason / Evaluation Strategy |
| :--- | :--- | :--- |
| `text_only` | **SUPPORTED** | Evaluated on test set |
| `image_only` | **SUPPORTED** | Evaluated on test set |
| `video_only` | **NOT AVAILABLE (0 labeled video files; technical verification only)** | Not evaluated due to 0 labeled assets |
| `audio_only` | **NOT AVAILABLE (0 labeled audio files; technical verification only)** | Not evaluated due to 0 labeled assets |
| `text_image_concat` | **SUPPORTED** | Evaluated on test set |
| `text_image_cross_attention` | **SUPPORTED** | Evaluated on test set |
| `text_video` | **NOT AVAILABLE (0 labeled video files)** | Not evaluated due to 0 labeled assets |
| `text_audio` | **NOT AVAILABLE (0 labeled audio files)** | Not evaluated due to 0 labeled assets |
| `text_image_audio` | **NOT AVAILABLE (0 labeled audio files)** | Not evaluated due to 0 labeled assets |

