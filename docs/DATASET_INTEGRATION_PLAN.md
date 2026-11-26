# TruthLens AI — Public Dataset Integration Plan

**Project:** TruthLens AI (Multimodal NLP for Deepfake Detection and Explainable Content Verification)  
**Author:** TruthLens AI Research & Engineering Team  
**Date:** November 2026  
**Status:** Audit Approved — Implementation Plan Active  

---

## 1. Executive Summary & Repository Audit

TruthLens AI is an end-to-end multimodal verification system integrating Natural Language Processing (DistilBERT) and Computer Vision (ResNet-18) via Bidirectional Cross-Attention fusion, augmented with temporal video feature pooling (8 frames) and acoustic spectral analysis (128-D STFT). 

An exhaustive audit of the existing codebase confirms the following baseline health:
- **Test Suite Status:** 96/96 automated tests passing across API, database, video/audio pipelines, calibration, contrastive alignment, and evaluation endpoints.
- **Frontend Build Status:** Clean Vite + TypeScript build (`npm --prefix frontend run build`) with zero compilation errors.
- **Backend Architecture:** Modular FastAPI application supporting single-modality and multi-modality verification with SQLite persistence and calibrated decision policies (`UNCERTAINTY_THRESHOLD = 0.65`).
- **Data Architecture Status:** The primary internal model was trained on a 320-sample synthetic course benchmark (`ml/data/metadata/dataset.csv`). In Phase 5, out-of-domain evaluation was initiated in `experiments/external_dataset/` across 180 samples.
- **Identified Gap:** The repository currently lacks a formalized, standardized `data/` root hierarchy (`data/raw/`, `data/interim/`, `data/processed/`, `data/metadata/`, `data/splits/`), dedicated automated verification and leakage detection CLI tools, and a self-contained, reproducible training and evaluation pipeline for public research benchmarks.

---

## 2. Dataset Selection Audit: Fakeddit Benchmark (Option D)

Among the four candidate datasets evaluated in accordance with project requirements:
1. **Option A (FakeAVCeleb):** Restricted/gated dataset requiring an individual Google Request Form, manual approval, and emailed download scripts. Contains facial video and speech audio only; **contains no textual news claims or NLP modality**, making it fundamentally mismatched with TruthLens AI's core DistilBERT NLP architecture.
2. **Option B (FaceForensics++):** Restricted academic agreement required from TUM. Video-only facial manipulation; no text modality.
3. **Option C (DFDC):** ~470 GB archive hosted on Kaggle with gated credentials. Video-only; no text modality.
4. **Option D (Fakeddit - Nakamura et al., LREC 2020):** **SELECTED.**
   - **Task Compatibility:** Directly addresses multimodal misinformation, combining natural text claims (Reddit submission titles) with visual media (images).
   - **Modality Alignment:** 100% compatible with TruthLens AI's text encoder (DistilBERT), vision encoder (ResNet18), and bidirectional cross-attention fusion.
   - **Licensing & Reusability:** Publicly accessible research benchmark (CC-BY / MIT code license, open public Reddit submissions).
   - **Semantic Mismatch Detection:** Specifically includes 6-way labels including "False Connection" (Condition 2 semantic contradiction), directly solving the out-of-distribution failure mode observed in real-world news.
   - **Storage & Compute Feasibility:** Highly practical for reproducible student research and automated CI testing.

---

## 3. What Must Be Changed & Added

To integrate Fakeddit without disrupting any existing production models, API endpoints, or test suites, the following components will be created:

```
TruthLens-AI/
├── config/
│   └── dataset_config.yaml             # Centralized YAML configuration for dataset paths & parameters
├── data/
│   ├── raw/
│   │   └── README.md                   # Instructions for obtaining raw Fakeddit data
│   ├── interim/                        # Temporary validation staging
│   ├── processed/                      # Preprocessed, normalized assets
│   ├── metadata/                       # Formal dataset metadata & schema definitions
│   └── splits/                         # Leakage-free train.csv, validation.csv, test.csv
├── docs/
│   ├── DATASET_INTEGRATION_PLAN.md     # This technical audit & roadmap
│   ├── DATASET_SELECTION.md            # In-depth comparative evaluation of Options A-D
│   └── DATASET_CARD.md                 # Complete provenance, license, and empirical statistics
├── reports/
│   ├── dataset_report.md               # Automated class balance & modality report
│   ├── evaluation_report.md            # Model evaluation metrics on public dataset
│   ├── confusion_matrix.png            # Visual confusion matrix
│   ├── roc_curve.png                   # Visual ROC-AUC curve
│   └── metrics.json                    # Machine-readable evaluation results
├── scripts/
│   ├── verify_dataset.py               # Raw asset integrity and hash validator
│   ├── analyze_dataset.py              # Automated class balance & modality reporter
│   ├── preprocess_dataset.py           # Preprocessing pipeline (raw -> interim -> processed)
│   ├── split_dataset.py                # Stratified, leakage-free split generator
│   ├── check_data_leakage.py           # Verification of hash, text, and identity separation
│   └── train_and_eval_public.py        # Reproducible training & evaluation on public benchmark
├── tests/
│   └── test_public_dataset_integration.py # Automated pytest test suite
└── requirements.txt                    # Root requirements file including PyYAML & dependencies
```

---

## 4. Implementation Phasing & Safety Measures

1. **Safety First (Zero Regressions):** The internal 320-sample course dataset (`ml/data/metadata/dataset.csv`) and existing model checkpoints (`models/production/selected_model.pt`) will remain completely untouched. The public dataset pipeline will operate in the standardized `data/` root directory.
2. **Git & Legal Compliance:** `.gitignore` will be configured to strictly prevent committing heavy media binaries (`*.jpg`, `*.png`, `*.mp4`, `*.wav`) while preserving split metadata CSVs, metadata JSONs, and configuration YAMLs.
3. **Reproducibility Guarantee:** All splits and training routines will use a fixed deterministic random seed (`seed = 42`).
4. **Verification Step:** The complete test suite will be expanded to include dedicated unit and integration tests for the public dataset pipeline, ensuring 100% test pass rates.
