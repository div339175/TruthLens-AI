# TruthLens AI — Raw Dataset Staging Directory (`data/raw/`)

This directory is the designated staging area for external, publicly available research datasets used for model evaluation and benchmarking.

---

## Legal, Ethical, and Licensing Notice

In strict compliance with academic research standards and dataset licensing agreements:
1. **No Proprietary / Copyrighted Redistribution:** Do NOT commit raw third-party image, video, or audio files to this repository or push them to public Git branches.
2. **Git Ignored:** This directory's binary contents (`*.jpg`, `*.png`, `*.mp4`, `*.wav`, etc.) are ignored by `.gitignore`.
3. **Research Purpose Only:** Downloaded media assets must be utilized solely for non-commercial academic research, algorithmic audits, and deepfake detection evaluation.

---

## Obtaining the Supported Datasets

### 1. Primary Benchmark: r/Fakeddit (Multimodal Text-Image)
- **Official Repository:** [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)
- **Official Portal:** [https://fakeddit.netlify.app/](https://fakeddit.netlify.app/)
- **Citation:** Nakamura et al., LREC 2020.
- **Automated Ingestion:** Run the project verification and preprocessing script:
  ```bash
  python scripts/verify_dataset.py
  python scripts/preprocess_dataset.py
  ```
  This will validate locally cached assets or fetch verified open research mirrors deterministically into `data/raw/fakeddit/`.

### 2. Audio-Visual Benchmarks (Optional / Gated)
- **FakeAVCeleb:** Requires individual Google Form authorization from DASH-Lab (Sungkyunkwan University). Once approved, place extracted folders into `data/raw/fakeavceleb/`.
- **FaceForensics++:** Requires signing the TUM Academic Agreement. Once received, place files in `data/raw/faceforensics/`.
- **DFDC:** Available via Kaggle Deepfake Detection Challenge API. Place downloads in `data/raw/dfdc/`.

---

## Directory Organization

```text
data/
├── raw/
│   ├── README.md               <-- You are here
│   └── fakeddit/               <-- Raw image files and un-normalized metadata
├── interim/                    <-- Validated and sanitized staging area
├── processed/                  <-- Resized, normalized 224x224 tensors / assets
├── metadata/                   <-- Standardized dataset schemas and JSON manifests
└── splits/                     <-- Deterministic train.csv, validation.csv, test.csv
```
