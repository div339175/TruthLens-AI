# TruthLens AI

**Multimodal NLP for Deepfake Detection and Explainable Content Verification**  
*Comprehensive Multimodal Verification System (Phases 1–7 Complete)*

---

## 1. Project Overview & Problem Statement

Digital manipulation and synthetic content generation have evolved beyond isolated modalities. Today's misinformation ecosystem combines:
- Generative text and manipulative social media captions
- Diffusion-generated or facial-reenactment images
- Deepfake video manipulations with temporal anomalies
- Voice-cloned synthetic audio and speech vocoder tampering
- Semantic mismatches and cross-modal contradictions (cheapfakes)

Existing detection systems predominantly evaluate content from a **single modality** (visual artifacts only or textual sentiment only). When visual manipulations are subtle or textual claims are syntactically well-formed, single-modality detectors suffer catastrophic degradation.

**TruthLens AI** solves this challenge by unifying:
1. **Natural Language Processing (DistilBERT)** for textual semantics
2. **Computer Vision (ResNet18)** for spatial manipulation detection
3. **Temporal Feature-Level Pooling** for multi-frame video deepfake evaluation
4. **Spectral Acoustic Analysis (128-D STFT)** for synthetic speech detection
5. **Bidirectional Cross-Attention** for deep cross-modal reasoning
6. **Probability Calibration & Guardrails** for reliable `REAL`, `FAKE`, and `UNCERTAIN` classification
7. **Transparent Explainability** via token gradient saliency, Layer4 Grad-CAM heatmaps, and acoustic diagnostics

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Inputs ["1. Multimodal Inputs (Simultaneous / Independent)"]
        T["Text Claim / Caption"]
        I["Image Source (PNG/JPG)"]
        V["Video Source (MP4)"]
        A["Audio Source (WAV/MP3)"]
    end

    subgraph Preprocessing ["2. Preprocessing Layer"]
        TP["Tokenizer (DistilBERT base uncased)"]
        IP["Image Normalization (224x224)"]
        VP["Uniform 8-Frame Linspace Sampler"]
        AP["16 kHz STFT Spectral Extractor (128-D)"]
    end

    subgraph Encoders ["3. Feature Extraction Backbones"]
        TE["Text Encoder -> 768-dim -> Linear 256-dim"]
        VE["Vision Encoder (ResNet18) -> 512-dim -> Linear 256-dim"]
        VFP["Video Feature-Level Temporal Pooling -> 256-dim"]
        AE["Audio Projection -> 256-dim"]
    end

    subgraph Fusion ["4. Multimodal Cross-Attention & Tri-Modal Fusion"]
        CA["Bidirectional Cross-Attention Layer\n(Text <-> Vision, Text <-> Audio)"]
        TMF["Tri-Modal Concatenation Projection\n(Text + Vision + Audio -> 256-dim)"]
    end

    subgraph Decision ["5. Calibrated Decision Policy (Threshold tau = 0.65)"]
        CL["MLP Classification Head -> Logits"]
        TS["Temperature Scaling (T = 1.00 Production)"]
        DP{"Operational Certainty Gate\n(Max Softmax Prob >= 0.65)"}
        REAL["REAL (Authentic)"]
        FAKE["FAKE (Manipulated)"]
        UNCERTAIN["UNCERTAIN (Routed to Human Review)"]
    end

    subgraph Explainability ["6. Explainability & Persistence"]
        TXE["Text Saliency (Gradient Attribution)"]
        VXE["Visual Saliency (Layer4 Grad-CAM)"]
        AXE["Acoustic Spectral Anomaly Metrics"]
        DB[(SQLite Persistent History\ntruthlens.db)]
        REP["Structured Verification Report"]
    end

    T --> TP --> TE --> CA
    I --> IP --> VE --> CA
    V --> VP --> VE --> VFP --> CA
    A --> AP --> AE --> CA
    CA --> CL --> TS --> DP
    DP -->|Prob >= 0.65 (Real)| REAL
    DP -->|Prob >= 0.65 (Fake)| FAKE
    DP -->|Prob < 0.65| UNCERTAIN
    CL -.-> TXE -.-> REP
    CL -.-> VXE -.-> REP
    CL -.-> AXE -.-> REP
    DP --> REP --> DB
```

---

## 3. Production Model Selection (Phase 7 Audit)

A multi-criteria empirical audit evaluated six candidate configurations across in-domain preservation, semantic mismatch robustness, out-of-domain Fakeddit generalization, generative audio/video detection, and calibration reliability:

- **Selected Production Model**: **Config A (Baseline Multimodal Cross-Attention)**
- **Source Checkpoint**: `models/multimodal_cross_attention.pt`
- **Production Alias**: `models/production/selected_model.pt` ($304.3\text{ MB}$)
- **Runtime Configuration**: `models/production/production_config.json`
- **Operational Parameters**: Temperature $T = 1.00$, Uncertainty Threshold $\tau = 0.65$
- **Composite Score**: **8.32 / 10.0**
- **Specialized Cross-Modal Variant**: **Config C** (Contrastive Alignment $\lambda=0.05$, achieving $100.0\%$ Condition 2 mismatch detection) is preserved and documented for targeted cross-modal deployment.

---

## 4. Key Capabilities & Deliverables

- **Simultaneous Tri-Modal Verification**: Reviewers can supply Text, an Image or Video, AND an Audio file simultaneously.
- **Evaluation Hub Dashboard**: Interactive 7-tab React dashboard (`/evaluation`) backed by `/api/evaluation/*` endpoints presenting live benchmark KPIs, in-domain results, Fakeddit generalization, acoustic/video suites, 4-condition semantic mismatch, calibration reliability diagrams, and the 6-model ablation matrix.
- **9 Curated Demonstration Benchmarks**: One-click sample loading for Easy REAL, Easy FAKE, Semantic Mismatch, Face Manipulation, Diffusion Synthesis, Text-Image Conflict, Video Deepfake, Audio Voice Clone, and UNCERTAIN Ambiguity.
- **First-Class UNCERTAIN Outcome**: Rather than forcing an unconfident binary guess, ambiguous inputs ($<65\%$ confidence) trigger an explicit amber guardrail banner explaining the uncertainty margin and routing to human reviewers.
- **Persistent SQLite History**: Full CRUD history interface (`/history`) with case search, modality filtering, and deletion.
- **Strict Academic Integrity**: Zero fabricated metrics. The KAIST FakeAVCeleb dataset is transparently disclosed as restricted/gated.

---

## 5. Repository Structure

```text
truthlens-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application (dual / and /api routing)
│   │   ├── config.py                   # App configuration & production model paths
│   │   ├── api/
│   │   │   ├── routes.py               # /verify/*, /verifications, /model/info
│   │   │   ├── evaluation.py           # /api/evaluation/* (7 evaluation endpoints)
│   │   │   └── health.py               # /health check endpoint
│   │   ├── database/                   # SQLite database & SQLAlchemy ORM
│   │   ├── schemas/                    # Pydantic validation schemas
│   │   └── services/                   # Verification & history service layers
│   └── requirements.txt
│
├── ml/
│   ├── data/samples/                   # Reference images, videos (.mp4), audios (.wav)
│   ├── preprocessing/                  # Text, Image, Video (8-frame), Audio (STFT)
│   ├── models/                         # DistilBERT, ResNet18, Cross-Attention, TriModal
│   ├── explainability/                 # Grad-CAM, Text Saliency, Audio Diagnostics
│   └── inference/                      # TruthLensPredictor (singleton inference engine)
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── VerificationPage.tsx    # Live verification with 9 demo presets & tri-modal
│   │   │   ├── EvaluationHubPage.tsx   # 7-tab interactive benchmark dashboard
│   │   │   ├── HistoryPage.tsx         # SQLite verification history browser
│   │   │   └── ProfessorDemoPage.tsx   # Curated professor demonstration suite
│   │   ├── components/                 # PredictionCard, EvidenceSection, Navbar
│   │   └── services/api.ts             # API client connecting to backend
│   └── vite.config.ts
│
├── models/
│   ├── production/
│   │   ├── selected_model.pt           # Production model alias (Config A, 304 MB)
│   │   └── production_config.json      # Production parameters and thresholds
│   ├── multimodal_cross_attention.pt   # Checkpoint A (304 MB)
│   └── phase6/                         # Checkpoints B, C, D, E, F
│
├── experiments/
│   ├── reports/                        # Phase 4/5 evaluation reports (JSON/CSV)
│   ├── phase6/results/                 # Phase 6 ablation and calibration data
│   └── phase7/                         # Model selection script and matrix
│
├── docs/                               # Comprehensive technical documentation
│   ├── PHASE7_FINAL_MODEL_SELECTION.md # Multi-criteria selection audit
│   ├── PHASE7_EVALUATION_HUB.md        # Evaluation Hub architecture guide
│   ├── PHASE7_DEMO_HARDENING.md        # Demo hardening & benchmark case guide
│   └── PHASE7_COMPLETION_REPORT.md     # Full Phase 7 report (Sections A–S)
│
└── tests/                              # Automated test suite (96 tests passing)
```

---

## 6. Quick Start Guide

### Step 1: Environment Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install backend dependencies (if needed)
pip install -r backend/requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..
```

### Step 2: Start the FastAPI Backend

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
- OpenAPI Documentation: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`
- Evaluation Overview: `http://localhost:8000/api/evaluation/overview`

### Step 3: Start the React Frontend

In a separate terminal window:
```bash
cd frontend
npm run dev
```
Open your browser to: `http://localhost:5173`

---

## 7. Public Dataset Pipeline & Reproducibility Guide

TruthLens AI integrates the peer-reviewed **r/Fakeddit Benchmark (LREC 2020)** for real-world multimodal misinformation evaluation. Follow these exact steps to reproduce the dataset verification, preprocessing, splitting, training, and evaluation pipeline from scratch:

### Step 1: Download the Dataset from the Official Source
The benchmark is curated from the official UCSB NLP Lab Fakeddit release. Download or fetch the metadata and verified media links:
- Official Portal: [https://fakeddit.netlify.app/](https://fakeddit.netlify.app/)
- Official Repository: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)

### Step 2: Place Raw Media in `data/raw/`
Place the downloaded raw images and metadata in:
```bash
data/raw/fakeddit/
```
*(In accordance with licensing and Git safety guidelines, raw image binaries are excluded from Git via `.gitignore` while keeping `data/raw/README.md`).*

### Step 3: Run Dataset Verification
Audit raw image integrity, header decodability, and MD5 cryptographic checksums:
```bash
python scripts/verify_dataset.py
```

### Step 4: Run Preprocessing
Standardize raw visual inputs into non-destructive 224x224 RGB tensors in `data/processed/` and normalize text claims:
```bash
python scripts/preprocess_dataset.py
```

### Step 5: Create Train / Validation / Test Splits
Generate deterministic, stratified 70/15/15 partitions with fixed random seed `42`:
```bash
python scripts/split_dataset.py
```
Outputs: `data/splits/train.csv`, `data/splits/validation.csv`, `data/splits/test.csv`.

### Step 6: Train the Model on the Public Benchmark
Fine-tune the Bidirectional Cross-Attention fusion layer using DistilBERT and ResNet-18:
```bash
python scripts/train_and_eval_public.py
```
Checkpoint saved to: `models/fakeddit_finetuned.pt`.

### Step 7: Evaluate the Model & Audit for Leakage
Run the automated class balance analysis and strict zero-leakage auditor:
```bash
python scripts/check_data_leakage.py
python scripts/analyze_dataset.py
```
Reports and figures generated in `reports/`:
- `reports/dataset_report.md`
- `reports/evaluation_report.md`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/metrics.json`

---

## 8. Running the Automated Test Suite

TruthLens AI includes **104 comprehensive automated tests** covering unit components, neural forward passes, gradient backpropagation, video temporal pooling, audio STFT projection, database persistence, API endpoints, and the public dataset pipeline:

```bash
.venv/bin/pytest tests/ -v
```

**Result**:
```text
============================= 104 passed in 16.07s =============================
```

To verify the frontend TypeScript and build:
```bash
npm --prefix frontend run build
```

**Result**:
```text
✓ built in 860ms (0 errors, 0 warnings)
```

---

## 9. Academic Citations & Dataset Transparency

### Official Dataset Citation (r/Fakeddit)
```bibtex
@inproceedings{nakamura2020fakeddit,
  title     = {r/{F}akeddit: A New Multimodal Benchmark Dataset for Fine-Grained Fake News Detection},
  author    = {Nakamura, Kai and Levy, Sharon and Wang, William Yang},
  booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
  month     = may,
  year      = {2020},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {6149--6157},
  url       = {https://aclanthology.org/2020.lrec-1.755}
}
```

- **Zero Fabricated Results**: Every displayed accuracy, F1 score, ROC-AUC, ECE, and confusion matrix derives strictly from reproducible experimental evaluations on authentic stored artifacts.
- **FakeAVCeleb Dataset Disclosure**: The KAIST FakeAVCeleb dataset is explicitly identified across the UI and documentation as **"Not Available / Restricted Gated Dataset"**, requiring formal institutional approval. Surrogate evaluations are conducted using verified synthetic acoustic and video benchmarks.
- **Uncertainty Policy**: The system prioritizes human editorial review over unconfident algorithmic predictions, enforcing an operational 65% certainty threshold.
