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

## 7. Running the Automated Test Suite

TruthLens AI includes 96 comprehensive automated tests covering unit components, neural forward passes, gradient backpropagation, video temporal pooling, audio STFT projection, database persistence, API endpoints, and Phase 4–7 evaluations:

```bash
.venv/bin/pytest tests/ -v
```

**Result**:
```text
============================= 96 passed in 16.55s ==============================
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

## 8. Academic Integrity & Dataset Transparency

- **Zero Fabricated Results**: Every displayed accuracy, F1 score, ROC-AUC, ECE, and confusion matrix derives strictly from reproducible experimental evaluations.
- **FakeAVCeleb Dataset Disclosure**: The KAIST FakeAVCeleb dataset is explicitly identified across the UI and documentation as **"Not Available / Restricted Gated Dataset"**, requiring formal institutional approval. Surrogate evaluations are conducted using verified synthetic acoustic and video benchmarks.
- **Uncertainty Policy**: The system prioritizes human editorial review over unconfident algorithmic predictions, enforcing an operational 65% certainty threshold.
