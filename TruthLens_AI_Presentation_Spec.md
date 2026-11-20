# TRUTHLENS AI — 10-SLIDE ACADEMIC RESEARCH PRESENTATION SPECIFICATION
**Course:** CSET 346 — Natural Language Processing  
**Institution:** Bennett University, Greater Noida, India  
**Project:** TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Authors:**
- Divakar Maurya — S24CSEU0807 (s24cseu0807@bennett.edu.in)
- Vrishank Pathak — S24CSEU0950 (s24cseu0950@bennett.edu.in)
- Hrishit Goel — S24CSEU2861 (s24cseu2861@bennett.edu.in)
- Dhruv Agarwal — S24CSEU0916 (s24cseu0916@bennett.edu.in)
- Vishwas Bhargava — S24CSEU0190 (s24cseu0190@bennett.edu.in)

---

## GLOBAL DESIGN SPECIFICATIONS & PALETTE
- **Theme & Atmosphere:** Deep navy research laboratory aesthetic.
- **Background Color:** Radial gradient from `#070D1E` (edges) to `#0F1D38` (center).
- **Typography:** Crisp white headers and body text (`#FFFFFF`); secondary labels in slate gray (`#94A3B8`).
- **Card Styling:** Glassmorphic dark blue cards (`#132247`) with 1px border (`#1E3A8A`).
- **Primary Accent / Glow:** Electric Cyan (`#00D2FF`).
- **Secondary Accent:** Cobalt Blue (`#3A86FF`).
- **Status Badges:** Emerald Green (`#10B981`) for Real/Success; Crimson Red (`#EF4444`) for Fake/Defects; Amber (`#F59E0B`) for Uncertain/Warnings.
- **Strict Visual Directive:** Native layout structures (flowcharts, 2x2 grids, comparison tables, process chevrons, metric cards). No generic stock clipart.

---

## SLIDE 1: TITLE & TEAM IDENTITY
- **Visual Structure:** Centered Hero Title + 2-Column Balanced Author Card Grid + Bottom 4-Stream Convergence Flow.
- **Header Badge (Pill with cyan border):** `MILESTONE 1 · CSET 346 · NATURAL LANGUAGE PROCESSING`
- **Main Title (Bold white, cyan glow):** **TruthLens AI**
- **Subtitle (Cyan #00D2FF):** Multimodal NLP for Deepfake Detection and Explainable Content Verification
- **Card 1 (Authors - Left Column):**
  * Divakar Maurya — S24CSEU0807 (`s24cseu0807@bennett.edu.in`)
  * Vrishank Pathak — S24CSEU0950 (`s24cseu0950@bennett.edu.in`)
  * Hrishit Goel — S24CSEU2861 (`s24cseu2861@bennett.edu.in`)
- **Card 2 (Authors & Affiliation - Right Column):**
  * Dhruv Agarwal — S24CSEU0916 (`s24cseu0916@bennett.edu.in`)
  * Vishwas Bhargava — S24CSEU0190 (`s24cseu0190@bennett.edu.in`)
  * Department of Computer Science and Engineering, Bennett University, Greater Noida, India
- **Native Flowchart Component (Bottom):**
  `[Text Stream (DistilBERT)]` + `[Image Stream (ResNet18)]` + `[Video Stream (8-Frame Pool)]` + `[Audio Stream (STFT)]` $\rightarrow$ `[TruthLens Multimodal Core]` $\rightarrow$ `[Calibrated Tri-State: REAL · FAKE · UNCERTAIN (τ = 0.65)]`.

---

## SLIDE 2: THE PROBLEM & LITERATURE GAP: DEEPFAKES ARE MULTIMODAL
- **Visual Structure:** 2-Column Asymmetric Split (40% Left / 60% Right) + Full-Width Bottom Callout Ribbon.
- **Left Column (3 Problem Cards with Amber/Red Accents):**
  * **The Cheapfake Paradox:** Real, unmanipulated photos paired with fabricated claims bypass computer vision filters completely.
  * **Visual Hegemony:** High-capacity vision backbones overpower text signals, misclassifying composite fakes as authentic.
  * **Overconfident Black Boxes:** Classifiers output >0.99 probability without admitted uncertainty or explainable causal evidence.
- **Right Column (Styled Native Table - Dark Fill #132247, Header #1E3A8A):**
  | Work & Reference | Modality | Core Mechanism | Critical Limitation |
  | :--- | :--- | :--- | :--- |
  | Rössler et al. (FaceForensics++) | Vision | CNN pixel artifacts | Blind to unmanipulated image reuse |
  | Guera & Delp (AVSS 2018) | Video | Frame temporal feature CNN | Lacked multimodal text context |
  | Aneja et al. (COSMOS 2021) | Text + Image | Contrastive cheapfake match | High false alarms; uncalibrated |
  | Nakamura et al. (Fakeddit 2020) | Text + Image | Concatenation / Late fusion | Severe collapse on informal slang |
  | Li et al. (IEEE TASLP 2021) | Audio | STFT spectral vocoder cues | Unimodal speech without video sync |
  | **TruthLens AI (This Work)** | **T + I + V + A** | **Bidirectional Cross-Attn** | **Unifies 4 streams, XAI, τ = 0.65** |
- **Bottom Ribbon:**
  * **Research Gap:** Current systems operate in unimodal silos and force speculative binary labels.
  * **TruthLens Solution:** Joint cross-modal reasoning + calibrated uncertainty ($\tau = 0.65$) + causal explainability.

---

## SLIDE 3: AUDITED DATASETS & MULTI-SUITE SETUP
- **Visual Structure:** 3-Column Metric Dashboard Grid (33% / 33% / 33%).
- **Column 1 Card (Primary Course Dataset - N = 320 Balanced):**
  * Visual Diagram: 3-part split meter showing 70% Train (Cyan `#00D2FF`), 15% Val (Blue `#3A86FF`), 15% Test (Emerald `#10B981`). Zero visual/textual leakage.
  * 4 Typologies (40 pairs each): Facial Deepfakes, Latent Diffusion, Spliced Headlines, Cloned Speech.
- **Column 2 Card (External Generalization Suites - N = 180):**
  * 3 stacked pill badges with icons:
    - `[Reddit Icon]` Fakeddit Benchmark ($N = 60$): 30 Real / 30 Fake uncurated social media pairs.
    - `[Audio Icon]` Deepfake Audio v4 ($N = 60$): 30 Human Speech / 30 ElevenLabs & Hume AI clones.
    - `[Video Icon]` Generative Video Sora ($N = 60$): 30 Documentary / 30 OpenAI Sora & Google Veo clips.
  * Transparency Disclosure: FakeAVCeleb is institutional gated data; surrogated via verified Audio v4 & Video Sora suites.
- **Column 3 Card (Controlled Semantic Mismatch Suite - 30 Pairs / 60 Evals):**
  * 4 controlled conditions:
    - Condition 1: Authentic Baseline (Real Text + Real Matching Image)
    - Condition 2: The Cheapfake Conflict (Fabricated Headline + Real Image - Highlighted in Amber `#F59E0B`)
    - Condition 3: Contextual Swap (Real Headline + Synthetic Spliced Media)
    - Condition 4: Cross-Modal Synthetic (Fabricated Claim + Generative Media)

---

## SLIDE 4: OVERALL SYSTEM ARCHITECTURE
- **Visual Structure:** Full-Width Horizontal Pipeline Flowchart (Left-to-Right) with 4 sequential stages.
- **Stage 1 (Inputs):** Text Claim $T$ · Static Image $I$ · Video Stream $V$ (8 Frames) · Audio Waveform $A$ (16 kHz Mono).
- **Stage 2 (Feature Encoders & Projectors):**
  * Text: DistilBERT $\rightarrow [CLS]$ Token $h_{[CLS]} \in \mathbb{R}^{768} \rightarrow$ Linear + GELU $\rightarrow z_T \in \mathbb{R}^{256}$.
  * Image: ResNet18 $\rightarrow$ Spatial $H_V \in \mathbb{R}^{512 \times 7 \times 7} \rightarrow$ AvgPool $\rightarrow z_I \in \mathbb{R}^{256}$.
  * Video: 8 Uniform Frames $\rightarrow$ ResNet18 Features $f_k \rightarrow$ Latent Temporal Mean Pool $\rightarrow z_V \in \mathbb{R}^{256}$.
  * Audio: STFT Spectral Vector in $\mathbb{R}^{128}$ (Mel Bins + ZCR + HF Energy) $\rightarrow$ Linear $\rightarrow z_A \in \mathbb{R}^{256}$.
- **Stage 3 (Fusion & Alignment):**
  * Bidirectional Cross-Attention ($h = 4$ heads, $d_k = 64$): Text $\leftrightarrow$ Vision mutual attention.
  * InfoNCE Contrastive Loss ($\lambda = 0.05$): Aligns matching pairs, penalizes semantic mismatches.
  * Tri-Modal Concatenation Projection: Maps combined representations $[B, 768] \rightarrow [B, 256]$.
- **Stage 4 (Calibrated Verdict & Explainability):**
  * Temperature Scaling ($T^* = 1.4628$) $\rightarrow$ Uncertainty Guardrail ($\tau = 0.65$) $\rightarrow$ REAL / FAKE / UNCERTAIN.
  * Causal Evidence: ResNet Layer4 Grad-CAM Heatmaps + WordPiece Token Saliency + SQLite Logging.

---

## SLIDE 5: MULTIMODAL METHODOLOGY & TEMPORAL VIDEO FIX
- **Visual Structure:** 2x2 Feature Card Grid.
- **Top-Left (Card 1: Text DistilBERT):**
  * WordPiece Tokenizer ($L = 128$) $\rightarrow$ Pretrained DistilBERT sequence $H_T \in \mathbb{R}^{128 \times 768}$.
  * Shared Latent Projection: $z_T = \text{LayerNorm}(W_T h_{[CLS]} + b_T) \in \mathbb{R}^{256}$.
  * Elastic Regularization: $\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{CE}}(T_{\text{fakeddit}}) + 0.5 \cdot \mathcal{L}_{\text{CE}}(T_{\text{course}})$ (Prevents catastrophic forgetting).
- **Top-Right (Card 2: Image ResNet18):**
  * $224 \times 224$ RGB image normalized to ImageNet statistics $\rightarrow$ ResNet18 convolutional layers.
  * Projection: $z_I = \text{LayerNorm}(W_I \text{AvgPool}(H_V) + b_I) \in \mathbb{R}^{256}$.
- **Bottom-Left (Card 3: Video Temporal Pooling — Architectural Fix):**
  * Highlight Badge: `CRITICAL DEFECT RESOLVED (Phase 2)` in Amber (`#F59E0B`).
  * *Before (Defective):* Video Frames $\rightarrow$ Pixel Mean (Severe Motion Blur) $\rightarrow$ CNN.
  * *After (Fixed):* Video Frames $\rightarrow$ Independent ResNet Passes $\rightarrow$ Latent Feature Mean Pool $\rightarrow z_V \in \mathbb{R}^{256}$.
- **Bottom-Right (Card 4: Audio STFT Projections — Phase 3):**
  * 16 kHz mono waveform $\rightarrow$ 128-D acoustic vector (STFT log-mel bins + ZCR + HF Ratio $> 4\text{ kHz}$).
  * Trainable linear projection: $z_A = \text{LayerNorm}(W_A a + b_A) \in \mathbb{R}^{256}$ with verified gradient backpropagation.

---

## SLIDE 6: FROM PREDICTION TO EVIDENCE: ATTENTION & XAI
- **Visual Structure:** 2-Column Split Screen (50% Left / 50% Right).
- **Left Column (Bidirectional Cross-Attention Mechanics):**
  * Bipartite network diagram showing Text Word Tokens connecting via glowing bidirectional cyan vectors to Visual Feature Patches.
  * Multi-Head Cross-Attention (4 heads, $d_k = 64$):
    - $\text{Attn}(Q_T, K_V, V_V) = \text{softmax}\left(\frac{Q_T K_V^T}{\sqrt{d_k}}\right) V_V$
    - $\text{Attn}(Q_V, K_T, V_T) = \text{softmax}\left(\frac{Q_V K_T^T}{\sqrt{d_k}}\right) V_T$
  * Contrastive InfoNCE Alignment ($\lambda = 0.05$): Pulls matching pairs together, pushes mismatched pairs apart.
- **Right Column (Multimodal Causal Explainability Engine):**
  * **Evidence Card 1 (Text Saliency):** News headline highlighting manipulation triggers: *"Terrorist explosion [DESTROYED] central [HOSPITAL]"* ($S(w_i) = \|\partial y_c / \partial e_i\|_1$).
  * **Evidence Card 2 (Visual Grad-CAM):** Localized heatmap overlay pinpointing synthetic seams on face boundaries ($\alpha_k^c$ Layer4 gradient pooling).
  * **Evidence Card 3 (Acoustic Diagnostics):** STFT zero-crossing rates and high-frequency spectral ratios flagging vocoder artifacts.

---

## SLIDE 7: SOFTWARE ARCHITECTURE & PRODUCTION TECH STACK
- **Visual Structure:** Layered Vertical Stack Diagram (Left 65%) + Tech Badges & Health Card (Right 35%).
- **Left Column (4-Tier Vertical Layered Architecture):**
  * Tier 1 (Client): React 18, TypeScript, Vite, Tailwind CSS. Features Verification Workspace, Evaluation Hub, History Viewer.
  * Tier 2 (API Gateway): FastAPI Asynchronous Backend, Uvicorn, RESTful routes (`/api/verify`, `/api/evaluation/*`, `/api/history`).
  * Tier 3 (ML Inference): PyTorch 2.x, Hugging Face Transformers (DistilBERT), Torchvision (ResNet18), Librosa, OpenCV.
  * Tier 4 (Persistence): SQLite database (`truthlens.db`) storing full multi-modal payloads, confidence margins, and XAI maps.
- **Right Column (Tech Badges & Verification Status):**
  * Tech Grid Badges: Python 3.13 · PyTorch 2.x · FastAPI · React 18 · TypeScript · Vite · Tailwind · SQLite 3 · OpenCV · Librosa.
  * Large Status Card (Emerald Green `#10B981`): **96 / 96 Automated Tests Passing** (Unit, Integration, API, Regressions). 100% Test Health · Zero TypeScript Build Errors.

---

## SLIDE 8: EXPERIMENTAL RESULTS & BASELINES
- **Visual Structure:** Styled Data Table (Top 60%) + 3 Metric Callout Cards (Bottom 40%).
- **Top Table (Native Table with highlighted TruthLens row):**
  | Model Architecture | Modalities | In-Domain Acc. | Fakeddit Acc. | Audio ROC-AUC | Video ROC-AUC | Mean ECE |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | DistilBERT Baseline | Text Only | 100.0% | 33.33% | — | — | 0.2840 |
  | ResNet18 Baseline | Vision Only | 100.0% | 50.00% | — | — | 0.2210 |
  | Acoustic Spectral Baseline | Audio Only | 100.0% | — | 0.8100 | — | 0.1840 |
  | Video Temporal Pooling | Video Only | 100.0% | — | — | 0.8200 | 0.1790 |
  | Concatenation Fusion | Text + Vision | 100.0% | 43.33% | — | — | 0.1870 |
  | **TruthLens Cross-Attention** | **T + V + A** | **100.0%** | **46.67%** | **0.8200** | **0.8200** | **0.1324** |
- **Bottom Callout Cards:**
  * **Card 1 (In-Domain Saturation - 100.0%):** In-domain laboratory metrics alone cannot verify real-world robustness.
  * **Card 2 (Unimodal Collapse - 33.33%):** Text-only drops drastically on Reddit slang, proving language models cannot stand alone.
  * **Card 3 (Calibration Superiority - 0.1324 ECE):** Cross-attention achieves the lowest Expected Calibration Error across all baselines.

---

## SLIDE 9: SEMANTIC MISMATCH, CALIBRATION & ABLATION MATRIX
- **Visual Structure:** 2-Column Comparative Split (45% Left / 55% Right).
- **Left Column (Visual Dominance Trap & Calibration):**
  * *Condition 2 Cheapfake:* Authentic unmanipulated photo + Fabricated headline ("Explosion levels hospital").
  * *Baseline Failure:* Config A achieves 0.0% Accuracy on Condition 2 (model assumes authentic image implies authentic claim).
  * *Visual Bar Metric:* Condition 2 Accuracy: Baseline 0.0% [Red Bar] $\rightarrow$ Contrastive Alignment 100.0% [Green Bar] (+100.0% Gain!).
  * *Probability Calibration:* Temperature scaling ($T^* = 1.4628$) reduces ECE from 0.3406 to 0.2701, routing 38.3% of borderline cases to UNCERTAIN.
- **Right Column (Six-Configuration Ablation Matrix - N = 276):**
  | Configuration | In-Domain | Mismatch | Fakeddit | Audio/Video | Mean ECE | Composite Score |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Config A (Production Baseline)** | **100.0%** | **55.0%** | **46.7%** | **0.8200** | **0.1324** | **8.32 / 10.0 (WINNER)** |
  | Config D (Calibrated Baseline) | 100.0% | 55.0% | 46.7% | 0.8200 | 0.1338 | 8.31 / 10.0 |
  | Config C (Contrastive Aligned) | 100.0% | 60.0% | 46.7% | 0.8300 | 0.1571 | 8.26 / 10.0 (Cheapfake Def.) |
  | Config B (Domain-Adapted Text) | 100.0% | 73.3% | 56.7% | 0.8100 | 0.2453 | 8.01 / 10.0 (Calib. Drift) |
  | Config F (Full Proposed Pipe) | 100.0% | 48.3% | 46.7% | 0.8100 | 0.1882 | 7.82 / 10.0 (Negative Transfer) |
  | Config E (Adapted + Contrast) | 100.0% | 48.3% | 46.7% | 0.8100 | 0.2104 | 7.74 / 10.0 (Negative Transfer) |
  * Bottom Alert (Crimson `#EF4444`): Stacking all modifications blindly (Config F) causes negative transfer drift (48.3% mismatch), confirming Config A as superior.

---

## SLIDE 10: END-TO-END DEMO, LIMITATIONS & FUTURE ROADMAP
- **Visual Structure:** 3-Column Structured Dashboard + Full-Width Bottom Quote Banner.
- **Column 1 (Verification Workflow - 5 Step Flowchart):**
  1. Ingest Media $\rightarrow$ Text claim + Image, Video, or Audio file.
  2. Deep Inference $\rightarrow$ DistilBERT, ResNet18, 8-frame pooling, STFT.
  3. Calibrated Verdict $\rightarrow$ Applies $\tau = 0.65$ threshold (REAL / FAKE / UNCERTAIN).
  4. Explainability $\rightarrow$ Generates Grad-CAM heatmaps & token saliency highlights.
  5. SQLite Logging $\rightarrow$ Full audit payload persisted to database.
  *Mini Result Preview Badge:* `Verdict: FAKE (Confidence: 87.4% · Conflict: 0.82)`.
- **Column 2 (Honest Research Limitations):**
  * Dataset Scale: Primary corpus limited to 320 samples (requires expansion to tens of thousands).
  * Video/Audio Labels: Primary training set contains no labeled video/audio records (relies on zero-shot projection).
  * Calibration Drift: Rehearsal fine-tuning on social media language causes minor calibration drift (ECE = 0.2453).
- **Column 3 (Future Research Roadmap):**
  * **RAG Fact-Checking:** Connect text encoder to live Wikipedia and Google Fact Check APIs.
  * **Spatial Frequency Transforms:** Incorporate DCT/FFT to detect invisible diffusion pixel lattice grids.
  * **Multilingual Cross-Attention:** Expand beyond English WordPiece to low-resource regional dialects.
- **Full-Width Bottom Banner (Glowing Cyan Border #00D2FF):**
  *“Deepfake verification is not an isolated computer vision problem; it is a cross-modal consistency and uncertainty reasoning challenge.”*
