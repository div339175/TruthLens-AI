# TRUTHLENS AI — TEAM ROLE DIVISION & CHATGPT PROMPTS
**Course:** CSET 346 — Natural Language Processing  
**Institution:** Bennett University, Greater Noida, India  
**Project:** TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  

---

## SERIAL SLIDE DISTRIBUTION ACROSS THE 5 TEAM MEMBERS

| Member | Name & Enrollment | Assigned Slides | Primary Focus & Domain |
| :--- | :--- | :--- | :--- |
| **Member 1** | **Divakar Maurya** (S24CSEU0807) | **Slide 1 & Slide 2** | Project Lead, Problem Framing, Multimodal Cheapfakes & Literature Survey |
| **Member 2** | **Vrishank Pathak** (S24CSEU0950) | **Slide 3 & Slide 4** | Dataset Auditing, Multi-Suite Setup & End-to-End System Pipeline Flow |
| **Member 3** | **Hrishit Goel** (S24CSEU2861) | **Slide 5 & Slide 6** | Multimodal Encoders, Video Temporal Pooling Fix, Cross-Attention & XAI |
| **Member 4** | **Dhruv Agarwal** (S24CSEU0916) | **Slide 7 & Slide 8** | Full-Stack Engineering, 4-Tier Architecture & Empirical Baseline Benchmarking |
| **Member 5** | **Vishwas Bhargava** (S24CSEU0190) | **Slide 9 & Slide 10** | Visual Dominance Trap, Temperature Calibration, Ablation Matrix & Future Roadmap |

---

## MEMBER 1 PROMPT (Divakar Maurya — S24CSEU0807)
*Copy and paste the entire block below into ChatGPT:*

```markdown
You are an expert AI/NLP research mentor and oral defense coach. I am Divakar Maurya (Enrollment: S24CSEU0807), a student at Bennett University in the CSET 346 (Natural Language Processing) course. 

Our team project is "TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification". 

My assigned role in the team is:
- Role: Team Lead, Problem Formulation & Literature Survey Specialist
- Assigned Presentation Slides: SLIDE 1 and SLIDE 2 (Serial Wise)

Please act as my personal project mentor and thoroughly explain:
1. WHAT IS MY ROLE & EXACT CONTRIBUTION:
   - How I led the conceptual framing of multimodal manipulation.
   - How I identified the "Cheapfake Paradox" (authentic unmanipulated images paired with fabricated, emotionally charged text claims) that completely bypass computer vision detectors.
   - How I identified "Visual Hegemony" (high-capacity vision backbones overpowering weaker linguistic signals during late fusion).
   - How I framed the "Overconfident Black Box" problem where deep learning classifiers output >0.99 confidence without admitting epistemic uncertainty.
   - My comparative analysis of prior literature (Rössler FaceForensics++, Guera & Delp AVSS 2018, Aneja COSMOS 2021, Nakamura Fakeddit 2020, Li TASLP 2021) and how TruthLens AI addresses their critical limitations.

2. HOW TRUTHLENS AI WORKS AT A HIGH LEVEL:
   - Provide an end-to-end walkthrough of the system: how 4 modalities (Text, Image, Video, Audio) are ingested, encoded into a shared 256-dimensional latent space, aligned via bidirectional cross-attention, calibrated with temperature scaling (T* = 1.4628), and routed through an uncertainty threshold (tau = 0.65) to produce REAL / FAKE / UNCERTAIN verdicts with Grad-CAM and token saliency explainability.

3. MY PPT PRESENTATION SCRIPT (SLIDES 1 & 2):
   - Give me an exact, highly professional oral script for Slide 1 (Title, Team Identity, Milestone 1, and the bottom 4-stream convergence flow diagram).
   - Give me an exact oral script for Slide 2 (The Problem & Literature Gap, 3 problem cards, literature comparison table, and bottom research gap ribbon).
   - Provide a smooth verbal transition handing over the presentation to Member 2 (Vrishank Pathak) for Slide 3.

4. ANTICIPATED VIVA / DEFENSE QUESTIONS & ANSWERS:
   - Give me the top 5 challenging questions the professor might ask me about problem framing, cheapfakes, literature survey, and why unimodal detection fails, along with crisp, technically rigorous answers.
```

---

## MEMBER 2 PROMPT (Vrishank Pathak — S24CSEU0950)
*Copy and paste the entire block below into ChatGPT:*

```markdown
You are an expert AI/NLP research mentor and oral defense coach. I am Vrishank Pathak (Enrollment: S24CSEU0950), a student at Bennett University in the CSET 346 (Natural Language Processing) course.

Our team project is "TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification".

My assigned role in the team is:
- Role: Data Engineering Lead, Multi-Suite Benchmark Auditor & System Pipeline Architect
- Assigned Presentation Slides: SLIDE 3 and SLIDE 4 (Serial Wise)

Please act as my personal project mentor and thoroughly explain:
1. WHAT IS MY ROLE & EXACT CONTRIBUTION:
   - How I designed, curated, and partitioned the datasets used in TruthLens AI.
   - Primary Course Dataset (N = 320 balanced samples): 70% Train (224), 15% Val (48), 15% Test (48) with zero visual/textual leakage across splits, covering 4 forensic typologies (Facial Deepfakes, Latent Diffusion, Spliced Headlines, Cloned Speech - 40 pairs each).
   - External Generalization Suites (N = 180 out-of-domain instances): Fakeddit social media benchmark (60), Deepfake Audio v4 (60), and Generative Video Sora/Veo (60).
   - Institutional transparency regarding FakeAVCeleb: explaining why it was surrogated with verified Audio v4 and Sora video suites due to institutional access gating.
   - Controlled Semantic Mismatch Suite: 30 pairs / 60 evaluations isolating cross-modal conflict across 4 conditions (Condition 1: Authentic, Condition 2: Cheapfake conflict, Condition 3: Contextual swap, Condition 4: Cross-modal synthetic).
   - Architectural flow design: How I laid out the 4-stage pipeline (Stage 1 Inputs -> Stage 2 Feature Encoders -> Stage 3 Fusion & Alignment -> Stage 4 Calibrated Verdict & XAI).

2. HOW TRUTHLENS AI WORKS AT A SYSTEM LEVEL:
   - Explain how raw text claims, 224x224 RGB images, 8-frame video sequences, and 16 kHz audio waveforms flow into independent feature projectors, map to R^256, fuse via cross-attention, and get persisted to SQLite with explainability maps.

3. MY PPT PRESENTATION SCRIPT (SLIDES 3 & 4):
   - Give me a smooth opening acknowledging Member 1's handover.
   - Give me an exact, professional oral script for Slide 3 (Audited Datasets, 3-part split meter, 4 typologies, external suites, and semantic mismatch suite).
   - Give me an exact oral script for Slide 4 (End-to-End System Architecture, explaining Stage 1 Inputs, Stage 2 Encoders, Stage 3 Fusion, and Stage 4 Verdict/XAI).
   - Provide a smooth verbal transition handing over the presentation to Member 3 (Hrishit Goel) for Slide 5.

4. ANTICIPATED VIVA / DEFENSE QUESTIONS & ANSWERS:
   - Give me the top 5 challenging questions the professor might ask me about dataset leakage, sample size constraints, why Fakeddit is out-of-domain, how the 4-condition suite isolates visual dominance, and pipeline modularity, along with crisp, mathematically grounded answers.
```

---

## MEMBER 3 PROMPT (Hrishit Goel — S24CSEU2861)
*Copy and paste the entire block below into ChatGPT:*

```markdown
You are an expert AI/NLP research mentor and oral defense coach. I am Hrishit Goel (Enrollment: S24CSEU2861), a student at Bennett University in the CSET 346 (Natural Language Processing) course.

Our team project is "TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification".

My assigned role in the team is:
- Role: Multimodal Feature Engineer, Cross-Attention Specialist & Explainable AI (XAI) Lead
- Assigned Presentation Slides: SLIDE 5 and SLIDE 6 (Serial Wise)

Please act as my personal project mentor and thoroughly explain:
1. WHAT IS MY ROLE & EXACT CONTRIBUTION:
   - Modality Encoders Implementation:
     * Text: DistilBERT with WordPiece (L=128), [CLS] token pooling into R^768, projected to z_T in R^256. Rehearsal fine-tuning loss L_adapt = L_CE(fakeddit) + 0.5 * L_CE(course) preventing catastrophic forgetting (+23.34% social media gain).
     * Image: ResNet18 Layer4 spatial tensor H_V in R^{512 x 7 x 7}, global average pooled and projected to z_I in R^256.
     * Video Temporal Pooling (CRITICAL PHASE 2 FIX): Identifying and resolving the fatal flaw where early prototypes averaged raw frame pixels before CNN (causing severe motion blur). How I replaced it with 8 independent ResNet frame forward passes followed by latent feature mean pooling (f_video = 1/K sum f_k), achieving Video ROC-AUC = 0.8200.
     * Audio Branch: 16 kHz mono waveform converted via STFT into a 128-D acoustic vector (log-mel bands, Zero Crossing Rate, High-Frequency ratio >4 kHz) with verified PyTorch gradient backpropagation.
   - Bidirectional Cross-Attention Mechanics:
     * Multi-Head Cross-Attention (4 heads, d_k=64) allowing text to query vision and vision to query text simultaneously:
       Attn(Q_T, K_V, V_V) = softmax((Q_T K_V^T) / sqrt(d_k)) V_V
     * InfoNCE Contrastive Loss (lambda = 0.05): Pulling matching pairs together and repelling mismatched cheapfakes.
   - Multimodal Causal Explainability Engine:
     * WordPiece Token Gradient Saliency: S(w_i) = || partial y_c / partial e_i ||_1.
     * Visual Layer4 Grad-CAM: alpha_k^c gradient pooling overlaid on facial boundaries.
     * Acoustic Spectral Diagnostics: Zero-crossing and high-frequency vocoder flags.

2. HOW TRUTHLENS AI WORKS MATHEMATICALLY:
   - Explain the tensor shapes at every step: [B, 128] -> [B, 128, 768] -> [B, 256], [B, 8, 3, 224, 224] -> [B, 8, 512] -> [B, 256], cross-attention residuals, and contrastive alignment.

3. MY PPT PRESENTATION SCRIPT (SLIDES 5 & 6):
   - Give me a smooth opening acknowledging Member 2's handover.
   - Give me an exact, technically deep oral script for Slide 5 (Multimodal Methodology, highlighting the Phase 2 Video Temporal Pooling Fix before vs after, DistilBERT adaptation, ResNet18, and Audio STFT).
   - Give me an exact oral script for Slide 6 (Cross-Attention Bipartite Network, mathematical formulas, InfoNCE contrastive alignment, and the 3 XAI evidence cards).
   - Provide a smooth verbal transition handing over the presentation to Member 4 (Dhruv Agarwal) for Slide 7.

4. ANTICIPATED VIVA / DEFENSE QUESTIONS & ANSWERS:
   - Give me the top 5 technical questions the professor might ask about cross-attention vs self-attention, why pixel averaging causes motion blur, why InfoNCE uses temperature tau, how Grad-CAM is computed on Layer4, and how gradient saliency avoids saturation, with rigorous mathematical answers.
```

---

## MEMBER 4 PROMPT (Dhruv Agarwal — S24CSEU0916)
*Copy and paste the entire block below into ChatGPT:*

```markdown
You are an expert AI/NLP research mentor and oral defense coach. I am Dhruv Agarwal (Enrollment: S24CSEU0916), a student at Bennett University in the CSET 346 (Natural Language Processing) course.

Our team project is "TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification".

My assigned role in the team is:
- Role: Full-Stack Systems Engineer, Production Infrastructure & Empirical Baseline Lead
- Assigned Presentation Slides: SLIDE 7 and SLIDE 8 (Serial Wise)

Please act as my personal project mentor and thoroughly explain:
1. WHAT IS MY ROLE & EXACT CONTRIBUTION:
   - Four-Tier Production Software Architecture:
     * Tier 1 (Client): React 18, TypeScript, Vite, Tailwind CSS with responsive Verification Workspace, Interactive Evaluation Hub, and History Viewer.
     * Tier 2 (API Gateway): FastAPI asynchronous backend with Uvicorn, Pydantic schema validation, handling multipart media uploads on /api/verify, /api/evaluation/metrics, /api/history.
     * Tier 3 (ML Inference Runtime): PyTorch 2.x, Hugging Face Transformers, Torchvision, Librosa, OpenCV integration with unified batch inference.
     * Tier 4 (Persistence & Audit): SQLite 3 (truthlens.db) storing full multimodal payloads, confidence scores, and serialized Grad-CAM heatmaps for zero-loss compliance.
   - Rigorous Engineering Quality Assurance:
     * 96 / 96 Automated Tests Passing (Unit, Integration, API, Persistence, Audio/Video Regressions).
     * 100% Test Suite Health and Zero TypeScript build errors.
   - Empirical Baseline Evaluation & Benchmarking:
     * Designed and ran systematic evaluations comparing 5 unimodal/simple baselines against TruthLens Cross-Attention:
       DistilBERT (Text), ResNet18 (Vision), Acoustic Spectral (Audio), Video Temporal Pooling (Video), Concatenation Fusion, and TruthLens Cross-Attention.
     * Key Empirical Findings: In-domain saturation (all hit 100% in lab), Unimodal text collapse (DistilBERT drops to 33.33% on Fakeddit), and Calibration superiority (Cross-attention hits lowest ECE of 0.1324 vs 0.2840 for text).

2. HOW THE SOFTWARE & DATA FLOW WORK IN PRODUCTION:
   - Detail the end-to-end request-response lifecycle from file upload in React UI to FastAPI background processing, PyTorch tensor execution, database write, and reactive UI rendering.

3. MY PPT PRESENTATION SCRIPT (SLIDES 7 & 8):
   - Give me a smooth opening acknowledging Member 3's handover.
   - Give me an exact, production-focused oral script for Slide 7 (Software Architecture 4 tiers, React/FastAPI/PyTorch/SQLite stack, 96/96 automated tests health card).
   - Give me an exact oral script for Slide 8 (Experimental Results Table, In-Domain Saturation card, Unimodal Collapse card, and Calibration Superiority card).
   - Provide a smooth verbal transition handing over the presentation to Member 5 (Vishwas Bhargava) for Slide 9.

4. ANTICIPATED VIVA / DEFENSE QUESTIONS & ANSWERS:
   - Give me the top 5 questions the professor might ask about software engineering trade-offs, FastAPI async vs Celery, SQLite scalability vs PostgreSQL, test coverage strategy, and why unimodal text collapses on informal Reddit slang, with confident, industry-standard answers.
```

---

## MEMBER 5 PROMPT (Vishwas Bhargava — S24CSEU0190)
*Copy and paste the entire block below into ChatGPT:*

```markdown
You are an expert AI/NLP research mentor and oral defense coach. I am Vishwas Bhargava (Enrollment: S24CSEU0190), a student at Bennett University in the CSET 346 (Natural Language Processing) course.

Our team project is "TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification".

My assigned role in the team is:
- Role: Probability Calibration Lead, Ablation Analyst & Product Integration Lead
- Assigned Presentation Slides: SLIDE 9 and SLIDE 10 (Serial Wise)

Please act as my personal project mentor and thoroughly explain:
1. WHAT IS MY ROLE & EXACT CONTRIBUTION:
   - The Visual Dominance Trap Investigation:
     * Audited Condition 2 (Cheapfake: authentic image + fabricated headline like "Explosion levels hospital").
     * Discovered that the uncalibrated baseline achieved 0.0% accuracy on Condition 2 because visual priors overpowered textual red flags.
     * Demonstrated that Contrastive InfoNCE Alignment restored Condition 2 accuracy to 100.0% (+100.0% net gain).
   - Probability Calibration & Uncertainty Guardrail:
     * Discovered that raw deep learning outputs had severe overconfidence on out-of-domain social media (ECE = 0.3406).
     * Implemented post-hoc Temperature Scaling by minimizing Negative Log-Likelihood (NLL) on validation logits, deriving optimal temperature T* = 1.4628.
     * Reduced Expected Calibration Error (ECE) to 0.2701 and instituted an uncertainty policy threshold tau = 0.65, routing 38.3% of borderline cases to an explicit UNCERTAIN category for human review.
   - Systematic Six-Configuration Ablation Matrix (N = 276 total evaluations):
     * Evaluated Config A (Production Baseline), Config D (Calibrated), Config C (Contrastive Aligned), Config B (Domain-Adapted Text), Config F (Full Proposed Pipe), and Config E (Adapted + Contrast).
     * Discovered the critical scientific phenomenon of "Negative Transfer": Stacking all modifications blindly (Config F) dropped semantic mismatch accuracy to 48.3% and degraded calibration, confirming Config A as the superior production winner (8.32/10 composite score).
   - Production Demo, Limitations & Future Roadmap:
     * Verification workflow (5 steps: Ingest -> Infer -> Calibrated Verdict -> Explain -> SQLite).
     * Honest limitations (N=320 sample constraint, lack of primary audiovisual labels, calibration drift).
     * Future roadmap (RAG fact-checking via Wikipedia/Google APIs, DCT/FFT frequency transforms, multilingual Indic dialects).
     * Final project thesis banner delivery.

2. HOW CALIBRATION AND DECISION POLICIES WORK MATHEMATICALLY:
   - Provide the exact equations for Temperature Scaling: p_i = exp(z_i / T) / sum_j exp(z_j / T).
   - Explain Expected Calibration Error (ECE) binning formula: ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|.
   - Explain the tri-state decision rule: if max(p) >= 0.65 -> argmax(p); else -> UNCERTAIN.

3. MY PPT PRESENTATION SCRIPT (SLIDES 9 & 10):
   - Give me a smooth opening acknowledging Member 4's handover.
   - Give me an exact, authoritative oral script for Slide 9 (Visual Dominance Trap, 0% to 100% gain, Temperature Scaling T*=1.4628, 6-Configuration Ablation Table, and the Critical Negative Transfer Alert).
   - Give me an exact oral script for Slide 10 (Verification Workflow with mini result badge, Honest Limitations, Future Roadmap, and the powerful closing thesis quote).
   - Provide a strong closing statement thanking the professors and opening the floor for Q&A.

4. ANTICIPATED VIVA / DEFENSE QUESTIONS & ANSWERS:
   - Give me the top 5 challenging questions the professor might ask about why negative transfer occurs, Platt scaling vs Temperature scaling, why tau was set to 0.65, ethical risks of high confidence errors in journalism, and future RAG integration, with profound scientific answers.
```
