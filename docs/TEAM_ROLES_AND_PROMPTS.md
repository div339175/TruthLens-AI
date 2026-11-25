# TruthLens AI — Team Role Distribution & ChatGPT Preparation Prompts
**Course:** CSET 346 — Natural Language Processing  
**Institution:** Bennett University, Greater Noida, India  
**Project:** TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Presentation:** 10 Slides (2 Slides per Team Member)

---

## 1. Team Roster & Slide Allocation

| # | Team Member | Student ID & Email | Designated Role | Slides Owned |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Divakar Maurya** | S24CSEU0807 (`s24cseu0807@bennett.edu.in`) | **Project Lead & Core Multimodal Architect** | **Slide 1 & Slide 4** |
| **2** | **Vrishank Pathak** | S24CSEU0950 (`s24cseu0950@bennett.edu.in`) | **NLP & Theoretical Foundations Lead** | **Slide 2 & Slide 5** |
| **3** | **Hrishit Goel** | S24CSEU2861 (`s24cseu2861@bennett.edu.in`) | **Benchmark & Data Engineering Lead** | **Slide 3 & Slide 8** |
| **4** | **Dhruv Agarwal** | S24CSEU0916 (`s24cseu0916@bennett.edu.in`) | **Explainable AI (XAI) & Alignment Lead** | **Slide 6 & Slide 9** |
| **5** | **Vishwas Bhargava** | S24CSEU0190 (`s24cseu0190@bennett.edu.in`) | **Full-Stack Systems & Production Deployment Lead** | **Slide 7 & Slide 10** |

---

## 2. Individual Prompts to Paste into ChatGPT

---

### PROMPT FOR TEAM MEMBER 1: DIVAKAR MAURYA (S24CSEU0807)
*Role: Project Lead & Core Multimodal Architect*  
*Assigned Slides: Slide 1 (Title & Team Identity) & Slide 4 (Overall System Architecture)*

```text
You are my personal AI Research Defense Coach and NLP Professor. I am Divakar Maurya (Student ID: S24CSEU0807) at Bennett University, presenting our final semester academic research project for CSET 346: Natural Language Processing.

Our Project:
"TruthLens AI: Multimodal NLP for Deepfake Detection and Explainable Content Verification"
Authors: Divakar Maurya (Lead), Vrishank Pathak, Hrishit Goel, Dhruv Agarwal, Vishwas Bhargava.

My Official Role in the Team:
PROJECT LEAD & CORE MULTIMODAL ARCHITECT
I am responsible for the overall vision, cross-modal integration architecture, bidirectional cross-attention formulation, and master system synthesis across all 7 development phases.
I present two pivotal slides in our 10-slide deck:
- SLIDE 1: Title & Team Identity (Opening the presentation, establishing motivation, introducing the team, and presenting the 4-stream convergence flow).
- SLIDE 4: Overall System Architecture (The core blueprint: 4-stage pipeline connecting Inputs -> Modality Encoders -> Cross-Attention Fusion -> Calibrated Verdict & XAI).

Act as my mentor and provide me with a comprehensive, master-level preparation briefing covering:
1. EXECUTIVE SUMMARY OF MY ROLE & OWNERSHIP:
   - Explain exactly what my role as Project Lead and Core Multimodal Architect means.
   - Explain how I coordinated the 7 development phases (SQLite history, temporal video pooling, acoustic STFT branch, baseline evaluations, external generalization suites, contrastive cross-modal calibration, and final product hardening).
   - What did I build under the hood? (Bidirectional cross-attention fusion projecting text, vision, video, and audio into a shared 256-D latent space; temperature-scaled uncertainty routing at tau=0.65).

2. COMPLETE SYSTEM UNDERSTANDING:
   - Explain how TruthLens AI works from the moment a user uploads a claim with media to the final verdict.
   - How do the 4 modality representations (DistilBERT text z_T in R^256, ResNet18 image z_I in R^256, 8-frame latent pooled video z_V in R^256, and 128-D STFT audio z_A in R^256) interact?
   - Why did naive late concatenation fail, and why did our bidirectional cross-attention succeed?

3. DETAILED SLIDE-BY-SLIDE PRESENTATION SCRIPT (SLIDE 1 & SLIDE 4):
   - For SLIDE 1: Give me the exact professional opening monologue, how to introduce my teammates, and how to explain the horizontal convergence diagram ([Text] + [Image] + [Video] + [Audio] -> Multimodal Core -> Tri-State Verdict). Provide the exact transition handing over to Vrishank Pathak for Slide 2.
   - For SLIDE 4: Give me an authoritative technical walkthrough of the 4 sequential stages (Stage 1: Inputs, Stage 2: Encoders, Stage 3: Fusion & InfoNCE Alignment, Stage 4: Calibrated Decision Policy & XAI). Provide the transition handing over to Vrishank Pathak for Slide 5.

4. VIVA / PROFESSOR DEFENSE QUESTIONS & BULLETPROOF ANSWERS:
   - Give me the top 5 toughest questions a professor could ask me about the system architecture, mathematical fusion, and my leadership decisions, along with concise, mathematically sound answers.
   - Specifically include: "Why choose 256 dimensions for the projection space?", "How does cross-attention handle missing modalities (e.g. text + image only)?", and "Why introduce a tri-state UNCERTAIN class instead of a simple binary classification?".
```

---

### PROMPT FOR TEAM MEMBER 2: VRISHANK PATHAK (S24CSEU0950)
*Role: NLP & Theoretical Foundations Lead*  
*Assigned Slides: Slide 2 (The Problem & Literature Gap) & Slide 5 (Multimodal Methodology & Temporal Video Fix)*

```text
You are my personal AI Research Defense Coach and NLP Professor. I am Vrishank Pathak (Student ID: S24CSEU0950) at Bennett University, presenting our final semester academic research project for CSET 346: Natural Language Processing.

Our Project:
"TruthLens AI: Multimodal NLP for Deepfake Detection and Explainable Content Verification"
Authors: Divakar Maurya, Vrishank Pathak, Hrishit Goel, Dhruv Agarwal, Vishwas Bhargava.

My Official Role in the Team:
NLP & THEORETICAL FOUNDATIONS LEAD
I am responsible for formulating the core problem definition, the state-of-the-art literature survey, the DistilBERT language representation pipeline with domain-adaptive rehearsal fine-tuning, and co-engineering the architectural video temporal pooling fix.
I present two critical methodology slides in our 10-slide deck:
- SLIDE 2: The Problem & Literature Gap — Deepfakes Are Multimodal (The Cheapfake Paradox, Visual Hegemony, Overconfident Black Boxes, and the SOTA Comparison Table).
- SLIDE 5: Multimodal Methodology & Temporal Video Fix (DistilBERT WordPiece pipeline with elastic regularization, ResNet18 visual pipeline, Phase 2 Video Temporal Pooling Fix Before vs After, and Audio STFT feature extraction).

Act as my mentor and provide me with a comprehensive, master-level preparation briefing covering:
1. EXECUTIVE SUMMARY OF MY ROLE & OWNERSHIP:
   - Explain exactly what my role as NLP & Theoretical Foundations Lead means.
   - How did I identify the literature gap showing that unimodal forensic detectors (FaceForensics++, AVSS, COSMOS, Fakeddit) fail on cross-modal manipulations?
   - How did I build the DistilBERT language pipeline, and what is the exact mathematics behind our domain adaptation loss: L_adapt = L_CE(T_fakeddit) + 0.5 * L_CE(T_course)?
   - How did we resolve the Phase 2 video temporal defect where pixel-mean averaging caused severe motion blur, replacing it with latent feature mean pooling?

2. COMPLETE TECHNICAL DEEP DIVE:
   - Explain the "Cheapfake Paradox": why authentic photos paired with fabricated text trick visual classifiers.
   - Explain "Visual Hegemony": why high-capacity vision backbones overpower text signals during naive late fusion.
   - Detail the video temporal fix: Video frames -> independent ResNet forward passes -> latent mean pooling f_video = (1/K) sum_{k=1}^K f_k in R^512 -> z_V in R^256.

3. DETAILED SLIDE-BY-SLIDE PRESENTATION SCRIPT (SLIDE 2 & SLIDE 5):
   - For SLIDE 2: Give me an articulate academic delivery explaining the 3 problem cards and walking row-by-row through the literature table (comparing Rössler, Guera & Delp, Aneja/COSMOS, Nakamura/Fakeddit, Li/TASLP against TruthLens AI). Hand over smoothly to Hrishit Goel for Slide 3.
   - For SLIDE 5: Give me an authoritative technical walkthrough of the 2x2 grid (Text DistilBERT, Image ResNet18, Video Temporal Pooling Before vs After, Audio STFT). Hand over smoothly to Dhruv Agarwal for Slide 6.

4. VIVA / PROFESSOR DEFENSE QUESTIONS & BULLETPROOF ANSWERS:
   - Give me the top 5 toughest questions a professor could ask me about the NLP pipeline, domain shift, and the video temporal pooling fix, with rigorous academic answers.
   - Specifically include: "Why use DistilBERT instead of larger models like RoBERTa or GPT?", "Why does alpha=0.5 elastic regularization prevent catastrophic forgetting on informal Reddit slang?", and "Why does latent feature averaging preserve temporal boundaries better than pixel averaging?".
```

---

### PROMPT FOR TEAM MEMBER 3: HRISHIT GOEL (S24CSEU2861)
*Role: Benchmark & Data Engineering Lead*  
*Assigned Slides: Slide 3 (Audited Datasets & Multi-Suite Setup) & Slide 8 (Experimental Results & Baselines)*

```text
You are my personal AI Research Defense Coach and NLP Professor. I am Hrishit Goel (Student ID: S24CSEU2861) at Bennett University, presenting our final semester academic research project for CSET 346: Natural Language Processing.

Our Project:
"TruthLens AI: Multimodal NLP for Deepfake Detection and Explainable Content Verification"
Authors: Divakar Maurya, Vrishank Pathak, Hrishit Goel, Dhruv Agarwal, Vishwas Bhargava.

My Official Role in the Team:
BENCHMARK & DATA ENGINEERING LEAD
I am responsible for dataset curation, cross-modal data auditing, split stratification with zero leakage, benchmark harness construction, and empirical baseline evaluation across in-domain and out-of-domain suites.
I present two foundational empirical slides in our 10-slide deck:
- SLIDE 3: Audited Datasets & Multi-Suite Setup (Primary course dataset N=320 with 70/15/15 split, 4 forensic typologies, external generalization suites N=180 [Fakeddit, Audio v4, Video Sora], and the 4-condition Semantic Mismatch suite).
- SLIDE 8: Experimental Results & Baselines (The master evaluation table comparing DistilBERT, ResNet18, Acoustic, Video Temporal, Concatenation, and TruthLens Cross-Attention; plus the 3 callouts: In-Domain Saturation, Unimodal Collapse, and Calibration Superiority).

Act as my mentor and provide me with a comprehensive, master-level preparation briefing covering:
1. EXECUTIVE SUMMARY OF MY ROLE & OWNERSHIP:
   - Explain exactly what my role as Benchmark & Data Engineering Lead entails.
   - How did I ensure zero visual and textual leakage across our 70% Train (N=224), 15% Val (N=48), and 15% Test (N=48) splits?
   - Why was it critical to curate external benchmarks (N=180) rather than relying only on the 320 course samples?
   - How did I construct the 4 controlled conditions of the Semantic Mismatch Suite (30 pairs / 60 evaluations)?

2. EMPIRICAL BENCHMARK ANALYSIS:
   - Why do all baselines achieve 100.0% in-domain accuracy, and why is this "in-domain saturation" dangerous if untested on out-of-domain data?
   - How did the DistilBERT baseline collapse to 33.33% on uncurated Fakeddit Reddit slang, and what does that prove?
   - Walk through the numbers: Audio ROC-AUC (0.8200), Video ROC-AUC (0.8200), and Mean ECE (0.1324 for Cross-Attention vs 0.2840 for text).

3. DETAILED SLIDE-BY-SLIDE PRESENTATION SCRIPT (SLIDE 3 & SLIDE 8):
   - For SLIDE 3: Give me a sharp, clear presentation script explaining the 3 dashboard columns (Primary Dataset, External Generalization Suites, Semantic Mismatch Suite) and addressing the FakeAVCeleb institutional gating disclosure. Hand over smoothly to Divakar Maurya for Slide 4.
   - For SLIDE 8: Give me an authoritative delivery walking through the master baseline results table, contrasting unimodal vs multimodal performance, and explaining the 3 bottom callout cards. Hand over smoothly to Dhruv Agarwal for Slide 9.

4. VIVA / PROFESSOR DEFENSE QUESTIONS & BULLETPROOF ANSWERS:
   - Give me the top 5 toughest questions a professor could ask me about dataset leakage, split stratification, benchmark validity, and metric interpretations, with rock-solid answers.
   - Specifically include: "With only 320 primary samples, how do you defend statistical significance?", "Why did you use ROC-AUC for audio/video but Accuracy for text/image?", and "What exactly does an Expected Calibration Error (ECE) of 0.1324 represent?".
```

---

### PROMPT FOR TEAM MEMBER 4: DHRUV AGARWAL (S24CSEU0916)
*Role: Explainable AI (XAI) & Alignment Lead*  
*Assigned Slides: Slide 6 (From Prediction to Evidence: Attention & XAI) & Slide 9 (Semantic Mismatch, Calibration & Ablation Matrix)*

```text
You are my personal AI Research Defense Coach and NLP Professor. I am Dhruv Agarwal (Student ID: S24CSEU0916) at Bennett University, presenting our final semester academic research project for CSET 346: Natural Language Processing.

Our Project:
"TruthLens AI: Multimodal NLP for Deepfake Detection and Explainable Content Verification"
Authors: Divakar Maurya, Vrishank Pathak, Hrishit Goel, Dhruv Agarwal, Vishwas Bhargava.

My Official Role in the Team:
EXPLAINABLE AI (XAI) & ALIGNMENT LEAD
I am responsible for the mathematical formulation of the bidirectional cross-attention mechanism, the InfoNCE contrastive cross-modal alignment objective, the multimodal explainability engine (token saliency, Grad-CAM, acoustic diagnostics), and the 6-configuration ablation study.
I present two of the most mathematically rigorous slides in our 10-slide deck:
- SLIDE 6: From Prediction to Evidence — Attention & XAI (Cross-attention mechanics: Attn(Q_T, K_V, V_V) and Attn(Q_V, K_T, V_T), InfoNCE contrastive alignment loss lambda=0.05, and the 3 causal explainability cards: Token Saliency, Layer4 Grad-CAM, and Acoustic Diagnostics).
- SLIDE 9: Semantic Mismatch, Calibration & Ablation Matrix (The Visual Dominance Trap: Condition 2 cheapfake failure at 0.0% accuracy restored to 100.0% via contrastive learning; Temperature scaling T*=1.4628 reducing ECE from 0.3406 to 0.2701; the 6-configuration ablation matrix [N=276]; and the critical negative transfer warning on Config F).

Act as my mentor and provide me with a comprehensive, master-level preparation briefing covering:
1. EXECUTIVE SUMMARY OF MY ROLE & OWNERSHIP:
   - Explain exactly what my role as Explainable AI & Alignment Lead entails.
   - What is the mathematical formulation of our Bidirectional Cross-Attention (4 heads, d_k=64)?
   - How does the InfoNCE contrastive loss (lambda=0.05) pull matching semantic pairs together and repel fabricated mismatches?
   - How did I build the 3 explainability hooks: WordPiece token gradient saliency S(w_i) = || ∂y_c / ∂e_i ||_1, ResNet18 Layer4 Grad-CAM heatmaps, and STFT acoustic spectral zero-crossing diagnostics?

2. RIGOROUS SCIENTIFIC ANALYSIS OF THE ABLATION STUDY:
   - Explain the "Visual Dominance Trap": why Config A scored 0.0% on Condition 2 cheapfakes (authentic photo + fabricated headline), and how Config C's contrastive alignment restored it to 100.0% (+100% gain).
   - Walk through the 6 configurations (Config A, B, C, D, E, F) across the 276 evaluations. Why did Config A win overall with 8.32/10?
   - What is "Negative Transfer Drift", and why did blindly combining text adaptation + contrastive alignment in Config F drop mismatch accuracy to 48.3%?

3. DETAILED SLIDE-BY-SLIDE PRESENTATION SCRIPT (SLIDE 6 & SLIDE 9):
   - For SLIDE 6: Give me an articulate academic delivery explaining the bipartite cross-attention network, the mathematical equations, and walking through the 3 evidence cards. Hand over smoothly to Vishwas Bhargava for Slide 7.
   - For SLIDE 9: Give me an authoritative delivery explaining the visual dominance trap bar metric, temperature scaling calibration, the ablation matrix table, and delivering the critical scientific warning on negative transfer. Hand over smoothly to Vishwas Bhargava for Slide 10.

4. VIVA / PROFESSOR DEFENSE QUESTIONS & BULLETPROOF ANSWERS:
   - Give me the top 5 toughest questions a professor could ask me about cross-attention, Grad-CAM backpropagation, InfoNCE loss, and negative transfer, with airtight mathematical answers.
   - Specifically include: "How are gradients propagated back to input token embeddings for saliency?", "Why set lambda=0.05 for InfoNCE loss?", and "Why does temperature scaling modify probabilities without changing argmax class predictions?".
```

---

### PROMPT FOR TEAM MEMBER 5: VISHWAS BHARGAVA (S24CSEU0190)
*Role: Full-Stack Systems & Production Deployment Lead*  
*Assigned Slides: Slide 7 (Software Architecture & Production Tech Stack) & Slide 10 (End-to-End Demo, Limitations & Future Roadmap)*

```text
You are my personal AI Research Defense Coach and NLP Professor. I am Vishwas Bhargava (Student ID: S24CSEU0190) at Bennett University, presenting our final semester academic research project for CSET 346: Natural Language Processing.

Our Project:
"TruthLens AI: Multimodal NLP for Deepfake Detection and Explainable Content Verification"
Authors: Divakar Maurya, Vrishank Pathak, Hrishit Goel, Dhruv Agarwal, Vishwas Bhargava.

My Official Role in the Team:
FULL-STACK SYSTEMS & PRODUCTION DEPLOYMENT LEAD
I am responsible for the end-to-end software engineering architecture, the asynchronous FastAPI backend gateway, the React 18 / TypeScript / Tailwind CSS verification frontend, the SQLite zero-loss audit database, CI test suite maintenance (96/96 automated tests passing), and formulating our honest research limitations and future extensions.
I present two critical production and closing slides in our 10-slide deck:
- SLIDE 7: Software Architecture & Production Tech Stack (The 4-tier vertical production architecture: Client, API Gateway, ML Inference Runtime, SQLite Persistence; Tech Badges; and the 96/96 Automated Tests Health Card).
- SLIDE 10: End-to-End Demo, Limitations & Future Roadmap (The 5-step Verification Workflow: Ingest Media -> Deep Inference -> Calibrated Verdict tau=0.65 -> Explainability -> SQLite Logging; Live Result Preview Badge; Honest Limitations: Dataset Scale, Video/Audio Labels, Calibration Drift; Future Roadmap: RAG Fact-Checking, DCT/FFT Frequency Transforms, Multilingual Cross-Attention; and the Grand Concluding Quote).

Act as my mentor and provide me with a comprehensive, master-level preparation briefing covering:
1. EXECUTIVE SUMMARY OF MY ROLE & OWNERSHIP:
   - Explain exactly what my role as Full-Stack Systems & Production Deployment Lead means.
   - How did I architect the 4 tiers: Client (React 18 + Vite), API Gateway (FastAPI + Uvicorn), Inference Runtime (PyTorch + Torchvision + Librosa), and Persistence (SQLite 3 truthlens.db)?
   - How did I guarantee 100% test health with 96/96 automated tests passing across unit, integration, database persistence, and API routes?
   - How does our calibrated decision policy tau=0.65 route uncertain cases to human analysts?

2. END-TO-END DEMO WORKFLOW & PRODUCTION PIPELINE:
   - Walk through the exact operational sequence when a user drops an image/video/audio and text claim into our web app:
     POST /api/verify -> Multipart payload parsing -> Tensor transformation -> Parallel modality forward passes -> Cross-attention fusion -> Temperature calibration -> Verdict generation -> Grad-CAM overlay generation -> SQLite commit -> JSON response.
   - Explain the honest research limitations with candor and academic integrity, showing that we are realistic researchers who understand the real-world boundaries of our model.

3. DETAILED SLIDE-BY-SLIDE PRESENTATION SCRIPT (SLIDE 7 & SLIDE 10):
   - For SLIDE 7: Give me a crisp, confident presentation script walking through the 4-tier software architecture and emphasizing our production stability, zero TypeScript errors, and 96/96 test coverage. Hand over smoothly to Hrishit Goel for Slide 8.
   - For SLIDE 10: Give me an authoritative, commanding closing delivery walking through the 5-step verification workflow, presenting the live verdict preview, candidly discussing our 3 limitations, outlining the 3 future research directions (RAG, DCT/FFT, Multilingual), and delivering the powerful concluding banner.

4. VIVA / PROFESSOR DEFENSE QUESTIONS & BULLETPROOF ANSWERS:
   - Give me the top 5 toughest questions a professor could ask me about API throughput, asynchronous execution, database schema, production scalability, and system limitations, with rock-solid software engineering answers.
   - Specifically include: "How does the system handle high-concurrency requests in FastAPI during heavy PyTorch GPU inference?", "Why choose SQLite over PostgreSQL or MongoDB for this stage?", and "How would you integrate live RAG fact-checking without introducing unacceptable API latency?".
```
