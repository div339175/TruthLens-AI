# Phase 5: External Dataset Feasibility & Benchmarking Audit

**Project:** TruthLens AI — Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Phase:** Phase 5 — Real-World Benchmark Expansion and Generalization  
**Date:** November 2026  
**Status:** Audit Complete & Validated  

---

## 1. Executive Summary

In Phase 4, the TruthLens AI evaluation framework verified the multimodal architecture on the in-domain course dataset (320 samples, 50% Real / 50% Fake). While unimodal and multimodal models achieved 100% metrics on the clean synthetic test split, evaluation on 15 challenging edge cases revealed an accuracy drop to **73.33%**, exposing a critical vulnerability to **cross-modal semantic mismatches** (authentic image paired with fabricated text captions). Furthermore, the course dataset contained 0 labeled video records and 0 labeled audio records, preventing quantitative benchmarking of the video temporal pooling and audio projection branches.

Phase 5 investigates external real-world benchmarks to evaluate true out-of-domain generalization across:
1. **Multimodal Text + Image**: r/Fakeddit Benchmark (Nakamura et al., LREC 2020)
2. **Audio-Video Deepfakes**: FakeAVCeleb (Khalid et al., NeurIPS 2021) and Public Audio/Video Benchmarks
3. **Controlled Semantic Mismatches**: Cross-modal contradiction suite

To maintain strict scientific integrity and avoid downloading multi-gigabyte or credentials-gated archives into the workspace, we conduct this formal feasibility audit before initiating any data ingestion.

---

## 2. Candidate Dataset 1: Fakeddit (Multimodal Text-Image)

### Overview & Source
- **Official Title:** r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection
- **Authors & Institution:** Kai Nakamura, Sharon Levy, William Yang Wang (UC Santa Barbara NLP Lab)
- **Publication:** LREC 2020 (arXiv:1911.03854)
- **Official Repository:** `https://github.com/entitize/Fakeddit` / `https://fakeddit.netlify.app/`
- **Validated Mirror Source:** `ams-99/fakeddit_9k` (HuggingFace, balanced evaluation split)

### Technical Specifications
- **Full Dataset Size:** > 1,000,000 Reddit submissions (~400 GB full image archive, 496 MB curated sample cache)
- **Modalities:** Text (Reddit submission titles) + Visual (224x224 RGB image submissions)
- **File Formats:** Tab-separated text/CSV metadata (`test_balanced.csv`), JPEG/PNG images
- **Original Labels:**
  - *2-way:* 0 = Real, 1 = Fake
  - *3-way:* 0 = Completely True, 1 = Misleading/Manipulated, 2 = Satire
  - *6-way:* 
    - 0: True
    - 1: Satire / Parody
    - 2: Misleading Content
    - 3: Manipulated Content
    - 4: False Connection (Semantic Mismatch / Caption Misalignment)
    - 5: Imposter Content
- **Mapping to TruthLens Target:**
  - Label 0 $\rightarrow$ **REAL (0)**
  - Labels 1, 2, 3, 4, 5 $\rightarrow$ **FAKE (1)**
  - Label 4 specifically isolates **Semantic Mismatch** (authentic imagery combined with contradictory headline)
- **Access & Authentication:** Open research access (CC-BY / MIT license on code, Reddit public data). No gated authentication required for metadata or image URLs.
- **Academic Suitability:** Standard benchmark in multimodal NLP and misinformation detection literature.
- **TruthLens Compatibility:** 100% compatible. Text passes directly into DistilBERT tokenizer (`TextPreprocessor`), images pass directly into ResNet18 (`ImagePreprocessor`).
- **Necessity of Full Dataset:** **Not Necessary.** The full 1M-sample archive is excessive and storage-prohibitive. A balanced, deterministic subset of 60 samples (30 Real, 30 Fake) captures out-of-domain real-world social media distribution with zero storage overhead (~10 MB).

### Subset Selection Rule
- **Dataset Split:** `test_balanced.csv` from Fakeddit.
- **Selection Algorithm:** Deterministic sort by `(6_way_label, hash(text))` with fixed random seed $42$.
- **Quota:** 30 samples with Label 0 (REAL), 30 samples with Labels 1–5 (FAKE), ensuring inclusion of Label 4 ("False Connection") to directly test semantic contradiction.
- **File Verification:** Every image URL is resolved via HTTP HEAD/GET; corrupted or unreachable HTTP 404 links are rejected and logged.

---

## 3. Candidate Dataset 2: FakeAVCeleb (Audio-Video Deepfakes)

### Overview & Source
- **Official Title:** FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset
- **Authors & Institution:** Hasam Khalid, Minha Kim, Shahroz Tariq, Simon S. Woo (DASH-Lab, Sungkyunkwan University)
- **Publication:** NeurIPS 2021 Datasets & Benchmarks Track
- **Official Repository:** `https://github.com/DASH-Lab/FakeAVCeleb`
- **Dataset Site:** `https://sites.google.com/view/fakeavcelebdash-lab/`

### Technical Specifications & Feasibility Audit
- **Full Dataset Size:** ~30 GB (21,566 video clips, 500 celebrities)
- **Modalities:** Video (MP4, H.264) + Audio (16 kHz WAV/AAC)
- **File Formats:** MP4 video containers with synchronized audio tracks
- **Original Labels:**
  - `0`: FakeVideo-FakeAudio
  - `1`: FakeVideo-RealAudio
  - `2`: RealVideo-FakeAudio
  - `3`: RealVideo-RealAudio
- **Access & Licensing Requirements:**
  - **Gated Access:** The authors require researchers to submit an individual Google Request Form (`https://docs.google.com/forms/u/1/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform`).
  - Upon manual review and approval, a unique bash download script (`download_FakeAVCeleb.sh`) is emailed with time-limited signed cloud storage links.
  - Public HuggingFace mirrors (e.g. `thenewsupercell/fixed-fakeavceleb`): Parquet inspection revealed that the `video` feature dictionary contains `bytes: None`, pointing to private local paths (`/kaggle/working/...`). No raw video bytes are publicly mirrored.
- **Feasibility Determination:**
  - **STATUS: GATED / NOT PROGRAMMATICALLY AVAILABLE**
  - In strict compliance with the project's **Scientific Integrity Guidelines** ("Never claim a dataset was used if it was not; if a dataset cannot be accessed, mark it NOT AVAILABLE and explain why"), official FakeAVCeleb cannot be downloaded in an automated, non-credentialed academic environment.

---

## 4. Alternative Real-World Benchmarks for Video & Audio

To fulfill the objective of empirical generalization without fabricating access to gated datasets, we evaluate open, public, peer-reviewed benchmarks:

### 4.1 Audio Generalization Benchmark: Deepfake Audio Detection Dataset (v4)
- **Source:** `garystafford/deepfake-audio-detection` (HuggingFace)
- **License:** CC-BY-4.0
- **Size:** 1,866 audio clips (933 Real human speech, 933 Synthetic cloned speech)
- **Audio Generators:** ElevenLabs, Hume AI, Tortoise TTS, Bark
- **Labels:** 0 = Real (YouTube speech), 1 = Fake (AI voice clones)
- **Usable Subset:** 20 Real + 20 Fake = 40 samples (~4 MB total)
- **Compatibility:** Processed through TruthLens 128-D acoustic feature extractor and `audio_projection`.

### 4.2 Video Generalization Benchmark: Generative Video Deepfake Dataset
- **Source:** `Sarim-Hash/video_DEEPFAKE_dataset` (HuggingFace / Public)
- **Modalities:** Video (MP4)
- **Synthesis Engine:** OpenAI Sora generative synthesis vs authentic high-definition video captures
- **Labels:** 0 = Real (17 videos), 1 = Fake (15 videos)
- **Usable Subset:** 10 Real + 10 Fake = 20 samples (~50 MB total)
- **Compatibility:** Processed through TruthLens uniform 8-frame sampling and feature-level ResNet18 temporal pooling.

---

## 5. Dataset Availability & Modality Support Matrix

| Benchmark Dataset | Modalities Supported | Labels Available | Access Status | Phase 5 Role |
| :--- | :--- | :--- | :--- | :--- |
| **Course Dataset (Phase 4)** | Text, Image | 160 Real, 160 Fake | Local In-Domain | Baseline Benchmark ($N=320$) |
| **Fakeddit Subset** | Text, Image | 30 Real, 30 Fake | Public / Open | External Cross-Dataset Benchmark ($N=60$) |
| **Controlled Semantic Mismatch** | Text, Image | 15 Congruent Real, 15 Mismatch Fake, 15 Mismatch Img, 15 Unrelated | Curated Benchmark | Vulnerability & Attention Analysis ($N=60$) |
| **Deepfake Audio Benchmark** | Audio | 20 Real, 20 Fake | Public / Open | Audio Modality Benchmark ($N=40$) |
| **Generative Video Benchmark** | Video | 10 Real, 10 Fake | Public / Open | Video Modality Benchmark ($N=20$) |
| **Official FakeAVCeleb** | Video, Audio | 4-way Multimodal | **Gated / Auth Required** | **NOT AVAILABLE (Documented Feasibility)** |

---

## 6. Selection Protocol & Integrity Guarantees

1. **No Data Fabrication:** Every sample is downloaded from its original published source or generated via documented deterministic rules.
2. **Deterministic Reproducibility:** All subsets use fixed random seed `42` and explicit index sorting.
3. **No Metric Inflation:** Results on external datasets will be reported raw without cherry-picking or threshold gaming.
4. **Zero Overlap:** External samples are hashed (MD5) and checked against the course dataset to guarantee 0.0% data leakage.
