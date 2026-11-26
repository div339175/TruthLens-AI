# TruthLens AI — Public Dataset Selection & Comparative Evaluation

**Project:** TruthLens AI (Multimodal NLP for Deepfake Detection and Explainable Content Verification)  
**Document:** Dataset Selection & Comparative Feasibility Audit  
**Author:** TruthLens AI Engineering & Research Group  
**Date:** November 2026  
**Status:** Approved & Validated  

---

## 1. Selected Dataset

- **Primary Selected Dataset:** **r/Fakeddit Benchmark (Multimodal Text-Image)**
- **Authors & Affiliation:** Kai Nakamura, Sharon Levy, and William Yang Wang (UC Santa Barbara NLP Lab)
- **Official Publication:** *r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection*, Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), European Language Resources Association (ELRA), Marseille, France, pp. 6149–6157.
- **Official Repository:** `https://github.com/entitize/Fakeddit`
- **Official Portal:** `https://fakeddit.netlify.app/`
- **Standard Research Mirror:** HuggingFace Datasets `ams-99/fakeddit_9k` (curated balanced evaluation split)

---

## 2. Why Fakeddit Was Selected

TruthLens AI was designed for **Multimodal NLP and Content Verification** (Course: CSET 346 - Natural Language Processing). Its core architectural innovation is the **Bidirectional Cross-Attention Fusion** connecting a Transformer text encoder (**DistilBERT**) and a Convolutional visual stem (**ResNet-18**).

Fakeddit was selected because it uniquely satisfies every architectural, scientific, and practical criterion:

1. **Direct Modality Alignment (Text + Image):** Every sample in Fakeddit consists of a natural language text claim (Reddit submission title) paired with a high-resolution image. This directly engages both the text tokenizer/encoder and the vision backbone.
2. **Task Alignment (Cross-Modal Contradiction / Cheapfakes):** The primary failure mode of multimodal verifiers on real-world news (such as the BBC article tested recently) is semantic mismatch—where an authentic photograph is paired with a misleading caption. Fakeddit specifically isolates this phenomenon in its Label 4 ("False Connection"), enabling empirical testing and training against multimodal misinformation.
3. **Open Research Licensing & No Gated Credentialing:** Unlike video deepfake datasets that require signing university NDAs or Google Forms, Fakeddit is distributed under open research terms (MIT License for code, Creative Commons/Fair Use research access for metadata and image references).
4. **Computational Feasibility for Student Research:** The dataset can be processed and evaluated rapidly on commodity GPU or Apple Silicon hardware without requiring terabytes of storage or cluster compute.
5. **Class Balance & Fine-Grained Granularity:** Fakeddit provides mutually mapped 2-way (Real vs. Fake), 3-way, and 6-way classification labels, enabling rigorous binary benchmarking alongside fine-grained manipulation analysis.

---

## 3. Alternative Datasets Considered

| Candidate Dataset | Primary Modalities | Access / Licensing Status | Compatibility with TruthLens AI | Reason for Rejection / Deferred Role |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: FakeAVCeleb** (Khalid et al., NeurIPS 2021) | Video + Audio (Facial deepfakes, cloned speech) | **Gated Access:** Requires submitting an individual Google Request Form to Sungkyunkwan University; unique download scripts emailed upon manual review. | **No Text Modality:** Does not contain textual claims, news headlines, or captions. Leaves DistilBERT completely unused. | Incompatible with core Multimodal NLP architecture; gated download cannot be automated in public student repositories. |
| **Option B: FaceForensics++** (Rössler et al., ICCV 2019) | Video only (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | **Restricted Access:** Requires submitting a formal application to Technical University of Munich (TUM) for access credentials. | **No Text Modality:** Purely computer-vision facial reenactment without linguistic or cross-modal context. | Unsuitable for NLP course objectives; restricted distribution policy. |
| **Option C: DFDC** (Dolhansky et al., 2020) | Video + Audio (Kaggle Deepfake Detection Challenge) | **Gated Access:** ~470 GB archive requiring Kaggle API authentication (`kaggle.json`) or massive download bandwidth. | **No Text Modality:** Focuses exclusively on raw video artifacts; storage-prohibitive for student workstations. | Exceeds realistic workstation storage limits; no text or cross-modal NLP grounding. |

---

## 4. Advantages of Fakeddit

- **Natural Social Media Distribution:** Real-world text written by humans, reflecting authentic syntactic variety, colloquialisms, and journalistic headlines rather than template-generated strings.
- **Multi-Modal Synergy:** Exercises the full cross-attention matrix between DistilBERT token representations and ResNet-18 spatial grid features.
- **Reproducible Verification:** Public URLs and deterministic subsets allow any student, researcher, or evaluator to reproduce exact results using seed `42`.
- **Interpretable Saliency:** Gradient saliency highlights deceptive tokens (e.g., clickbait keywords) while Grad-CAM highlights manipulated or misaligned image regions.

---

## 5. Limitations of Fakeddit

- **Image Link Attrition (404 Drift):** Because the original dataset links to external Reddit and Imgur image hosts, some URLs inevitably decay over time. *Mitigation:* The TruthLens ingestion pipeline incorporates strict HTTP validation and MD5 verification, discarding broken URLs and storing local processed caches.
- **No Native Audio Stream:** Fakeddit does not provide audio tracks. *Mitigation:* TruthLens AI retains its synthetic acoustic pipeline (`ml/preprocessing/audio.py`) and Deepfake Audio Detection suite for tri-modal evaluations while using Fakeddit as its primary text-image public benchmark.

---

## 6. License Information

- **Code License:** MIT License (official UCSB NLP Lab repository).
- **Metadata & Submissions:** Public domain / Fair Use for non-commercial academic research. Reddit content is subject to Reddit's API Terms of Service.
- **Redistribution Policy:** Raw third-party images must not be bundled in git repositories. Instead, metadata, URLs, hashes, and deterministic preprocessing scripts are versioned.

---

## 7. Research-Use Conditions

- **Permitted Use:** Academic evaluation, reproducible research benchmarks, algorithmic audits, and educational demonstrations.
- **Prohibited Use:** Commercial resale of raw data, re-identification of private Reddit users, or targeted harassment.
- **Citation Requirement:** Formal citation of the original LREC 2020 publication by Nakamura et al.

---

## 8. Dataset Provenance

- **Origin:** Curated from Reddit submissions across multiple diverse subreddits (including news, political satire, photo manipulation communities, and official channels).
- **Annotation Methodology:** Multi-stage labeling based on subreddit credibility, fact-checking verifications (Snopes, PolitiFact), and manual quality control.

---

## 9. Original Source

- **URL:** `https://github.com/entitize/Fakeddit`
- **Hosted Portal:** `https://fakeddit.netlify.app/`
- **Dataset Paper:** `https://arxiv.org/abs/1911.03854`

---

## 10. Official Citation Information

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

---

## 11. Expected Modalities

1. **Text Modality:** Submission titles, captions, or claim statements (20 to 180 characters), tokenized via `distilbert-base-uncased` to $[B, 128]$.
2. **Visual Modality:** Natural photographs, digital illustrations, or screenshots, normalized and resized via `ImagePreprocessor` to $[B, 3, 224, 224]$.

---

## 12. Real / Fake Class Structure

Fakeddit supports multi-tiered classification:

### Binary Mapping (Used by TruthLens AI):
- **REAL ($y=0$):** True submissions verified as authentic representations of events.
- **FAKE ($y=1$):** Submissions falling into any category of misattribution, satire, manipulation, or fabrication.

### Fine-Grained Typologies (6-Way Classification):
- **0: True (REAL)**
- **1: Satire / Parody (FAKE)**
- **2: Misleading Content (FAKE)**
- **3: Manipulated Content / Photo Tampering (FAKE)**
- **4: False Connection / Semantic Mismatch (FAKE)** — Authentic image paired with a fabricated headline.
- **5: Imposter Content (FAKE)**

---

## 13. Known Ethical and Privacy Considerations

- **User Privacy:** The dataset uses publicly submitted Reddit post titles and images. Personal identifiable information (usernames, IP addresses) is stripped from all local metadata indexes.
- **No Private Surveillance:** In accordance with Section 4 of the project guidelines, no scraping of private individuals' personal files or non-public communications was conducted.
- **Content Sensitivity:** A small fraction of social media posts may touch on sensitive political or cultural topics. The dataset is restricted strictly to academic benchmarking of content verification algorithms.
