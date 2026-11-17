# TruthLens AI — Dataset Description and Preprocessing Documentation

**Course**: CSET 346 – Natural Language Processing  
**Project**: Multimodal NLP for Deepfake Detection and Explainable Content Verification  
**Instructor**: Dr. Sakshi (sakshi@bennett.edu.in)  
**Deliverable**: Dataset & Preprocessing Technical Report  

---

## 1. Dataset Overview and Specification

In accordance with Section 3 and Section 9 of the CSET 346 Course Syllabus, the project requires a documented dataset containing genuine and manipulated multimodal samples with a recommended minimum of 300 labeled instances and balanced representation between REAL and FAKE classes.

The TruthLens AI corpus comprises **320 meticulously curated, paired multimodal samples**:

| Metric | Specification |
| :--- | :--- |
| **Total Samples** | **320** (Exceeds $\ge 300$ requirement) |
| **REAL Samples ($y=0$)** | **160** (50.0%) |
| **FAKE Samples ($y=1$)** | **160** (50.0%) |
| **Train Split** | **224 samples** (70.0%) |
| **Validation Split** | **48 samples** (15.0%) |
| **Test Split** | **48 samples** (15.0%) |
| **Metadata File** | [`ml/data/metadata/dataset.csv`](file:///Users/divakarmaurya/nlp_project/ml/data/metadata/dataset.csv) |
| **Media Directory** | [`ml/data/samples/`](file:///Users/divakarmaurya/nlp_project/ml/data/samples/) |

---

## 2. Dataset Schema and Metadata Format

The central index [`ml/data/metadata/dataset.csv`](file:///Users/divakarmaurya/nlp_project/ml/data/metadata/dataset.csv) stores structured attributes for every sample:

| Column Header | Data Type | Description | Allowed Values |
| :--- | :--- | :--- | :--- |
| `id` | String | Unique sample identifier | `sample_001` to `sample_320` |
| `text` | String | Textual claim, headline, broadcast transcript, or caption | 20 to 180 characters |
| `image_path` | String | Relative filepath to image PNG file in `ml/data/` | `samples/real_*.png` or `samples/fake_*.png` |
| `video_path` | String | Relative filepath to video file (if applicable) | `*.mp4` or empty string |
| `audio_path` | String | Relative filepath to audio WAV file (if applicable) | `*.wav` or empty string |
| `label` | Integer | Ground truth binary classification label | `0` (REAL) or `1` (FAKE) |
| `manipulation_type` | String | Specific manipulation class or authentic sub-type | See Section 3 below |
| `source` | String | Source provenance or curation tag | `course_curated_*` |
| `split` | String | Stratified partition assignment | `train`, `val`, or `test` |

---

## 3. Class Distribution & Manipulation Typologies

### A. Authentic Class (160 samples, label = 0)
Spans four representative factual domains to prevent domain memorization:
1. **`course_curated_official_press` (40 samples)**: Official ministry statements, public health releases, transport grants, and bilateral treaties.
2. **`course_curated_science_nature` (40 samples)**: NASA satellite telemetry, astronomical observatory reports, coral reef surveys, and geological records.
3. **`course_curated_verified_journalism` (40 samples)**: Accredited news dispatches, fact-checked archival reports, civic council debates, and election monitoring.
4. **`course_curated_academic_interview` (40 samples)**: University symposiums, Nobel laureate lectures, economics colloquia, and medical research presentations.

### B. Manipulated Class (160 samples, label = 1)
Spans five distinct manipulation categories reflecting modern digital threats:
1. **`deepfake_faceswap` (35 samples)**: Facial reenactments and identity swaps exhibiting subtle boundary blending and edge interpolation artifacts.
2. **`diffusion_synthesis` (35 samples)**: Entirely generated latent diffusion portraits, synthetic crowd scenes, and hallucinated disaster images.
3. **`spliced_banner` (35 samples)**: Digital tampering where legitimate video frames or images have fabricated breaking news banners, headlines, or cloned elements inserted.
4. **`semantic_mismatch` (35 samples)**: Cross-modal contradiction where genuine, authentic photographs are deliberately paired with false, sensationalized captions ("cheapfakes").
5. **`synthetic_speech` (20 samples)**: Audio-visual clips featuring voice-cloned phone calls and synthetic vocal timbre anomalies.

---

## 4. Preprocessing Pipelines by Modality

### A. Text Modality Preprocessing (`ml/preprocessing/text.py`)
- **Cleaning**: Whitespace normalization, URL stripping, and control character removal.
- **Tokenization**: Hugging Face `distilbert-base-uncased` WordPiece tokenizer.
- **Sequence Truncation & Padding**: Padded or truncated to fixed maximum length $L=128$.
- **Tensor Format**: Outputs `input_ids` and `attention_mask` tensors of shape $[B, 128]$.

### B. Image Modality Preprocessing (`ml/preprocessing/image.py`)
- **Resolution Standardization**: Resized to $224 \times 224$ pixels.
- **Normalization**: Normalized using ImageNet channel distributions:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$
- **Tensor Format**: Outputs normalized float32 tensor of shape $[B, 3, 224, 224]$.

### C. Video Modality Preprocessing (`ml/preprocessing/video.py`)
- **OpenCV Capture**: Reads video streams and retrieves total frame counts.
- **Uniform Temporal Sampling**: Extracts $K=8$ uniformly spaced frames across video duration:
  $$\text{frame\_indices} = \left\lfloor \text{linspace}\left(0, \text{total\_frames}-1, 8\right) \right\rfloor$$
- **Tensor Format**: Preprocesses each frame through the image pipeline and stacks into $[8, 3, 224, 224]$.

### D. Audio Modality Preprocessing (`ml/preprocessing/audio.py`)
- **Sample Rate**: $16\text{ kHz}$ mono PCM audio.
- **Spectral Feature Extraction**: Short-Time Fourier Transform (STFT) with 512-point window and 256-point hop size.
- **Acoustic Representation**: 128-dimensional vector combining sub-band energy statistics, zero-crossing rates, and high-frequency spectral ratios.

---

## 5. Train / Validation / Test Stratification

Data splitting was executed using stratified partitioning ([`ml/datasets/split_dataset.py`](file:///Users/divakarmaurya/nlp_project/ml/datasets/split_dataset.py)) to ensure identical class distributions across all three splits and prevent data leakage:

```text
Class Distribution Across Splits:
- Train Split:      112 REAL (50%) + 112 FAKE (50%) = 224 samples (70.0%)
- Validation Split:  24 REAL (50%) +  24 FAKE (50%) =  48 samples (15.0%)
- Test Split:        24 REAL (50%) +  24 FAKE (50%) =  48 samples (15.0%)
```

Validation script [`scripts/prepare_data.py`](file:///Users/divakarmaurya/nlp_project/scripts/prepare_data.py) verifies every file's integrity on disk before training or evaluation starts.
