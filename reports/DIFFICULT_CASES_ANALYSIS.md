# TruthLens AI — Qualitative Error Analysis of 15 Difficult Cases

**Course Requirement (CSET 346 - NLP)**: Detailed case study across semantic mismatch, identity manipulation, diffusion synthesis, spliced media, and synthetic voice cues.

- **Total Challenging Cases Analyzed**: 15
- **Direct Concordance with Ground Truth**: 11/15 (73.3%)
- **Uncertainty Policy Invocations**: 0/15 (0.0%)

| Case ID | Category | Title | Ground Truth | Verdict | Conf | Status | Failure / Robustness Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CASE_01** | `semantic_mismatch` | Peaceful Festival Miscaptioned as Street Riot | **FAKE** | **REAL** | 74.5% | MISCLASSIFIED | Single-modality bias dominated prediction. |
| **CASE_02** | `semantic_mismatch` | Historical Archival Space Launch Claimed as New Ballistic Test | **FAKE** | **REAL** | 84.1% | MISCLASSIFIED | Single-modality bias dominated prediction. |
| **CASE_03** | `semantic_mismatch` | Natural Wildfire Drill Repurposed as Catastrophic Disaster | **FAKE** | **REAL** | 67.5% | MISCLASSIFIED | Single-modality bias dominated prediction. |
| **CASE_04** | `semantic_mismatch` | Commercial Hospital Wing Opening Claimed as Secret Military Biolab | **FAKE** | **REAL** | 87.6% | MISCLASSIFIED | Single-modality bias dominated prediction. |
| **CASE_05** | `deepfake_faceswap` | Subtle Diplomatic Face-Swap with Blurred Boundary | **FAKE** | **FAKE** | 93.3% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_06** | `deepfake_faceswap` | Celebrity Deepfake Financial Endorsement | **FAKE** | **FAKE** | 92.4% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_07** | `deepfake_faceswap` | Synthesized Politician Debate Reenactment | **FAKE** | **FAKE** | 92.5% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_08** | `diffusion_synthesis` | Photorealistic AI Portrait with Asymmetric Ocular Reflections | **FAKE** | **FAKE** | 88.8% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_09** | `diffusion_synthesis` | Synthesized Non-Existent Disaster Scene | **FAKE** | **FAKE** | 91.3% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_10** | `diffusion_synthesis` | AI Composite Military Hardware Prototype | **FAKE** | **FAKE** | 90.2% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_11** | `spliced_banner` | Authentic Television Broadcast with Spliced Bank Run Banner | **FAKE** | **FAKE** | 93.5% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_12** | `spliced_banner` | Cloned Military Convoy Vehicle Duplication | **FAKE** | **FAKE** | 92.9% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_13** | `synthetic_speech` | Voice-Cloned Election Official Phone Call | **FAKE** | **FAKE** | 92.3% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_14** | `synthetic_speech` | Synthesized Executive Authorization Call | **FAKE** | **FAKE** | 91.8% | CORRECT | Accurately resolved multimodal cues. |
| **CASE_15** | `authentic_edge_case` | Genuine Scientific Glacial Ice Shelf Telemetry | **REAL** | **REAL** | 92.0% | CORRECT | Accurately resolved multimodal cues. |


## Detailed Per-Case Evidence Breakdown

### CASE_01: Peaceful Festival Miscaptioned as Street Riot
- **Category**: `semantic_mismatch`
- **Text Claim**: *"Breaking news: violent riots and civil unrest break out in downtown capital, teargas deployed."*
- **Difficulty Rationale**: The visual modality is authentic, but the textual claim describes violent conflict. Detectors relying only on visual artifacts would classify it as REAL.
- **Model Output**: Predicted **REAL** with **74.5%** confidence (Real Prob: 0.745, Fake Prob: 0.255).
- **Explanation**: The content was verified as REAL with 74.5% model confidence. Analysis across text, visual demonstrates coherence with authentic reference distributions.

### CASE_02: Historical Archival Space Launch Claimed as New Ballistic Test
- **Category**: `semantic_mismatch`
- **Text Claim**: *"Unverified surveillance footage captures secret intercontinental ballistic missile launch test today."*
- **Difficulty Rationale**: Archival footage is completely unmanipulated, but out-of-context caption fabricates modern active warfare.
- **Model Output**: Predicted **REAL** with **84.1%** confidence (Real Prob: 0.841, Fake Prob: 0.159).
- **Explanation**: The content was verified as REAL with 84.1% model confidence. Analysis across text, visual demonstrates coherence with authentic reference distributions.

### CASE_03: Natural Wildfire Drill Repurposed as Catastrophic Disaster
- **Category**: `semantic_mismatch`
- **Text Claim**: *"Emergency declared as uncontrolled chemical fire engulfs municipal residential sectors."*
- **Difficulty Rationale**: Standard training drill photos reused with catastrophic panic-inducing headlines.
- **Model Output**: Predicted **REAL** with **67.5%** confidence (Real Prob: 0.675, Fake Prob: 0.325).
- **Explanation**: The content was verified as REAL with 67.5% model confidence. Analysis across text, visual demonstrates coherence with authentic reference distributions.

### CASE_04: Commercial Hospital Wing Opening Claimed as Secret Military Biolab
- **Category**: `semantic_mismatch`
- **Text Claim**: *"Leaked photograph exposes clandestine bioweapon research laboratory operating under commercial cover."*
- **Difficulty Rationale**: Authentic medical facility imagery coupled with fabricated biosecurity conspiracy claims.
- **Model Output**: Predicted **REAL** with **87.6%** confidence (Real Prob: 0.876, Fake Prob: 0.124).
- **Explanation**: The content was verified as REAL with 87.6% model confidence. Analysis across text, visual demonstrates coherence with authentic reference distributions.

### CASE_05: Subtle Diplomatic Face-Swap with Blurred Boundary
- **Category**: `deepfake_faceswap`
- **Text Claim**: *"Manipulated facial reenactment video shows prominent diplomat uttering unverified threats."*
- **Difficulty Rationale**: Facial boundaries blended smoothly, requiring fine-grained spatial feature extraction to localize manipulation margins.
- **Model Output**: Predicted **FAKE** with **93.3%** confidence (Real Prob: 0.067, Fake Prob: 0.933).
- **Explanation**: The content was classified as FAKE with 93.3% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_06: Celebrity Deepfake Financial Endorsement
- **Category**: `deepfake_faceswap`
- **Text Claim**: *"Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme."*
- **Difficulty Rationale**: High-fidelity generative faceswap where lighting matches background context reasonably well.
- **Model Output**: Predicted **FAKE** with **92.4%** confidence (Real Prob: 0.076, Fake Prob: 0.924).
- **Explanation**: The content was classified as FAKE with 92.4% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_07: Synthesized Politician Debate Reenactment
- **Category**: `deepfake_faceswap`
- **Text Claim**: *"Synthesized face-swap interview shows opposition politician making fabricated confessions."*
- **Difficulty Rationale**: Subtle expression manipulation where audio and video must be evaluated jointly for semantic coherence.
- **Model Output**: Predicted **FAKE** with **92.5%** confidence (Real Prob: 0.075, Fake Prob: 0.925).
- **Explanation**: The content was classified as FAKE with 92.5% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_08: Photorealistic AI Portrait with Asymmetric Ocular Reflections
- **Category**: `diffusion_synthesis`
- **Text Claim**: *"Diffusion-generated portrait exhibiting telltale asymmetric iris reflections and blurred hair boundaries."*
- **Difficulty Rationale**: State-of-the-art diffusion models produce convincing skin textures, leaving only subtle corneal reflection asymmetries.
- **Model Output**: Predicted **FAKE** with **88.8%** confidence (Real Prob: 0.112, Fake Prob: 0.888).
- **Explanation**: The content was classified as FAKE with 88.8% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_09: Synthesized Non-Existent Disaster Scene
- **Category**: `diffusion_synthesis`
- **Text Claim**: *"Fully synthesized image of non-existent natural disaster flood published to deceive audiences."*
- **Difficulty Rationale**: Generative synthetic disaster images containing realistic lighting and water reflections without human faces.
- **Model Output**: Predicted **FAKE** with **91.3%** confidence (Real Prob: 0.087, Fake Prob: 0.913).
- **Explanation**: The content was classified as FAKE with 91.3% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_10: AI Composite Military Hardware Prototype
- **Category**: `diffusion_synthesis`
- **Text Claim**: *"AI-generated composite depicting fabricated futuristic military prototype in desert testing."*
- **Difficulty Rationale**: Technical object rendering where background is convincing and subject lacks biological facial cues.
- **Model Output**: Predicted **FAKE** with **90.2%** confidence (Real Prob: 0.098, Fake Prob: 0.902).
- **Explanation**: The content was classified as FAKE with 90.2% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_11: Authentic Television Broadcast with Spliced Bank Run Banner
- **Category**: `spliced_banner`
- **Text Claim**: *"Authentic television broadcast altered with spliced banner claiming nationwide bank collapse."*
- **Difficulty Rationale**: 90% of the image is genuine news anchor footage, but the lower third banner text is forged.
- **Model Output**: Predicted **FAKE** with **93.5%** confidence (Real Prob: 0.065, Fake Prob: 0.935).
- **Explanation**: The content was classified as FAKE with 93.5% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_12: Cloned Military Convoy Vehicle Duplication
- **Category**: `spliced_banner`
- **Text Claim**: *"Image showing military convoy altered by digitally cloning vehicles to exaggerate troop numbers."*
- **Difficulty Rationale**: Copy-move forgery within the same image, preserving natural lighting, camera grain, and color temperature.
- **Model Output**: Predicted **FAKE** with **92.9%** confidence (Real Prob: 0.071, Fake Prob: 0.929).
- **Explanation**: The content was classified as FAKE with 92.9% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_13: Voice-Cloned Election Official Phone Call
- **Category**: `synthetic_speech`
- **Text Claim**: *"Cloned audio clip imitating election official claiming polling station closure in key district."*
- **Difficulty Rationale**: Neural vocoder audio clones imitate vocal timbres with minimal acoustic artifacts, deceiving unimodal listeners.
- **Model Output**: Predicted **FAKE** with **92.3%** confidence (Real Prob: 0.077, Fake Prob: 0.923).
- **Explanation**: The content was classified as FAKE with 92.3% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_14: Synthesized Executive Authorization Call
- **Category**: `synthetic_speech`
- **Text Claim**: *"Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer."*
- **Difficulty Rationale**: Business context with plausible formal diction; requires cross-checking metadata and spectral anomalies.
- **Model Output**: Predicted **FAKE** with **91.8%** confidence (Real Prob: 0.082, Fake Prob: 0.918).
- **Explanation**: The content was classified as FAKE with 91.8% model confidence. Features extracted across text, visual exhibit synthetic characteristics consistent with generative or spliced manipulations.

### CASE_15: Genuine Scientific Glacial Ice Shelf Telemetry
- **Category**: `authentic_edge_case`
- **Text Claim**: *"Verified NASA satellite imagery documenting glacial ice shelf variations over five-year observation cycle."*
- **Difficulty Rationale**: Complex geometric patterns and irregular natural satellite textures that uncalibrated detectors frequently false-positive as synthetic noise.
- **Model Output**: Predicted **REAL** with **92.0%** confidence (Real Prob: 0.920, Fake Prob: 0.080).
- **Explanation**: The content was verified as REAL with 92.0% model confidence. Analysis across text, visual demonstrates coherence with authentic reference distributions.

