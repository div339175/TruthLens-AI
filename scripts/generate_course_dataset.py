"""Course Dataset Generator for TruthLens AI (CSET 346 - NLP).

Generates a balanced dataset of 320 multimodal samples meeting course specifications:
- >= 300 labeled samples
- Balanced REAL (160) and FAKE (160)
- Multiple manipulation categories:
    * authentic (none)
    * deepfake_faceswap
    * diffusion_synthesis
    * spliced_banner
    * semantic_mismatch (cross-modal contradiction)
    * synthetic_speech
- Stratified 70/15/15 train/val/test splits
- Generates distinct visual image files in ml/data/samples/
- Creates ml/data/metadata/dataset.csv
"""

import os
import sys
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "ml" / "data" / "samples"
METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"

# Set deterministic seed for reproducibility
random.seed(42)
np.random.seed(42)

# Templates for Authentic Samples (REAL = 0)
REAL_DOMAINS = [
    {
        "domain": "official_press",
        "title": "Government Briefing",
        "bg_color": (30, 58, 95),
        "texts": [
            "Ministry of Health releases updated quarterly public health statistics and preventive guidelines.",
            "Department of Transportation announces federal infrastructure modernization grant allocations.",
            "Official statement from economic committee on inflation indices and fiscal monetary adjustments.",
            "Prime Minister delivers address on international trade cooperation and regional peace accords.",
            "National Weather Service confirms seasonal rainfall patterns across agricultural regions.",
            "Civil aviation authority publishes verified commercial airline safety inspection audits.",
            "Department of Education unveils digital literacy curriculum for secondary school students.",
            "Official press conference highlighting bilateral renewable energy agreements signed in summit.",
        ]
    },
    {
        "domain": "science_nature",
        "title": "Scientific Observation",
        "bg_color": (24, 76, 56),
        "texts": [
            "NASA satellite imagery documents polar ice shelf calving patterns over a five-year cycle.",
            "Peer-reviewed astronomical observatory captures high-resolution stellar corona eruption.",
            "Marine biological survey confirms coral reef restoration progress in protected marine park.",
            "Geological survey team records baseline seismic telemetry in tectonic subduction zone.",
            "Meteorological satellite tracking reveals cyclonic vortex dissipation over open ocean.",
            "Forestry conservation agency publishes biodiversity census across temperate biosphere reserves.",
            "Agricultural university research verifies drought-resistant grain yield improvements.",
            "Deep ocean submersible photographs hydrothermal vent communities along mid-ocean ridge.",
        ]
    },
    {
        "domain": "verified_journalism",
        "title": "Fact-Checked News",
        "bg_color": (45, 55, 72),
        "texts": [
            "Reuters fact-check confirms authentic archival footage from United Nations General Assembly.",
            "Accredited photojournalists document election polling precinct operations in capital district.",
            "Associated Press dispatch verifies opening of regional hospital surgical wing.",
            "Verified civic broadcast shows municipal council debate on public transit funding.",
            "Investigative report corroborates verified financial filings of public utility corporation.",
            "Archival television recording of international aerospace treaty ratification ceremony.",
            "Standard investigative documentation of commercial shipping traffic in deepwater canal.",
            "Local news organization covers charity marathon opening ceremony with verified attendee counts.",
        ]
    },
    {
        "domain": "academic_interview",
        "title": "Academic Interview",
        "bg_color": (52, 42, 75),
        "texts": [
            "University chancellor delivers commencement keynote address on artificial intelligence ethics.",
            "Nobel laureate in physics explains quantum entanglement experiments in televised symposium.",
            "Panel discussion among infectious disease specialists regarding public vaccine efficacy.",
            "Economics faculty seminar analyzes labor market elasticity during industrial transitions.",
            "Computer science symposium keynote on verifiable cryptographic privacy protocols.",
            "Distinguished scholar presents archival historical research on ancient Mediterranean trade.",
            "Biochemistry professor describes CRISPR gene therapy clinical trials in medical journal forum.",
            "Philosophy symposium debate on technological determinism and epistemic agency.",
        ]
    }
]

# Templates for Manipulated Samples (FAKE = 1)
FAKE_CATEGORIES = [
    {
        "category": "deepfake_faceswap",
        "title": "Deepfake Faceswap",
        "bg_color": (98, 30, 42),
        "artifact": "boundary_blur",
        "texts": [
            "Manipulated facial reenactment video shows prominent diplomat uttering unverified threats.",
            "Deepfake video swap depicts celebrity actor endorsing fraudulent cryptocurrency token scheme.",
            "Synthesized face-swap interview shows opposition politician making fabricated confessions.",
            "Altered televised debate video where speaker's facial geometry was replaced using deep learning.",
            "Facially reenacted video clip of corporate CEO announcing fictitious bankruptcy filing.",
            "Synthetic face-swapped footage falsely depicting international athlete in doping scandal.",
            "Manipulated speech video showing state governor declaring unauthorized state of emergency.",
            "Doctored press conference video featuring altered lip motion and facial identity swap.",
        ]
    },
    {
        "category": "diffusion_synthesis",
        "title": "Generative Diffusion AI",
        "bg_color": (85, 34, 85),
        "artifact": "high_frequency_noise",
        "texts": [
            "Latent diffusion model creates entirely fabricated portrait of non-existent whistleblower.",
            "AI-generated synthetic image depicting fictitious explosion near international embassy complex.",
            "Text-to-image synthesis produces fake historical photograph of secret military expedition.",
            "Generative model hallucinates photorealistic image of political figures at unverified meeting.",
            "Fully synthesized image of non-existent natural disaster flood published to deceive audiences.",
            "AI-generated composite depicting fabricated futuristic military prototype in desert testing.",
            "Diffusion-generated portrait exhibiting telltale asymmetric iris reflections and blurred hair boundaries.",
            "Synthetic crowd image generated with distorted peripheral hands and repeating facial patterns.",
        ]
    },
    {
        "category": "spliced_banner",
        "title": "Spliced Media & Tampering",
        "bg_color": (102, 51, 0),
        "artifact": "spliced_strip",
        "texts": [
            "Authentic television broadcast altered with spliced banner claiming nationwide bank collapse.",
            "Doctored newspaper front page with digitally inserted sensational headline and fabricated byline.",
            "Image showing military convoy altered by digitally cloning vehicles to exaggerate troop numbers.",
            "Spliced emergency alert banner falsely warning of water contamination in metropolitan area.",
            "Digitally edited scoreboard and crowd photo spliced to misrepresent stadium attendance figures.",
            "Doctored credential card and security badge composite image circulated on social messaging groups.",
            "Tampered pharmaceutical packaging image featuring altered expiration dates and fake approval seals.",
            "Spliced infographic with altered statistical charts designed to spread economic panic.",
        ]
    },
    {
        "category": "semantic_mismatch",
        "title": "Cross-Modal Contradiction",
        "bg_color": (70, 40, 20),
        "artifact": "authentic_visual_fake_caption",
        "texts": [
            "Stock photo of peaceful summer festival paired with false caption claiming violent street riots.",
            "Archival rocket launch from 2012 falsely captioned as new ballistic missile test in active conflict.",
            "Photo of standard wildfire drill captioned as uncontrolled metropolitan fire emergency.",
            "Ordinary pharmaceutical laboratory image captioned as secret bioweapon manufacturing facility.",
            "Archival photo of flooded village in 2005 repurposed as current consequence of recent dam breach.",
            "Standard military training exercise photo paired with caption claiming imminent border invasion.",
            "Peaceful protest photo captioned as armed insurrection storming parliament building.",
            "Satellite photo of solar farm development falsely described as clandestine missile silo installation.",
        ]
    },
    {
        "category": "synthetic_speech",
        "title": "Cloned Synthetic Audio",
        "bg_color": (80, 20, 60),
        "artifact": "spectral_anomaly",
        "texts": [
            "Cloned audio clip imitating election official claiming polling station closure in key district.",
            "Voice-cloned phone call impersonating executive officer authorizing fraudulent wire transfer.",
            "Synthetic audio recording falsely claiming military mobilization issued by defense secretary.",
            "AI voice synthesis replicating broadcast anchor's cadence to spread manufactured currency devaluation rumors.",
            "Deepfake audio clip of high-ranking diplomat ostensibly discussing confidential territorial negotiations.",
        ]
    }
]


def create_sample_image(sample_id: str, title: str, subtitle: str, bg_color: tuple, artifact: str, out_path: Path):
    """Generates an image sample with realistic geometric components and optional manipulation artifacts."""
    w, h = 320, 320
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Base design: Clean border, header banner, typography elements
    draw.rectangle([12, 12, w - 12, h - 12], outline=(220, 220, 230), width=2)
    draw.rectangle([12, 12, w - 12, 54], fill=(bg_color[0] + 25, bg_color[1] + 25, bg_color[2] + 25))

    draw.text((22, 22), title.upper(), fill=(255, 255, 255))
    draw.text((22, 38), f"TRUTHLENS VERIFIED CORPUS • ID: {sample_id}", fill=(180, 200, 220))

    # Center decorative frame representing video frame or portrait
    center_box = [40, 75, w - 40, 240]
    draw.rectangle(center_box, outline=(180, 190, 205), width=2, fill=(bg_color[0] - 10, bg_color[1] - 10, bg_color[2] - 10))

    # Abstract facial or scene geometry inside center box
    draw.ellipse([w // 2 - 45, 95, w // 2 + 45, 185], outline=(200, 210, 225), width=2)
    draw.arc([w // 2 - 25, 145, w // 2 + 25, 170], start=0, end=180, fill=(200, 210, 225), width=2)
    draw.line([w // 2 - 25, 125, w // 2 - 10, 125], fill=(240, 240, 250), width=2)
    draw.line([w // 2 + 10, 125, w // 2 + 25, 125], fill=(240, 240, 250), width=2)

    # Footer banner
    draw.rectangle([12, h - 45, w - 12, h - 12], fill=(20, 25, 35))
    short_caption = subtitle[:44] + "..." if len(subtitle) > 44 else subtitle
    draw.text((18, h - 35), short_caption, fill=(200, 210, 220))

    # Apply manipulation artifacts
    arr = np.array(img).astype(np.int16)

    if artifact == "high_frequency_noise":
        # Latent diffusion high frequency noise & color speckling
        noise = np.random.normal(0, 28, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    elif artifact == "boundary_blur":
        # Facial boundary blending artifact
        img = Image.fromarray(arr.astype(np.uint8))
        box = (w // 2 - 50, 90, w // 2 + 50, 190)
        region = img.crop(box)
        region = region.filter(ImageFilter.GaussianBlur(radius=2.5))
        img.paste(region, box)

    elif artifact == "spliced_strip":
        # Insert sharp mismatched spliced alert banner
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 195, w - 20, 230], fill=(210, 30, 30))
        draw.text((28, 205), "BREAKING NEWS ALERT [SPLICED]", fill=(255, 255, 255))

    elif artifact == "spectral_anomaly":
        # Audio-visual frequency artifact: horizontal scanlines across lower half
        arr = np.array(img)
        for row in range(160, 240, 4):
            arr[row, :, :] = np.clip(arr[row, :, :].astype(int) + 50, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    img.save(out_path, "PNG")


def generate_dataset():
    """Generates 320 balanced multimodal samples (160 REAL, 160 FAKE)."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    sample_counter = 1

    print("[INFO] Generating 160 REAL samples...")
    # Generate 160 REAL samples across 4 domains (40 each)
    for domain_info in REAL_DOMAINS:
        domain = domain_info["domain"]
        title = domain_info["title"]
        bg = domain_info["bg_color"]
        texts = domain_info["texts"]

        for i in range(40):
            sid = f"sample_{sample_counter:03d}"
            text_base = texts[i % len(texts)]
            # Add natural phrasing variations
            variations = [
                f"{text_base}",
                f"Official dispatch: {text_base}",
                f"Verified archival record: {text_base}",
                f"Published documentation: {text_base}",
                f"Accredited broadcast: {text_base}",
            ]
            text = variations[i % len(variations)]
            filename = f"real_{domain}_{i+1:02d}.png"
            img_path = SAMPLES_DIR / filename

            create_sample_image(
                sample_id=sid,
                title=title,
                subtitle=text,
                bg_color=bg,
                artifact="none",
                out_path=img_path,
            )

            records.append({
                "id": sid,
                "text": text,
                "image_path": f"samples/{filename}",
                "video_path": "",
                "audio_path": "",
                "label": "0",  # REAL
                "manipulation_type": "none",
                "source": f"course_curated_{domain}",
            })
            sample_counter += 1

    print("[INFO] Generating 160 FAKE samples...")
    # Generate 160 FAKE samples across 5 categories:
    # 35 deepfake_faceswap, 35 diffusion_synthesis, 35 spliced_banner, 35 semantic_mismatch, 20 synthetic_speech = 160
    fake_targets = [
        ("deepfake_faceswap", 35),
        ("diffusion_synthesis", 35),
        ("spliced_banner", 35),
        ("semantic_mismatch", 35),
        ("synthetic_speech", 20),
    ]

    for cat_name, count in fake_targets:
        cat_info = next(c for c in FAKE_CATEGORIES if c["category"] == cat_name)
        title = cat_info["title"]
        bg = cat_info["bg_color"]
        artifact = cat_info["artifact"]
        texts = cat_info["texts"]

        for i in range(count):
            sid = f"sample_{sample_counter:03d}"
            text_base = texts[i % len(texts)]
            variations = [
                f"{text_base}",
                f"Alert: {text_base}",
                f"Social media circulation: {text_base}",
                f"Unverified transmission: {text_base}",
                f"Reported incident: {text_base}",
            ]
            text = variations[i % len(variations)]
            filename = f"fake_{cat_name}_{i+1:02d}.png"
            img_path = SAMPLES_DIR / filename

            create_sample_image(
                sample_id=sid,
                title=title,
                subtitle=text,
                bg_color=bg,
                artifact=artifact,
                out_path=img_path,
            )

            records.append({
                "id": sid,
                "text": text,
                "image_path": f"samples/{filename}",
                "video_path": "",
                "audio_path": "",
                "label": "1",  # FAKE
                "manipulation_type": cat_name,
                "source": f"course_curated_{cat_name}",
            })
            sample_counter += 1

    # Stratified Split (70% Train = 224, 15% Val = 48, 15% Test = 48)
    real_records = [r for r in records if r["label"] == "0"]
    fake_records = [r for r in records if r["label"] == "1"]

    random.shuffle(real_records)
    random.shuffle(fake_records)

    # 160 REAL: 112 Train, 24 Val, 24 Test
    for idx, r in enumerate(real_records):
        if idx < 112:
            r["split"] = "train"
        elif idx < 112 + 24:
            r["split"] = "val"
        else:
            r["split"] = "test"

    # 160 FAKE: 112 Train, 24 Val, 24 Test
    for idx, r in enumerate(fake_records):
        if idx < 112:
            r["split"] = "train"
        elif idx < 112 + 24:
            r["split"] = "val"
        else:
            r["split"] = "test"

    all_records = real_records + fake_records
    # Sort deterministically by id
    all_records.sort(key=lambda x: int(x["id"].split("_")[1]))

    csv_path = METADATA_DIR / "dataset.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "text", "image_path", "video_path", "audio_path", "label", "manipulation_type", "source", "split"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    total_samples = len(all_records)
    train_count = sum(1 for r in all_records if r["split"] == "train")
    val_count = sum(1 for r in all_records if r["split"] == "val")
    test_count = sum(1 for r in all_records if r["split"] == "test")
    real_count = sum(1 for r in all_records if r["label"] == "0")
    fake_count = sum(1 for r in all_records if r["label"] == "1")

    print("\n" + "=" * 55)
    print("TRUTHLENS AI - COURSE DATASET SUMMARY")
    print("=" * 55)
    print(f"Total labeled samples: {total_samples} (Requirement: >= 300)")
    print(f"REAL samples:          {real_count} (50.0%)")
    print(f"FAKE samples:          {fake_count} (50.0%)")
    print(f"Train split:           {train_count} (70.0%)")
    print(f"Validation split:      {val_count} (15.0%)")
    print(f"Test split:            {test_count} (15.0%)")
    print(f"CSV Metadata path:     {csv_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    generate_dataset()
