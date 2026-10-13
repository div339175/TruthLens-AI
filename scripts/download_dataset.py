"""Dataset downloader, registrar, and sample development generator for TruthLens AI.

Supported candidate datasets:
1. FaceForensics++ (Requires formal academic agreement from TUM)
2. DFDC (Deepfake Detection Challenge - Kaggle / AWS)
3. FakeAVCeleb (Multimodal Audio-Visual Celeb dataset)
4. Fakeddit (Multimodal Reddit dataset with text + image claims)
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "ml" / "data" / "metadata"
SAMPLES_DIR = BASE_DIR / "ml" / "data" / "samples"

DATASET_GUIDES = {
    "faceforensics": """
FaceForensics++ (FF++)
----------------------
Licensing & Download:
1. Submit an academic request form at: https://github.com/ondyari/FaceForensics
2. TUM will email your unique download script (download_FaceForensics_compressed.py).
3. Download the Deepfakes / Face2Face / FaceSwap manipulations and original videos.
4. Mount or extract to: ml/data/raw/faceforensics/
5. Run: python scripts/prepare_data.py --dataset faceforensics --source ml/data/raw/faceforensics
""",
    "dfdc": """
Deepfake Detection Challenge (DFDC)
-----------------------------------
Licensing & Download:
1. Available on Kaggle: https://www.kaggle.com/c/deepfake-detection-challenge/data
2. Requires Kaggle API token (~/.kaggle/kaggle.json) or manual archive download (~470 GB full, 4 GB preview).
3. Place extracted video folders in: ml/data/raw/dfdc/
4. Run: python scripts/prepare_data.py --dataset dfdc --source ml/data/raw/dfdc
""",
    "fakeavceleb": """
FakeAVCeleb
-----------
Licensing & Download:
1. Video + Audio multimodal deepfake benchmark: https://github.com/DASH-Lab/FakeAVCeleb
2. Sign Google Form license on the repo to receive download links.
3. Extract files into: ml/data/raw/fakeavceleb/
4. Run: python scripts/prepare_data.py --dataset fakeavceleb --source ml/data/raw/fakeavceleb
""",
    "fakeddit": """
Fakeddit
--------
Licensing & Download:
1. 1M+ Multimodal Reddit posts: https://fakeddit.netlify.app/ / https://github.com/entitize/Fakeddit
2. Download TSV metadata and image archive (or scrape using Reddit IDs).
3. Extract into: ml/data/raw/fakeddit/
4. Run: python scripts/prepare_data.py --dataset fakeddit --source ml/data/raw/fakeddit
""",
}


def create_development_samples():
    """Generates a small, legitimate, clearly labeled development set for MVP demonstration."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Creating development sample images in ml/data/samples/...")

    # We generate clean visual assets with distinct visual artifacts for testing
    sample_items = [
        {
            "id": "sample_001",
            "filename": "real_press_conference.png",
            "title": "Press Briefing",
            "subtitle": "Authentic government press conference on renewable energy policy.",
            "color": (28, 54, 84),
            "noise": False,
            "label": "0",  # REAL
            "manipulation_type": "none",
            "source": "dev_sample",
            "split": "train",
        },
        {
            "id": "sample_002",
            "filename": "fake_celebrity_speech.png",
            "title": "Deepfake Speech",
            "subtitle": "AI-generated speech showing prominent actor endorsing unverified token.",
            "color": (94, 28, 38),
            "noise": True,
            "label": "1",  # FAKE
            "manipulation_type": "deepfake_faceswap",
            "source": "dev_sample",
            "split": "train",
        },
        {
            "id": "sample_003",
            "filename": "real_scientific_report.png",
            "title": "Satellite Observation",
            "subtitle": "Verified NASA satellite imagery documenting glacial ice shelf variations.",
            "color": (22, 68, 50),
            "noise": False,
            "label": "0",  # REAL
            "manipulation_type": "none",
            "source": "dev_sample",
            "split": "val",
        },
        {
            "id": "sample_004",
            "filename": "fake_breaking_news.png",
            "title": "Fabricated Incident",
            "subtitle": "Manipulated broadcast banner claiming emergency evacuation in capital.",
            "color": (88, 35, 20),
            "noise": True,
            "label": "1",  # FAKE
            "manipulation_type": "spliced_banner",
            "source": "dev_sample",
            "split": "val",
        },
        {
            "id": "sample_005",
            "filename": "real_wildlife_photo.png",
            "title": "Authentic Wildlife",
            "subtitle": "Natural geographic photograph of migration patterns in national sanctuary.",
            "color": (40, 60, 40),
            "noise": False,
            "label": "0",  # REAL
            "manipulation_type": "none",
            "source": "dev_sample",
            "split": "test",
        },
        {
            "id": "sample_006",
            "filename": "fake_synthetic_portrait.png",
            "title": "Generative Face Portrait",
            "subtitle": "Diffusion model generated composite portrait with asymmetric ocular reflections.",
            "color": (70, 20, 60),
            "noise": True,
            "label": "1",  # FAKE
            "manipulation_type": "diffusion_synthesis",
            "source": "dev_sample",
            "split": "test",
        },
    ]

    import numpy as np

    csv_rows = []
    for item in sample_items:
        img_path = SAMPLES_DIR / item["filename"]
        # Create image
        w, h = 320, 320
        img = Image.new("RGB", (w, h), color=item["color"])
        draw = ImageDraw.Draw(img)

        # Draw decorative geometric elements
        draw.rectangle([20, 20, w - 20, h - 20], outline=(200, 200, 200), width=2)
        draw.text((30, 40), item["title"], fill=(255, 255, 255))
        draw.text((30, 70), f"ID: {item['id']}", fill=(180, 180, 180))
        draw.text((30, 95), f"TruthLens Dev Sample", fill=(140, 180, 220))

        if item["noise"]:
            # Add synthetic manipulation artifacts pattern
            arr = np.array(img)
            noise = np.random.randint(-40, 40, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        img.save(img_path)
        print(f"  [+] Saved sample: {img_path.name}")

        rel_img_path = f"samples/{item['filename']}"
        csv_rows.append({
            "id": item["id"],
            "text": item["subtitle"],
            "image_path": rel_img_path,
            "video_path": "",
            "audio_path": "",
            "label": item["label"],
            "manipulation_type": item["manipulation_type"],
            "source": item["source"],
            "split": item["split"],
        })

    # Write initial dataset.csv
    import csv
    csv_file = METADATA_DIR / "dataset.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id", "text", "image_path", "video_path", "audio_path",
            "label", "manipulation_type", "source", "split"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[INFO] Initialized verified development dataset at: {csv_file}")
    print(f"[INFO] Total verified samples: {len(csv_rows)}")


def main():
    parser = argparse.ArgumentParser(description="TruthLens AI Dataset Manager")
    parser.add_argument(
        "--dataset",
        choices=["faceforensics", "dfdc", "fakeavceleb", "fakeddit"],
        help="Target benchmark dataset name",
    )
    parser.add_argument("--source", type=str, help="Local directory path containing downloaded dataset")
    parser.add_argument("--sample", action="store_true", help="Generate small local development sample set")
    parser.add_argument("--info", action="store_true", help="Show instructions for all supported academic datasets")

    args = parser.parse_args()

    if args.info or (not args.dataset and not args.sample):
        print("=" * 60)
        print("TruthLens AI - Multimodal Dataset Information")
        print("=" * 60)
        for key, guide in DATASET_GUIDES.items():
            print(guide)
        print("To generate a local development sample set now, run:")
        print("  python scripts/download_dataset.py --sample")
        return

    if args.sample:
        create_development_samples()
        return

    if args.dataset:
        print(DATASET_GUIDES.get(args.dataset, "Dataset not recognized."))
        if args.source:
            print(f"[INFO] Path provided: {args.source}")
            print(f"Run 'python scripts/prepare_data.py --dataset {args.dataset} --source {args.source}' to parse.")


if __name__ == "__main__":
    main()
