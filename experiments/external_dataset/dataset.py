"""PyTorch External Benchmark Dataset loader for TruthLens AI (Phase 5).

Supports loading:
1. Fakeddit text+image benchmark (source_dataset='fakeddit')
2. Deepfake Audio Detection benchmark (source_dataset='deepfake_audio_v4')
3. Generative Video Deepfake benchmark (source_dataset='generative_video_benchmark')
4. Semantic Mismatch 4-condition benchmark (source_dataset='semantic_mismatch_suite')
"""

import os
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from ml.preprocessing.audio import AudioPreprocessor
from ml.preprocessing.video import VideoPreprocessor


class TruthLensExternalDataset(Dataset):
    """PyTorch Dataset loading samples from external_dataset_manifest.csv."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        source_dataset: Optional[str] = None,
        condition: Optional[str] = None,
        text_model_name: str = "distilbert-base-uncased",
        image_size: int = 224,
        max_text_len: int = 128,
    ):
        self.manifest_path = Path(manifest_path)
        self.source_dataset = source_dataset
        self.condition = condition
        self.base_dir = self.manifest_path.resolve().parent.parent.parent  # project root

        self.text_preprocessor = TextPreprocessor(model_name=text_model_name, max_length=max_text_len)
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.audio_preprocessor = AudioPreprocessor()
        self.video_preprocessor = VideoPreprocessor(sample_frames=8, image_size=image_size)

        self.records: List[Dict[str, Any]] = []
        self._load_records()

    def _load_records(self):
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_ds = str(row.get("source_dataset", "")).strip()
                if self.source_dataset and src_ds != self.source_dataset:
                    continue

                src_id = str(row.get("source_id", "")).strip()
                if self.condition and self.condition not in src_id:
                    continue

                label_raw = str(row.get("label", "0")).strip()
                try:
                    label_int = int(label_raw)
                except ValueError:
                    label_int = 1 if label_raw.upper() == "FAKE" else 0

                # Resolve file paths relative to project root
                img_path = str(row.get("image_path", "")).strip()
                if img_path and not os.path.isabs(img_path):
                    img_path = str(self.base_dir / img_path)

                aud_path = str(row.get("audio_path", "")).strip()
                if aud_path and not os.path.isabs(aud_path):
                    aud_path = str(self.base_dir / aud_path)

                vid_path = str(row.get("video_path", "")).strip()
                if vid_path and not os.path.isabs(vid_path):
                    vid_path = str(self.base_dir / vid_path)

                self.records.append({
                    "sample_id": row.get("sample_id", ""),
                    "source_dataset": src_ds,
                    "label": label_int,
                    "text": str(row.get("text", "")).strip(),
                    "image_path": img_path,
                    "video_path": vid_path,
                    "audio_path": aud_path,
                    "split": row.get("split", "external_test"),
                    "source_id": src_id,
                    "manipulation_type": row.get("manipulation_type", "none"),
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        text = item["text"]
        img_path = item["image_path"]
        aud_path = item["audio_path"]
        vid_path = item["video_path"]
        label = item["label"]

        # Text encoding
        if text:
            encoded_text = self.text_preprocessor.tokenize(text)
            input_ids = encoded_text["input_ids"].squeeze(0)
            attention_mask = encoded_text["attention_mask"].squeeze(0)
            has_text = True
        else:
            # Neutral empty string fallback
            encoded_text = self.text_preprocessor.tokenize("")
            input_ids = encoded_text["input_ids"].squeeze(0)
            attention_mask = encoded_text["attention_mask"].squeeze(0)
            has_text = False

        # Image encoding
        has_image = False
        if img_path and os.path.exists(img_path):
            try:
                img_tensor = self.image_preprocessor.preprocess(img_path)
                has_image = True
            except Exception:
                blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
                img_tensor = self.image_preprocessor.preprocess(blank)
        else:
            blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
            img_tensor = self.image_preprocessor.preprocess(blank)

        # Audio feature extraction (128-D)
        has_audio = False
        if aud_path and os.path.exists(aud_path):
            try:
                audio_res = self.audio_preprocessor.extract_features(aud_path)
                audio_tensor = audio_res["feature_tensor"]
                has_audio = True
            except Exception:
                audio_tensor = torch.zeros(128, dtype=torch.float32)
        else:
            audio_tensor = torch.zeros(128, dtype=torch.float32)

        # Video frame extraction [8, 3, 224, 224]
        has_video = False
        if vid_path and os.path.exists(vid_path):
            try:
                vid_frames, _, _ = self.video_preprocessor.preprocess_video(vid_path)
                vid_tensor = vid_frames  # [8, 3, 224, 224]
                has_video = True
            except Exception:
                vid_tensor = torch.zeros((8, 3, 224, 224), dtype=torch.float32)
        else:
            vid_tensor = torch.zeros((8, 3, 224, 224), dtype=torch.float32)

        return {
            "sample_id": item["sample_id"],
            "label": torch.tensor(label, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "has_text": torch.tensor(1.0 if has_text else 0.0, dtype=torch.float32),
            "image": img_tensor,
            "has_image": torch.tensor(1.0 if has_image else 0.0, dtype=torch.float32),
            "audio": audio_tensor,
            "has_audio": torch.tensor(1.0 if has_audio else 0.0, dtype=torch.float32),
            "video": vid_tensor,
            "has_video": torch.tensor(1.0 if has_video else 0.0, dtype=torch.float32),
            "raw_text": text,
            "source_dataset": item["source_dataset"],
            "manipulation_type": item["manipulation_type"],
            "image_path": img_path,
            "audio_path": aud_path,
            "video_path": vid_path,
        }
