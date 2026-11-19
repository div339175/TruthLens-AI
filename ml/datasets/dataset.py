"""PyTorch Multimodal Dataset implementation for TruthLens AI."""

import os
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from ml.preprocessing.audio import AudioPreprocessor


class TruthLensMultimodalDataset(Dataset):
    """PyTorch Dataset loading paired Text, Image, Video, and Audio data from metadata CSV."""

    def __init__(
        self,
        metadata_csv: Path,
        split: Optional[str] = None,
        text_model_name: str = "distilbert-base-uncased",
        image_size: int = 224,
        max_text_len: int = 128,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.text_preprocessor = TextPreprocessor(model_name=text_model_name, max_length=max_text_len)
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.audio_preprocessor = AudioPreprocessor()

        self.records: List[Dict[str, Any]] = []
        self._load_records()

    def _load_records(self):
        if not self.metadata_csv.exists():
            return

        base_dir = self.metadata_csv.parent.parent  # ml/data

        with open(self.metadata_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = str(row.get("split", "")).lower().strip()
                if self.split and row_split != self.split.lower():
                    continue

                label_raw = str(row.get("label", "0")).strip()
                try:
                    label_int = int(label_raw)
                except ValueError:
                    label_int = 1 if label_raw.upper() == "FAKE" else 0

                # Resolve relative image paths
                img_path = row.get("image_path", "")
                if img_path and not os.path.isabs(img_path):
                    img_path = str(base_dir / img_path)

                # Resolve relative audio paths
                aud_path = row.get("audio_path", "")
                if aud_path and not os.path.isabs(aud_path):
                    aud_path = str(base_dir / aud_path)

                self.records.append({
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "image_path": img_path,
                    "video_path": row.get("video_path", ""),
                    "audio_path": aud_path,
                    "label": label_int,
                    "manipulation_type": row.get("manipulation_type", "unknown"),
                    "source": row.get("source", "unknown"),
                    "split": row_split,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        text = item["text"]
        img_path = item["image_path"]
        aud_path = item.get("audio_path", "")
        label = item["label"]

        # Text encoding
        encoded_text = self.text_preprocessor.tokenize(text)
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        # Image encoding (or blank canvas fallback if missing)
        if img_path and os.path.exists(img_path):
            img_tensor = self.image_preprocessor.preprocess(img_path)
        else:
            # Neutral blank canvas tensor [3, 224, 224]
            blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
            img_tensor = self.image_preprocessor.preprocess(blank)

        # Audio feature extraction (or neutral zeros fallback [128])
        has_audio = False
        if aud_path and os.path.exists(aud_path):
            try:
                audio_res = self.audio_preprocessor.extract_features(aud_path)
                audio_tensor = audio_res["feature_tensor"]  # [128]
                has_audio = True
            except Exception:
                audio_tensor = torch.zeros(128, dtype=torch.float32)
        else:
            audio_tensor = torch.zeros(128, dtype=torch.float32)

        return {
            "id": item["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img_tensor,
            "audio": audio_tensor,
            "has_audio": torch.tensor(1.0 if has_audio else 0.0, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "raw_text": text,
        }
