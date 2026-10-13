"""Preprocessing service providing lightweight diagnostics and validation."""

from pathlib import Path
from typing import Dict, Any
from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor


class PreprocessingService:
    def __init__(self):
        self.text_prep = TextPreprocessor()
        self.image_prep = ImagePreprocessor()

    def inspect_text(self, text: str) -> Dict[str, Any]:
        cleaned = self.text_prep.clean_text(text)
        tokenized = self.text_prep.tokenize(text)
        tokens = self.text_prep.decode_tokens(tokenized["input_ids"])
        non_pad = [t for t in tokens if t != "[PAD]"]
        return {
            "cleaned_text": cleaned,
            "raw_length": len(text),
            "tokens_count": len(non_pad),
            "tokens_sample": non_pad[:15],
        }

    def inspect_image(self, image_path: Path) -> Dict[str, Any]:
        img = self.image_prep.load_image(image_path)
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format or "RGB",
            "mode": img.mode,
        }
