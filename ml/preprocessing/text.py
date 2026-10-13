"""Text preprocessing pipeline for TruthLens AI."""

import re
from typing import Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import torch


class TextPreprocessor:
    """Handles text cleaning, tokenization, and tensor preparation."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def clean_text(self, text: str) -> str:
        """Standard cleaning: strip extra whitespace, normalize urls/mentions."""
        if not text:
            return ""
        text = text.strip()
        # Collapse multiple spaces and newlines
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Tokenize text into input_ids, attention_mask, etc."""
        cleaned = self.clean_text(text)
        return self._tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def decode_tokens(self, input_ids: torch.Tensor) -> list[str]:
        """Convert input_ids back to list of string tokens."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        return self._tokenizer.convert_ids_to_tokens(input_ids.tolist())
