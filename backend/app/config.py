"""Configuration module for TruthLens AI Backend."""

import os
from pathlib import Path
from pydantic import BaseModel
import torch

# Prevent slow Hugging Face network retries when local cached weights exist
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "TruthLens AI"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Multimodal NLP for Deepfake Detection and Explainable Content Verification"

    # Execution Mode: 'real' (actual deep learning model pipeline) or 'demo' (explicit mock)
    MODEL_MODE: str = os.getenv("MODEL_MODE", "real").lower()

    # Hardware device
    DEVICE: str = os.getenv("DEVICE", "auto")

    # NLP & Vision model configurations
    TEXT_MODEL_NAME: str = os.getenv("TEXT_MODEL_NAME", "distilbert-base-uncased")
    VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "resnet18")

    # Dimensions
    TEXT_EMBED_DIM: int = int(os.getenv("TEXT_EMBED_DIM", "768"))
    VISION_EMBED_DIM: int = int(os.getenv("VISION_EMBED_DIM", "512"))
    PROJECTION_DIM: int = int(os.getenv("PROJECTION_DIM", "256"))

    # Multimodal fusion architecture: 'cross_attention' or 'concat'
    FUSION_TYPE: str = os.getenv("FUSION_TYPE", "cross_attention")

    # Uncertainty decision threshold
    # In binary REAL/FAKE, if max(real_prob, fake_prob) < UNCERTAINTY_THRESHOLD, class is UNCERTAIN
    UNCERTAINTY_THRESHOLD: float = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.65"))

    # Video uniform sampling
    VIDEO_SAMPLE_FRAMES: int = int(os.getenv("VIDEO_SAMPLE_FRAMES", "8"))

    # Data paths
    DATA_DIR: Path = BASE_DIR / "ml" / "data"
    METADATA_PATH: Path = BASE_DIR / "ml" / "data" / "metadata" / "dataset.csv"
    MODELS_DIR: Path = BASE_DIR / "models"
    UPLOAD_TEMP_DIR: Path = BASE_DIR / "backend" / "app" / "uploads"
    MEDIA_STORAGE_DIR: Path = BASE_DIR / "backend" / "app" / "storage" / "media"
    DATABASE_PATH: Path = BASE_DIR / "backend" / "app" / "truthlens.db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'backend' / 'app' / 'truthlens.db'}")

    def get_torch_device(self) -> torch.device:
        """Resolve torch device based on config and platform availability."""
        if self.DEVICE == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        elif self.DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif self.DEVICE == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


settings = Settings()
# Ensure directories exist
settings.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
settings.MEDIA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
