"""Service layer orchestrating verification requests and model introspection."""

from pathlib import Path
from typing import Optional, Dict, Any
from backend.app.config import settings
from backend.app.schemas.verification import VerificationReport, ModelInfoResponse
from ml.inference.predictor import TruthLensPredictor


class VerificationService:
    def __init__(self):
        self.predictor = TruthLensPredictor.get_instance()

    def verify_text(self, text: str) -> VerificationReport:
        return self.predictor.predict_multimodal(text=text)

    def verify_image(self, image_path: Path) -> VerificationReport:
        return self.predictor.predict_multimodal(image_source=image_path)

    def verify_video(self, video_path: Path) -> VerificationReport:
        return self.predictor.predict_multimodal(video_path=video_path)

    def verify_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
    ) -> VerificationReport:
        return self.predictor.predict_multimodal(
            text=text,
            image_source=image_path,
            video_path=video_path,
        )

    def get_model_info(self) -> ModelInfoResponse:
        checkpoint_path = settings.MODELS_DIR / "multimodal_baseline.pt"
        return ModelInfoResponse(
            project_name=settings.PROJECT_NAME,
            version=settings.VERSION,
            execution_mode=settings.MODEL_MODE,
            text_backbone=settings.TEXT_MODEL_NAME,
            vision_backbone=settings.VISION_MODEL_NAME,
            fusion_architecture=settings.FUSION_TYPE,
            device=str(self.predictor.device),
            uncertainty_threshold=settings.UNCERTAINTY_THRESHOLD,
            is_trained_checkpoint=self.predictor.is_checkpoint_loaded,
            checkpoint_path=str(checkpoint_path) if self.predictor.is_checkpoint_loaded else None,
            architecture_summary={
                "text_embed_dim": settings.TEXT_EMBED_DIM,
                "vision_embed_dim": settings.VISION_EMBED_DIM,
                "projection_dim": settings.PROJECTION_DIM,
                "video_sample_frames": settings.VIDEO_SAMPLE_FRAMES,
                "num_classes": 2,
                "decision_classes": ["REAL", "FAKE", "UNCERTAIN"],
            },
        )
