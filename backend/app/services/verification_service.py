"""Service layer orchestrating verification requests, model inference, and persistence."""

from pathlib import Path
from typing import Optional, Dict, Any
from backend.app.config import settings
from backend.app.schemas.verification import VerificationReport, ModelInfoResponse
from backend.app.services.history_service import HistoryService
from ml.inference.predictor import TruthLensPredictor


class VerificationService:
    def __init__(self):
        self.predictor = TruthLensPredictor.get_instance()
        self.history_service = HistoryService()

    def verify_text(self, text: str, case_name: Optional[str] = None) -> VerificationReport:
        report = self.predictor.predict_multimodal(text=text)
        self.history_service.save_verification(
            report=report,
            case_name=case_name,
            text_snippet=text,
        )
        return report

    def verify_image(self, image_path: Path, case_name: Optional[str] = None) -> VerificationReport:
        report = self.predictor.predict_multimodal(image_source=image_path)
        self.history_service.save_verification(
            report=report,
            case_name=case_name,
            image_ref=str(image_path),
        )
        return report

    def verify_video(self, video_path: Path, case_name: Optional[str] = None) -> VerificationReport:
        report = self.predictor.predict_multimodal(video_path=video_path)
        self.history_service.save_verification(
            report=report,
            case_name=case_name,
            video_ref=str(video_path),
        )
        return report

    def verify_audio(self, audio_path: Path, case_name: Optional[str] = None) -> VerificationReport:
        report = self.predictor.predict_multimodal(audio_path=audio_path)
        self.history_service.save_verification(
            report=report,
            case_name=case_name,
            audio_ref=str(audio_path),
        )
        return report

    def verify_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        audio_path: Optional[Path] = None,
        case_name: Optional[str] = None,
    ) -> VerificationReport:
        report = self.predictor.predict_multimodal(
            text=text,
            image_source=image_path,
            video_path=video_path,
            audio_path=audio_path,
        )
        self.history_service.save_verification(
            report=report,
            case_name=case_name,
            text_snippet=text,
            image_ref=str(image_path) if image_path else None,
            video_ref=str(video_path) if video_path else None,
            audio_ref=str(audio_path) if audio_path else None,
        )
        return report

    def get_model_info(self) -> ModelInfoResponse:
        checkpoint_path = self.predictor.loaded_checkpoint_path or (settings.MODELS_DIR / "multimodal_baseline.pt")
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
                "calibration_temperature": getattr(self.predictor, "temperature", 1.0),
                "num_classes": 2,
                "decision_classes": ["REAL", "FAKE", "UNCERTAIN"],
            },
        )
