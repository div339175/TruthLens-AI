"""Predictor module managing model loading, pipeline inference, and explainability generation."""

import logging
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import torch

from backend.app.config import settings
from backend.app.schemas.verification import VerificationReport
from ml.models.multimodal_model import MultimodalVerificationModel
from ml.models.classifier import DecisionPolicy
from ml.preprocessing.text import TextPreprocessor
from ml.preprocessing.image import ImagePreprocessor
from ml.preprocessing.video import VideoPreprocessor
from ml.explainability.text_explanation import TextExplainer
from ml.explainability.visual_explanation import VisualGradCAM
from ml.explainability.evidence import EvidenceEngine

logger = logging.getLogger("truthlens.predictor")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class TruthLensPredictor:
    """End-to-end inference engine for TruthLens AI."""

    _instance = None

    def __init__(self):
        self.device = settings.get_torch_device()
        logger.info(f"Using compute device: {self.device} (configured: {settings.DEVICE})")

        self.mode = settings.MODEL_MODE
        self.decision_policy = DecisionPolicy(uncertainty_threshold=settings.UNCERTAINTY_THRESHOLD)
        self.evidence_engine = EvidenceEngine()

        logger.info(f"Loading Text Preprocessor ({settings.TEXT_MODEL_NAME})...")
        self.text_preprocessor = TextPreprocessor(model_name=settings.TEXT_MODEL_NAME)

        logger.info(f"Loading Image Preprocessor (224x224)...")
        self.image_preprocessor = ImagePreprocessor(image_size=224)

        logger.info(f"Loading Video Preprocessor (sample_frames={settings.VIDEO_SAMPLE_FRAMES})...")
        self.video_preprocessor = VideoPreprocessor(sample_frames=settings.VIDEO_SAMPLE_FRAMES)

        logger.info(f"Initializing Multimodal Model (fusion={settings.FUSION_TYPE})...")
        self.model = MultimodalVerificationModel(
            text_model_name=settings.TEXT_MODEL_NAME,
            vision_model_name=settings.VISION_MODEL_NAME,
            projection_dim=settings.PROJECTION_DIM,
            fusion_type=settings.FUSION_TYPE,
            pretrained_vision=True,
        ).to(self.device)

        # Check if saved checkpoint exists
        checkpoint_path = settings.MODELS_DIR / "multimodal_baseline.pt"
        self.is_checkpoint_loaded = False
        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.is_checkpoint_loaded = True
                logger.info(f"Loaded trained checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        self.model.eval()

        self.text_explainer = TextExplainer(self.text_preprocessor)
        self.grad_cam = VisualGradCAM(self.model)

    @classmethod
    def get_instance(cls):
        """Singleton accessor to prevent reloading heavy models."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_multimodal(
        self,
        text: Optional[str] = None,
        image_source: Optional[Union[str, Path, Image.Image]] = None,
        video_path: Optional[Union[str, Path]] = None,
    ) -> VerificationReport:
        """Run multimodal inference across text, image, and/or video."""
        if self.mode == "demo":
            return self._demo_response(text=text, has_image=image_source is not None, has_video=video_path is not None)

        modalities: List[str] = []
        input_ids = None
        attention_mask = None
        image_tensor = None
        pil_image = None
        video_meta = None

        # 1. Process Text if provided
        if text and text.strip():
            logger.info("Processing text input...")
            encoded = self.text_preprocessor.tokenize(text)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            modalities.append("text")

        # 2. Process Video if provided
        if video_path:
            logger.info(f"Processing video input from {video_path}...")
            frame_tensor, pil_frames, indices = self.video_preprocessor.preprocess_video(video_path)
            # Pool video frames using vision encoder
            frame_tensor = frame_tensor.to(self.device)
            modalities.append("video")
            # Select key middle frame for Grad-CAM visualization
            pil_image = pil_frames[len(pil_frames) // 2]
            image_tensor = frame_tensor[len(frame_tensor) // 2 : len(frame_tensor) // 2 + 1]
            video_meta = {
                "sampled_frames_count": len(pil_frames),
                "frame_indices": indices,
            }
        # 3. Process Image if provided (and no video)
        elif image_source:
            logger.info("Processing image input...")
            pil_image = self.image_preprocessor.load_image(image_source)
            image_tensor = self.image_preprocessor.preprocess(pil_image).unsqueeze(0).to(self.device)
            modalities.append("visual")

        if not modalities:
            raise ValueError("No valid input provided. Please supply text, image, or video.")

        logger.info(f"Running multimodal forward pass on device={self.device} with modalities: {modalities}")
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
            )
            logits = out["logits"]

        decision = self.decision_policy.evaluate(logits)
        logger.info(f"Raw prediction: {decision['prediction']} (confidence: {decision['confidence']})")

        # Saliency text explanation
        text_tokens = None
        if input_ids is not None:
            try:
                # Target class 1: FAKE (or class with highest score)
                target_cls = 1 if decision["prediction"] == "FAKE" else 0
                text_tokens = self.text_explainer.explain_instance(
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_class=target_cls,
                )
            except Exception as e:
                logger.warning(f"Could not compute text saliency: {e}")

        # Visual Grad-CAM explanation
        cam_data_uri = None
        cam_meta = None
        if image_tensor is not None and pil_image is not None:
            try:
                target_cls = 1 if decision["prediction"] == "FAKE" else 0
                cam_data_uri, cam_meta = self.grad_cam.generate_cam_overlay(
                    image_tensor=image_tensor,
                    pil_image=pil_image,
                    target_class=target_cls,
                )
            except Exception as e:
                logger.warning(f"Could not compute Grad-CAM: {e}")

        version_label = f"TruthLens Multimodal Baseline ({'Checkpoint' if self.is_checkpoint_loaded else 'Initialized'})"

        return self.evidence_engine.build_report(
            decision=decision,
            modalities_used=modalities,
            text_tokens=text_tokens,
            cam_data_uri=cam_data_uri,
            cam_meta=cam_meta,
            video_meta=video_meta,
            cross_attention_data=out.get("cross_attention"),
            execution_mode="real",
            model_version=version_label,
            fusion_type=self.model.fusion_type,
            device_used=str(self.device),
        )

    def _demo_response(self, text: Optional[str], has_image: bool, has_video: bool) -> VerificationReport:
        """Demo response with explicit DEMO labelling."""
        mods = []
        if text:
            mods.append("text")
        if has_video:
            mods.append("video")
        elif has_image:
            mods.append("visual")

        decision = {
            "prediction": "FAKE",
            "confidence": 0.88,
            "real_probability": 0.12,
            "fake_probability": 0.88,
            "uncertain": False,
            "threshold_used": settings.UNCERTAINTY_THRESHOLD,
        }
        tokens = [{"token": "breaking", "score": 0.9}, {"token": "unverified", "score": 0.75}] if text else None

        return self.evidence_engine.build_report(
            decision=decision,
            modalities_used=mods or ["text", "visual"],
            text_tokens=tokens,
            execution_mode="demo",
            model_version="TruthLens DEMO MODE (Explicit Mock)",
            fusion_type=settings.FUSION_TYPE,
            device_used="cpu",
        )
