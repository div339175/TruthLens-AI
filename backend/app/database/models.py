"""SQLAlchemy ORM models for TruthLens AI persistent storage."""

from datetime import datetime, timezone
import json
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, JSON
from backend.app.database.session import Base


class VerificationRecord(Base):
    """Persistent database model storing multimodal verification outcomes."""

    __tablename__ = "verifications"

    id = Column(String(36), primary_key=True, index=True)
    case_name = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Classification outcomes
    prediction = Column(String(50), nullable=False)  # REAL, FAKE, UNCERTAIN
    confidence = Column(Float, nullable=False)
    real_probability = Column(Float, nullable=False)
    fake_probability = Column(Float, nullable=False)
    uncertain = Column(Boolean, default=False, nullable=False)

    # Modalities and references
    modalities_used = Column(JSON, nullable=False)  # e.g. ["text", "visual"]
    manipulation_type = Column(String(255), nullable=True)
    text_snippet = Column(Text, nullable=True)
    image_reference = Column(String(255), nullable=True)
    video_reference = Column(String(255), nullable=True)
    audio_reference = Column(String(255), nullable=True)

    # Explainability and metadata
    explanation = Column(Text, nullable=False)
    model_version = Column(String(100), nullable=True)
    fusion_type = Column(String(50), nullable=True)
    device_used = Column(String(50), nullable=True)
    execution_mode = Column(String(50), default="real")

    # Complete serialized VerificationReport JSON
    report_json = Column(Text, nullable=False)

    def to_summary_dict(self):
        """Return compact summary dictionary for history table listing."""
        return {
            "id": self.id,
            "case_name": self.case_name,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "prediction": self.prediction,
            "confidence": self.confidence,
            "real_probability": self.real_probability,
            "fake_probability": self.fake_probability,
            "uncertain": self.uncertain,
            "modalities_used": self.modalities_used if isinstance(self.modalities_used, list) else json.loads(self.modalities_used or "[]"),
            "manipulation_type": self.manipulation_type or "None detected",
            "text_snippet": self.text_snippet or "",
            "has_image": bool(self.image_reference),
            "has_video": bool(self.video_reference),
            "has_audio": bool(self.audio_reference),
            "execution_mode": self.execution_mode,
        }

    def to_full_report_dict(self):
        """Parse and return full stored VerificationReport dictionary."""
        data = json.loads(self.report_json)
        data["id"] = self.id
        data["case_name"] = self.case_name
        return data
