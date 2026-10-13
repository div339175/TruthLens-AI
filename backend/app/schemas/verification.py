"""Pydantic schemas for TruthLens AI verification and analysis requests/responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TokenImportance(BaseModel):
    token: str
    score: float


class ModalityEvidence(BaseModel):
    modality: str = Field(..., description="'text', 'visual', 'video', or 'cross-modal'")
    type: str = Field(..., description="'model-derived' or 'heuristic'")
    description: str
    details: Optional[Dict[str, Any]] = None
    token_importance: Optional[List[TokenImportance]] = None
    heatmap_available: Optional[bool] = False
    heatmap_base64: Optional[str] = None


class VerificationReport(BaseModel):
    prediction: str = Field(..., description="REAL, FAKE, or UNCERTAIN")
    confidence: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    uncertain: bool
    modalities_used: List[str]
    manipulation_type: str = Field(..., description="e.g. 'Potential visual manipulation', 'Semantic mismatch', 'None detected'")
    evidence: List[ModalityEvidence]
    explanation: str
    model_version: str
    fusion_type: str
    device_used: str
    execution_mode: str = Field(..., description="'real' or 'demo'")
    timestamp: str


class TextVerificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    device: str
    execution_mode: str
    torch_version: str
    models_loaded: bool


class DatasetStatsResponse(BaseModel):
    status: str
    total_samples: int
    real_count: int
    fake_count: int
    uncertain_count: int
    train_count: int
    val_count: int
    test_count: int
    sources: Dict[str, int]
    message: Optional[str] = None


class ModelInfoResponse(BaseModel):
    project_name: str
    version: str
    execution_mode: str
    text_backbone: str
    vision_backbone: str
    fusion_architecture: str
    device: str
    uncertainty_threshold: float
    is_trained_checkpoint: bool
    checkpoint_path: Optional[str] = None
    architecture_summary: Dict[str, Any]
