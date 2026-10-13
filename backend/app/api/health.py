"""Health check API endpoint."""

from fastapi import APIRouter
import torch
from backend.app.config import settings
from backend.app.schemas.verification import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint exposing runtime environment and hardware configuration."""
    torch_device = str(settings.get_torch_device())
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        device=torch_device,
        execution_mode=settings.MODEL_MODE,
        torch_version=torch.__version__,
        models_loaded=True,
    )
