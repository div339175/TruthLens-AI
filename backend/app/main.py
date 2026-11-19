"""TruthLens AI Backend FastAPI Application."""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database.session import init_db
from backend.app.api.health import router as health_router
from backend.app.api.routes import router as verification_router

logger = logging.getLogger("truthlens.app")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Initialize database tables on module import / startup
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(verification_router)

# Mount sample media files if available
samples_dir = settings.DATA_DIR / "samples"
if samples_dir.exists():
    app.mount("/api/samples", StaticFiles(directory=str(samples_dir)), name="samples")


@app.get("/")
def read_root():
    """Root endpoint welcoming clients and indicating operational mode."""
    return {
        "project": settings.PROJECT_NAME,
        "title": "Multimodal NLP for Deepfake Detection and Explainable Content Verification",
        "version": settings.VERSION,
        "status": "online",
        "mode": settings.MODEL_MODE,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
