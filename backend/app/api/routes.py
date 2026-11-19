"""API routes for TruthLens AI verification, dataset metrics, and model inspection."""

import csv
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.app.config import settings
from backend.app.schemas.verification import (
    VerificationReport,
    TextVerificationRequest,
    ModelInfoResponse,
    DatasetStatsResponse,
    VerificationSummary,
    SaveVerificationRequest,
    VerificationListResponse,
)
from backend.app.services.verification_service import VerificationService
from backend.app.utils.file_utils import save_upload_to_temp, cleanup_file

router = APIRouter(tags=["Verification"])
verification_service = VerificationService()


@router.post("/verify/text", response_model=VerificationReport)
def verify_text_endpoint(payload: TextVerificationRequest):
    """Verify textual claim or caption and auto-persist to history."""
    try:
        return verification_service.verify_text(text=payload.text, case_name=payload.case_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify/image", response_model=VerificationReport)
async def verify_image_endpoint(image: UploadFile = File(...)):
    """Verify image for deepfake or synthetic manipulation."""
    temp_path, file_type = save_upload_to_temp(image, settings.UPLOAD_TEMP_DIR)
    if file_type != "image":
        cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        report = verification_service.verify_image(temp_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)


@router.post("/verify/video", response_model=VerificationReport)
async def verify_video_endpoint(video: UploadFile = File(...)):
    """Verify video for deepfake manipulation using sampled frames."""
    temp_path, file_type = save_upload_to_temp(video, settings.UPLOAD_TEMP_DIR)
    if file_type != "video":
        cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not a video.")

    try:
        report = verification_service.verify_video(temp_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)


@router.post("/verify/audio", response_model=VerificationReport)
async def verify_audio_endpoint(audio: UploadFile = File(...)):
    """Verify audio file for synthetic voice cloning or manipulation."""
    temp_path, file_type = save_upload_to_temp(audio, settings.UPLOAD_TEMP_DIR)
    if file_type != "audio":
        cleanup_file(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is not an audio file.")

    try:
        report = verification_service.verify_audio(temp_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_file(temp_path)


@router.post("/verify/multimodal", response_model=VerificationReport)
async def verify_multimodal_endpoint(
    text: Optional[str] = Form(None),
    case_name: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    """Main verification endpoint accepting text, image, video, and/or audio."""
    if not text and not image and not video and not audio:
        raise HTTPException(
            status_code=400,
            detail="At least one modality (text, image, video, or audio) must be supplied.",
        )

    img_temp_path: Optional[Path] = None
    vid_temp_path: Optional[Path] = None
    aud_temp_path: Optional[Path] = None

    try:
        if image and image.filename:
            img_temp_path, ftype = save_upload_to_temp(image, settings.UPLOAD_TEMP_DIR)
            if ftype != "image":
                raise HTTPException(status_code=400, detail="Uploaded image file is not an image.")

        if video and video.filename:
            vid_temp_path, ftype = save_upload_to_temp(video, settings.UPLOAD_TEMP_DIR)
            if ftype != "video":
                raise HTTPException(status_code=400, detail="Uploaded video file is not a video.")

        if audio and audio.filename:
            aud_temp_path, ftype = save_upload_to_temp(audio, settings.UPLOAD_TEMP_DIR)
            if ftype != "audio":
                raise HTTPException(status_code=400, detail="Uploaded audio file is not an audio file.")

        report = verification_service.verify_multimodal(
            text=text if (text and text.strip()) else None,
            image_path=img_temp_path,
            video_path=vid_temp_path,
            audio_path=aud_temp_path,
            case_name=case_name,
        )
        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if img_temp_path:
            cleanup_file(img_temp_path)
        if vid_temp_path:
            cleanup_file(vid_temp_path)
        if aud_temp_path:
            cleanup_file(aud_temp_path)


# ==============================================================================
# Persistent Verification History Endpoints (SQLite / SQLAlchemy)
# ==============================================================================


@router.get("/verifications", response_model=VerificationListResponse)
def list_verifications_endpoint(limit: int = 100, offset: int = 0):
    """Retrieve previous verification records ordered newest-first from SQLite database."""
    records = verification_service.history_service.list_verifications(limit=limit, offset=offset)
    items = [VerificationSummary(**r.to_summary_dict()) for r in records]
    return VerificationListResponse(total=len(items), items=items)


@router.get("/verifications/{verification_id}", response_model=VerificationReport)
def get_verification_endpoint(verification_id: str):
    """Retrieve complete saved verification result by unique ID."""
    record = verification_service.history_service.get_verification(verification_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Verification record '{verification_id}' not found.")
    return VerificationReport(**record.to_full_report_dict())


@router.post("/verifications", response_model=VerificationReport)
def save_verification_endpoint(payload: SaveVerificationRequest):
    """Explicitly save or update a completed verification report in persistent history."""
    try:
        record = verification_service.history_service.save_verification(
            report=payload.report,
            case_name=payload.case_name,
        )
        return VerificationReport(**record.to_full_report_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/verifications/{verification_id}")
def delete_verification_endpoint(verification_id: str):
    """Delete a verification record from persistent history."""
    deleted = verification_service.history_service.delete_verification(verification_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Verification record '{verification_id}' not found.")
    return {"status": "success", "message": f"Verification '{verification_id}' deleted successfully."}


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info_endpoint():
    """Returns technical architecture specifications and backbone metadata."""
    return verification_service.get_model_info()


@router.get("/dataset/stats", response_model=DatasetStatsResponse)
def get_dataset_stats_endpoint():
    """Inspects metadata CSV file and reports REAL verified statistics without fabricating numbers."""
    csv_path = settings.METADATA_PATH

    if not csv_path.exists():
        return DatasetStatsResponse(
            status="uninitialized",
            total_samples=0,
            real_count=0,
            fake_count=0,
            uncertain_count=0,
            train_count=0,
            val_count=0,
            test_count=0,
            sources={},
            message="Dataset not initialized. Run scripts/download_dataset.py or scripts/prepare_data.py to configure.",
        )

    total = 0
    real = 0
    fake = 0
    uncertain = 0
    train = 0
    val = 0
    test = 0
    sources = {}

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                lbl = str(row.get("label", "")).strip()
                if lbl in ["0", "REAL", "real"]:
                    real += 1
                elif lbl in ["1", "FAKE", "fake"]:
                    fake += 1
                elif lbl in ["2", "UNCERTAIN", "uncertain"]:
                    uncertain += 1

                split = str(row.get("split", "")).lower().strip()
                if split == "train":
                    train += 1
                elif split in ["val", "validation", "valid"]:
                    val += 1
                elif split == "test":
                    test += 1

                src = str(row.get("source", "unknown")).strip()
                sources[src] = sources.get(src, 0) + 1

        return DatasetStatsResponse(
            status="initialized",
            total_samples=total,
            real_count=real,
            fake_count=fake,
            uncertain_count=uncertain,
            train_count=train,
            val_count=val,
            test_count=test,
            sources=sources,
            message=f"Dataset verified: {total} records in {csv_path.name}",
        )
    except Exception as e:
        return DatasetStatsResponse(
            status="error",
            total_samples=0,
            real_count=0,
            fake_count=0,
            uncertain_count=0,
            train_count=0,
            val_count=0,
            test_count=0,
            sources={},
            message=f"Error reading dataset metadata: {str(e)}",
        )
