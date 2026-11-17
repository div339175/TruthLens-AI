"""Utility helpers for file validation, saving, and cleanup."""

import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def save_upload_to_temp(upload_file: UploadFile, temp_dir: Path) -> Tuple[Path, str]:
    """Save an incoming UploadFile to a temporary directory with a unique name.

    Returns the path and detected file type ('image', 'video', or 'audio').
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload_file.filename or "file").suffix.lower()

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        file_type = "image"
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        file_type = "video"
    elif ext in ALLOWED_AUDIO_EXTENSIONS:
        file_type = "audio"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed image: {ALLOWED_IMAGE_EXTENSIONS}, video: {ALLOWED_VIDEO_EXTENSIONS}, audio: {ALLOWED_AUDIO_EXTENSIONS}"
        )

    unique_filename = f"{uuid.uuid4()}{ext}"
    dest_path = temp_dir / unique_filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return dest_path, file_type


def cleanup_file(path: Path) -> None:
    """Safely delete a temporary file if it exists."""
    try:
        if path.exists():
            os.remove(path)
    except Exception:
        pass
