"""Service layer managing verification persistence and history retrieval."""

import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from backend.app.database.session import SessionLocal
from backend.app.database.models import VerificationRecord
from backend.app.schemas.verification import VerificationReport

logger = logging.getLogger("truthlens.history")


class HistoryService:
    """Provides CRUD operations for persistent verification records in SQLite."""

    def __init__(self, db_factory=SessionLocal):
        self.db_factory = db_factory

    def save_verification(
        self,
        report: VerificationReport,
        case_name: Optional[str] = None,
        text_snippet: Optional[str] = None,
        image_ref: Optional[str] = None,
        video_ref: Optional[str] = None,
        audio_ref: Optional[str] = None,
    ) -> VerificationRecord:
        """Persist a completed VerificationReport into the database."""
        db: Session = self.db_factory()
        try:
            # Generate deterministic or random ID if report doesn't already have one
            record_id = report.id or str(uuid.uuid4())[:12]
            report.id = record_id

            # Determine a meaningful default case name if none was supplied
            if not case_name:
                now_str = datetime.now(timezone.utc).strftime("%b %d, %H:%M")
                mods = "+".join([m.capitalize() for m in report.modalities_used])
                if text_snippet and text_snippet.strip():
                    snippet = text_snippet.strip().replace("\n", " ")
                    if len(snippet) > 40:
                        snippet = snippet[:37] + "..."
                    case_name = f"{mods}: \"{snippet}\""
                elif report.modalities_used:
                    case_name = f"{mods} Verification ({now_str})"
                else:
                    case_name = f"Case #{record_id} ({now_str})"

            report.case_name = case_name

            # Serialize full report
            report_dict = report.model_dump()
            report_json = json.dumps(report_dict)

            # Check if record already exists (update vs insert)
            existing = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
            if existing:
                existing.case_name = case_name
                existing.prediction = report.prediction
                existing.confidence = report.confidence
                existing.real_probability = report.real_probability
                existing.fake_probability = report.fake_probability
                existing.uncertain = report.uncertain
                existing.modalities_used = report.modalities_used
                existing.manipulation_type = report.manipulation_type
                existing.explanation = report.explanation
                existing.model_version = report.model_version
                existing.fusion_type = report.fusion_type
                existing.device_used = report.device_used
                existing.execution_mode = report.execution_mode
                existing.report_json = report_json
                if text_snippet:
                    existing.text_snippet = text_snippet[:500]
                if image_ref:
                    existing.image_reference = str(image_ref)
                if video_ref:
                    existing.video_reference = str(video_ref)
                if audio_ref:
                    existing.audio_reference = str(audio_ref)
                db.commit()
                db.refresh(existing)
                logger.info(f"Updated verification record {record_id} ('{case_name}') in database.")
                return existing

            record = VerificationRecord(
                id=record_id,
                case_name=case_name,
                created_at=datetime.now(timezone.utc),
                prediction=report.prediction,
                confidence=report.confidence,
                real_probability=report.real_probability,
                fake_probability=report.fake_probability,
                uncertain=report.uncertain,
                modalities_used=report.modalities_used,
                manipulation_type=report.manipulation_type,
                text_snippet=text_snippet[:500] if text_snippet else "",
                image_reference=str(image_ref) if image_ref else None,
                video_reference=str(video_ref) if video_ref else None,
                audio_reference=str(audio_ref) if audio_ref else None,
                explanation=report.explanation,
                model_version=report.model_version,
                fusion_type=report.fusion_type,
                device_used=report.device_used,
                execution_mode=report.execution_mode,
                report_json=report_json,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(f"Saved verification record {record_id} ('{case_name}') to database.")
            return record
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to persist verification record: {e}")
            raise
        finally:
            db.close()

    def list_verifications(self, limit: int = 100, offset: int = 0) -> List[VerificationRecord]:
        """Retrieve previous verifications ordered newest-first."""
        db: Session = self.db_factory()
        try:
            return (
                db.query(VerificationRecord)
                .order_by(VerificationRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    def get_verification(self, verification_id: str) -> Optional[VerificationRecord]:
        """Fetch a single verification record by unique ID."""
        db: Session = self.db_factory()
        try:
            return db.query(VerificationRecord).filter(VerificationRecord.id == verification_id).first()
        finally:
            db.close()

    def delete_verification(self, verification_id: str) -> bool:
        """Delete a verification record by ID."""
        db: Session = self.db_factory()
        try:
            record = db.query(VerificationRecord).filter(VerificationRecord.id == verification_id).first()
            if not record:
                return False
            db.delete(record)
            db.commit()
            logger.info(f"Deleted verification record {verification_id} from database.")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete verification record {verification_id}: {e}")
            raise
        finally:
            db.close()
