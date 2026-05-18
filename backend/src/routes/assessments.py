"""Assessment routes: create, list, latest, processing status, PHQ-8 questions."""

import json
import logging
import math
import asyncio
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from database import get_db, async_session_factory
from src.models import User, Assessment, AssessmentAnswer, AssessmentMLDetail, Doctor, DoctorAssignment, MediaFile, ProcessingJob
from src.middleware.deps import get_current_user, require_patient
from src.services.ml_client import MLClient

logger = logging.getLogger("mindscope")

router = APIRouter(tags=["assessments"])

# ── PHQ-8 question data ───────────────────────────────

PHQ8_QUESTIONS = [
    {"id": 1, "text": "Over the last two weeks, how often have you felt little interest or pleasure in doing things?", "instruction": "Choose the option that best matches your experience."},
    {"id": 2, "text": "Over the last two weeks, how often have you felt down, depressed, or hopeless?", "instruction": "Choose the option that best matches your experience."},
    {"id": 3, "text": "Over the last two weeks, how often have you had trouble falling or staying asleep, or sleeping too much?", "instruction": "Choose the option that best matches your experience."},
    {"id": 4, "text": "Over the last two weeks, how often have you felt tired or had little energy?", "instruction": "Choose the option that best matches your experience."},
    {"id": 5, "text": "Over the last two weeks, how often have you had a poor appetite or been overeating?", "instruction": "Choose the option that best matches your experience."},
    {"id": 6, "text": "Over the last two weeks, how often have you felt bad about yourself, or that you are a failure?", "instruction": "Choose the option that best matches your experience."},
    {"id": 7, "text": "Over the last two weeks, how often have you had trouble concentrating on things, such as reading or watching television?", "instruction": "Choose the option that best matches your experience."},
    {"id": 8, "text": "Over the last two weeks, how often have you been moving or speaking so slowly that other people have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?", "instruction": "Choose the option that best matches your experience."},
]

PHQ8_OPTIONS = [
    {"label": "Not at all", "value": 0},
    {"label": "Several days", "value": 1},
    {"label": "More than half the days", "value": 2},
    {"label": "Nearly every day", "value": 3},
]


def get_severity_label(score: int) -> str:
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    if score <= 19:
        return "Moderately Severe"
    return "Severe"


def score_from_ml_output(
    question_id: int,
    ml_phq8_score: Optional[float],
    ml_item_scores: Optional[list],
) -> int:
    """Convert model output to a single PHQ item score in the 0-3 range."""
    if isinstance(ml_item_scores, list) and ml_item_scores:
        idx = min(max(question_id - 1, 0), len(ml_item_scores) - 1)
        try:
            return max(0, min(3, int(round(float(ml_item_scores[idx])))))
        except (TypeError, ValueError):
            pass

    normalized = float(ml_phq8_score or 0.0) / 8.0
    return max(0, min(3, int(round(normalized))))


def normalize_severity_label(value: Optional[str], score: Optional[float] = None) -> str:
    if value:
        normalized = value.strip().lower().replace("_", " ")
        mapping = {
            "none/minimal": "Minimal",
            "minimal": "Minimal",
            "mild": "Mild",
            "moderate": "Moderate",
            "moderately severe": "Moderately Severe",
            "severe": "Severe",
        }
        if normalized in mapping:
            return mapping[normalized]
    return get_severity_label(int(round(score or 0)))


def confidence_score_from_interval(mean: Optional[float], std: Optional[float]) -> Optional[float]:
    if mean is None or std is None or std <= 0:
        return None
    spread = abs(std or 0.0)
    # PHQ-8 spans 0-24. A 12-point std is effectively low confidence.
    return max(0.0, min(1.0, 1.0 - (spread / 12.0)))


def _safe_json_loads(value: str | None, default=None):
    fallback = {} if default is None else default
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


async def _has_video_recordings_for_assessment(assessment_id: str, db: AsyncSession) -> bool:
    result = await db.execute(
        select(MediaFile.mime_type)
        .select_from(AssessmentAnswer)
        .join(MediaFile, AssessmentAnswer.audio_file_id == MediaFile.id)
        .where(
            AssessmentAnswer.assessment_id == assessment_id,
            MediaFile.mime_type.like("video/%"),
        )
        .limit(1)
    )
    return bool(result.scalar_one_or_none())


def distribute_total_score(total_score: int, answers: list, media_by_id: dict) -> dict[int, int]:
    """Allocate the model's total PHQ-8 score into 0-3 item estimates."""
    if not answers:
        return {}

    capped_total = max(0, min(int(total_score), len(answers) * 3))
    weights = []
    for answer in answers:
        media = media_by_id.get(answer.audio_file_id)
        duration = max(float(answer.duration_sec or 0), 0.0)
        size_kb = max(float(getattr(media, "file_size", 0) or 0) / 1024.0, 0.0)
        # Keep the item allocation lightweight: model output controls the total,
        # recording duration/size only decide how that total is spread per item.
        weights.append(max(0.1, math.log1p(size_kb) + min(duration, 120.0) / 30.0))

    weight_sum = sum(weights) or len(answers)
    raw_scores = [(capped_total * weight / weight_sum) for weight in weights]
    scores = [min(3, int(math.floor(value))) for value in raw_scores]
    remaining = capped_total - sum(scores)

    order = sorted(
        range(len(answers)),
        key=lambda idx: (raw_scores[idx] - math.floor(raw_scores[idx]), weights[idx]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for idx in order:
            if scores[idx] >= 3:
                continue
            scores[idx] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            break

    return {answers[idx].question_id: scores[idx] for idx in range(len(answers))}


def ml_detail_payload(detail: AssessmentMLDetail | None, assessment: Assessment | None = None) -> Optional[dict]:
    if not detail:
        return None

    confidence_mean = detail.confidence_mean
    confidence_std = detail.confidence_std
    return {
        "phq8Score": assessment.ml_score if assessment else None,
        "severity": assessment.ml_severity if assessment else None,
        "numChunks": assessment.ml_num_chunks if assessment else None,
        "confidenceMean": confidence_mean,
        "confidenceStd": confidence_std,
        "confidenceScore": confidence_score_from_interval(confidence_mean, confidence_std),
        "ciLower": detail.ci_lower,
        "ciUpper": detail.ci_upper,
        "audioQualityScore": detail.audio_quality_score,
        "audioSnrDb": detail.audio_snr_db,
        "audioSpeechProb": detail.audio_speech_prob,
        "behavioral": _safe_json_loads(detail.behavioral_json),
        "inferenceTimeMs": detail.inference_time_ms,
    }


def _is_report_ready(assessment: Assessment) -> bool:
    return bool(
        assessment.status == "completed"
        or assessment.is_report_ready
        or assessment.report_status == "available"
        or assessment.ml_score is not None
    )


def _as_utc(dt: datetime | None) -> datetime | None:
    """Make a datetime timezone-aware (UTC). SQLite returns naive datetimes."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def _sync_report_state(db: AsyncSession, assessment: Assessment) -> bool:
    if _is_report_ready(assessment):
        assessment.status = "completed"
        assessment.report_status = "available"
        assessment.is_report_ready = True
        await db.flush()
        return True

    job = (await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.assessment_id == assessment.id)
        .order_by(desc(ProcessingJob.created_at))
        .limit(1)
    )).scalar_one_or_none()

    if job and job.progress_pct >= 100 and job.status == "succeeded":
        assessment.status = "completed"
        assessment.report_status = "available"
        assessment.is_report_ready = True
        await db.flush()
        return True

    # Timeout recovery: if stuck for > 5 minutes, force-complete with existing scores
    if assessment.status in ("processing", "preparing"):
        now = datetime.now(timezone.utc)
        created_at = _as_utc(assessment.created_at) or now
        elapsed = (now - created_at).total_seconds()
        if elapsed > 300:  # 5 minutes
            assessment.status = "completed"
            assessment.report_status = "available"
            assessment.is_report_ready = True
            if job:
                job.status = "timeout_recovered"
                job.progress_pct = 100
                job.stage = "Timeout recovery"
                job.finished_at = now
                job.error_message = f"Timed out after {elapsed:.0f}s, force-completed with existing scores"
            await db.flush()
            logger.warning(
                f"[SYNC] Assessment {assessment.id[:8]}... timeout-recovered from stuck state after {elapsed:.0f}s"
            )
            return True

    return False


# ── Schemas ────────────────────────────────────────────

class AnswerInput(BaseModel):
    questionId: int = Field(..., ge=1, le=8)
    score: int = Field(..., ge=0, le=3)
    durationSec: Optional[float] = None
    audioFileId: Optional[str] = None


class CreateAssessmentRequest(BaseModel):
    questionSetVersion: str = "phq8_v1"
    answers: List[AnswerInput] = Field(..., min_length=1, max_length=8)
    recordingCount: int = Field(default=0, ge=0)
    skipBackgroundInference: bool = False


class ScoreQuestionAudioRequest(BaseModel):
    questionId: int = Field(..., ge=1, le=8)
    audioFileId: str = Field(..., min_length=1)
    durationSec: Optional[float] = Field(default=None, ge=0)


# ── GET /phq8/questions ────────────────────────────────

@router.get("/phq8/questions")
async def get_questions():
    # Questions never change — clients may cache for up to 1 hour
    content = {
        "version": "phq8_v1",
        "questions": PHQ8_QUESTIONS,
        "options": PHQ8_OPTIONS,
    }
    return JSONResponse(content=content, headers={"Cache-Control": "public, max-age=3600"})


# ── POST /assessments ─────────────────────────────────

@router.post("/assessments", status_code=201)
async def create_assessment(
    body: CreateAssessmentRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    # Compute total score
    score_total = sum(a.score for a in body.answers)
    severity = get_severity_label(score_total)
    has_audio = any(bool(a.audioFileId) for a in body.answers)
    should_run_background_ml = has_audio and not body.skipBackgroundInference

    assessment = Assessment(
        user_id=user.id,
        question_set_version=body.questionSetVersion,
        score_total=score_total,
        severity=severity,
        recording_count=body.recordingCount,
        status="processing" if should_run_background_ml else "completed",
        report_status="pending" if should_run_background_ml else "available",
        is_report_ready=not should_run_background_ml,
    )
    db.add(assessment)
    await db.flush()

    # Save individual answers
    audio_file_ids = []
    for ans in body.answers:
        answer = AssessmentAnswer(
            assessment_id=assessment.id,
            question_id=ans.questionId,
            score=ans.score,
            duration_sec=ans.durationSec,
            audio_file_id=ans.audioFileId,
        )
        db.add(answer)
        if ans.audioFileId:
            audio_file_ids.append(ans.audioFileId)
    await db.flush()

    assessment_id = assessment.id
    user_id = user.id
    has_video = await _has_video_recordings_for_assessment(assessment_id, db)

    # If audio was recorded, trigger background ML inference
    if should_run_background_ml and audio_file_ids:
        db.add(
            ProcessingJob(
                assessment_id=assessment.id,
                status="running",
                progress_pct=5,
                stage="Loading voice responses",
                started_at=datetime.now(timezone.utc),
            )
        )
        await db.flush()
        background_tasks.add_task(_run_ml_inference, assessment_id, user_id, audio_file_ids)

    return {
        "assessment": {
            "id": assessment.id,
            "userId": assessment.user_id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "status": assessment.status,
            "reportStatus": assessment.report_status,
            "reportReady": assessment.is_report_ready,
            "isReportReady": assessment.is_report_ready,
            "hasVideoRecordings": has_video,
            "stage": "Completed" if assessment.is_report_ready else "Loading voice responses",
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
        }
    }


# ── POST /assessments/score/question ───────────────────

@router.post("/assessments/score/question")
async def score_question_audio(
    body: ScoreQuestionAudioRequest,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    from pathlib import Path
    from config.settings import get_settings

    settings = get_settings()
    media = (await db.execute(
        select(MediaFile).where(
            MediaFile.id == body.audioFileId,
            MediaFile.owner_user_id == user.id,
        )
    )).scalar_one_or_none()
    if not media:
        raise HTTPException(status_code=404, detail="Audio file not found")

    audio_path = Path(settings.STORAGE_LOCAL_PATH) / media.storage_key
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing from storage")

    client = MLClient()
    try:
        ml_result = await client.predict_extended(
            audio_path=str(audio_path),
            participant_id=user.id,
        )
    except Exception as exc:
        message = str(exc)
        logger.error(f"[ML] Question scoring failed for user={user.id}, q={body.questionId}: {message}")
        if "No usable audio chunks" in message or "No speech detected" in message:
            raise HTTPException(
                status_code=422,
                detail="No clear speech detected in this recording. Please re-record and speak clearly for at least a few seconds.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Voice model service is unavailable. {message}",
        ) from exc

    ml_score = max(0.0, min(24.0, float(ml_result.get("phq_total") or ml_result.get("phq8_score") or 0.0)))
    question_score = score_from_ml_output(
        question_id=body.questionId,
        ml_phq8_score=ml_score,
        ml_item_scores=ml_result.get("item_scores"),
    )
    inference_time_ms = float(ml_result.get("inference_time_s") or 0.0) * 1000.0

    return {
        "questionId": body.questionId,
        "score": question_score,
        "audioFileId": body.audioFileId,
        "mlScore": round(ml_score, 2),
        "inferenceTimeMs": round(inference_time_ms, 2),
    }


# ── GET /assessments/latest ───────────────────────────

@router.get("/assessments/latest")
async def get_latest_assessment(
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id == user.id)
        .order_by(desc(Assessment.created_at))
        .limit(1)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="No assessment found")

    await _sync_report_state(db, assessment)

    # Build answers map
    answers_result = await db.execute(
        select(AssessmentAnswer).where(AssessmentAnswer.assessment_id == assessment.id)
    )
    answers_map = {str(a.question_id): a.score for a in answers_result.scalars().all()}
    has_video = await _has_video_recordings_for_assessment(assessment.id, db)

    return {
        "assessment": {
            "id": assessment.id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "answers": answers_map,
            "recordingCount": assessment.recording_count,
            "hasVideoRecordings": has_video,
            "status": assessment.status,
            "reportStatus": assessment.report_status,
            "isReportReady": assessment.is_report_ready,
            "doctorRemarks": assessment.doctor_remarks,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
            "mlScore": assessment.ml_score,
            "mlSeverity": assessment.ml_severity,
        }
    }


# ── GET /assessments ──────────────────────────────────

@router.get("/assessments")
async def list_assessments(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    # Count
    count_q = select(func.count(Assessment.id)).where(Assessment.user_id == user.id)
    total = (await db.execute(count_q)).scalar() or 0

    # Fetch page
    offset = (page - 1) * pageSize
    result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id == user.id)
        .order_by(desc(Assessment.created_at))
        .offset(offset)
        .limit(pageSize)
    )
    assessments = result.scalars().all()

    video_map = {}
    if assessments:
        assessment_ids = [a.id for a in assessments]
        video_rows = await db.execute(
            select(AssessmentAnswer.assessment_id, MediaFile.mime_type)
            .join(MediaFile, AssessmentAnswer.audio_file_id == MediaFile.id)
            .where(AssessmentAnswer.assessment_id.in_(assessment_ids))
        )
        for assessment_id, mime_type in video_rows.all():
            if mime_type and mime_type.startswith("video/"):
                video_map[assessment_id] = True

    items = [
        {
            "id": a.id,
            "score": a.score_total,
            "severity": a.severity,
            "recordingCount": a.recording_count,
            "hasVideoRecordings": bool(video_map.get(a.id)),
            "status": "completed" if _is_report_ready(a) else a.status,
            "reportStatus": "available" if _is_report_ready(a) else (a.report_status or "pending"),
            "isReportReady": _is_report_ready(a),
            "doctorRemarks": a.doctor_remarks,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
            "mlScore": a.ml_score,
            "mlSeverity": a.ml_severity,
        }
        for a in assessments
    ]

    return {
        "items": items,
        "pagination": {
            "page": page,
            "pageSize": pageSize,
            "total": total,
        },
    }


# ── GET /assessments/{id} ─────────────────────────────

@router.get("/assessments/{assessment_id}")
async def get_assessment_detail(
    assessment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    assessment = (await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )).scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if assessment.user_id != user.id and user.role == "doctor":
        doctor = (await db.execute(select(Doctor).where(Doctor.user_id == user.id))).scalar_one_or_none()
        assigned = None
        if doctor:
            assigned = (await db.execute(
                select(DoctorAssignment).where(
                    DoctorAssignment.doctor_id == doctor.id,
                    DoctorAssignment.assessment_id == assessment.id,
                    DoctorAssignment.status.in_(["pending", "accepted", "completed"]),
                )
            )).scalar_one_or_none()
        if not assigned:
            raise HTTPException(status_code=403, detail="Not authorized")
    elif assessment.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")

    await _sync_report_state(db, assessment)

    answers = (await db.execute(
        select(AssessmentAnswer)
        .where(AssessmentAnswer.assessment_id == assessment_id)
        .order_by(AssessmentAnswer.question_id)
    )).scalars().all()

    media_ids = [answer.audio_file_id for answer in answers if answer.audio_file_id]
    media_by_id = {}
    if media_ids:
        media_files = (await db.execute(
            select(MediaFile).where(MediaFile.id.in_(media_ids))
        )).scalars().all()
        media_by_id = {media.id: media for media in media_files}

    detail = (await db.execute(
        select(AssessmentMLDetail).where(AssessmentMLDetail.assessment_id == assessment_id)
    )).scalar_one_or_none()

    questions_by_id = {item["id"]: item for item in PHQ8_QUESTIONS}
    has_video = any(
        getattr(media_by_id.get(answer.audio_file_id), "mime_type", "",).startswith("video/")
        for answer in answers
        if answer.audio_file_id
    )

    # ── Fetch associated MultimodalSessions for real ML metrics ──────────
    from src.routes.multimodal import MultimodalSession
    _ts = assessment.created_at
    _mm_sessions = (await db.execute(
        select(MultimodalSession)
        .where(
            MultimodalSession.user_id == assessment.user_id,
            MultimodalSession.status == "completed",
            MultimodalSession.created_at >= _ts - timedelta(minutes=1),
            MultimodalSession.created_at <= _ts + timedelta(minutes=15),
        )
        .order_by(MultimodalSession.created_at)
    )).scalars().all()

    # Build question_id → per-question ML score map (real: each session analysed ONE question)
    _q_score_map: dict = {}
    _inference_times = []
    for _s in _mm_sessions:
        if _s.inference_time_ms:
            _inference_times.append(_s.inference_time_ms)
        try:
            _dbg = json.loads(_s.debug_json or "{}")
            _qid = _dbg.get("question_id")
            if _qid and _s.phq8_score is not None:
                # Each session predicts full PHQ-8 total; divide by 8 to get per-Q estimate (0-3)
                _q_score_map[int(_qid)] = round(min(3.0, max(0.0, float(_s.phq8_score) / 8.0)), 2)
        except Exception:
            pass

    # Pick the "primary" session (last = combined reeval if reeval was run, else latest question)
    _primary = _mm_sessions[-1] if _mm_sessions else None
    _ml_model_details = None
    if _primary:
        _ml_model_details = {
            "confidence": round(float(_primary.confidence or 0), 4),
            "depressionProbability": round(float(_primary.depression_probability or 0), 4) if _primary.depression_probability is not None else None,
            "predictedLabel": _primary.predicted_label,
            "modalityContributions": {
                "audio": round(float(_primary.audio_contribution or 0), 4),
                "video": round(float(_primary.video_contribution or 0), 4),
                "text":  round(float(_primary.text_contribution  or 0), 4),
            },
            "avgInferenceTimeMs": round(sum(_inference_times) / len(_inference_times), 1) if _inference_times else None,
            "totalInferenceTimeMs": round(sum(_inference_times), 1) if _inference_times else None,
            "sessionCount": len(_mm_sessions),
            "perQuestionScores": _q_score_map,  # {1: 1.58, 2: 1.58, ...} — real per-Q from individual sessions
        }

    return {
        "assessment": {
            "id": assessment.id,
            "userId": assessment.user_id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "recordingCount": assessment.recording_count,
            "hasVideoRecordings": has_video,
            "status": assessment.status,
            "reportStatus": assessment.report_status,
            "isReportReady": assessment.is_report_ready,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
            "mlScore": assessment.ml_score,
            "mlSeverity": assessment.ml_severity,
            "doctorRemarks": assessment.doctor_remarks,
            "answers": [
                {
                    "questionId": answer.question_id,
                    "questionText": questions_by_id.get(answer.question_id, {}).get("text", ""),
                    "score": answer.score,
                    "mlScore": _q_score_map.get(answer.question_id),
                    "durationSec": answer.duration_sec,
                    "audioFileId": answer.audio_file_id,
                    "isVideo": bool(
                        answer.audio_file_id
                        and getattr(media_by_id.get(answer.audio_file_id), "mime_type", "",).startswith("video/")
                    ),
                    "audioUrl": f"/api/v1/files/audio/{answer.audio_file_id}" if answer.audio_file_id in media_by_id else None,
                    "fileName": getattr(media_by_id.get(answer.audio_file_id), "original_filename", None),
                    "fileSize": getattr(media_by_id.get(answer.audio_file_id), "file_size", None),
                }
                for answer in answers
            ],
            "mlDetails": ml_detail_payload(detail, assessment),
            "mlModelDetails": _ml_model_details,
        }
    }


# ── GET /assessments/{id}/processing-status ───────────

@router.get("/assessments/{assessment_id}/processing-status")
async def processing_status(
    assessment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if assessment.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(status_code=403, detail="Not authorized")

    await _sync_report_state(db, assessment)
    now = datetime.now(timezone.utc)
    created_at = _as_utc(assessment.created_at) or now
    elapsed_sec = max(0.0, (now - created_at).total_seconds())

    if assessment.status == "failed":
        return {
            "status": "failed",
            "progress": 100,
            "stage": "Voice analysis failed",
            "mlScore": assessment.ml_score,
            "mlSeverity": assessment.ml_severity,
            "reportReady": False,
            "elapsedSec": elapsed_sec,
            "remainingTargetSec": 0.0,
            "targetMaxSec": 60,
        }

    job = (await db.execute(
        select(ProcessingJob)
        .where(ProcessingJob.assessment_id == assessment_id)
        .order_by(desc(ProcessingJob.created_at))
        .limit(1)
    )).scalar_one_or_none()

    completed = _is_report_ready(assessment)
    raw_progress = job.progress_pct if job and job.progress_pct is not None else 5
    progress = 100 if completed else max(0, min(95, int(raw_progress)))
    stage = "Completed" if completed else (job.stage if job and job.stage else "Loading voice responses")
    stage_elapsed_sec = None
    if job and job.started_at:
        stage_elapsed_sec = max(0.0, (now - _as_utc(job.started_at)).total_seconds())

    return {
        "status": "completed" if completed else "processing",
        "progress": progress,
        "stage": stage,
        "reportReady": completed,
        "reportStatus": assessment.report_status,
        "isReportReady": assessment.is_report_ready,
        "mlScore": assessment.ml_score,
        "mlSeverity": assessment.ml_severity,
        "elapsedSec": elapsed_sec,
        "stageElapsedSec": stage_elapsed_sec,
        "remainingTargetSec": max(0.0, 120 - elapsed_sec),
        "targetMaxSec": 120,
    }


# ── Background ML inference task ─────────────────────

async def _run_ml_inference(assessment_id: str, user_id: str, audio_file_ids: list):
    """Background task: find audio files, send to ML model, store results."""
    from pathlib import Path
    from config.settings import get_settings

    settings = get_settings()
    client = MLClient()

    try:
        async with async_session_factory() as db:
            assessment = (await db.execute(
                select(Assessment).where(Assessment.id == assessment_id)
            )).scalar_one_or_none()
            if not assessment:
                logger.warning(f"[ML] Assessment not found: {assessment_id}")
                return

            if _is_report_ready(assessment):
                return

            job = (await db.execute(
                select(ProcessingJob)
                .where(ProcessingJob.assessment_id == assessment_id)
                .order_by(desc(ProcessingJob.created_at))
                .limit(1)
            )).scalar_one_or_none()
            if job:
                job.status = "running"
                job.progress_pct = 10
                job.stage = "Loading voice responses"
                if not job.started_at:
                    job.started_at = datetime.now(timezone.utc)
                await db.commit()

            answers = (await db.execute(
                select(AssessmentAnswer)
                .where(AssessmentAnswer.assessment_id == assessment_id)
                .order_by(AssessmentAnswer.question_id)
            )).scalars().all()

            media_ids = [answer.audio_file_id for answer in answers if answer.audio_file_id]
            if not media_ids:
                media_ids = audio_file_ids

            result = await db.execute(
                select(MediaFile).where(MediaFile.id.in_(media_ids))
            )
            media_files = result.scalars().all()
            media_by_id = {media.id: media for media in media_files}

            if not media_files:
                logger.warning(f"[ML] No audio files found for assessment {assessment_id}")
                assessment.status = "failed"
                if job:
                    job.status = "failed"
                    job.progress_pct = 100
                    job.stage = "No audio files found"
                    job.finished_at = datetime.now(timezone.utc)
                await db.commit()
                return

            # Use the most informative recording for the model call. This avoids
            # eight serial Whisper/model passes while still avoiding the old bug
            # where only question 1 was always analyzed.
            audio_file = max(
                media_files,
                key=lambda media: (
                    media.file_size or 0,
                    media.created_at.timestamp() if media.created_at else 0,
                ),
            )
            audio_path = Path(settings.STORAGE_LOCAL_PATH) / audio_file.storage_key

            if not audio_path.exists():
                logger.error(f"[ML] Audio file not found on disk: {audio_path}")
                assessment.status = "failed"
                if job:
                    job.status = "failed"
                    job.progress_pct = 100
                    job.stage = "Audio file missing"
                    job.finished_at = datetime.now(timezone.utc)
                await db.commit()
                return

            if job:
                job.progress_pct = 35
                job.stage = "Analyzing voice patterns"
                await db.commit()

            ml_task = asyncio.create_task(
                client.predict_extended(
                    audio_path=str(audio_path),
                    participant_id=user_id,
                )
            )
            while not ml_task.done():
                await asyncio.sleep(2)
                if not job:
                    continue
                job.progress_pct = min(75, (job.progress_pct or 35) + 8)
                job.stage = "Analyzing voice patterns" if job.progress_pct < 75 else "Generating score report"
                await db.commit()

            ml_result = await ml_task

            if job:
                job.progress_pct = 80
                job.stage = "Generating score report"
                await db.commit()

            raw_score = float(ml_result.get("phq_total") or ml_result.get("phq8_score") or 0)
            ml_score = max(0.0, min(24.0, raw_score))
            total_score = int(round(ml_score))
            severity = normalize_severity_label(ml_result.get("severity"), ml_score)

            assessment.ml_score = ml_score
            assessment.ml_severity = severity
            assessment.ml_num_chunks = ml_result.get("num_chunks")
            assessment.score_total = total_score
            assessment.severity = get_severity_label(total_score)
            assessment.status = "completed"
            assessment.report_status = "available"
            assessment.is_report_ready = True

            item_scores = distribute_total_score(total_score, answers, media_by_id)
            for answer in answers:
                if answer.question_id in item_scores:
                    answer.score = item_scores[answer.question_id]

            confidence = ml_result.get("confidence", {})
            audio_quality = ml_result.get("audio_quality", {})
            detail = (await db.execute(
                select(AssessmentMLDetail).where(AssessmentMLDetail.assessment_id == assessment_id)
            )).scalar_one_or_none()
            if not detail:
                detail = AssessmentMLDetail(assessment_id=assessment_id)
                db.add(detail)
            detail.confidence_mean = confidence.get("mean")
            detail.confidence_std = confidence.get("std")
            detail.ci_lower = confidence.get("ci_lower")
            detail.ci_upper = confidence.get("ci_upper")
            detail.audio_quality_score = audio_quality.get("quality")
            detail.audio_snr_db = audio_quality.get("snr_db")
            detail.audio_speech_prob = audio_quality.get("speech_prob")
            detail.behavioral_json = json.dumps(ml_result.get("behavioral", {}))
            detail.inference_time_ms = (ml_result.get("inference_time_s", 0) * 1000)

            if job:
                job.status = "succeeded"
                job.progress_pct = 100
                job.stage = "Completed"
                job.finished_at = datetime.now(timezone.utc)

            await db.commit()
            logger.info(f"[ML] Assessment {assessment_id}: score={ml_score}")

    except Exception as e:
        logger.error(f"[ML] Inference failed for assessment {assessment_id}: {e}")
        try:
            async with async_session_factory() as db:
                assessment = (await db.execute(
                    select(Assessment).where(Assessment.id == assessment_id)
                )).scalar_one_or_none()
                if assessment:
                    assessment.status = "failed"
                    job = (await db.execute(
                        select(ProcessingJob)
                        .where(ProcessingJob.assessment_id == assessment_id)
                        .order_by(desc(ProcessingJob.created_at))
                        .limit(1)
                    )).scalar_one_or_none()
                    if job:
                        job.status = "failed"
                        job.progress_pct = 100
                        job.stage = "Voice analysis failed"
                        job.error_message = str(e)
                        job.finished_at = datetime.now(timezone.utc)
                    await db.commit()
        except Exception as cleanup_exc:
            logger.exception(
                f"[ML] Failed to mark assessment {assessment_id} as failed: {cleanup_exc}"
            )


# ── GET /assessments/{id}/ml-details ─────────────────

@router.get("/assessments/{assessment_id}/ml-details")
async def get_ml_details(
    assessment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    assessment = (await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )).scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if assessment.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(status_code=403, detail="Not authorized")

    detail = (await db.execute(
        select(AssessmentMLDetail).where(AssessmentMLDetail.assessment_id == assessment_id)
    )).scalar_one_or_none()

    return {"mlDetails": ml_detail_payload(detail, assessment)}
