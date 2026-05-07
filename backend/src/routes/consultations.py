"""Patient-facing consultation lifecycle management routes."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload
from typing import Optional

from database import get_db
from src.middleware.deps import require_patient
from src.models import Assessment, Consultation, Doctor, User

router = APIRouter(prefix="/consultations", tags=["consultations"])


def _doctor_payload(doctor: Doctor) -> dict:
    """Safe doctor payload conversion."""
    if not doctor:
        return None
    
    specialization = None
    try:
        if doctor.user:
            specialization = doctor.user.specialization
    except Exception:
        pass
    
    return {
        "id": doctor.id,
        "name": doctor.name or "Doctor",
        "email": doctor.email or "",
        "phone": doctor.phone or "",
        "fee": doctor.fee or 0,
        "specialization": specialization,
    }


def _assessment_payload(assessment: Assessment) -> dict:
    """Safe assessment payload conversion."""
    if not assessment:
        return None
    
    return {
        "id": assessment.id,
        "score": assessment.score_total,
        "severity": assessment.severity or "Unknown",
        "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
    }


def _consultation_payload(consultation: Consultation) -> dict:
    """Convert Consultation ORM to API payload with safe null handling."""
    if not consultation:
        return None
    
    return {
        "id": consultation.id,
        "status": consultation.status,
        "doctor": _doctor_payload(consultation.doctor),
        "assessment": _assessment_payload(consultation.assessment),
        "createdAt": consultation.created_at.isoformat() if consultation.created_at else None,
        "startedAt": consultation.started_at.isoformat() if consultation.started_at else None,
        "endedAt": consultation.ended_at.isoformat() if consultation.ended_at else None,
        "stopReason": consultation.stop_reason,
    }


async def _get_consultation_with_relations(
    db: AsyncSession, consultation_id: str, patient_id: str
) -> Consultation:
    """Fetch consultation with eagerly loaded relationships."""
    result = await db.execute(
        select(Consultation)
        .where(
            and_(
                Consultation.id == consultation_id,
                Consultation.patient_id == patient_id,
            )
        )
        .options(
            joinedload(Consultation.doctor).joinedload(Doctor.user),
            joinedload(Consultation.assessment),
        )
    )
    return result.unique().scalar_one_or_none()


@router.get("/active")
async def get_active_consultation(
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get the current active or pending consultation for the patient."""
    result = await db.execute(
        select(Consultation)
        .where(
            Consultation.patient_id == user.id,
            Consultation.status.in_(["pending", "active"]),
        )
        .options(
            joinedload(Consultation.doctor).joinedload(Doctor.user),
            joinedload(Consultation.assessment),
        )
        .order_by(desc(Consultation.created_at))
        .limit(1)
    )
    consultation = result.unique().scalar_one_or_none()

    return {"consultation": _consultation_payload(consultation) if consultation else None}


@router.get("/history")
async def get_consultation_history(
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get past (stopped, completed, rejected, cancelled) consultations for the patient."""
    result = await db.execute(
        select(Consultation)
        .where(
            Consultation.patient_id == user.id,
            Consultation.status.in_(["stopped", "completed", "rejected", "cancelled"]),
        )
        .options(
            joinedload(Consultation.doctor).joinedload(Doctor.user),
            joinedload(Consultation.assessment),
        )
        .order_by(desc(Consultation.created_at))
    )
    consultations = result.unique().scalars().all()

    return {
        "items": [_consultation_payload(c) for c in consultations],
    }


@router.post("/{consultation_id}/stop")
async def stop_consultation(
    consultation_id: str,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Stop an active or pending consultation."""
    consultation = await _get_consultation_with_relations(
        db, consultation_id, user.id
    )

    if not consultation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultation not found."
        )

    if consultation.status not in ("pending", "active"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot stop consultation with status '{consultation.status}'. Only pending or active consultations can be stopped.",
        )

    # Update consultation status and end time
    consultation.status = "stopped"
    consultation.ended_at = datetime.now(timezone.utc)
    
    await db.flush()
    await db.commit()
    await db.refresh(consultation)

    return {"consultation": _consultation_payload(consultation)}


@router.get("")
async def list_all_consultations(
    status_filter: Optional[str] = None,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """List all consultations for the patient with optional status filter."""
    query = select(Consultation).where(Consultation.patient_id == user.id)

    if status_filter and status_filter.strip():
        valid_statuses = ["pending", "active", "stopped", "completed", "rejected", "cancelled"]
        status_lower = status_filter.strip().lower()
        if status_lower in valid_statuses:
            query = query.where(Consultation.status == status_lower)

    query = query.options(
        joinedload(Consultation.doctor).joinedload(Doctor.user),
        joinedload(Consultation.assessment),
    ).order_by(desc(Consultation.created_at))

    result = await db.execute(query)
    consultations = result.unique().scalars().all()

    return {
        "items": [_consultation_payload(c) for c in consultations],
    }


@router.get("/{consultation_id}")
async def get_consultation_detail(
    consultation_id: str,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a specific consultation."""
    consultation = await _get_consultation_with_relations(
        db, consultation_id, user.id
    )

    if not consultation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Consultation not found."
        )

    return {"consultation": _consultation_payload(consultation)}
