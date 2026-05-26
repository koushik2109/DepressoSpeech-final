"""Doctor marketplace, profile, and assignment routes."""

from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from database import get_db
from src.middleware.deps import get_current_user, require_doctor, require_patient
from src.models import Assessment, AssessmentAnswer, AssessmentMLDetail, Consultation, Doctor, DoctorAssignment, MediaFile, User
from src.routes.assessments import PHQ8_QUESTIONS, ml_detail_payload
from src.services.email_service import send_card_email_async

router = APIRouter(tags=["doctors"])


class DoctorProfileRequest(BaseModel):
    email: EmailStr
    phone: str = Field(..., min_length=7, max_length=32)
    fee: float = Field(..., ge=0)
    isAvailable: bool = True


class AssignDoctorRequest(BaseModel):
    doctorId: Optional[str] = None
    assessmentId: Optional[str] = None
    autoAssign: bool = False


class AssignmentActionRequest(BaseModel):
    action: str = Field(..., pattern="^(accept|reject|complete|reassign)$")


class DoctorRemarksRequest(BaseModel):
    doctorRemarks: str = Field(default="", max_length=5000)


def _is_complete_profile(doctor: Doctor) -> bool:
    return bool(
        doctor.name
        and doctor.email
        and doctor.phone
        and doctor.fee is not None
    )


def _doctor_payload(doctor: Doctor) -> dict:
    return {
        "id": doctor.id,
        "name": doctor.name,
        "email": doctor.email,
        "phone": doctor.phone,
        "fee": doctor.fee,
        "isAvailable": doctor.is_available,
        "patientCount": doctor.patient_count or 0,
    }


def _patient_payload(patient: User) -> dict:
    if not patient:
        return {
            "id": None,
            "name": "Patient",
            "email": "Not provided",
            "phone": "Not provided",
        }
    return {
        "id": patient.id,
        "name": patient.name or "Patient",
        "email": patient.email or "Not provided",
        "phone": getattr(patient, "phone", None) or "Not provided",
    }


def _assignment_payload(assignment: DoctorAssignment, patient: User | None = None, assessment: Assessment | None = None, doctor: Doctor | None = None) -> dict:
    return {
        "id": assignment.id,
        "status": assignment.status,
        "createdAt": assignment.created_at.isoformat() if assignment.created_at else None,
        "assessmentId": assignment.assessment_id,
        "patient": _patient_payload(patient) if patient else None,
        "doctor": _doctor_payload(doctor) if doctor else None,
        "assessment": {
            "id": assessment.id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "status": "completed" if _assessment_report_ready(assessment) else assessment.status,
            "reportStatus": "available" if _assessment_report_ready(assessment) else (assessment.report_status or "pending"),
            "isReportReady": _assessment_report_ready(assessment),
            "doctorRemarks": assessment.doctor_remarks,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
        }
        if assessment
        else None,
    }


def _complete_profile_filters():
    return (
        Doctor.name != "",
        Doctor.email != "",
        Doctor.phone.is_not(None),
        Doctor.phone != "",
        Doctor.fee.is_not(None),
        User.role == "doctor",
        User.status == "active",
    )


def _assessment_report_ready(assessment: Assessment | None) -> bool:
    return bool(
        assessment
        and (
            assessment.status == "completed"
            or assessment.is_report_ready
            or assessment.report_status == "available"
            or assessment.ml_score is not None
        )
    )


@router.get("/doctors")
async def list_doctors(
    minFee: Optional[float] = Query(default=None, ge=0),
    maxFee: Optional[float] = Query(default=None, ge=0),
    isAvailable: Optional[bool] = Query(default=None),
    _user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List doctors with complete public profiles, sorted by lowest fee."""
    query = (
        select(Doctor)
        .join(User, Doctor.user_id == User.id)
        .where(*_complete_profile_filters())
    )

    if minFee is not None:
        query = query.where(Doctor.fee >= minFee)
    if maxFee is not None:
        query = query.where(Doctor.fee <= maxFee)
    if isAvailable is not None:
        query = query.where(Doctor.is_available == isAvailable)

    result = await db.execute(query.order_by(asc(Doctor.fee), asc(Doctor.name)))
    return {"items": [_doctor_payload(doctor) for doctor in result.scalars().all()]}


@router.get("/doctor/profile")
async def get_doctor_profile(
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Doctor).where(Doctor.user_id == user.id))
    doctor = result.scalar_one_or_none()

    if not doctor:
        return {
            "profile": {
                "id": None,
                "name": user.name,
                "email": user.email,
                "phone": "",
                "fee": 100.0,
                "isAvailable": False,
                "patientCount": 0,
                "profileComplete": False,
            }
        }

    payload = _doctor_payload(doctor)
    active_patient_count = (await db.execute(
        select(func.count(func.distinct(DoctorAssignment.patient_id))).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.status.in_(["accepted", "completed"]),
        )
    )).scalar() or 0
    if doctor.patient_count != active_patient_count:
        doctor.patient_count = active_patient_count
        await db.flush()
    payload["patientCount"] = active_patient_count
    payload["profileComplete"] = _is_complete_profile(doctor)
    return {"profile": payload}


@router.put("/doctor/profile")
async def update_doctor_profile(
    body: DoctorProfileRequest,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Doctor).where(Doctor.user_id == user.id))
    doctor = result.scalar_one_or_none()

    if not doctor:
        doctor = Doctor(user_id=user.id, name=user.name, email=body.email.lower())
        db.add(doctor)

    doctor.name = user.name
    doctor.email = body.email.lower()
    doctor.phone = body.phone.strip()
    doctor.fee = body.fee
    doctor.is_available = body.isAvailable
    await db.flush()

    payload = _doctor_payload(doctor)
    active_patient_count = (await db.execute(
        select(func.count(func.distinct(DoctorAssignment.patient_id))).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.status.in_(["accepted", "completed"]),
        )
    )).scalar() or 0
    if doctor.patient_count != active_patient_count:
        doctor.patient_count = active_patient_count
        await db.flush()
    payload["patientCount"] = active_patient_count
    payload["profileComplete"] = _is_complete_profile(doctor)
    return {"profile": payload}


@router.post("/assign-doctor", status_code=201)
async def assign_doctor(
    body: AssignDoctorRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    if not body.doctorId and not body.autoAssign:
        raise HTTPException(status_code=400, detail="Choose a doctor or enable auto-assign.")

    doctor = await _select_doctor(db, body.doctorId, body.autoAssign)
    assessment = await _select_assessment(db, user.id, body.assessmentId)

    existing = (await db.execute(
        select(DoctorAssignment).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.patient_id == user.id,
            DoctorAssignment.status.in_(["pending", "assigned", "accepted"]),
        )
    )).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=409,
            detail="An active assignment with this doctor already exists.",
        )

    assignment = DoctorAssignment(
        doctor_id=doctor.id,
        patient_id=user.id,
        assessment_id=assessment.id if assessment else None,
        status="pending",
    )
    db.add(assignment)
    await db.flush()

    # Create pending consultation
    consultation = Consultation(
        doctor_id=doctor.id,
        patient_id=user.id,
        assessment_id=assessment.id if assessment else None,
        doctor_assignment_id=assignment.id,
        status="pending",
    )
    db.add(consultation)
    await db.flush()
    await db.commit()

    background_tasks.add_task(
        _send_assignment_emails,
        patient=_patient_payload(user),
        doctor=_doctor_payload(doctor),
        assessment={
            "id": assessment.id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
        }
        if assessment
        else None,
    )

    return {
        "assignment": {
            "id": assignment.id,
            "status": assignment.status,
            "assessmentId": assignment.assessment_id,
            "doctor": _doctor_payload(doctor),
        }
    }


@router.get("/doctor/assignments")
async def doctor_assignments(
    status: Optional[str] = Query(default=None),
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    doctor = await _get_current_doctor(db, user)
    query = select(DoctorAssignment).where(DoctorAssignment.doctor_id == doctor.id)
    if status:
        # Treat legacy 'assigned' status as equivalent to 'pending' for queue display
        if status == "pending":
            query = query.where(DoctorAssignment.status.in_(["pending", "assigned"]))
        else:
            query = query.where(DoctorAssignment.status == status)

    assignments = (await db.execute(
        query.order_by(desc(DoctorAssignment.created_at))
    )).scalars().all()

    patient_ids = [item.patient_id for item in assignments]
    assessment_ids = [item.assessment_id for item in assignments if item.assessment_id]
    patients = {}
    assessments = {}
    if patient_ids:
        patient_rows = (await db.execute(select(User).where(User.id.in_(patient_ids)))).scalars().all()
        patients = {patient.id: patient for patient in patient_rows}
    if assessment_ids:
        assessment_rows = (await db.execute(select(Assessment).where(Assessment.id.in_(assessment_ids)))).scalars().all()
        assessments = {assessment.id: assessment for assessment in assessment_rows}

    return {
        "items": [
            _assignment_payload(
                assignment,
                patients.get(assignment.patient_id),
                assessments.get(assignment.assessment_id),
            )
            for assignment in assignments
        ]
    }


@router.patch("/doctor/assignments/{assignment_id}")
async def update_assignment_status(
    assignment_id: str,
    body: AssignmentActionRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    doctor = await _get_current_doctor(db, user)
    assignment = (await db.execute(
        select(DoctorAssignment).where(
            DoctorAssignment.id == assignment_id,
            DoctorAssignment.doctor_id == doctor.id,
        )
    )).scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found.")

    patient = (await db.execute(select(User).where(User.id == assignment.patient_id))).scalar_one_or_none()
    assessment = None
    if assignment.assessment_id:
        assessment = (await db.execute(select(Assessment).where(Assessment.id == assignment.assessment_id))).scalar_one_or_none()

    if body.action == "accept":
        if assignment.status == "accepted":
            return {"assignment": _assignment_payload(assignment, patient, assessment, doctor)}
        if assignment.status not in ("pending", "assigned"):
            raise HTTPException(status_code=409, detail="Assignment already handled.")
        assignment.status = "accepted"
        
        # Create or activate consultation when assignment is accepted
        existing_consultation = (await db.execute(
            select(Consultation).where(
                Consultation.doctor_id == doctor.id,
                Consultation.patient_id == assignment.patient_id,
                Consultation.status.in_(["pending", "active"]),
            )
        )).scalar_one_or_none()
        
        if existing_consultation:
            # Update existing pending consultation to active
            existing_consultation.status = "active"
            existing_consultation.started_at = datetime.now(timezone.utc)
        else:
            # Create new consultation
            consultation = Consultation(
                doctor_id=doctor.id,
                patient_id=assignment.patient_id,
                assessment_id=assignment.assessment_id,
                doctor_assignment_id=assignment.id,
                status="active",
                started_at=datetime.now(timezone.utc),
            )
            db.add(consultation)
        
        active_patient_count = (await db.execute(
            select(func.count(func.distinct(DoctorAssignment.patient_id))).where(
                DoctorAssignment.doctor_id == doctor.id,
                DoctorAssignment.status.in_(["accepted", "completed"]),
            )
        )).scalar() or 0
        doctor.patient_count = active_patient_count
        background_tasks.add_task(
            _send_acceptance_email,
            patient=_patient_payload(patient),
            doctor=_doctor_payload(doctor),
        )
    elif body.action == "complete":
        assignment.status = "completed"
    elif body.action == "reject":
        if assignment.status == "rejected":
            return {"assignment": _assignment_payload(assignment, patient, assessment, doctor)}
        if assignment.status == "completed":
            raise HTTPException(status_code=409, detail="Assignment already handled.")
        assignment.status = "rejected"
    elif body.action == "reassign":
        if assignment.status in ("rejected", "completed"):
            raise HTTPException(status_code=409, detail="Assignment already handled.")
        assignment.status = "rejected"
        next_doctor = await _select_reassignment_doctor(db, exclude_doctor_id=doctor.id)
        if not next_doctor:
            await db.flush()
            return {
                "assignment": _assignment_payload(assignment, patient, assessment),
                "reassigned": None,
            }
        reassigned = DoctorAssignment(
            doctor_id=next_doctor.id,
            patient_id=assignment.patient_id,
            assessment_id=assignment.assessment_id,
            status="pending",
        )
        db.add(reassigned)
        await db.flush()
        background_tasks.add_task(
            _send_assignment_emails,
            patient=_patient_payload(patient),
            doctor=_doctor_payload(next_doctor),
            assessment={
                "id": assessment.id,
                "score": assessment.score_total,
                "severity": assessment.severity,
                "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
            }
            if assessment
            else None,
        )
        return {
            "assignment": _assignment_payload(assignment, patient, assessment),
            "reassigned": _assignment_payload(reassigned, patient, assessment, next_doctor),
        }

    await db.flush()
    await db.commit()
    return {"assignment": _assignment_payload(assignment, patient, assessment)}


@router.get("/doctor/reports/{assessment_id}")
async def doctor_report(
    assessment_id: str,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    doctor = await _get_current_doctor(db, user)
    assessment = (await db.execute(select(Assessment).where(Assessment.id == assessment_id))).scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    assignment = (await db.execute(
        select(DoctorAssignment).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.patient_id == assessment.user_id,
        ).order_by(desc(DoctorAssignment.created_at)).limit(1)
    )).scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=403, detail="Report is not assigned to this doctor.")
    if assessment.user_id != assignment.patient_id:
        raise HTTPException(status_code=404, detail="Report mapping is invalid.")

    patient = (await db.execute(select(User).where(User.id == assessment.user_id))).scalar_one_or_none()
    answers = (await db.execute(
        select(AssessmentAnswer)
        .where(AssessmentAnswer.assessment_id == assessment_id)
        .order_by(AssessmentAnswer.question_id)
    )).scalars().all()

    media_ids = [answer.audio_file_id for answer in answers if answer.audio_file_id]
    media_by_id = {}
    if media_ids:
        media_files = (await db.execute(select(MediaFile).where(MediaFile.id.in_(media_ids)))).scalars().all()
        media_by_id = {media.id: media for media in media_files}

    detail = (await db.execute(
        select(AssessmentMLDetail).where(AssessmentMLDetail.assessment_id == assessment_id)
    )).scalar_one_or_none()

    questions_by_id = {item["id"]: item for item in PHQ8_QUESTIONS}
    return {
        "assignment": _assignment_payload(assignment, patient, assessment),
        "assessment": {
            "id": assessment.id,
            "userId": assessment.user_id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "recordingCount": assessment.recording_count,
            "status": "completed" if _assessment_report_ready(assessment) else assessment.status,
            "reportStatus": "available" if _assessment_report_ready(assessment) else (assessment.report_status or "pending"),
            "isReportReady": _assessment_report_ready(assessment),
            "doctorRemarks": assessment.doctor_remarks,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
            "mlScore": assessment.ml_score,
            "mlSeverity": assessment.ml_severity,
            "patient": _patient_payload(patient),
            "answers": [
                {
                    "questionId": answer.question_id,
                    "questionText": questions_by_id.get(answer.question_id, {}).get("text", ""),
                    "score": answer.score,
                    "audioFileId": answer.audio_file_id,
                    "audioUrl": f"/api/v1/files/audio/{answer.audio_file_id}" if answer.audio_file_id in media_by_id else None,
                    "fileName": getattr(media_by_id.get(answer.audio_file_id), "original_filename", None),
                    "fileSize": getattr(media_by_id.get(answer.audio_file_id), "file_size", None),
                    "mimeType": getattr(media_by_id.get(answer.audio_file_id), "mime_type", None),
                }
                for answer in answers
            ],
            "mlDetails": ml_detail_payload(detail, assessment),
        },
    }


@router.put("/doctor/reports/{assessment_id}/remarks")
async def update_doctor_report_remarks(
    assessment_id: str,
    body: DoctorRemarksRequest,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    doctor = await _get_current_doctor(db, user)
    assessment = (await db.execute(select(Assessment).where(Assessment.id == assessment_id))).scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    assignment = (await db.execute(
        select(DoctorAssignment).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.patient_id == assessment.user_id,
        ).order_by(desc(DoctorAssignment.created_at)).limit(1)
    )).scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=403, detail="Report is not assigned to this doctor.")
    if assessment.user_id != assignment.patient_id:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    assessment.doctor_remarks = body.doctorRemarks.strip() or None
    await db.flush()

    return {
        "assessment": {
            "id": assessment.id,
            "doctorRemarks": assessment.doctor_remarks,
        }
    }


@router.get("/doctor/patients/{patient_id}/reports")
async def doctor_patient_reports(
    patient_id: str,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    doctor = await _get_current_doctor(db, user)
    current_assignment = (await db.execute(
        select(DoctorAssignment)
        .where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.patient_id == patient_id,
        )
        .order_by(desc(DoctorAssignment.created_at))
        .limit(1)
    )).scalar_one_or_none()
    if not current_assignment:
        raise HTTPException(status_code=403, detail="Patient is not assigned to this doctor.")

    patient = (await db.execute(select(User).where(User.id == patient_id))).scalar_one_or_none()
    ordered_assessments = (await db.execute(
        select(Assessment)
        .where(Assessment.user_id == patient_id)
        .order_by(Assessment.created_at)
    )).scalars().all()
    ordered_assessments = sorted(
        ordered_assessments,
        key=lambda assessment: assessment.created_at.timestamp() if assessment.created_at else 0,
    )
    scores = [assessment.score_total or 0 for assessment in ordered_assessments]
    latest = ordered_assessments[-1] if ordered_assessments else None
    previous = ordered_assessments[-2] if len(ordered_assessments) > 1 else None
    first = ordered_assessments[0] if ordered_assessments else None
    improvement = (first.score_total if first and first.score_total is not None else 0) - (
        latest.score_total if latest and latest.score_total is not None else 0
    )
    average_score = (sum(scores) / len(scores)) if scores else 0
    best_score = min(scores) if scores else 0
    worst_score = max(scores) if scores else 0

    return {
        "patient": _patient_payload(patient),
        "assignment": _assignment_payload(current_assignment, patient, ordered_assessments[-1] if ordered_assessments else None, doctor),
        "metrics": {
            "totalSessions": len(ordered_assessments),
            "latestScore": latest.score_total if latest else None,
            "previousScore": previous.score_total if previous else None,
            "improvement": improvement,
            "averageScore": round(average_score, 2) if scores else 0,
            "bestScore": best_score,
            "worstScore": worst_score,
            "latestSeverity": latest.severity if latest else None,
            "latestStatus": "completed" if latest and _assessment_report_ready(latest) else (latest.status if latest else None),
        },
        "items": [
            {
                "id": assessment.id,
                "assessment": {
                    "id": assessment.id,
                    "score": assessment.score_total,
                    "severity": assessment.severity,
                    "status": "completed" if _assessment_report_ready(assessment) else assessment.status,
                    "reportStatus": "available" if _assessment_report_ready(assessment) else (assessment.report_status or "pending"),
                    "isReportReady": _assessment_report_ready(assessment),
                    "doctorRemarks": assessment.doctor_remarks,
                    "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
                    "patient": _patient_payload(patient),
                },
            }
            for assessment in ordered_assessments
        ]
    }


@router.get("/reports/{patient_id}")
async def reports_by_patient(
    patient_id: str,
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    return await doctor_patient_reports(patient_id, user, db)


@router.get("/patient/assignments")
async def patient_assignments(
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    assignments = (await db.execute(
        select(DoctorAssignment)
        .where(DoctorAssignment.patient_id == user.id)
        .order_by(desc(DoctorAssignment.created_at))
    )).scalars().all()
    doctor_ids = [item.doctor_id for item in assignments]
    doctors = {}
    if doctor_ids:
        rows = (await db.execute(select(Doctor).where(Doctor.id.in_(doctor_ids)))).scalars().all()
        doctors = {doctor.id: doctor for doctor in rows}
    return {
        "items": [
            _assignment_payload(assignment, user, None, doctors.get(assignment.doctor_id))
            for assignment in assignments
        ]
    }


async def _get_current_doctor(db: AsyncSession, user: User) -> Doctor:
    doctor = (await db.execute(select(Doctor).where(Doctor.user_id == user.id))).scalar_one_or_none()
    if not doctor:
        doctor = Doctor(
            user_id=user.id,
            name=user.name,
            email=user.email,
            fee=100.0,
            is_available=False,
        )
        db.add(doctor)
        await db.flush()
    return doctor


async def _select_doctor(
    db: AsyncSession,
    doctor_id: Optional[str],
    auto_assign: bool,
) -> Doctor:
    query = (
        select(Doctor)
        .join(User, Doctor.user_id == User.id)
        .where(
            *_complete_profile_filters(),
            Doctor.is_available.is_(True),
        )
    )

    if doctor_id:
        result = await db.execute(query.where(Doctor.id == doctor_id))
        doctor = result.scalar_one_or_none()
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor is unavailable or profile is incomplete.")
        return doctor

    if auto_assign:
        result = await db.execute(query.order_by(asc(Doctor.fee), asc(Doctor.name)).limit(1))
        doctor = result.scalar_one_or_none()
        if not doctor:
            raise HTTPException(status_code=404, detail="No available doctors with complete profiles.")
        return doctor

    raise HTTPException(status_code=400, detail="Doctor selection is required.")


async def _select_reassignment_doctor(db: AsyncSession, exclude_doctor_id: str) -> Doctor | None:
    result = await db.execute(
        select(Doctor)
        .join(User, Doctor.user_id == User.id)
        .where(
            *_complete_profile_filters(),
            Doctor.is_available.is_(True),
            Doctor.id != exclude_doctor_id,
        )
        .order_by(asc(Doctor.fee), asc(Doctor.name))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _select_assessment(
    db: AsyncSession,
    patient_id: str,
    assessment_id: Optional[str],
) -> Assessment | None:
    if assessment_id:
        result = await db.execute(
            select(Assessment).where(
                Assessment.id == assessment_id,
                Assessment.user_id == patient_id,
            )
        )
        assessment = result.scalar_one_or_none()
        if not assessment:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        return assessment

    result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id == patient_id)
        .order_by(desc(Assessment.created_at))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _send_assignment_emails(
    patient: dict,
    doctor: dict,
    assessment: dict | None,
) -> None:
    patient_summary = (
        f"Assessment: {assessment['score']}/24, {assessment['severity']}"
        if assessment
        else "No assessment was attached."
    )

    await send_card_email_async(
        to_email=doctor["email"],
        subject="MindScope patient assignment",
        title="New patient request",
        intro="A patient selected you for follow-up. Review the case in your doctor dashboard.",
        rows=[
            ("Patient name", patient["name"]),
            ("Patient email", patient["email"]),
            ("Patient phone", patient["phone"]),
            ("Assessment", patient_summary),
        ],
        footer="Accept the request to share your contact details with the patient.",
    )


async def _send_acceptance_email(patient: dict, doctor: dict) -> None:
    await send_card_email_async(
        to_email=patient["email"],
        subject="Your MindScope doctor accepted your request",
        title=f"Dr. {doctor['name']} accepted your request",
        intro="You can now contact your doctor directly using the details below.",
        rows=[
            ("Doctor name", f"Dr. {doctor['name']}"),
            ("Doctor email", doctor["email"]),
            ("Doctor phone", doctor["phone"]),
            ("Consultation fee", doctor["fee"]),
        ],
        footer="Use these details to schedule your follow-up.",
    )
