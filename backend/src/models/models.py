"""SQLAlchemy ORM models for MindScope."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    LargeBinary,
    String,
    Integer,
    SmallInteger,
    Float,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship

from database.base import Base


def _utcnow():
    return datetime.now(timezone.utc)


def _uuid():
    return str(uuid.uuid4())


# ── Users ──────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    role = Column(String(16), nullable=False)  # patient | doctor | admin
    name = Column(String(120), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(Text, nullable=False)

    # Patient fields
    age = Column(SmallInteger, nullable=True)
    basic_info = Column(Text, nullable=True)

    # Doctor fields
    specialization = Column(String(120), nullable=True)
    license_number = Column(String(80), nullable=True)
    clinic_name = Column(String(160), nullable=True)
    years_experience = Column(SmallInteger, nullable=True)

    # Email verification
    is_verified = Column(Boolean, default=False)
    verification_otp = Column(String(6), nullable=True)
    otp_expires_at = Column(DateTime(timezone=True), nullable=True)

    status = Column(String(16), default="active")
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    assessments = relationship("Assessment", back_populates="user", lazy="selectin")
    media_files = relationship("MediaFile", back_populates="owner", lazy="selectin")
    doctor_profile = relationship("Doctor", back_populates="user", lazy="selectin", uselist=False)

    __table_args__ = (
        Index("ix_users_role", "role"),
        Index("ix_users_created_at", "created_at"),
    )


# ── Doctor Profiles ───────────────────────────────────

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, unique=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(32), nullable=True)
    fee = Column(Float, nullable=False, default=100.0)
    is_available = Column(Boolean, default=False, nullable=False)
    patient_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user = relationship("User", back_populates="doctor_profile")
    assignments = relationship("DoctorAssignment", back_populates="doctor", lazy="selectin")

    __table_args__ = (
        Index("ix_doctors_fee_available", "fee", "is_available"),
    )


# ── Doctor Assignments (legacy doctor-facing queue) ───

class DoctorAssignment(Base):
    __tablename__ = "doctor_assignments"

    id = Column(String(36), primary_key=True, default=_uuid)
    doctor_id = Column(String(36), ForeignKey("doctors.id"), nullable=False, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    assessment_id = Column(String(36), ForeignKey("assessments.id"), nullable=True, index=True)
    status = Column(String(16), default="pending")  # pending | accepted | rejected | completed
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    doctor = relationship("Doctor", back_populates="assignments")
    patient = relationship("User", lazy="selectin", foreign_keys=[patient_id])
    assessment = relationship("Assessment", lazy="selectin")

    __table_args__ = (
        Index("ix_doctor_assignments_patient_created", "patient_id", "created_at"),
        Index("ix_doctor_assignments_doctor_created", "doctor_id", "created_at"),
    )


# ── Consultations (patient-facing lifecycle) ───────────

class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(String(36), primary_key=True, default=_uuid)
    doctor_id = Column(String(36), ForeignKey("doctors.id"), nullable=False, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    assessment_id = Column(String(36), ForeignKey("assessments.id"), nullable=True, index=True)
    doctor_assignment_id = Column(String(36), ForeignKey("doctor_assignments.id"), nullable=True, index=True)
    status = Column(
        String(16),
        default="pending",
        nullable=False,
    )  # pending | active | stopped | completed | rejected | cancelled
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    stop_reason = Column(Text, nullable=True)

    doctor = relationship("Doctor", lazy="selectin")
    patient = relationship("User", lazy="selectin", foreign_keys=[patient_id])
    assessment = relationship("Assessment", lazy="selectin")

    __table_args__ = (
        Index("ix_consultations_patient_status", "patient_id", "status"),
        Index("ix_consultations_doctor_status", "doctor_id", "status"),
        Index("ix_consultations_patient_created", "patient_id", "created_at"),
    )


# ── Assessments ────────────────────────────────────────

class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    question_set_version = Column(String(64), default="phq8_v1")
    score_total = Column(SmallInteger, nullable=True)
    severity = Column(String(32), nullable=True)
    recording_count = Column(SmallInteger, default=0)
    status = Column(String(16), default="completed")  # completed | processing | failed
    report_status = Column(String(16), default="pending")  # pending | available
    is_report_ready = Column(Boolean, default=False, nullable=False)
    doctor_remarks = Column(Text, nullable=True)

    # ML inference results
    ml_score = Column(Float, nullable=True)
    ml_severity = Column(String(32), nullable=True)
    ml_num_chunks = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="assessments")
    answers = relationship("AssessmentAnswer", back_populates="assessment", lazy="selectin")

    __table_args__ = (
        Index("ix_assessments_user_created", "user_id", "created_at"),
        Index("ix_assessments_severity", "severity", "created_at"),
    )


# ── Assessment Answers ─────────────────────────────────

class AssessmentAnswer(Base):
    __tablename__ = "assessment_answers"

    id = Column(String(36), primary_key=True, default=_uuid)
    assessment_id = Column(String(36), ForeignKey("assessments.id"), nullable=False, index=True)
    question_id = Column(Integer, nullable=False)
    score = Column(SmallInteger, nullable=False)
    duration_sec = Column(Float, nullable=True)
    audio_file_id = Column(String(36), ForeignKey("media_files.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    assessment = relationship("Assessment", back_populates="answers")
    audio_file = relationship("MediaFile", lazy="selectin")


# ── Media Files ────────────────────────────────────────

class MediaFile(Base):
    __tablename__ = "media_files"

    id = Column(String(36), primary_key=True, default=_uuid)
    owner_user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=True)
    storage_key = Column(Text, nullable=False)
    mime_type = Column(String(80), nullable=True)
    file_size = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    status = Column(String(16), default="available")
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    owner = relationship("User", back_populates="media_files")
    file_data = relationship("MediaFileData", back_populates="media_file", uselist=False, cascade="all, delete-orphan")


# ── Media File Binary Data (BYTEA / PostgreSQL file storage) ──

class MediaFileData(Base):
    """Stores raw audio/video bytes inside the database.

    One row per file — kept in a separate table so that metadata queries
    on media_files (listing, status checks) never load the binary payload.

    Used when STORAGE_PROVIDER=postgres.  The storage_key on the parent
    MediaFile row is set to 'db:<file_id>' to signal DB storage.
    """
    __tablename__ = "media_file_data"

    file_id = Column(String(36), ForeignKey("media_files.id", ondelete="CASCADE"), primary_key=True)
    data = Column(LargeBinary, nullable=False)

    media_file = relationship("MediaFile", back_populates="file_data")


# ── Assessment ML Details ─────────────────────────────

class AssessmentMLDetail(Base):
    __tablename__ = "assessment_ml_details"

    id = Column(String(36), primary_key=True, default=_uuid)
    assessment_id = Column(String(36), ForeignKey("assessments.id"), nullable=False, index=True)
    confidence_mean = Column(Float, nullable=True)
    confidence_std = Column(Float, nullable=True)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)
    audio_quality_score = Column(Float, nullable=True)
    audio_snr_db = Column(Float, nullable=True)
    audio_speech_prob = Column(Float, nullable=True)
    behavioral_json = Column(Text, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    assessment = relationship("Assessment", lazy="selectin")


# ── Request Metrics ───────────────────────────────────

class RequestMetric(Base):
    __tablename__ = "request_metrics"

    id = Column(String(36), primary_key=True, default=_uuid)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(8), nullable=False)
    status_code = Column(SmallInteger, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_request_metrics_created_at", "created_at"),
    )


# ── Processing Jobs ────────────────────────────────────

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    # Plain string — no FK so it can reference assessments OR multimodal_sessions
    assessment_id = Column(String(36), nullable=True, index=True)
    job_type = Column(String(32), default="inference")
    status = Column(String(16), default="queued")  # queued | running | succeeded | failed
    progress_pct = Column(SmallInteger, default=0)
    stage = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
