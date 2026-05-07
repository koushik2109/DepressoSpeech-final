"""
Database schema extension for assessment integrity monitoring
Add these models to the existing assessment database

Usage:
    from models import AssessmentIntegrityMetadata
    metadata = AssessmentIntegrityMetadata.create(
        assessment_id=123,
        alignment_score=85,
        quality_score=78,
        ...
    )
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from database.base import Base


class AssessmentIntegrityMetadata(Base):
    """
    Stores real-time face alignment and quality metrics for each assessment
    
    Fields:
        assessment_id: Foreign key to Assessment
        alignment_score: 0-100, based on continuous face pose validation
        quality_score: 0-100, based on lighting/blur/motion analysis
        motion_score: 0-100, stability of head movement
        continuous_validation_status: Current state of monitoring
        frame_count: Total frames processed during assessment
        degradation_events: Number of times quality dropped below threshold
        auto_pause_count: How many times recording auto-paused
        pause_resume_events: Detailed list of pause/resume events
        face_alignment_frames: Frames with valid face alignment
        total_frames: Total frames captured
        recalibration_required: Whether face re-alignment is needed
        integrity_token: Unique token for this assessment session
        created_at: Timestamp of metadata creation
    """
    
    __tablename__ = "assessment_integrity_metadata"
    
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey("assessment.id"), nullable=False, unique=True)
    
    # Scoring metrics
    alignment_score = Column(Float, default=0, nullable=False)  # 0-100
    quality_score = Column(Float, default=0, nullable=False)  # 0-100
    motion_score = Column(Float, default=0, nullable=False)  # 0-100
    composite_integrity_score = Column(Float, default=0, nullable=False)  # 0-100
    
    # Monitoring data
    continuous_validation_status = Column(String(50), default="INITIALIZING")
    frame_count = Column(Integer, default=0)
    face_alignment_frames = Column(Integer, default=0)
    degradation_events = Column(Integer, default=0)
    auto_pause_count = Column(Integer, default=0)
    recalibration_required = Column(Integer, default=0, nullable=False)
    alignment_dropout_count = Column(Integer, default=0)
    
    # Event logs
    pause_resume_events = Column(JSON, default=list)  # List of {timestamp, type, reason}
    degradation_event_log = Column(JSON, default=list)  # List of degradation events
    integrity_checkpoints = Column(JSON, default=list)  # List of state transitions
    
    # Session data
    alignment_percentage = Column(Float, default=0)  # Percentage of frames aligned
    integrity_token = Column(String(255), unique=True, nullable=False)
    
    # Metadata
    video_enabled = Column(Integer, default=0)  # Boolean
    continuous_alignment_required = Column(Integer, default=1)  # Boolean
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    assessment = relationship("Assessment", backref="integrity_metadata", uselist=False)
    
    def __repr__(self):
        return (
            f"<AssessmentIntegrityMetadata(assessment_id={self.assessment_id}, "
            f"alignment_score={self.alignment_score}, "
            f"quality_score={self.quality_score})>"
        )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "assessment_id": self.assessment_id,
            "alignment_score": self.alignment_score,
            "quality_score": self.quality_score,
            "motion_score": self.motion_score,
            "composite_integrity_score": self.composite_integrity_score,
            "frame_count": self.frame_count,
            "face_alignment_frames": self.face_alignment_frames,
            "alignment_percentage": self.alignment_percentage,
            "degradation_events": self.degradation_events,
            "auto_pause_count": self.auto_pause_count,
            "alignment_dropout_count": self.alignment_dropout_count,
            "continuous_validation_status": self.continuous_validation_status,
            "pause_resume_events": self.pause_resume_events,
            "degradation_event_log": self.degradation_event_log,
            "integrity_checkpoints": self.integrity_checkpoints,
            "integrity_token": self.integrity_token,
            "video_enabled": bool(self.video_enabled),
            "continuous_alignment_required": bool(self.continuous_alignment_required),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class IntegrityQualityMetrics(Base):
    """
    Frame-by-frame quality metrics for detailed analysis
    
    Useful for post-assessment analysis and debugging
    """
    
    __tablename__ = "integrity_quality_metrics"
    
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey("assessment.id"), nullable=False)
    
    # Per-frame metrics
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    
    # Quality measurements
    brightness = Column(Float)  # 0-255
    blur_score = Column(Float)  # Laplacian variance
    motion_pixels = Column(Float)  # Pixel displacement
    face_visibility = Column(Float)  # 0-1
    centering_score = Column(Float)  # 0-1
    lighting_quality = Column(Float)  # 0-1
    
    # Face geometry
    yaw_degrees = Column(Float)  # Head horizontal rotation
    pitch_degrees = Column(Float)  # Head vertical tilt
    roll_degrees = Column(Float)  # Head roll
    mouth_openness = Column(Float)  # 0-1
    eye_aspect_ratio = Column(Float)  # 0-1
    
    # Composite
    frame_quality_score = Column(Float)  # 0-100
    frame_issues = Column(JSON, default=list)  # List of detected issues
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class FaceMeshLandmarks(Base):
    """
    Store detected face mesh landmarks for each frame
    Optional: For detailed analysis and ML model training
    """
    
    __tablename__ = "face_mesh_landmarks"
    
    id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer, ForeignKey("assessment.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    
    # Serialized landmarks (468 points, 3D coordinates each)
    landmarks_json = Column(JSON, nullable=False)  # [{"x": 0.5, "y": 0.5, "z": 0}, ...]
    
    # Bounding box
    bbox_min_x = Column(Float)
    bbox_max_x = Column(Float)
    bbox_min_y = Column(Float)
    bbox_max_y = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# API ENDPOINT SCHEMAS
# ============================================================================

"""
API Endpoints for integrity monitoring

POST /api/v1/assessments/{id}/integrity-metadata
    Creates integrity metadata entry
    
GET /api/v1/assessments/{id}/integrity-metadata
    Retrieves stored integrity metadata
    
POST /api/v1/assessments/{id}/integrity-metrics
    Logs frame-by-frame metrics (called during assessment)
    
GET /api/v1/assessments/{id}/integrity-report
    Generates integrity report for assessment
"""

# Request/Response schemas

class IntegrityMetricsInput:
    """Frame-by-frame metrics submitted during assessment"""
    
    frame_number: int
    timestamp: float
    state: str  # SCANNING, ALIGNED, MONITORING, DEGRADED
    aligned: bool
    quality_score: float  # 0-100
    integrity_score: float  # 0-100
    issues: list  # [{"type": "MOTION_BLUR", "severity": "warning"}]
    
    # Face geometry
    yaw: float
    pitch: float
    roll: float
    
    # Quality metrics
    brightness: float
    blur_score: float
    motion: float
    face_visibility: float
    centering: float


class IntegrityReport:
    """Complete integrity report for an assessment"""
    
    assessment_id: int
    overall_alignment_score: float  # 0-100
    overall_quality_score: float  # 0-100
    overall_motion_score: float  # 0-100
    composite_integrity_score: float  # 0-100
    
    # Statistics
    total_frames: int
    aligned_frames: int
    alignment_percentage: float
    
    # Events
    degradation_events: int
    auto_pause_events: int
    recalibration_required: bool
    
    # Recommendations
    recording_valid: bool
    recommendations: list  # ["Improve lighting", "Reduce head movement"]
    
    # Timestamps
    assessment_start_time: str
    assessment_end_time: str
    duration_seconds: float


# ============================================================================
# MIGRATION SCRIPT
# ============================================================================

"""
Run this to create the new tables:

from sqlalchemy import create_engine
from database.base import Base
from models import (
    AssessmentIntegrityMetadata,
    IntegrityQualityMetrics,
    FaceMeshLandmarks
)

# Create tables
Base.metadata.create_all(bind=engine)

# Or use Alembic:
# alembic revision --autogenerate -m "Add integrity monitoring tables"
# alembic upgrade head
"""
