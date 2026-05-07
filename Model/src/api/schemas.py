"""
[LAYER_START] Session 9: API Schemas
Request/response models for the REST API.

[INFERENCE_PATH] HTTP request → validated schema → inference pipeline → response schema.
"""

from datetime import datetime
from typing import List, Optional, Any

from pydantic import BaseModel, Field


# =========================================================
# Response Schemas
# =========================================================


class PredictionResponse(BaseModel):
    """Single prediction result returned by the API."""

    participant_id: str = Field(..., description="Identifier for the audio sample")
    phq8_score: float = Field(..., ge=0.0, le=24.0, description="Predicted PHQ-8 score (0-24)")
    severity: str = Field(..., description="Clinical severity category")
    num_chunks: int = Field(..., ge=0, description="Number of audio chunks processed")
    item_scores: Optional[List[int]] = Field(
        default=None, description="Optional 0-3 item-level scores"
    )
    debug: Optional[dict[str, Any]] = Field(
        default=None, description="Optional debug payload with intermediate outputs"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 UTC timestamp of prediction",
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""

    predictions: List[PredictionResponse]
    total: int = Field(..., description="Total number of files processed")
    failed: List[str] = Field(
        default_factory=list, description="Files that failed processing"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Inference device (cpu/cuda)")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error message")


class ExtendedPredictionResponse(BaseModel):
    """Extended prediction with confidence, audio quality, and behavioral features."""

    participant_id: str = Field(..., description="Identifier for the audio sample")
    phq8_score: float = Field(..., ge=0.0, le=24.0, description="Predicted PHQ-8 score")
    severity: str = Field(..., description="Clinical severity category")
    num_chunks: int = Field(..., ge=0, description="Number of audio chunks processed")
    inference_time_s: float = Field(..., description="Total inference time in seconds")
    item_scores: Optional[List[int]] = Field(
        default=None, description="Optional 0-3 item-level scores"
    )
    debug: Optional[dict[str, Any]] = Field(
        default=None, description="Optional debug payload with intermediate outputs"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )
    confidence: dict = Field(..., description="MC Dropout uncertainty: mean, std, ci_lower, ci_upper")
    audio_quality: dict = Field(..., description="Audio quality metrics: rms, snr_db, speech_prob, quality")
    behavioral: dict = Field(..., description="Behavioral features extracted from audio")


# ── Multimodal Schemas ────────────────────────────────

class MultimodalAudioInput(BaseModel):
    """Audio features as JSON arrays."""
    mfcc: Optional[List[List[float]]] = Field(None, description="MFCC features (N×120)")
    egemaps: Optional[List[List[float]]] = Field(None, description="eGeMAPS features (N×88)")
    behavioral: Optional[List[float]] = Field(None, description="Behavioral features (16,)")


class MultimodalVideoInput(BaseModel):
    """Video features as JSON arrays."""
    openface: Optional[List[List[float]]] = Field(None, description="OpenFace features (T×49)")
    cnn_embed: Optional[List[List[float]]] = Field(None, description="CNN embeddings (T×512)")


class MultimodalTextInput(BaseModel):
    """Text features."""
    raw_text: Optional[str] = Field(None, description="Raw transcript text")
    embeddings: Optional[List[List[float]]] = Field(None, description="Pre-extracted text embeddings (N×384)")


class MultimodalRequest(BaseModel):
    """Full multimodal prediction request."""
    session_id: Optional[str] = Field(None, description="Session identifier")
    participant_id: str = Field("unknown", description="Participant identifier")
    audio_features: Optional[MultimodalAudioInput] = None
    video_features: Optional[MultimodalVideoInput] = None
    text_features: Optional[MultimodalTextInput] = None


class MultimodalPredictionResponse(BaseModel):
    """Multimodal prediction response."""
    session_id: Optional[str] = None
    participant_id: str
    phq8_score: float = Field(..., ge=0.0, le=24.0)
    severity: str
    confidence: float
    modalities_used: List[str]
    modality_contributions: dict
    inference_time_s: float
    debug: Optional[dict] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
    )

