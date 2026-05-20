from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    description: str


class AudioPayload(BaseModel):
    audio_path: Optional[str] = None
    audio_features: Optional[List[List[float]]] = None
    metadata: Optional[dict] = None


class VideoPayload(BaseModel):
    video_path: Optional[str] = None
    openface_csv_path: Optional[str] = None
    video_frames: Optional[List[List[int]]] = None
    video_features: Optional[List[List[float]]] = None
    metadata: Optional[dict] = None


class TextPayload(BaseModel):
    transcript: Optional[str] = None
    chunked_transcript: Optional[List[str]] = None
    metadata: Optional[dict] = None


class MultimodalPayload(BaseModel):
    audio_features: Optional[List[List[float]]] = None
    video_features: Optional[List[List[float]]] = None
    text_features: Optional[List[List[float]]] = None
    audio_mask: Optional[List[bool]] = None
    video_mask: Optional[List[bool]] = None
    text_mask: Optional[List[bool]] = None
    metadata: Optional[dict] = None


class PredictionResponse(BaseModel):
    phq_total: float
    phq_questions: List[float]
    classification: float
    confidence: float
    modality_scores: dict
    entropy: float
    metadata: Optional[dict] = None
    processing_details: Optional[dict] = None

    @field_validator("phq_questions")
    @classmethod
    def validate_phq_questions(cls, value: List[float]) -> List[float]:
        if len(value) != 8:
            raise ValueError("phq_questions must contain 8 values")
        return value
