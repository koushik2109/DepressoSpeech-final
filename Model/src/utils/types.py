from typing import TypedDict, Optional, Sequence, Any


class ModalityMask(TypedDict):
    audio: Sequence[bool]
    video: Sequence[bool]
    text: Sequence[bool]


class InferenceResult(TypedDict):
    phq_total: float
    phq_questions: Sequence[float]
    classification: float
    confidence: float
    modality_scores: dict[str, float]
    metadata: Optional[dict[str, Any]]
