from typing import Any, Dict, List
import torch


def _detect_dim(batch: List[Dict[str, Any]], key: str) -> int:
    for item in batch:
        value = item.get(key)
        if value is not None and value.ndim == 2:
            return value.shape[1]
    return 0


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    audio_lengths = [item["audio"].shape[0] if item["audio"] is not None else 0 for item in batch]
    video_lengths = [item["video"].shape[0] if item["video"] is not None else 0 for item in batch]
    text_lengths = [item["text"].shape[0] if item["text"] is not None else 0 for item in batch]

    max_audio = max(audio_lengths) if audio_lengths else 0
    max_video = max(video_lengths) if video_lengths else 0
    max_text = max(text_lengths) if text_lengths else 0

    audio_dim = _detect_dim(batch, "audio")
    video_dim = _detect_dim(batch, "video")
    text_dim = _detect_dim(batch, "text")

    audio_tensor = torch.zeros(batch_size, max_audio, audio_dim, dtype=torch.float32)
    video_tensor = torch.zeros(batch_size, max_video, video_dim, dtype=torch.float32)
    text_tensor = torch.zeros(batch_size, max_text, text_dim, dtype=torch.float32)
    audio_mask = torch.zeros(batch_size, max_audio, dtype=torch.bool)
    video_mask = torch.zeros(batch_size, max_video, dtype=torch.bool)
    text_mask = torch.zeros(batch_size, max_text, dtype=torch.bool)

    phq_totals = torch.zeros(batch_size, dtype=torch.float32)
    phq_questions = torch.zeros(batch_size, 8, dtype=torch.float32)
    classifications = torch.zeros(batch_size, dtype=torch.float32)
    participant_ids = []

    for i, item in enumerate(batch):
        if item["audio"] is not None:
            length = item["audio"].shape[0]
            audio_tensor[i, :length] = item["audio"]
            audio_mask[i, :length] = True
        if item["video"] is not None:
            length = item["video"].shape[0]
            video_tensor[i, :length] = item["video"]
            video_mask[i, :length] = True
        if item["text"] is not None:
            length = item["text"].shape[0]
            text_tensor[i, :length] = item["text"]
            text_mask[i, :length] = True
        phq_totals[i] = item["phq_total"]
        phq_questions[i] = item["phq_questions"]
        classifications[i] = item["classification"]
        participant_ids.append(item["participant_id"])

    return {
        "audio": audio_tensor,
        "video": video_tensor,
        "text": text_tensor,
        "audio_mask": audio_mask,
        "video_mask": video_mask,
        "text_mask": text_mask,
        "phq_total": phq_totals,
        "phq_questions": phq_questions,
        "classification": classifications,
        "participant_ids": participant_ids,
    }
