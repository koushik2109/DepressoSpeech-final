"""
Re-evaluate the most recent assessment by:
  1. Combining ALL feature files from every multimodal session created in the
     same time window as the most recent regular assessment.
  2. Running ONE combined ML inference for a full-session PHQ-8 score.
  3. Updating both multimodal_sessions and assessments tables.

Usage (from backend/ dir):
    /home/koushik_2109/.venvs/global/bin/python3 reeval.py
"""
import asyncio
import json
import sys
import numpy as np
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from database import async_session_factory
from sqlalchemy import select

settings = get_settings()
STORAGE_BASE = Path(settings.STORAGE_LOCAL_PATH).parent / "multimodal"


def _severity_label(score: float) -> str:
    if score < 5:
        return "Minimal"
    if score < 10:
        return "Mild"
    if score < 15:
        return "Moderate"
    if score < 20:
        return "Moderately Severe"
    return "Severe"


def _load_csv(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.nan_to_num(arr, nan=1e-6, posinf=1e-6, neginf=-1e-6)


def _combine_sessions(sessions, storage_base: Path):
    """Concatenate feature rows from all sessions to form one long sequence."""
    audios, videos, texts = [], [], []
    for s in sessions:
        # Prefer audio_combined.csv (62-dim MFCC+eGeMAPS) — what the preprocessor expects
        session_dir = storage_base / s.id
        combined_path = session_dir / "audio_combined.csv"
        if combined_path.exists():
            audios.append(_load_csv(combined_path))
        elif s.audio_mfcc_key and s.audio_egemaps_key:
            mfcc_p     = storage_base / s.audio_mfcc_key
            egemaps_p  = storage_base / s.audio_egemaps_key
            if mfcc_p.exists() and egemaps_p.exists():
                mfcc    = _load_csv(mfcc_p)
                egemaps = _load_csv(egemaps_p)
                n = min(len(mfcc), len(egemaps))
                audios.append(np.concatenate([mfcc[:n], egemaps[:n]], axis=1))
        elif s.audio_mfcc_key:
            p = storage_base / s.audio_mfcc_key
            if p.exists():
                audios.append(_load_csv(p))

        if s.video_openface_key:
            p = storage_base / s.video_openface_key
            if p.exists():
                videos.append(_load_csv(p))
        if s.text_key:
            p = storage_base / s.text_key
            if p.exists():
                texts.append(_load_csv(p))

    audio_combined = np.concatenate(audios, axis=0) if audios else None
    video_combined = np.concatenate(videos, axis=0) if videos else None
    text_combined  = np.concatenate(texts,  axis=0) if texts  else None
    return audio_combined, video_combined, text_combined


async def main():
    from src.routes.multimodal import MultimodalSession
    from src.models.models import Assessment, AssessmentAnswer
    import httpx

    # ── 1. Find the most-recent regular assessment ──────────────────────────
    async with async_session_factory() as db:
        latest_assessment = (await db.execute(
            select(Assessment).order_by(Assessment.created_at.desc()).limit(1)
        )).scalar_one_or_none()

        if not latest_assessment:
            print("No regular assessments found.")
            return

        print(f"Target assessment : {latest_assessment.id}")
        print(f"  current score   : {latest_assessment.score_total}  severity: {latest_assessment.severity}")
        print(f"  created_at      : {latest_assessment.created_at}")

        # ── 2. Grab multimodal sessions from ±5 min window ─────────────────
        ts = latest_assessment.created_at
        window_start = ts - timedelta(minutes=1)
        window_end   = ts + timedelta(minutes=10)

        mm_sessions = (await db.execute(
            select(MultimodalSession)
            .where(
                MultimodalSession.user_id == latest_assessment.user_id,
                MultimodalSession.status  == "completed",
                MultimodalSession.created_at >= window_start,
                MultimodalSession.created_at <= window_end,
            )
            .order_by(MultimodalSession.created_at)
        )).scalars().all()

        print(f"\nFound {len(mm_sessions)} multimodal sessions in window:")
        for s in mm_sessions:
            print(f"  {s.id[:8]}  phq8={s.phq8_score:.2f}  severity={s.severity}")

        if not mm_sessions:
            print("No multimodal sessions — nothing to do.")
            return

        # ── 3. Combine all features and run ONE inference ───────────────────
        audio_arr, video_arr, text_arr = _combine_sessions(mm_sessions, STORAGE_BASE)

        payload: dict = {"session_id": latest_assessment.id, "participant_id": "reeval"}
        if audio_arr is not None:
            payload["audio_features"] = audio_arr.tolist()
        if video_arr is not None:
            payload["video_features"] = video_arr.tolist()
        if text_arr is not None:
            payload["text_features"] = text_arr.tolist()

        print(f"\nCombined shapes — audio:{audio_arr.shape if audio_arr is not None else None}  "
              f"video:{video_arr.shape if video_arr is not None else None}  "
              f"text:{text_arr.shape if text_arr is not None else None}")

        ml_url = settings.ML_MODEL_URL.rstrip("/")
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{ml_url}/predict/multimodal", json=payload)
            resp.raise_for_status()
            result = resp.json()

        raw_phq   = result.get("phq_total") or result.get("phq8_score") or 0.0
        new_score = float(max(0.0, min(24.0, raw_phq)))
        severity  = _severity_label(new_score)
        conf      = result.get("confidence", 0.0)
        class_prob = result.get("classification") or result.get("depression_probability")
        modality_scores = result.get("modality_scores") or result.get("modality_contributions") or {}

        print("\nML result:")
        print(f"  phq_total       : {raw_phq:.4f}  → capped: {new_score:.2f}")
        print(f"  severity        : {severity}")
        print(f"  confidence      : {conf:.4f}")
        print(f"  depression_prob : {class_prob:.4f}" if class_prob else "  depression_prob : N/A")
        print(f"  modality_scores : {modality_scores}")

        # ── 4. Update regular assessment ─────────────────────────────────────
        setattr(latest_assessment, "score_total", round(new_score))
        setattr(latest_assessment, "ml_score", new_score)
        setattr(latest_assessment, "ml_severity", severity)
        setattr(latest_assessment, "severity", severity)
        setattr(latest_assessment, "status", "completed")
        setattr(latest_assessment, "report_status", "available")
        setattr(latest_assessment, "is_report_ready", True)

        # ── 5. Snapshot per-Q scores BEFORE overwriting session phq8_score ─────
        # Must happen before step 5 because step 5 overwrites phq8_score → combined value.
        q_score_map_pre: dict[int, int] = {}
        for s in mm_sessions:
            try:
                dbg_val = getattr(s, "debug_json", "{}") or "{}"
                dbg = json.loads(str(dbg_val))
                qid = dbg.get("question_id")
                phq_val = getattr(s, "phq8_score", None)
                if qid and phq_val is not None:
                    # phq8_score is full PHQ-8 total (0-24) from this one question's video
                    # divide by 8 to get per-question estimate in [0, 3]
                    q_score_map_pre[int(qid)] = max(0, min(3, round(float(phq_val) / 8.0)))
            except Exception:
                pass

        # ── 6. Update each multimodal session with combined inference result ──
        # NOTE: do NOT overwrite phq8_score — each session keeps its own per-question score.
        # The combined phq_total lives only in assessment.ml_score.
        for s in mm_sessions:
            setattr(s, "confidence", conf)
            setattr(s, "depression_probability", float(class_prob) if class_prob else None)
            setattr(s, "is_classification", class_prob is not None)
            setattr(s, "predicted_label", int(class_prob > 0.5) if class_prob else None)
            setattr(s, "audio_contribution", modality_scores.get("audio", getattr(s, "audio_contribution", 0.0)))
            setattr(s, "video_contribution", modality_scores.get("video", getattr(s, "video_contribution", 0.0)))
            setattr(s, "text_contribution", modality_scores.get("text", getattr(s, "text_contribution", 0.0)))

        # ── 7. Set per-question answer scores using pre-snapshot map ──────────
        answers = (await db.execute(
            select(AssessmentAnswer)
            .where(AssessmentAnswer.assessment_id == latest_assessment.id)
            .order_by(AssessmentAnswer.question_id)
        )).scalars().all()

        if answers:
            q_score_map = q_score_map_pre

            mapped_count = 0
            for answer in answers:
                qid = getattr(answer, "question_id", None)
                if qid and int(qid) in q_score_map:
                    setattr(answer, "score", q_score_map[int(qid)])
                    mapped_count += 1
                else:
                    # Fallback: distribute combined phq_total evenly for unmapped questions
                    n = len(answers)
                    total_int = round(new_score)
                    setattr(answer, "score", min(3, total_int // n))

            if mapped_count:
                print(f"\n  Per-Q from individual sessions (phq8/8): {q_score_map}")
                print(f"  Mapped {mapped_count}/{len(answers)} answers with real per-Q scores")
            else:
                n = len(answers)
                total_int = round(new_score)
                base, rem = total_int // n, total_int % n
                for i, answer in enumerate(answers):
                    setattr(answer, "score", min(3, base + (1 if i < rem else 0)))
                print(f"\n  No question_id in sessions (old data) — distributed {total_int} over {n} answers")
            print(f"  Updated {len(answers)} answer rows")

        await db.commit()
        print(f"\n✓ Assessment {latest_assessment.id[:8]} updated:")
        print(f"    score_total : {getattr(latest_assessment, 'score_total', 0)}/24")
        print(f"    severity    : {getattr(latest_assessment, 'severity', 'N/A')}")
        print(f"    ml_score    : {getattr(latest_assessment, 'ml_score', 0.0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
