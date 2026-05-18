import argparse
from pathlib import Path
from typing import Dict, Any

import yaml

from src.features.audio_features import AudioFeatureExtractor
from src.features.video_features import VideoFeatureExtractor
from src.features.text_features import TextFeatureExtractor
from src.features.feature_store import save_features
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.preprocessing.text_preprocessor import TextPreprocessor


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ModelV2 multimodal features")
    parser.add_argument("--config", type=Path, default=Path("configs/feature_extraction.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    audio_preprocessor = AudioPreprocessor(**config["audio"])
    audio_extractor = AudioFeatureExtractor(config["audio"])
    video_extractor = VideoFeatureExtractor(config["video"])
    text_preprocessor = TextPreprocessor()
    text_extractor = TextFeatureExtractor(config["text"]["transformer_model"])

    for subject_dir in args.input_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        participant_id = subject_dir.name
        audio_path = subject_dir / "audio.wav"
        transcript_path = subject_dir / "transcript.txt"
        openface_path = subject_dir / "openface.csv"
        if not audio_path.exists() or not transcript_path.exists() or not openface_path.exists():
            continue
        waveform = audio_preprocessor.load_audio(audio_path)
        audio_chunks = audio_preprocessor.preprocess(waveform)
        audio_features = {"audio": audio_extractor.extract_modern(waveform)}
        video_behavior = video_extractor.extract_openface(openface_path)["behavior"]
        text = transcript_path.read_text(encoding="utf-8")
        chunks = text_preprocessor.preprocess(text)
        text_features = {"text": text_extractor.encode(chunks)}
        save_features(args.output_dir / f"{participant_id}.npz", {
            **audio_features,
            "video": video_behavior,
            **text_features,
        })
