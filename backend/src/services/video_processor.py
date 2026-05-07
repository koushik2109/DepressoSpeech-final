"""
Video Processing Service — DepressoSpeech

Handles raw video files from webcam recordings:
    1. Save temporary video file
    2. Extract audio track via FFmpeg → WAV
    3. Extract video frames at configured FPS
    4. Generate audio features (eGeMAPS, MFCC)
    5. Generate video features (placeholder OpenFace + CNN embeddings)
    6. Optionally transcribe speech to text
    7. Clean up temporary files

No raw video is stored permanently — only extracted features survive.
"""

import asyncio
import logging
import time
import uuid
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import aiofiles

from config.settings import get_settings

logger = logging.getLogger("mindscope")
settings = get_settings()

# ── Constants ──────────────────────────────────────────

FRAME_EXTRACT_FPS = 2            # Extract 2 frames/sec (sufficient for AU analysis)
MAX_VIDEO_DURATION_SEC = 300     # 5 min max
AUDIO_SAMPLE_RATE = 16000
MFCC_DIM = 120
EGEMAPS_DIM = 88
BEHAVIORAL_DIM = 16
OPENFACE_DIM = 49
CNN_EMBED_DIM = 512
TEXT_EMBED_DIM = 384


def _temp_dir() -> Path:
    """Temporary directory for raw video processing. Cleaned after use."""
    p = Path(settings.STORAGE_LOCAL_PATH).parent / "tmp_video"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _feature_dir() -> Path:
    """Persistent directory for extracted features."""
    p = Path(settings.STORAGE_LOCAL_PATH).parent / "multimodal"
    p.mkdir(parents=True, exist_ok=True)
    return p


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


class VideoProcessor:
    """
    Processes raw webcam recordings into multimodal features.

    Pipeline:
        video.webm → FFmpeg split → audio.wav + frames/*.jpg
                   → feature extraction → {mfcc, egemaps, openface, cnn}.csv
                   → cleanup temp files
    """

    def __init__(self):
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg():
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                logger.warning("FFmpeg returned non-zero, video processing may fail")
        except FileNotFoundError:
            logger.error("FFmpeg not found! Install with: apt install ffmpeg")
            raise VideoProcessingError("FFmpeg is not installed")
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg version check timed out")

    async def save_upload(self, file_content: bytes, filename: str, session_id: str) -> Path:
        """Save uploaded video to temp directory.

        Returns:
            Path to saved temporary video file
        """
        tmp = _temp_dir() / session_id
        tmp.mkdir(parents=True, exist_ok=True)

        safe_name = f"recording_{uuid.uuid4().hex[:8]}{Path(filename).suffix or '.webm'}"
        video_path = tmp / safe_name

        async with aiofiles.open(video_path, "wb") as f:
            await f.write(file_content)

        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"Saved video upload: {video_path} ({file_size_mb:.1f} MB)")
        return video_path

    async def save_upload_stream(self, upload, filename: str, session_id: str, max_size: int) -> Tuple[Path, int]:
        """Stream an uploaded video to temp storage while enforcing a size limit."""
        tmp = _temp_dir() / session_id
        tmp.mkdir(parents=True, exist_ok=True)

        safe_name = f"recording_{uuid.uuid4().hex[:8]}{Path(filename).suffix or '.webm'}"
        video_path = tmp / safe_name
        total_size = 0

        try:
            async with aiofiles.open(video_path, "wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > max_size:
                        video_path.unlink(missing_ok=True)
                        raise VideoProcessingError(
                            f"Video too large ({total_size / 1024 / 1024:.1f} MB). "
                            f"Max: {settings.VIDEO_MAX_FILE_SIZE_MB} MB"
                        )
                    await f.write(chunk)
        except Exception:
            video_path.unlink(missing_ok=True)
            raise

        logger.info(f"Saved video upload: {video_path} ({total_size / 1024 / 1024:.1f} MB)")
        return video_path, total_size

    async def extract_audio(self, video_path: Path, session_id: str) -> Path:
        """Extract audio track from video using FFmpeg.

        Args:
            video_path: Path to input video file
            session_id: Session identifier

        Returns:
            Path to extracted WAV audio file
        """
        tmp = _temp_dir() / session_id
        audio_path = tmp / "extracted_audio.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                          # No video
            "-acodec", "pcm_s16le",         # PCM 16-bit
            "-ar", str(AUDIO_SAMPLE_RATE),  # 16kHz
            "-ac", "1",                     # Mono
            "-loglevel", "error",
            str(audio_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.error(f"FFmpeg audio extraction failed: {err_msg}")
            raise VideoProcessingError(f"Audio extraction failed: {err_msg[:200]}")

        if not audio_path.exists() or audio_path.stat().st_size < 100:
            raise VideoProcessingError("Audio extraction produced empty file")

        logger.info(f"Extracted audio: {audio_path} ({audio_path.stat().st_size / 1024:.1f} KB)")
        return audio_path

    async def extract_frames(self, video_path: Path, session_id: str, fps: int = FRAME_EXTRACT_FPS) -> Path:
        """Extract frames from video at specified FPS.

        Args:
            video_path: Path to input video
            session_id: Session identifier
            fps: Frames per second to extract

        Returns:
            Path to directory containing extracted frames
        """
        tmp = _temp_dir() / session_id
        frames_dir = tmp / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",                    # High quality JPEG
            "-loglevel", "error",
            str(frames_dir / "frame_%04d.jpg"),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")
            logger.error(f"FFmpeg frame extraction failed: {err_msg}")
            raise VideoProcessingError(f"Frame extraction failed: {err_msg[:200]}")

        frame_count = len(list(frames_dir.glob("*.jpg")))
        logger.info(f"Extracted {frame_count} frames at {fps} fps → {frames_dir}")
        return frames_dir

    async def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using FFprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")
                raise VideoProcessingError(f"Unable to read video duration: {err_msg[:200]}")
            duration = float(stdout.decode().strip())
            if not np.isfinite(duration) or duration <= 0:
                raise VideoProcessingError("Unable to read video duration")
            return duration
        except VideoProcessingError:
            raise
        except Exception as e:
            raise VideoProcessingError(f"Unable to read video duration: {e}") from e

    async def extract_audio_features(
        self, audio_path: Path, session_id: str,
    ) -> Dict[str, str]:
        """Extract eGeMAPS and MFCC features from audio.

        Uses numpy-based feature extraction as a fallback when
        specialized libraries aren't available.

        Returns:
            Dict with storage keys for mfcc.csv, egemaps.csv, behavioral.csv
        """
        feat_dir = _feature_dir() / session_id
        feat_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Try to load audio with scipy
            from scipy.io import wavfile
            sr, audio_data = wavfile.read(str(audio_path))
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        except Exception as e:
            logger.warning(f"Failed to read audio file: {e}")
            raise VideoProcessingError(f"Failed to read audio file: {e}") from e

        # Chunk audio into segments (3-second chunks with 1.5s overlap)
        chunk_size = sr * 3
        hop_size = int(sr * 1.5)
        num_chunks = max(1, (len(audio_data) - chunk_size) // hop_size + 1)

        # Generate MFCC-like features (N, 120)
        mfcc_features = np.zeros((num_chunks, MFCC_DIM), dtype=np.float32)
        for i in range(num_chunks):
            start = i * hop_size
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            if len(chunk) < 256:
                continue
            # Basic spectral features as MFCC proxy
            spectrum = np.abs(np.fft.rfft(chunk, n=256))
            # Resample spectrum to MFCC_DIM
            indices = np.linspace(0, len(spectrum) - 1, MFCC_DIM).astype(int)
            mfcc_features[i] = np.log1p(spectrum[indices])

        # Generate eGeMAPS-like features (N, 88)
        egemaps_features = np.zeros((num_chunks, EGEMAPS_DIM), dtype=np.float32)
        for i in range(num_chunks):
            start = i * hop_size
            end = min(start + chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            if len(chunk) < 256:
                continue
            # Basic prosodic features
            rms = np.sqrt(np.mean(chunk ** 2))
            zcr = np.mean(np.abs(np.diff(np.sign(chunk)))) / 2.0
            spectrum = np.abs(np.fft.rfft(chunk, n=512))
            spectral_centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-8)
            spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-8))) / (np.mean(spectrum) + 1e-8)

            egemaps_features[i, 0] = rms
            egemaps_features[i, 1] = zcr
            egemaps_features[i, 2] = spectral_centroid / len(spectrum)
            egemaps_features[i, 3] = spectral_flatness
            # Fill rest with spectral features
            indices = np.linspace(0, len(spectrum) - 1, EGEMAPS_DIM - 4).astype(int)
            egemaps_features[i, 4:] = np.log1p(spectrum[indices])

        # Behavioral features (16,)
        behavioral = np.zeros(BEHAVIORAL_DIM, dtype=np.float32)
        duration = len(audio_data) / sr
        behavioral[0] = duration                                      # total_duration
        behavioral[1] = np.mean(np.abs(audio_data))                    # mean_amplitude
        behavioral[2] = np.std(audio_data)                             # amplitude_std
        behavioral[3] = float(np.sum(np.abs(audio_data) < 0.01)) / len(audio_data)  # silence_ratio
        behavioral[9] = 1.0 - behavioral[3]                           # speaking_ratio

        # Save features
        np.savetxt(str(feat_dir / "mfcc.csv"), mfcc_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "egemaps.csv"), egemaps_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "behavioral.csv"), behavioral.reshape(1, -1), delimiter=",", fmt="%.6f")

        logger.info(f"Audio features: mfcc={mfcc_features.shape}, egemaps={egemaps_features.shape}")

        return {
            "mfcc_key": f"{session_id}/mfcc.csv",
            "egemaps_key": f"{session_id}/egemaps.csv",
            "behavioral_key": f"{session_id}/behavioral.csv",
        }

    async def extract_video_features(
        self, frames_dir: Path, session_id: str,
    ) -> Dict[str, str]:
        """Extract video features from frames.

        Generates OpenFace-style and CNN-style features.
        In production, this would call OpenFace and a CNN model.

        Returns:
            Dict with storage keys for openface.csv and cnn_embed.csv
        """
        feat_dir = _feature_dir() / session_id
        feat_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(frames_dir.glob("*.jpg"))
        num_frames = len(frame_files)

        if num_frames == 0:
            logger.warning("No frames extracted, generating minimal video features")
            num_frames = 1

        # OpenFace-style features (T, 49): pose(6) + gaze(8) + AUs(35)
        openface_features = np.zeros((num_frames, OPENFACE_DIM), dtype=np.float32)
        for i, frame_path in enumerate(frame_files):
            try:
                # Read frame to get basic visual statistics
                with open(frame_path, "rb") as f:
                    data = f.read()
                # Use file size as entropy proxy for visual activity
                file_kb = len(data) / 1024.0
                # Simulated pose (yaw, pitch, roll, x, y, z)
                openface_features[i, 0:6] = np.random.randn(6) * 0.1
                # Simulated gaze
                openface_features[i, 6:14] = np.random.randn(8) * 0.05
                # Simulated AUs based on image complexity
                au_base = np.clip(file_kb / 50.0, 0, 1)
                openface_features[i, 14:49] = np.abs(np.random.randn(35) * 0.1 + au_base * 0.3)
            except Exception:
                pass

        # CNN embedding features (T, 512)
        cnn_features = np.zeros((num_frames, CNN_EMBED_DIM), dtype=np.float32)
        for i, frame_path in enumerate(frame_files):
            try:
                with open(frame_path, "rb") as f:
                    data = f.read()
                # Generate deterministic-ish embedding from frame bytes
                seed = abs(hash(frame_path.name)) % (2**31)
                rng = np.random.default_rng(seed)
                cnn_features[i] = rng.standard_normal(CNN_EMBED_DIM).astype(np.float32) * 0.1
            except Exception:
                pass

        np.savetxt(str(feat_dir / "openface.csv"), openface_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "cnn_embed.csv"), cnn_features, delimiter=",", fmt="%.6f")

        logger.info(f"Video features: openface={openface_features.shape}, cnn={cnn_features.shape}")

        return {
            "openface_key": f"{session_id}/openface.csv",
            "cnn_key": f"{session_id}/cnn_embed.csv",
        }

    async def extract_text_features(
        self, audio_path: Path, session_id: str,
    ) -> Optional[Dict[str, str]]:
        """Generate text features from audio via speech-to-text (if available).

        Falls back to generating a placeholder transcript.
        In production, this would use Whisper or Google STT.

        Returns:
            Dict with text_key or None if STT is unavailable
        """
        feat_dir = _feature_dir() / session_id
        feat_dir.mkdir(parents=True, exist_ok=True)

        transcript = None

        # Try using Whisper if available
        try:
            model = self._get_whisper_model()
            result = model.transcribe(str(audio_path), language="en")
            transcript = result.get("text", "")
            logger.info(f"Whisper transcription: {len(transcript)} chars")
        except ImportError:
            logger.info("Whisper not available, generating placeholder text features")
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")

        if not transcript:
            # Generate placeholder text embedding (1, 384)
            text_embed = np.random.randn(1, TEXT_EMBED_DIM).astype(np.float32) * 0.1
            np.savetxt(str(feat_dir / "text_embeddings.csv"), text_embed, delimiter=",", fmt="%.6f")
            return {
                "text_key": f"{session_id}/text_embeddings.csv",
                "raw_text": None,
            }

        # If we have a transcript, generate SBERT-like embeddings
        try:
            model = self._get_sentence_transformer_model()
            # Split into sentences
            sentences = [s.strip() for s in transcript.split(".") if s.strip()]
            if not sentences:
                sentences = [transcript]
            embeddings = model.encode(sentences)
            np.savetxt(str(feat_dir / "text_embeddings.csv"), embeddings, delimiter=",", fmt="%.6f")
        except ImportError:
            text_embed = np.random.randn(1, TEXT_EMBED_DIM).astype(np.float32) * 0.1
            np.savetxt(str(feat_dir / "text_embeddings.csv"), text_embed, delimiter=",", fmt="%.6f")

        return {
            "text_key": f"{session_id}/text_embeddings.csv",
            "raw_text": transcript,
        }

    _whisper_model = None
    _sentence_transformer_model = None

    @classmethod
    def _get_whisper_model(cls):
        if cls._whisper_model is None:
            import whisper
            cls._whisper_model = whisper.load_model("base")
        return cls._whisper_model

    @classmethod
    def _get_sentence_transformer_model(cls):
        if cls._sentence_transformer_model is None:
            from sentence_transformers import SentenceTransformer
            cls._sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._sentence_transformer_model

    async def process_video(
        self, video_path: Path, session_id: str,
        enable_stt: bool = True,
    ) -> Dict[str, Any]:
        """Full video processing pipeline.

        1. Validate video
        2. Extract audio → audio features
        3. Extract frames → video features
        4. Optionally run STT → text features
        5. Clean up raw video/frames

        Returns:
            Dict with all feature storage keys
        """
        t_start = time.perf_counter()
        duration = None
        results = {
            "audio_features": None,
            "video_features": None,
            "text_features": None,
            "duration_sec": None,
        }

        audio_path = None
        frames_dir = None
        try:
            # Validate duration
            duration = await self.get_video_duration(video_path)
            if duration > MAX_VIDEO_DURATION_SEC:
                raise VideoProcessingError(
                    f"Video too long ({duration:.0f}s). Maximum is {MAX_VIDEO_DURATION_SEC}s."
                )
            results["duration_sec"] = duration

            # Extract audio
            audio_path = await self.extract_audio(video_path, session_id)
            results["audio_features"] = await self.extract_audio_features(audio_path, session_id)

            # Extract frames + video features
            frames_dir = await self.extract_frames(video_path, session_id)
            results["video_features"] = await self.extract_video_features(frames_dir, session_id)

            # STT + text features
            if enable_stt:
                results["text_features"] = await self.extract_text_features(audio_path, session_id)
        finally:
            # Clean up temporary files (raw video + frames + extracted audio)
            await self.cleanup(session_id, video_path)

        elapsed = time.perf_counter() - t_start
        results["processing_time_s"] = round(elapsed, 3)

        logger.info(
            f"Video processing complete: session={session_id}, "
            f"duration={duration:.1f}s, processing={elapsed:.2f}s"
        )
        return results

    async def cleanup(self, session_id: str, video_path: Optional[Path] = None):
        """Remove temporary files. Only extracted features in feature_dir persist."""
        tmp = _temp_dir() / session_id
        try:
            if tmp.exists():
                shutil.rmtree(str(tmp), ignore_errors=True)
                logger.info(f"Cleaned up temp files: {tmp}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {tmp}: {e}")

        # Also delete the original upload if it's outside tmp
        if video_path and video_path.exists() and str(video_path).startswith(str(_temp_dir())):
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass
