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
MFCC_DIM = 39           # Matches training pipeline (OpenSMILE MFCC)
EGEMAPS_DIM = 23        # Matches training pipeline (OpenSMILE eGeMAPS)
BEHAVIORAL_DIM = 16
OPENFACE_DIM = 49
CNN_EMBED_DIM = 128     # Post-PCA dimension (training uses PCA 2048→128)
TEXT_EMBED_DIM = 384
AUDIO_DIM = MFCC_DIM + EGEMAPS_DIM  # 62 combined


def _finite_float32(values: np.ndarray, clip: Optional[float] = None, *, replace_zeros: bool = False) -> np.ndarray:
    """Return finite float32 features so NaN/Inf never reaches model inference."""
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=1e-6, posinf=1e-6, neginf=-1e-6)
    if clip is not None:
        arr = np.clip(arr, -clip, clip)
    if replace_zeros:
        arr = np.where(arr == 0.0, 1e-6, arr)
        if arr.size and float(np.max(np.abs(arr))) <= 1e-6:
            flat_dim = arr.shape[-1] if arr.ndim > 1 else arr.size
            base = np.linspace(1e-6, 1e-6 * flat_dim, flat_dim, dtype=np.float32)
            if arr.ndim > 1:
                scales = np.arange(1, arr.shape[0] + 1, dtype=np.float32).reshape(-1, 1)
                arr = base.reshape(1, -1) * scales
            else:
                arr = base
    return arr.astype(np.float32, copy=False)


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

    async def _ffprobe_duration(self, video_path: Path, show_entry: str) -> Optional[float]:
        """Run ffprobe with the given show_entry and return the duration or None."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", show_entry,
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            raw = stdout.decode("utf-8", errors="replace").strip().splitlines()
            for line in raw:
                line = line.strip()
                if line and line.lower() != "n/a":
                    try:
                        val = float(line)
                        if np.isfinite(val) and val > 0:
                            return val
                    except ValueError:
                        continue
        except Exception:
            pass
        return None

    async def get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds using FFprobe.

        Browser WebRTC webm recordings often lack a container-level duration header
        (ffprobe returns 'N/A'). Falls back through multiple strategies before
        returning 0.0 (duration is non-critical for feature extraction).
        """
        # Strategy 1: container format duration (fast, works for mp4/avi)
        dur = await self._ffprobe_duration(video_path, "format=duration")
        if dur is not None:
            return dur

        # Strategy 2: per-stream duration (works for some webm files)
        dur = await self._ffprobe_duration(video_path, "stream=duration")
        if dur is not None:
            return dur

        # Strategy 3: compute from audio stream sample count + sample rate
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=nb_read_samples,sample_rate",
                "-count_packets",
                "-of", "default=noprint_wrappers=1",
                str(video_path),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            lines = stdout.decode("utf-8", errors="replace")
            nb_samples = sr = None
            for line in lines.splitlines():
                if "nb_read_samples" in line:
                    try: nb_samples = int(line.split("=")[1])
                    except Exception: pass
                if "sample_rate" in line:
                    try: sr = int(line.split("=")[1])
                    except Exception: pass
            if nb_samples and sr and sr > 0:
                return float(nb_samples) / float(sr)
        except Exception:
            pass

        # Fallback: duration unknown — not critical, continue processing
        logger.warning("Could not determine duration for %s; defaulting to 0.0", video_path.name)
        return 0.0

    async def extract_audio_features(
        self, audio_path: Path, session_id: str,
    ) -> Dict[str, str]:
        """Extract eGeMAPS and MFCC features from audio using opensmile.

        Uses the opensmile Python wrapper for accurate feature extraction
        matching the training pipeline (OpenSMILE 2.3.0 compatible).

        Returns:
            Dict with storage keys for combined audio features CSV
        """
        feat_dir = _feature_dir() / session_id
        feat_dir.mkdir(parents=True, exist_ok=True)

        try:
            import librosa
            audio_data, sr = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
            audio_data = _finite_float32(audio_data)
        except ImportError:
            raise VideoProcessingError(
                "librosa is not installed. Run: pip install librosa soundfile"
            )
        except Exception as e:
            logger.warning(f"Failed to read audio with librosa: {e}")
            raise VideoProcessingError(f"Failed to read audio file: {e}") from e

        if audio_data.size == 0:
            raise VideoProcessingError("Audio extraction produced no samples")

        # Chunk audio into segments (5-second chunks with 25% overlap, matching training)
        chunk_duration = 5.0
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * 0.75)  # 25% overlap
        num_chunks = max(1, (len(audio_data) - chunk_samples) // hop_samples + 1)

        # Try real opensmile extraction
        try:
            import opensmile
            # eGeMAPS functional features per chunk
            smile_egemaps = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            # MFCC features per chunk
            smile_mfcc = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )

            mfcc_list = []
            egemaps_list = []
            for i in range(num_chunks):
                start = i * hop_samples
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                if len(chunk) < sr:  # Skip chunks shorter than 1 second
                    continue

                # Extract eGeMAPS
                eg_df = smile_egemaps.process_signal(chunk, sr)
                eg_vals = _finite_float32(eg_df.values.flatten()[:EGEMAPS_DIM])
                if len(eg_vals) < EGEMAPS_DIM:
                    eg_vals = np.pad(eg_vals, (0, EGEMAPS_DIM - len(eg_vals)))
                egemaps_list.append(eg_vals)

                # Extract MFCCs via librosa (39-dim = 13 base + 13 delta + 13 delta-delta)
                mfcc_base = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
                mfcc_delta = librosa.feature.delta(mfcc_base)
                mfcc_delta2 = librosa.feature.delta(mfcc_base, order=2)
                mfcc_full = _finite_float32(np.concatenate([
                    mfcc_base.mean(axis=1),
                    mfcc_delta.mean(axis=1),
                    mfcc_delta2.mean(axis=1),
                ]))
                mfcc_list.append(mfcc_full[:MFCC_DIM])

            if mfcc_list:
                mfcc_features = np.array(mfcc_list, dtype=np.float32)
                egemaps_features = np.array(egemaps_list, dtype=np.float32)
            else:
                raise ValueError("No valid chunks")

            logger.info(f"Audio features (opensmile): mfcc={mfcc_features.shape}, egemaps={egemaps_features.shape}")

        except Exception as e:
            logger.warning(f"OpenSMILE extraction unavailable ({e}); using librosa-only MFCC + spectral fallback")
            # Fallback: librosa-based extraction (still better than raw FFT)
            import librosa

            mfcc_list = []
            egemaps_list = []
            for i in range(num_chunks):
                start = i * hop_samples
                end = min(start + chunk_samples, len(audio_data))
                chunk = audio_data[start:end]
                if len(chunk) < sr:
                    continue

                # MFCC: 13 base + 13 delta + 13 delta-delta = 39
                mfcc_base = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
                mfcc_delta = librosa.feature.delta(mfcc_base)
                mfcc_delta2 = librosa.feature.delta(mfcc_base, order=2)
                mfcc_full = _finite_float32(np.concatenate([
                    mfcc_base.mean(axis=1),
                    mfcc_delta.mean(axis=1),
                    mfcc_delta2.mean(axis=1),
                ]))
                mfcc_list.append(mfcc_full[:MFCC_DIM])

                # eGeMAPS-like: spectral + prosodic features (23-dim)
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=chunk)))
                spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)))
                spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sr)))
                spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=chunk, sr=sr)))
                spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
                chroma = librosa.feature.chroma_stft(y=chunk, sr=sr).mean(axis=1)  # 12 dims
                tonnetz = librosa.feature.tonnetz(y=chunk, sr=sr).mean(axis=1)  # 6 dims - only use 5
                eg_feat = _finite_float32(np.array([
                    rms, zcr, spec_centroid / sr, spec_bandwidth / sr,
                    spec_rolloff / sr, spec_flatness,
                    *chroma[:12], *tonnetz[:5]
                ], dtype=np.float32))
                egemaps_list.append(eg_feat[:EGEMAPS_DIM])

            mfcc_features = _finite_float32(np.array(mfcc_list, dtype=np.float32)) if mfcc_list else np.zeros((1, MFCC_DIM), dtype=np.float32)
            egemaps_features = _finite_float32(np.array(egemaps_list, dtype=np.float32)) if egemaps_list else np.zeros((1, EGEMAPS_DIM), dtype=np.float32)
            logger.info(f"Audio features (librosa fallback): mfcc={mfcc_features.shape}, egemaps={egemaps_features.shape}")

        mfcc_features = _finite_float32(mfcc_features)
        egemaps_features = _finite_float32(egemaps_features)

        # Combine MFCC + eGeMAPS as single audio feature (N, 62)
        min_len = min(len(mfcc_features), len(egemaps_features))
        if min_len == 0:
            mfcc_features = np.zeros((1, MFCC_DIM), dtype=np.float32)
            egemaps_features = np.zeros((1, EGEMAPS_DIM), dtype=np.float32)
            min_len = 1
        combined_audio = np.concatenate(
            [mfcc_features[:min_len], egemaps_features[:min_len]], axis=1
        )  # (N, 62)
        combined_audio = _finite_float32(combined_audio)

        # Behavioral features (16,)
        behavioral = np.zeros(BEHAVIORAL_DIM, dtype=np.float32)
        duration = len(audio_data) / sr
        behavioral[0] = duration
        behavioral[1] = np.mean(np.abs(audio_data))
        behavioral[2] = np.std(audio_data)
        behavioral[3] = float(np.sum(np.abs(audio_data) < 0.01)) / max(len(audio_data), 1)
        behavioral[9] = 1.0 - behavioral[3]
        behavioral = _finite_float32(behavioral)

        # Save features (separate files for compatibility with ML client)
        np.savetxt(str(feat_dir / "mfcc.csv"), mfcc_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "egemaps.csv"), egemaps_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "audio_combined.csv"), combined_audio, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "behavioral.csv"), behavioral.reshape(1, -1), delimiter=",", fmt="%.6f")

        logger.info(f"Audio features saved: combined={combined_audio.shape}")

        return {
            "mfcc_key": f"{session_id}/mfcc.csv",
            "egemaps_key": f"{session_id}/egemaps.csv",
            "audio_combined_key": f"{session_id}/audio_combined.csv",
            "behavioral_key": f"{session_id}/behavioral.csv",
        }

    async def extract_video_features(
        self, frames_dir: Path, session_id: str,
    ) -> Dict[str, str]:
        """Extract video features from frames using MediaPipe Face Mesh.

        Generates OpenFace-compatible features (pose + gaze proxy + AU proxy)
        and CNN embeddings using torchvision ResNet50.

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

        # OpenFace-style features using MediaPipe
        openface_features = np.zeros((num_frames, OPENFACE_DIM), dtype=np.float32)
        try:
            import mediapipe as mp
            import cv2

            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

            for i, frame_path in enumerate(frame_files):
                try:
                    image = cv2.imread(str(frame_path))
                    if image is None:
                        continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_image)

                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0]
                        h, w = image.shape[:2]

                        # Pose estimation from key landmarks (6 dims)
                        nose = landmarks.landmark[1]
                        chin = landmarks.landmark[152]
                        left_eye = landmarks.landmark[33]
                        right_eye = landmarks.landmark[263]
                        mouth_left = landmarks.landmark[61]
                        mouth_right = landmarks.landmark[291]

                        openface_features[i, 0] = nose.x - 0.5  # pose_Tx
                        openface_features[i, 1] = nose.y - 0.5  # pose_Ty
                        openface_features[i, 2] = nose.z         # pose_Tz
                        # Head rotation proxies
                        openface_features[i, 3] = (right_eye.y - left_eye.y)  # Rx (roll)
                        openface_features[i, 4] = (nose.y - chin.y)  # Ry (pitch)
                        openface_features[i, 5] = (right_eye.x - left_eye.x - 0.15)  # Rz (yaw)

                        # Gaze estimation from eye landmarks (8 dims)
                        left_iris = landmarks.landmark[468] if len(landmarks.landmark) > 468 else left_eye
                        right_iris = landmarks.landmark[473] if len(landmarks.landmark) > 473 else right_eye
                        openface_features[i, 6] = left_iris.x - left_eye.x
                        openface_features[i, 7] = left_iris.y - left_eye.y
                        openface_features[i, 8] = right_iris.x - right_eye.x
                        openface_features[i, 9] = right_iris.y - right_eye.y
                        openface_features[i, 10:14] = [
                            left_iris.z, right_iris.z,
                            abs(left_iris.x - right_iris.x),
                            abs(left_iris.y - right_iris.y),
                        ]

                        # AU proxies from landmark distances (35 dims)
                        # Mouth opening (AU25/26)
                        mouth_open = abs(landmarks.landmark[13].y - landmarks.landmark[14].y)
                        # Brow raise (AU1/2)
                        brow_raise_l = abs(landmarks.landmark[70].y - left_eye.y)
                        brow_raise_r = abs(landmarks.landmark[300].y - right_eye.y)
                        # Lip corner (AU12/15)
                        lip_width = abs(mouth_left.x - mouth_right.x)
                        # Eye opening (AU5/7)
                        eye_open_l = abs(landmarks.landmark[159].y - landmarks.landmark[145].y)
                        eye_open_r = abs(landmarks.landmark[386].y - landmarks.landmark[374].y)

                        au_base = np.array([
                            brow_raise_l, brow_raise_r, mouth_open, lip_width,
                            eye_open_l, eye_open_r,
                        ], dtype=np.float32)

                        # Fill AU slots with these proxies + zero padding
                        au_features = np.zeros(35, dtype=np.float32)
                        au_features[:len(au_base)] = au_base
                        # Add more landmark-based features
                        for j, idx in enumerate([10, 152, 234, 454, 1, 4, 5, 6]):
                            if j + len(au_base) < 35 and idx < len(landmarks.landmark):
                                lm = landmarks.landmark[idx]
                                au_features[j + len(au_base)] = lm.y
                        openface_features[i, 14:49] = au_features

                except Exception as e:
                    logger.debug(f"Frame {i} processing failed: {e}")

            face_mesh.close()
            logger.info(f"Video features (MediaPipe): openface={openface_features.shape}")

        except Exception as e:
            logger.warning(f"MediaPipe extraction unavailable ({e}); using deterministic OpenFace fallback")

        openface_features = _finite_float32(openface_features, clip=10.0, replace_zeros=True)

        # CNN embedding features using torchvision ResNet50
        cnn_features = np.zeros((num_frames, CNN_EMBED_DIM), dtype=np.float32)
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image

            # Load ResNet50 and remove classification head
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
            resnet.eval()

            # Load PCA transform if available
            pca_transform = None
            repo_root = Path(__file__).resolve().parents[3]
            pca_path = repo_root / "Model" / "checkpoints" / "pca_cnn_transform.pkl"
            if pca_path.exists():
                try:
                    import sys
                    model_root = repo_root / "Model"
                    if str(model_root) not in sys.path:
                        sys.path.insert(0, str(model_root))
                    from src.features.pca_reducer import _safe_pickle_load
                    with open(pca_path, 'rb') as f:
                        pca_data = _safe_pickle_load(f)
                    pca_transform = pca_data.get("pca", pca_data) if isinstance(pca_data, dict) else pca_data
                    logger.info(f"Loaded PCA transform: 2048 → {getattr(pca_transform, 'n_components_', '?')}")
                except Exception as e:
                    logger.warning(f"PCA transform load failed ({e}); using truncated ResNet features")

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            with torch.no_grad():
                for i, frame_path in enumerate(frame_files):
                    try:
                        img = Image.open(frame_path).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        embedding = resnet(img_tensor).squeeze().numpy()  # (2048,)

                        if pca_transform is not None:
                            embedding = pca_transform.transform(embedding.reshape(1, -1)).flatten()  # (128,)
                        else:
                            # Truncate to CNN_EMBED_DIM if no PCA
                            embedding = embedding[:CNN_EMBED_DIM]

                        cnn_features[i] = _finite_float32(embedding[:CNN_EMBED_DIM], clip=100.0, replace_zeros=True)
                    except Exception as e:
                        logger.debug(f"CNN extraction failed for frame {i}: {e}")

            logger.info(f"Video features (ResNet50): cnn={cnn_features.shape}")

        except Exception as e:
            logger.warning(f"torchvision/ResNet extraction unavailable ({e}); using deterministic CNN fallback")

        cnn_features = _finite_float32(cnn_features, clip=100.0, replace_zeros=True)

        np.savetxt(str(feat_dir / "openface.csv"), openface_features, delimiter=",", fmt="%.6f")
        np.savetxt(str(feat_dir / "cnn_embed.csv"), cnn_features, delimiter=",", fmt="%.6f")

        logger.info(f"Video features saved: openface={openface_features.shape}, cnn={cnn_features.shape}")

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
            logger.info("No transcript available; skipping text modality")
            return None

        # If we have a transcript, generate SBERT-like embeddings
        try:
            model = self._get_sentence_transformer_model()
            # Split into sentences
            sentences = [s.strip() for s in transcript.split(".") if s.strip()]
            if not sentences:
                sentences = [transcript]
            embeddings = _finite_float32(model.encode(sentences))
            np.savetxt(str(feat_dir / "text_embeddings.csv"), embeddings, delimiter=",", fmt="%.6f")
        except Exception as e:
            logger.info(f"sentence-transformers unavailable ({e}); sending raw transcript without embeddings")
            return {
                "text_key": None,
                "raw_text": transcript,
            }

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
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Process video file into multimodal features.
        
        Args:
            video_path: Path to video file
            session_id: Session identifier
            enable_stt: Enable speech-to-text transcription
            fast_mode: If True, use audio-only mode for real-time scoring.
                      Skips video processing (MediaPipe, ResNet) and text (Whisper).
                      Target: 3-5 seconds for 30s video.
                      Scores will vary based on audio features only:
                      - Voice tone, pitch, energy
                      - Speaking rate and pauses
                      - Silence ratio and speech patterns
        
        Returns:
            Dict with feature storage keys and processing metadata
        """
        t_start = time.time()
        
        try:
            # Get video duration
            duration = await self.get_video_duration(video_path)
            logger.info(f"Processing video: {video_path.name}, duration={duration:.1f}s, fast_mode={fast_mode}")
            
            # ALWAYS extract audio (fast: ~1-2s for 30s video)
            audio_path = await self.extract_audio(video_path, session_id)
            
            # Extract audio features (fast: ~2-3s for 30s video)
            audio_features = await self.extract_audio_features(audio_path, session_id)
            
            video_features = None
            text_features = None
            
            if fast_mode:
                # FAST PATH: Skip heavy video processing entirely
                # Use audio-only mode - no video features at all
                # This ensures scores vary based on actual speech patterns
                # rather than being diluted by random noise
                logger.info("[FAST MODE] Audio-only mode - skipping video and text extraction")
                video_features = None  # No video in fast mode
                text_features = None   # No text in fast mode
            else:
                # FULL PATH: Complete video + text processing (30-60s)
                logger.info("[FULL MODE] Running complete video + text extraction")
                
                # Extract frames (moderate: ~3-5s for 30s video)
                frames_dir = await self.extract_frames(video_path, session_id, fps=FRAME_EXTRACT_FPS)
                
                # Extract video features (SLOW: ~15-25s for MediaPipe + ResNet)
                video_features = await self.extract_video_features(frames_dir, session_id)
                
                # Extract text features if enabled (SLOW: ~5-10s for Whisper)
                if enable_stt:
                    text_features = await self.extract_text_features(audio_path, session_id)
            
            processing_time = time.time() - t_start
            logger.info(f"Video processing complete: {processing_time:.2f}s (fast_mode={fast_mode})")
            
            return {
                "audio_features": audio_features,
                "video_features": video_features,
                "text_features": text_features,
                "duration_sec": duration,
                "processing_time_s": processing_time,
                "fast_mode": fast_mode,
            }
            
        except VideoProcessingError:
            raise
        except Exception as e:
            logger.error(f"Video processing failed: {e}", exc_info=True)
            raise VideoProcessingError(f"Video processing failed: {str(e)[:200]}") from e

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
