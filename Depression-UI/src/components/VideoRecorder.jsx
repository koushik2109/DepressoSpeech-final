import { useEffect, useRef, useState, useCallback } from "react";

const MAX_RECORDING_SECONDS = 180;
const MIN_RECORDING_SECONDS = 5;

/**
 * VideoRecorder — Captures webcam + microphone using MediaRecorder API.
 *
 * Features:
 *  - Live camera preview
 *  - Start/Stop recording
 *  - Recording timer with max duration
 *  - Playback preview of recorded video
 *  - Blob output for upload
 */
export default function VideoRecorder({ onRecordingComplete, onRecordingCleared }) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [videoURL, setVideoURL] = useState("");
  const [hasRecording, setHasRecording] = useState(false);
  const [blobSize, setBlobSize] = useState(0);
  const [cameraError, setCameraError] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const videoRef = useRef(null);
  const playbackRef = useRef(null);
  const recordedBlobRef = useRef(null);
  const videoURLRef = useRef("");
  const secondsRef = useRef(0);

  const formatTime = (value) => {
    const m = Math.floor(value / 60).toString().padStart(2, "0");
    const s = (value % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  const clearCurrentRecording = useCallback(() => {
    if (videoURLRef.current) {
      URL.revokeObjectURL(videoURLRef.current);
      videoURLRef.current = "";
    }
    setVideoURL("");
    setHasRecording(false);
    setBlobSize(0);
    recordedBlobRef.current = null;
    onRecordingCleared?.();
  }, [onRecordingCleared]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      setCameraError("");
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          facingMode: "user",
          frameRate: { ideal: 30, max: 30 },
        },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraOn(true);
    } catch (err) {
      const message =
        err.name === "NotAllowedError"
          ? "Camera and microphone access denied. Please allow permissions in your browser settings."
          : err.name === "NotFoundError"
            ? "No camera or microphone found on this device."
            : err.name === "NotReadableError"
              ? "Camera or microphone is in use by another application."
              : `Camera access failed: ${err.message}`;
      setCameraError(message);
      setIsCameraOn(false);
    }
  }, []);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    setIsRecording(false);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const startRecording = useCallback(async () => {
    if (!streamRef.current) {
      await startCamera();
      if (videoRef.current) {
        await new Promise((resolve) => {
          if (videoRef.current.readyState >= HTMLMediaElement.HAVE_METADATA) {
            resolve();
            return;
          }
          videoRef.current.addEventListener("loadedmetadata", resolve, { once: true });
        });
      }
      if (!streamRef.current) return;
    }

    clearCurrentRecording();

    const mimeType = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ].find((t) => MediaRecorder.isTypeSupported(t)) || "video/webm";

    const recorder = new MediaRecorder(streamRef.current, {
      mimeType,
      videoBitsPerSecond: 1500000,
      audioBitsPerSecond: 128000,
    });
    mediaRecorderRef.current = recorder;

    const chunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      if (!blob.size) {
        clearCurrentRecording();
        return;
      }

      recordedBlobRef.current = blob;
      setBlobSize(blob.size);
      setHasRecording(true);

      const url = URL.createObjectURL(blob);
      videoURLRef.current = url;
      setVideoURL(url);

      // Determine file extension from mime
      const ext = mimeType.includes("mp4") ? ".mp4" : ".webm";

      onRecordingComplete?.(blob, url, secondsRef.current, `recording${ext}`);
    };

    recorder.start(1000); // Collect data every second
    setIsRecording(true);
    setSeconds(0);
    secondsRef.current = 0;

    timerRef.current = setInterval(() => {
      secondsRef.current += 1;
      setSeconds(secondsRef.current);
      if (secondsRef.current >= MAX_RECORDING_SECONDS) {
        stopRecording();
      }
    }, 1000);
  }, [startCamera, clearCurrentRecording, onRecordingComplete, stopRecording]);

  // Cleanup on unmount
  useEffect(
    () => () => {
      clearInterval(timerRef.current);
      stopCamera();
      if (videoURLRef.current) URL.revokeObjectURL(videoURLRef.current);
    },
    [stopCamera],
  );

  const canPlayback = hasRecording && videoURL;
  const recordingTooShort = hasRecording && seconds < MIN_RECORDING_SECONDS;
  const progressPct = Math.min(100, (seconds / MAX_RECORDING_SECONDS) * 100);

  return (
    <div className="video-recorder">
      {/* Camera Preview / Playback */}
      <div className="video-recorder-preview-container">
        {/* Live camera feed */}
        <video
          ref={videoRef}
          className="video-recorder-preview"
          muted
          playsInline
          style={{
            display: canPlayback ? "none" : "block",
            transform: "scaleX(-1)",
          }}
        />

        {/* Playback */}
        {canPlayback && (
          <video
            ref={playbackRef}
            className="video-recorder-preview"
            src={videoURL}
            controls
            playsInline
          />
        )}

        {/* No camera overlay */}
        {!isCameraOn && !canPlayback && (
          <div className="video-recorder-no-camera">
            <svg className="w-12 h-12 text-[#B5B5B5]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
            </svg>
            <p className="text-sm text-[#B5B5B5] mt-3">Camera preview will appear here</p>
          </div>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <div className="video-recorder-rec-badge">
            <span className="video-recorder-rec-dot" />
            REC {formatTime(seconds)}
          </div>
        )}

        {/* Duration progress bar */}
        {isRecording && (
          <div className="video-recorder-progress-bar">
            <div
              className="video-recorder-progress-fill"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}
      </div>

      {/* Error */}
      {cameraError && (
        <div className="rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-4 py-3 mt-4">
          <p className="text-sm text-[#A94442]">{cameraError}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col items-center gap-4 mt-6">
        {/* Timer */}
        <div className="text-center">
          <span className="text-3xl font-bold text-[#1B1B1B] font-mono tracking-tight">
            {formatTime(seconds)}
          </span>
          <p className="text-xs text-[#B5B5B5] mt-1">
            {isRecording ? "Recording" : hasRecording ? "Recorded" : "Ready"} •
            Max {formatTime(MAX_RECORDING_SECONDS)}
          </p>
        </div>

        {/* Buttons */}
        <div className="flex items-center gap-4">
          {/* Camera toggle */}
          {!isRecording && !hasRecording && (
            <button
              type="button"
              onClick={isCameraOn ? stopCamera : startCamera}
              className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 ${
                isCameraOn
                  ? "bg-[#E8E8E8] text-[#555] hover:bg-[#D9D9D9]"
                  : "bg-[#2D6A4F] text-white hover:bg-[#1B3A2D] shadow-lg"
              }`}
              aria-label={isCameraOn ? "Turn off camera" : "Turn on camera"}
            >
              {isCameraOn ? (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
              )}
            </button>
          )}

          {/* Main record / stop button */}
          <div className="relative">
            {isRecording && (
              <div className="absolute -inset-3 rounded-full bg-red-400/20 animate-pulse" />
            )}
            <button
              type="button"
              onClick={isRecording ? stopRecording : startRecording}
              className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg hover:shadow-xl active:scale-95 font-semibold text-white text-sm ${
                isRecording
                  ? "bg-red-500 hover:bg-red-600 shadow-red-200"
                  : "bg-gradient-to-br from-[#7C3AED] to-[#2D6A4F] hover:from-[#6D28D9] hover:to-[#1B3A2D] shadow-[#7C3AED]/30"
              }`}
              aria-label={isRecording ? "Stop recording" : "Start recording"}
            >
              {isRecording ? (
                <div className="w-7 h-7 rounded-md bg-white" />
              ) : (
                <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
              )}
            </button>
          </div>

          {/* Re-record button */}
          {hasRecording && !isRecording && (
            <button
              type="button"
              onClick={() => {
                clearCurrentRecording();
                startRecording();
              }}
              className="w-12 h-12 rounded-full flex items-center justify-center bg-[#E8E8E8] text-[#555] hover:bg-[#D9D9D9] transition-all"
              aria-label="Re-record"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
              </svg>
            </button>
          )}
        </div>

        {/* Status text */}
        <div className="text-center">
          {isRecording ? (
            <p className="text-sm font-medium text-[#777]">
              Recording in progress. Look at the camera and speak naturally.
            </p>
          ) : recordingTooShort ? (
            <p className="text-sm text-[#A94442]">
              Recording too short ({seconds}s). Minimum {MIN_RECORDING_SECONDS}s required.
            </p>
          ) : canPlayback ? (
            <div className="space-y-2">
              <p className="text-sm font-medium text-[#52B788]">
                ✓ Recording saved ({seconds}s)
              </p>
              <p className="text-xs text-[#B5B5B5]">
                File size: {(blobSize / (1024 * 1024)).toFixed(1)} MB
              </p>
            </div>
          ) : (
            <p className="text-sm font-medium text-[#777]">
              {isCameraOn
                ? "Camera ready. Tap the button to start recording."
                : "Tap the camera button to enable your webcam, then record."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
