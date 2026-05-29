import { useEffect, useRef, useState, useCallback } from "react";

const MAX_RECORDING_SECONDS = 120;
const MIN_RECORDING_SECONDS = 3;
const FFT_SIZE = 2048;
const BAR_COUNT = 64;

/**
 * VoiceRecorder — captures audio (with optional webcam video).
 *
 * When `enableVideo` is true, records webcam + microphone simultaneously
 * and outputs a video blob (webm/mp4) containing both tracks.
 * When false, records audio-only like the original implementation.
 *
 * Props:
 *   onRecordingComplete(blob, previewUrl, seconds)
 *   onRecordingCleared()
 *   enableVideo — boolean, default false
 */
export default function VoiceRecorder({
  onRecordingComplete,
  onRecordingCleared,
  enableVideo = false,
  isPaused = false,
  onPauseStateChange,
  onRecordingStart,
  onRecordingStop,
  onMediaRecorderReady,
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [audioURL, setAudioURL] = useState("");
  const [recordingBlobSize, setRecordingBlobSize] = useState(0);
  const [level, setLevel] = useState(0);
  const [hasRecording, setHasRecording] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [effectiveVideoEnabled, setEffectiveVideoEnabled] = useState(enableVideo);
  const [isPlaybackPlaying, setIsPlaybackPlaying] = useState(false);

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const canvasRef = useRef(null);
  const videoPreviewRef = useRef(null);
  const playbackVideoRef = useRef(null);
  const animFrameRef = useRef(null);
  const dataArrayRef = useRef(null);
  const freqDataArrayRef = useRef(null);
  const recordedBlobRef = useRef(null);
  const audioURLRef = useRef("");
  const secondsRef = useRef(0);
  const isRecordingRef = useRef(false);
  const enableVideoRef = useRef(enableVideo);
  const isPausedRef = useRef(isPaused);

  useEffect(() => {
    enableVideoRef.current = enableVideo;
    setEffectiveVideoEnabled(enableVideo);
  }, [enableVideo]);

  useEffect(() => {
    isPausedRef.current = isPaused;
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") return;

    if (isPaused) {
      // Keep MediaRecorder running continuously to prevent browser-level keyframe desync and video freezing.
      // Simply suspend the AudioContext to flatline the visual waveform during the pause.
      if (audioContextRef.current && audioContextRef.current.state === "running") {
        audioContextRef.current.suspend().catch(() => {});
      }
    } else {
      if (audioContextRef.current && audioContextRef.current.state === "suspended") {
        audioContextRef.current.resume().catch(() => {});
      }
    }
    onPauseStateChange?.(isPaused);
  }, [isPaused, onPauseStateChange]);

  const formatTime = (v) => {
    const m = Math.floor(v / 60).toString().padStart(2, "0");
    const s = (v % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  const drawIdleLine = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#FAFAF7";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#D8F3DC";
    ctx.lineWidth = Math.max(2, w / 240);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
  }, []);

  const clearCurrentRecording = useCallback(() => {
    if (audioURLRef.current) {
      URL.revokeObjectURL(audioURLRef.current);
      audioURLRef.current = "";
    }
    setAudioURL("");
    setHasRecording(false);
    setRecordingBlobSize(0);
    recordedBlobRef.current = null;
    onRecordingCleared?.();
    onRecordingStop?.();
  }, [onRecordingCleared, onRecordingStop]);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    const freqDataArray = freqDataArrayRef.current;
    if (!canvas || !analyser || !dataArray || !freqDataArray) return;

    if (isPausedRef.current) {
      drawIdleLine();
      if (isRecordingRef.current) {
        animFrameRef.current = requestAnimationFrame(drawWaveform);
      }
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    analyser.getByteTimeDomainData(dataArray);

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const bg = ctx.createLinearGradient(0, 0, w, h);
    bg.addColorStop(0, "#FAFAF7");
    bg.addColorStop(1, "#F0FAF4");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    analyser.getByteFrequencyData(freqDataArray);
    const barW = w / BAR_COUNT;
    const useVideo = enableVideoRef.current;
    for (let i = 0; i < BAR_COUNT; i++) {
      const idx = Math.floor((i / BAR_COUNT) * freqDataArray.length);
      const mag = freqDataArray[idx] / 255;
      const barH = Math.max(2, mag * (h * 0.8));
      const x = i * barW;
      const y = (h - barH) / 2;
      const g = ctx.createLinearGradient(0, y, 0, y + barH);
      g.addColorStop(0, useVideo ? "#74C69D" : "#52B788");
      g.addColorStop(1, useVideo ? "#2D6A4F" : "#2D6A4F");
      ctx.fillStyle = g;
      ctx.fillRect(x + barW * 0.15, y, barW * 0.7, barH);
    }

    const centerY = h / 2;
    const sliceW = w / dataArray.length;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    for (let i = 0; i < dataArray.length; i++) {
      ctx.lineTo(i * sliceW, (dataArray[i] / 128.0) * centerY);
    }
    ctx.strokeStyle = "#2D6A4F";
    ctx.lineWidth = Math.max(2, w / 220);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();

    const sum = freqDataArray.reduce((a, b) => a + b, 0);
    setLevel(Math.min(1, sum / (freqDataArray.length * 255)));

    if (isRecordingRef.current) {
      animFrameRef.current = requestAnimationFrame(drawWaveform);
    }
  }, [drawIdleLine]);

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;
    clearInterval(timerRef.current);
    timerRef.current = null;
    setIsRecording(false);
    onRecordingStop?.();

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setCameraReady(false);
    drawIdleLine();
  }, [drawIdleLine, onRecordingStop]);

  const startRecording = useCallback(async () => {
    try {
      clearCurrentRecording();
      setCameraError("");

      const useVideo = enableVideoRef.current;
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      };
      if (useVideo) {
        constraints.video = {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          facingMode: "user",
          frameRate: { ideal: 30 },
        };
      }

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        if (useVideo) {
          setCameraError("Camera not available — recording audio only.");
          enableVideoRef.current = false;
          setEffectiveVideoEnabled(false);
          stream = await navigator.mediaDevices.getUserMedia({
            audio: constraints.audio,
          });
        } else {
          throw err;
        }
      }

      streamRef.current = stream;
      const hasVideoTrack = stream.getVideoTracks().length > 0;
      if (hasVideoTrack && videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        await videoPreviewRef.current.play().catch(() => undefined);
        setCameraReady(true);
      }

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = FFT_SIZE;
      analyser.smoothingTimeConstant = 0.85;
      audioCtx.createMediaStreamSource(stream).connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.fftSize);
      freqDataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

      let mimeType;
      if (hasVideoTrack) {
        mimeType = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"]
          .find((t) => MediaRecorder.isTypeSupported(t)) || "video/webm";
      } else {
        mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
          .find((t) => MediaRecorder.isTypeSupported(t)) || "audio/webm";
      }

      const recorder = new MediaRecorder(stream, {
        mimeType,
        ...(hasVideoTrack
          ? { videoBitsPerSecond: 1500000, audioBitsPerSecond: 128000 }
          : { audioBitsPerSecond: 128000 }),
      });
      mediaRecorderRef.current = recorder;
      onMediaRecorderReady?.(recorder);

      const chunks = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        if (!blob.size) { clearCurrentRecording(); return; }
        recordedBlobRef.current = blob;
        setRecordingBlobSize(blob.size);
        setHasRecording(true);
        setIsPlaybackPlaying(false);
        const url = URL.createObjectURL(blob);
        audioURLRef.current = url;
        setAudioURL(url);
        onRecordingComplete?.(blob, url, secondsRef.current);
      };

      recorder.start(1000);
      isRecordingRef.current = true;
      setIsRecording(true);
      setSeconds(0);
      secondsRef.current = 0;
      drawWaveform();
      onRecordingStart?.();

      timerRef.current = setInterval(() => {
        if (isPausedRef.current) return;
        secondsRef.current += 1;
        setSeconds(secondsRef.current);
        if (secondsRef.current >= MAX_RECORDING_SECONDS) stopRecording();
      }, 1000);
    } catch (error) {
      const msg =
        error.name === "NotAllowedError" ? "Permission denied. Please allow camera/microphone access."
        : error.name === "NotFoundError" ? "No microphone found."
        : error.name === "NotReadableError" ? "Microphone/camera in use by another app."
        : "Device access failed. Please try again.";
      setCameraError(msg);
    }
  }, [clearCurrentRecording, onRecordingComplete, stopRecording, drawWaveform, onRecordingStart, onMediaRecorderReady]);

  // Canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const r = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(parent.clientWidth * r));
      canvas.height = Math.max(1, Math.floor(parent.clientHeight * r));
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      if (!isRecordingRef.current) drawIdleLine();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [drawIdleLine]);

  // Cleanup
  useEffect(() => () => {
    clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioContextRef.current) audioContextRef.current.close().catch(() => {});
    if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current);
  }, []);

  const canPlayback = hasRecording && audioURL;
  const isVideoBlob = (recordedBlobRef.current?.type || "").startsWith("video/");

  const togglePlayback = async () => {
    const video = playbackVideoRef.current;
    if (!video) return;
    if (video.paused) {
      await video.play().catch(() => {});
      setIsPlaybackPlaying(true);
    } else {
      video.pause();
      setIsPlaybackPlaying(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full">
      {/* Camera preview when video is enabled */}
      {effectiveVideoEnabled && (
        <div className="video-recorder-preview-container" style={{ maxWidth: "480px", aspectRatio: "16/9" }}>
          <video
            ref={videoPreviewRef}
            className="video-recorder-preview"
            muted
            playsInline
            style={{ display: isRecording && cameraReady ? "block" : "none", transform: "scaleX(-1)" }}
          />
          {canPlayback && isVideoBlob && !isRecording && (
            <>
              <video
                ref={playbackVideoRef}
                className="video-recorder-preview"
                src={audioURL}
                playsInline
                onEnded={() => setIsPlaybackPlaying(false)}
              />
              <button
                type="button"
                onClick={togglePlayback}
                className="absolute left-1/2 top-1/2 z-10 inline-flex h-14 w-14 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white/90 text-[#1B3A2D] shadow-lg backdrop-blur transition hover:bg-white"
                aria-label={isPlaybackPlaying ? "Pause video preview" : "Play video preview"}
              >
                {isPlaybackPlaying ? (
                  <svg className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5h3v14H8V5zm5 0h3v14h-3V5z" />
                  </svg>
                ) : (
                  <svg className="ml-0.5 h-7 w-7" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7L8 5z" />
                  </svg>
                )}
              </button>
              <div className="absolute bottom-3 left-3 rounded-full bg-white/90 px-3 py-1 text-xs font-semibold text-[#1B3A2D]">
                Preview ready
              </div>
            </>
          )}
          {!isRecording && !canPlayback && (
            <div className="video-recorder-no-camera">
              <svg className="w-10 h-10 text-[#B5B5B5]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
              <p className="text-xs text-[#B5B5B5] mt-2">Camera activates on record</p>
            </div>
          )}
          {isRecording && (
            <div className="video-recorder-rec-badge"><span className="video-recorder-rec-dot" /> Recording</div>
          )}
        </div>
      )}

      {/* Waveform */}
      <div className="w-full">
        <p className="text-xs font-medium text-[#777] uppercase tracking-wider mb-2.5">
          {effectiveVideoEnabled ? "Audio Waveform" : "Voice Waveform"}
        </p>
        <div className="w-full h-28 rounded-2xl bg-[#FAFAF7] border border-[#E8E8E8] overflow-hidden shadow-sm">
          <canvas ref={canvasRef} className="w-full h-full" />
        </div>
      </div>

      {/* Timer + level */}
      <div className="flex items-center gap-4 w-full">
        <div className="w-24 flex items-center justify-start">
          {isRecording && <span className="inline-block w-3 h-3 rounded-full bg-red-500 animate-pulse shadow-lg shadow-red-200" />}
        </div>
        <div className="text-center flex-1">
          <span className="text-3xl font-bold text-[#1B1B1B] font-mono tracking-tight">{formatTime(seconds)}</span>
          <p className="text-xs text-[#B5B5B5] mt-1">{isRecording ? "Recording" : "Ready"} • Max {formatTime(MAX_RECORDING_SECONDS)}</p>
        </div>
        <div className="w-24 text-right">
          <p className="text-xs uppercase tracking-[0.16em] text-[#777]">Level</p>
          <div className="mt-2 h-2 rounded-full bg-gray-100 overflow-hidden">
            <div className="h-full rounded-full transition-all duration-150" style={{
              width: `${Math.max(8, level * 100)}%`,
              background: "linear-gradient(to right, #52B788, #2D6A4F)",
            }} />
          </div>
        </div>
      </div>

      {/* Record button */}
      <div className="relative">
        {isRecording && <div className="absolute -inset-3 rounded-full bg-red-400/20 animate-pulse" />}
        <button
          type="button"
          onClick={isRecording ? stopRecording : startRecording}
          className={`relative w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg hover:shadow-xl active:scale-95 font-semibold text-white text-sm ${
            isRecording ? "bg-red-500 hover:bg-red-600 shadow-red-200"
            : effectiveVideoEnabled ? "bg-[#2D6A4F] hover:bg-[#1B3A2D] shadow-[#2D6A4F]/20"
            : "bg-[#2D6A4F] hover:bg-[#1B3A2D] shadow-[#2D6A4F]/20"
          }`}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
        >
          {isRecording ? (
            <div className="w-7 h-7 rounded-md bg-white" />
          ) : effectiveVideoEnabled ? (
            <svg className="w-9 h-9" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
            </svg>
          ) : (
            <svg className="w-9 h-9" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
            </svg>
          )}
        </button>
      </div>

      {/* Status text */}
      <div className="text-center w-full">
        {isRecording ? (
          <p className="text-sm font-medium text-[#777]">
            {effectiveVideoEnabled ? "Recording video & audio. Look at the camera and speak naturally." : "Recording... Tap stop when finished."}
          </p>
        ) : canPlayback && seconds >= MIN_RECORDING_SECONDS ? (
          <div className="space-y-3">
            <p className="text-sm font-medium text-[#52B788]">✓ Recording saved ({seconds}s)</p>
            {!isVideoBlob && (
              <div className="bg-white p-4 rounded-2xl border border-[#E8E8E8] shadow-sm">
                <audio controls controlsList="nodownload" src={audioURL} className="w-full h-12" />
              </div>
            )}
            <p className="text-xs text-[#B5B5B5]">
              File size: {(recordingBlobSize / 1024).toFixed(1)} KB{isVideoBlob && " (video + audio)"}
            </p>
          </div>
        ) : canPlayback && seconds < MIN_RECORDING_SECONDS ? (
          <p className="text-sm text-[#A94442]">Too short ({seconds}s). Min {MIN_RECORDING_SECONDS}s required.</p>
        ) : (
          <p className="text-sm font-medium text-[#777]">
            {effectiveVideoEnabled ? "Tap to start recording with camera & microphone." : "Tap the microphone to begin recording."}
          </p>
        )}
      </div>

      {cameraError && (
        <div className="rounded-xl border border-[#FEE2B3] bg-[#FFFBEB] px-4 py-2.5">
          <p className="text-xs text-[#92400E]">{cameraError}</p>
        </div>
      )}

      {hasRecording && audioURL && (
        <button type="button" onClick={startRecording}
          className="text-sm text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold flex items-center gap-2 transition-colors group hover:bg-[#D8F3DC] px-4 py-2 rounded-lg">
          <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
          </svg>
          Re-record
        </button>
      )}
    </div>
  );
}
