import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import VideoRecorder from "../components/VideoRecorder.jsx";
import ModalityContribution from "../components/ModalityContribution.jsx";
import Loader from "../components/Loader.jsx";
import { processVideoRecording } from "../services/api.js";

const SEVERITY_COLORS = {
  Minimal: "#52B788",
  Mild: "#95D5B2",
  Moderate: "#FBBF24",
  "Moderately Severe": "#FB923C",
  Severe: "#EF4444",
};

const PROCESSING_STEPS = [
  { id: "upload", label: "Uploading video", icon: "☁️" },
  { id: "extract", label: "Extracting audio & frames", icon: "🎬" },
  { id: "features", label: "Generating features", icon: "📊" },
  { id: "predict", label: "Running multimodal analysis", icon: "🧠" },
  { id: "done", label: "Complete", icon: "✓" },
];

export default function LiveAssessment() {
  const navigate = useNavigate();

  const [recording, setRecording] = useState(null);     // { blob, url, duration, filename }
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [enableSTT, setEnableSTT] = useState(true);

  const handleRecordingComplete = useCallback((blob, url, duration, filename) => {
    setRecording({ blob, url, duration, filename });
    setError("");
  }, []);

  const handleRecordingCleared = useCallback(() => {
    setRecording(null);
    setError("");
  }, []);

  const handleSubmit = async () => {
    if (!recording?.blob) {
      setError("Please record a video first.");
      return;
    }
    if (recording.duration < 5) {
      setError("Recording is too short. Please record at least 5 seconds.");
      return;
    }

    setProcessing(true);
    setError("");
    setCurrentStep(0);

    try {
      // Step 1: Upload
      setCurrentStep(0);

      // Simulate step progression (server processes synchronously)
      const stepTimer = setInterval(() => {
        setCurrentStep((prev) => Math.min(prev + 1, PROCESSING_STEPS.length - 2));
      }, 3000);

      const response = await processVideoRecording(
        recording.blob,
        recording.filename || "recording.webm",
        enableSTT,
      );

      clearInterval(stepTimer);
      setCurrentStep(PROCESSING_STEPS.length - 1);

      // Small delay for visual satisfaction
      await new Promise((r) => setTimeout(r, 500));

      setResult(response);
    } catch (err) {
      setError(err.message || "Processing failed. Please try again.");
    } finally {
      setProcessing(false);
    }
  };

  const handleNewAssessment = () => {
    setRecording(null);
    setResult(null);
    setError("");
    setCurrentStep(0);
  };

  // ─── Results View ───
  if (result) {
    return (
      <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
        <div className="max-w-5xl mx-auto animate-fade-in">
          {/* Header */}
          <div className="text-center mb-10">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#D8F3DC] text-[#2D6A4F] text-sm font-semibold mb-4">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Analysis Complete
            </div>
            <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
              Your Results
            </h1>
          </div>

          {/* Score Card */}
          <div className="multimodal-section-card mb-8">
            <div className="grid lg:grid-cols-2 gap-8 items-center">
              <div>
                <h2 className="text-3xl font-bold text-[#1B1B1B]">
                  PHQ-8 Score: <span style={{ color: SEVERITY_COLORS[result.severity] || "#2D6A4F" }}>{result.phq8_score}</span>/24
                </h2>
                <p
                  className="text-xl font-semibold mt-2"
                  style={{ color: SEVERITY_COLORS[result.severity] || "#2D6A4F" }}
                >
                  {result.severity}
                </p>

                <div className="flex flex-wrap gap-2 mt-4">
                  {(result.modalities_used || []).map((mod) => (
                    <span
                      key={mod}
                      className="px-3 py-1.5 rounded-full text-xs font-semibold"
                      style={{
                        backgroundColor: mod === "audio" ? "#D8F3DC" : mod === "video" ? "#EDE9FE" : "#FEF3C7",
                        color: mod === "audio" ? "#2D6A4F" : mod === "video" ? "#7C3AED" : "#D97706",
                      }}
                    >
                      {mod.charAt(0).toUpperCase() + mod.slice(1)} ✓
                    </span>
                  ))}
                </div>

                <div className="mt-6 grid grid-cols-2 gap-4">
                  <div className="rounded-xl bg-[#F7F7F2] p-4">
                    <p className="text-xs text-[#777] uppercase tracking-wider mb-1">Confidence</p>
                    <p className="text-lg font-bold text-[#1B1B1B]">{Math.round((result.confidence || 0) * 100)}%</p>
                  </div>
                  <div className="rounded-xl bg-[#F7F7F2] p-4">
                    <p className="text-xs text-[#777] uppercase tracking-wider mb-1">Processing</p>
                    <p className="text-lg font-bold text-[#1B1B1B]">{(result.processing_time_s || 0).toFixed(1)}s</p>
                  </div>
                </div>
              </div>

              {/* PHQ-8 Visual Scale */}
              <div className="flex justify-center">
                <div className="relative w-48 h-48">
                  <svg viewBox="0 0 200 200" className="w-full h-full">
                    {/* Background arc */}
                    <circle cx="100" cy="100" r="80" fill="none" stroke="#E8E8E8" strokeWidth="12" strokeDasharray="502" strokeDashoffset="126" transform="rotate(135 100 100)" strokeLinecap="round" />
                    {/* Score arc */}
                    <circle
                      cx="100" cy="100" r="80" fill="none"
                      stroke={SEVERITY_COLORS[result.severity] || "#52B788"}
                      strokeWidth="12"
                      strokeDasharray="502"
                      strokeDashoffset={502 - (376 * Math.min(result.phq8_score, 24) / 24)}
                      transform="rotate(135 100 100)"
                      strokeLinecap="round"
                      className="transition-all duration-1000"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-4xl font-bold text-[#1B1B1B]">{result.phq8_score}</span>
                    <span className="text-xs text-[#777]">out of 24</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Modality Contributions */}
          <div className="multimodal-section-card mb-8">
            <ModalityContribution
              contributions={result.modality_contributions || {}}
              modalitiesUsed={result.modalities_used || []}
              confidence={result.confidence || 0}
              phq8Score={result.phq8_score}
              severity={result.severity}
            />
          </div>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="multimodal-process-btn" onClick={handleNewAssessment}>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
              </svg>
              New Assessment
            </button>
            <button className="results-btn-outline" onClick={() => navigate("/assessment-history")}>
              View History
            </button>
            <button className="results-btn-outline" onClick={() => navigate("/")}>
              Return Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ─── Processing View ───
  if (processing) {
    return (
      <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
        <div className="max-w-2xl mx-auto">
          <div className="multimodal-section-card text-center py-16">
            <div className="inline-flex flex-col items-center gap-8">
              <div className="multimodal-processing-spinner" />

              <div>
                <h2 className="text-2xl font-bold text-[#1B1B1B] mb-2">Processing Your Recording</h2>
                <p className="text-sm text-[#777] max-w-md mx-auto">
                  Extracting audio, analyzing facial features, and running our
                  trimodal fusion model on your recording.
                </p>
              </div>

              {/* Processing steps */}
              <div className="w-full max-w-sm space-y-3 text-left">
                {PROCESSING_STEPS.map((step, i) => {
                  const isActive = i === currentStep;
                  const isDone = i < currentStep;
                  return (
                    <div
                      key={step.id}
                      className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 ${isActive
                          ? "bg-gradient-to-r from-[#EDE9FE] to-[#D8F3DC] border border-[#7C3AED]/20 shadow-sm"
                          : isDone
                            ? "bg-[#F0FAF4] border border-[#D8F3DC]"
                            : "bg-[#FAFAF7] border border-transparent"
                        }`}
                    >
                      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${isDone
                          ? "bg-[#52B788] text-white"
                          : isActive
                            ? "bg-[#7C3AED] text-white"
                            : "bg-[#E8E8E8] text-[#999]"
                        }`}>
                        {isDone ? (
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        ) : isActive ? (
                          <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                        ) : (
                          i + 1
                        )}
                      </div>
                      <span className={`text-sm font-medium ${isActive ? "text-[#7C3AED]" : isDone ? "text-[#2D6A4F]" : "text-[#B5B5B5]"
                        }`}>
                        {step.label}
                      </span>
                      {isActive && <Loader size="sm" />}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ─── Recording View (Default) ───
  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
      <div className="max-w-4xl mx-auto animate-fade-in">
        {/* Header */}
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.18em] uppercase font-semibold text-[#7C3AED] mb-3">
            Live Assessment
          </p>
          <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
            Video Recording Analysis
          </h1>
          <p className="mt-3 text-base text-[#777] max-w-2xl mx-auto">
            Record a short video of yourself speaking naturally. Our AI will analyze
            your voice, facial expressions, and speech patterns simultaneously.
          </p>
        </div>

        {/* Main Card */}
        <div className="multimodal-section-card">
          <VideoRecorder
            onRecordingComplete={handleRecordingComplete}
            onRecordingCleared={handleRecordingCleared}
          />
        </div>

        {/* Options */}
        <div className="mt-6 multimodal-section-card">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-[#1B1B1B]">Speech-to-Text</h3>
              <p className="text-xs text-[#777] mt-0.5">
                Transcribe your speech for text-based analysis (adds processing time)
              </p>
            </div>
            <button
              type="button"
              onClick={() => setEnableSTT(!enableSTT)}
              className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${enableSTT ? "bg-[#7C3AED]" : "bg-[#D9D9D9]"
                }`}
            >
              <span
                className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200 ${enableSTT ? "translate-x-[22px]" : "translate-x-0.5"
                  }`}
              />
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-5 py-4">
            <p className="text-sm font-medium text-[#A94442]">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex justify-center mt-8">
          <button
            className="multimodal-process-btn"
            onClick={handleSubmit}
            disabled={!recording?.blob || processing}
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
            </svg>
            Analyze Recording
          </button>
        </div>

        {/* Info Cards */}
        <div className="mt-10 grid md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-[#E8E8E8] bg-white p-5">
            <div className="w-10 h-10 rounded-xl bg-[#D8F3DC] flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51" />
              </svg>
            </div>
            <h4 className="text-sm font-bold text-[#1B1B1B] mb-1">Voice Analysis</h4>
            <p className="text-xs text-[#777] leading-relaxed">
              Extracts prosodic features like pitch, energy, and speaking rate from your audio.
            </p>
          </div>
          <div className="rounded-xl border border-[#E8E8E8] bg-white p-5">
            <div className="w-10 h-10 rounded-xl bg-[#EDE9FE] flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-[#7C3AED]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <h4 className="text-sm font-bold text-[#1B1B1B] mb-1">Facial Analysis</h4>
            <p className="text-xs text-[#777] leading-relaxed">
              Detects facial action units, gaze patterns, and head pose from video frames.
            </p>
          </div>
          <div className="rounded-xl border border-[#E8E8E8] bg-white p-5">
            <div className="w-10 h-10 rounded-xl bg-[#FEF3C7] flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-[#D97706]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <h4 className="text-sm font-bold text-[#1B1B1B] mb-1">Speech Content</h4>
            <p className="text-xs text-[#777] leading-relaxed">
              Transcribes your speech and analyzes language patterns and sentiment.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
