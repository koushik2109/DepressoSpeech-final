import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Loader from "../components/Loader.jsx";
import {
  getAssessmentDetail,
  getProcessingStatus,
  invalidateCache,
} from "../services/api.js";

const MAX_REVEAL_SECONDS = 120;

const processingSteps = [
  {
    label: "Loading Voice Responses",
    description: "Preparing the recorded answers for analysis",
    threshold: 10,
  },
  {
    label: "Analyzing Voice Patterns",
    description: "Running the ML model on audio features",
    threshold: 35,
  },
  {
    label: "Generating Report",
    description: "Building the PHQ-8 score report",
    threshold: 80,
  },
  {
    label: "Completed",
    description: "Report is ready to open",
    threshold: 100,
  },
];

const multimodalSteps = [
  {
    label: "Loading Recordings",
    description: "Preparing audio and video data",
    threshold: 8,
  },
  {
    label: "Extracting Audio Features",
    description: "Generating eGeMAPS, MFCC, and prosodic features",
    threshold: 20,
  },
  {
    label: "Analyzing Facial Features",
    description: "Processing video frames for action units and gaze",
    threshold: 40,
  },
  {
    label: "Transcribing Speech",
    description: "Converting speech to text for linguistic analysis",
    threshold: 55,
  },
  {
    label: "Trimodal Fusion",
    description: "Running cross-modal attention prediction",
    threshold: 80,
  },
  {
    label: "Completed",
    description: "Multimodal report is ready to open",
    threshold: 100,
  },
];

const fastSteps = [
  {
    label: "Scoring Answers",
    description: "Adding the selected PHQ-8 values",
    threshold: 40,
  },
  {
    label: "Preparing Report",
    description: "Formatting the score summary",
    threshold: 85,
  },
  {
    label: "Completed",
    description: "Report is ready to open",
    threshold: 100,
  },
];

function readAssessment() {
  try {
    return JSON.parse(sessionStorage.getItem("latestAssessment") || "{}");
  } catch {
    return {};
  }
}

export default function Processing() {
  const navigate = useNavigate();
  const latestAssessment = useMemo(() => readAssessment(), []);
  const hasAssessment = Boolean(latestAssessment.id);
  const hasAudio = (latestAssessment.recordingCount || 0) > 0;
  const hasVideo = Boolean(latestAssessment.hasVideoRecordings);
  const hasMultimodalReady = Boolean(
    latestAssessment.hasMultimodal && latestAssessment.multimodalResult,
  );
  const steps = hasVideo
    ? multimodalSteps
    : hasAudio
      ? processingSteps
      : fastSteps;
  const initialProgress = hasMultimodalReady
    ? 0  // animate 0→100 in useEffect
    : latestAssessment.status === "completed" ||
        latestAssessment.reportStatus === "available" ||
        latestAssessment.isReportReady
      ? 100
      : hasAudio
        ? 5
        : 0;
  const [progress, setProgress] = useState(initialProgress);
  const [status, setStatus] = useState(
    !hasAssessment
      ? "failed"
      : hasMultimodalReady
        ? "processing"  // will transition to completed after animation
        : latestAssessment.status === "completed" ||
            latestAssessment.reportStatus === "available" ||
            latestAssessment.isReportReady
          ? "completed"
          : "processing",
  );
  const [stage, setStage] = useState(
    hasMultimodalReady
      ? (steps[0]?.label ?? "Preparing results")
      : latestAssessment.status === "completed" ||
          latestAssessment.reportStatus === "available" ||
          latestAssessment.isReportReady
        ? "Completed"
        : hasAudio
          ? "Loading voice responses"
          : "Preparing results",
  );
  const [error, setError] = useState(
    hasAssessment ? "" : "Assessment not found. Please retake the assessment.",
  );
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [stepReachedAt, setStepReachedAt] = useState(() => {
    const reached = {};
    for (const step of steps) {
      if (initialProgress >= step.threshold) {
        reached[step.threshold] = 0;
      }
    }
    return reached;
  });
  const [forcedReady, setForcedReady] = useState(false);
  const startedAtRef = useRef(0);
  const hasNavigatedRef = useRef(false);
  const navigateTimerRef = useRef(null);

  const isCompleted = status === "completed";
  const isFailed = status === "failed";
  const activeStep = isCompleted
    ? steps.length
    : steps.findIndex((step) => progress < step.threshold);
  const displayStep =
    activeStep === -1 ? steps.length - 1 : Math.max(0, activeStep);

  const markCompletedSteps = useCallback(
    (nextProgress, atSeconds = 0) => {
      setStepReachedAt((previous) => {
        let changed = false;
        const next = { ...previous };
        for (const step of steps) {
          if (nextProgress >= step.threshold && next[step.threshold] == null) {
            next[step.threshold] = atSeconds;
            changed = true;
          }
        }
        return changed ? next : previous;
      });
    },
    [steps],
  );

  useEffect(() => {
    if (!latestAssessment.id) return undefined;
    startedAtRef.current = Date.now();

    let stopped = false;
    let timerId = null;

    const finishWithReport = async ({
      forceOpen = false,
      elapsedAt = 0,
    } = {}) => {
      const detail = await getAssessmentDetail(latestAssessment.id);
      if (stopped) return;
      sessionStorage.setItem("latestAssessment", JSON.stringify(detail));
      invalidateCache("GET:/assessments");
      setProgress(100);
      setStatus("completed");
      setStage(
        forceOpen ? "Score ready (voice analysis continues)" : "Completed",
      );
      setForcedReady(forceOpen);
      markCompletedSteps(100, elapsedAt);
      if (!hasNavigatedRef.current) {
        hasNavigatedRef.current = true;
        navigateTimerRef.current = window.setTimeout(() => {
          navigateTimerRef.current = null;
          navigate("/results");
        }, 500);
        timerId = navigateTimerRef.current;
      }
    };

    if (hasMultimodalReady) {
      // Animate stages to 100% before navigating so user sees the processing page.
      let p = 0;
      const stepTargets = steps.map((s) => s.threshold);
      let stepIdx = 0;
      const animate = () => {
        if (stopped) return;
        p = Math.min(p + 3, 100);
        setProgress(p);
        setElapsedSeconds((s) => s + 0.1);
        const target = stepTargets[stepIdx] ?? 100;
        if (p >= target) {
          setStage(steps[stepIdx]?.label ?? "Completed");
          markCompletedSteps(p, Math.round(p / 3));
          stepIdx = Math.min(stepIdx + 1, steps.length - 1);
        }
        if (p < 100) {
          timerId = window.setTimeout(animate, 60);
        } else {
          setStatus("completed");
          setStage("Completed");
          if (!hasNavigatedRef.current) {
            hasNavigatedRef.current = true;
            navigateTimerRef.current = window.setTimeout(() => {
              navigateTimerRef.current = null;
              navigate("/results");
            }, 800);
            timerId = navigateTimerRef.current;
          }
        }
      };
      timerId = window.setTimeout(animate, 60);
      return () => {
        stopped = true;
        if (timerId) window.clearTimeout(timerId);
        if (navigateTimerRef.current) window.clearTimeout(navigateTimerRef.current);
      };
    }

    const pollStatus = async () => {
      const nowMs = Date.now();
      const elapsed = Math.floor((nowMs - startedAtRef.current) / 1000);
      setElapsedSeconds(elapsed);

      if (elapsed >= MAX_REVEAL_SECONDS && !hasNavigatedRef.current) {
        try {
          await finishWithReport({ forceOpen: true, elapsedAt: elapsed });
        } catch (err) {
          if (!stopped) {
            setError(err.message || "Unable to open report.");
          }
        }
        return;
      }

      try {
        const data = await getProcessingStatus(latestAssessment.id);
        if (stopped) return;

        const nextProgress = Math.min(Number(data.progress ?? 0), 100);
        setProgress((current) => Math.max(current, nextProgress));
        markCompletedSteps(nextProgress, elapsed);
        setStage(data.stage || "Generating report");
        setStatus(
          data.status === "failed"
            ? "failed"
            : data.reportReady ||
                data.isReportReady ||
                data.reportStatus === "available" ||
                nextProgress >= 100
              ? "completed"
              : "processing",
        );
        setError("");

        if (data.status === "failed") {
          await finishWithReport({ forceOpen: true, elapsedAt: elapsed });
          return;
        }

        if (
          data.status === "completed" ||
          data.reportReady ||
          data.isReportReady ||
          data.reportStatus === "available" ||
          nextProgress >= 100
        ) {
          await finishWithReport({ forceOpen: false, elapsedAt: elapsed });
          return;
        }
      } catch (err) {
        if (!stopped) {
          setError(err.message || "Unable to refresh processing status.");
        }
      }

      if (!stopped) {
        timerId = window.setTimeout(pollStatus, 1000);
      }
    };

    pollStatus();

    return () => {
      stopped = true;
      if (timerId) window.clearTimeout(timerId);
      if (navigateTimerRef.current)
        window.clearTimeout(navigateTimerRef.current);
    };
  }, [latestAssessment.id, markCompletedSteps, navigate, hasMultimodalReady, steps]);

  const openReport = () => {
    if (!isCompleted) return;
    if (navigateTimerRef.current) {
      window.clearTimeout(navigateTimerRef.current);
      navigateTimerRef.current = null;
      navigate("/results");
      return;
    }
    if (hasNavigatedRef.current) return;
    hasNavigatedRef.current = true;
    navigate("/results");
  };

  const getStepDuration = (index) => {
    const endAt = stepReachedAt[steps[index].threshold];
    if (endAt == null) return null;
    const prevThreshold = index > 0 ? steps[index - 1].threshold : null;
    const startAt = prevThreshold ? stepReachedAt[prevThreshold] || 0 : 0;
    return Math.max(0, endAt - startAt);
  };

  const getActiveStepElapsed = (index) => {
    const prevThreshold = index > 0 ? steps[index - 1].threshold : null;
    const startAt = prevThreshold ? stepReachedAt[prevThreshold] || 0 : 0;
    return Math.max(0, elapsedSeconds - startAt);
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-[#F7F7F2]">
      <style>{`
        @keyframes shimmer {
          0%, 100% { opacity: 0.25; transform: translateY(0); }
          50% { opacity: 1; transform: translateY(-2px); }
        }
        @keyframes pulse-ring-anim {
          0% { transform: scale(1); opacity: 1; }
          100% { transform: scale(1.4); opacity: 0; }
        }
        @keyframes slide-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-shimmer { animation: shimmer 2s infinite; }
        .animate-pulse-ring-anim { animation: pulse-ring-anim 2s infinite; }
        .animate-slide-in { animation: slide-in 0.6s ease-out; }
      `}</style>

      <div className="w-full max-w-4xl text-center bg-white/80 backdrop-blur-md border border-[#E8E8E8] rounded-3xl shadow-[0_20px_60px_rgba(45,106,79,0.08)] px-6 py-10 md:px-10">
        <div className="flex justify-center mb-10 relative h-32">
          {!isCompleted ? (
            <Loader
              size="lg"
              text={`${isFailed ? "Failed" : "Generating"}... ${Math.round(progress)}%`}
            />
          ) : (
            <div className="w-24 h-24 rounded-full bg-[#D8F3DC] flex items-center justify-center border-2 border-[#B7E4C7] animate-slide-in">
              <svg
                className="w-12 h-12 text-[#2D6A4F]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
          )}
        </div>

        <div className="mb-8">
          <p className="text-sm font-bold uppercase tracking-[0.14em] text-[#52B788] mb-2">
            Status: {isCompleted ? "Completed" : isFailed ? "Failed" : stage}
          </p>
          <h1 className="text-3xl font-bold text-[#1B1B1B] mb-2 animate-slide-in">
            {isCompleted
              ? "Report ready"
              : isFailed
                ? "Generation failed"
                : "Preparing your result..."}
          </h1>
          <p
            className="text-base text-[#777] animate-slide-in"
            style={{ animationDelay: "0.1s" }}
          >
            {error || (isCompleted ? "Your score report is ready" : stage)}
          </p>
          {/* Modality indicator */}
          {!isCompleted && !isFailed && (
            <div className={`mt-4 inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-semibold ${hasVideo ? "border-[#93C5FD] bg-[#EFF6FF] text-[#1D4ED8]" : "border-[#B7E4C7] bg-[#F0FAF4] text-[#2D6A4F]"}`}>
              {hasVideo ? (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                  Trimodal analysis — Video + Audio + Speech
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
                  </svg>
                  Audio analysis — Voice patterns
                </>
              )}
            </div>
          )}
        </div>

        <div className="mb-8 px-2">
          <div className="h-2.5 bg-[#D8F3DC] rounded-full overflow-hidden border border-[#B7E4C7]">
            <div
              className="h-full bg-gradient-to-r from-[#52B788] to-[#2D6A4F] rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-[#B5B5B5] mt-2 font-medium">
            {Math.round(progress)}% complete
          </p>
          <p className="text-xs text-[#6A766F] mt-1">
            Elapsed {elapsedSeconds}s · Target ≤ {MAX_REVEAL_SECONDS}s
          </p>
          {forcedReady && (
            <p className="text-xs text-[#2D6A4F] mt-2">
              Score opened while voice analysis continues in the background.
            </p>
          )}
        </div>

        <div className="space-y-3 mb-6">
          {steps.map((step, i) => {
            const isDone = isCompleted || progress >= step.threshold;
            const isActive = !isCompleted && !isFailed && i === displayStep;
            const stepDuration = getStepDuration(i);

            return (
              <div
                key={step.label}
                className={`relative flex items-center gap-3 p-3 rounded-lg border transition-all duration-300 ${
                  isDone
                    ? "bg-[#F0FAF4] border-[#B7E4C7]"
                    : isActive
                      ? "bg-[#FAFAF7] border-[#2D6A4F]/40 shadow-sm scale-[1.02]"
                      : "bg-white/60 border-[#E8E8E8] opacity-70"
                }`}
                style={isActive ? { animation: "slide-in 0.3s ease-out" } : {}}
              >
                <div className="flex-shrink-0 relative w-8 h-8">
                  {isDone ? (
                    <div className="w-full h-full rounded-full bg-[#D8F3DC] flex items-center justify-center border border-[#52B788]">
                      <svg
                        className="w-4 h-4 text-[#2D6A4F]"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2.5}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    </div>
                  ) : isActive ? (
                    <>
                      <div className="absolute inset-0 rounded-full border-2 border-[#52B788]/30 animate-pulse-ring-anim" />
                      <div className="w-full h-full rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] flex items-center justify-center text-white text-sm font-semibold">
                        {i + 1}
                      </div>
                    </>
                  ) : (
                    <div className="w-full h-full rounded-full bg-gray-100 flex items-center justify-center border border-gray-200">
                      <div className="w-2 h-2 rounded-full bg-gray-400" />
                    </div>
                  )}
                </div>

                <div className="text-left flex-1">
                  <p
                    className={`text-sm font-semibold transition-colors ${isDone ? "text-[#2D6A4F]" : isActive ? "text-[#1B1B1B]" : "text-[#B5B5B5]"}`}
                  >
                    {step.label}
                  </p>
                  <p
                    className={`text-xs transition-colors ${isDone ? "text-[#52B788]" : isActive ? "text-[#777]" : "text-[#D1D5DB]"}`}
                  >
                    {step.description}
                  </p>
                </div>

                <p className="text-[11px] font-semibold text-[#6A766F] min-w-14 text-right">
                  {stepDuration != null
                    ? `${stepDuration.toFixed(1)}s`
                    : isActive
                      ? `${getActiveStepElapsed(i).toFixed(1)}s`
                      : "—"}
                </p>

                {isActive && (
                  <div className="flex gap-1">
                    {[0, 1, 2].map((j) => (
                      <div
                        key={j}
                        className="w-1.5 h-1.5 rounded-full bg-[#2D6A4F] animate-shimmer"
                        style={{ animationDelay: `${j * 0.2}s` }}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <button
            type="button"
            onClick={openReport}
            disabled={!isCompleted}
            className={`w-full sm:w-auto rounded-xl px-6 py-3 text-sm font-bold transition-colors ${
              isCompleted
                ? "bg-[#1B3A2D] text-white hover:bg-[#2D6A4F]"
                : "cursor-not-allowed bg-[#E8E8E8] text-[#9AA49F]"
            }`}
          >
            Open Report
          </button>
          {!isCompleted && !isFailed && (
            <p className="text-xs text-[#777] font-medium tracking-wide">
              Please keep this window open.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
