import { useMemo, useState, useCallback, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/Card.jsx";
import Button from "../components/Button.jsx";
import VoiceRecorder from "../components/VoiceRecorder.jsx";
import EnhancedDeviceCheck from "../components/EnhancedDeviceCheck.jsx";
import { buildQuestionSet, getSeverityLabel } from "../data/questionsData.js";
import {
  getCurrentUser,
  scoreQuestionAudio,
  saveAssessment,
  uploadAudio,
  processVideoRecording,
} from "../services/api.js";

function clampScore3(v) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.min(3, Math.round(n))) : 0;
}

/* ────────────────────────────────────────────
   MAIN ASSESSMENT PAGE
   ──────────────────────────────────────────── */
export default function Assessment() {
  const navigate = useNavigate();
  const [phase, setPhase] = useState("check"); // "check" | "assess"
  const [enableVideo, setEnableVideo] = useState(true);
  const [currentQ, setCurrentQ] = useState(0);
  const [voiceScores, setVoiceScores] = useState({});
  const [recordings, setRecordings] = useState({});
  const previewUrlsRef = useRef(new Set());
  const [audioFileIds, setAudioFileIds] = useState({});
  const [scoringQuestionId, setScoringQuestionId] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastLatencyMs, setLastLatencyMs] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const user = useMemo(() => getCurrentUser(), []);
  const questions = useMemo(() => buildQuestionSet(), []);
  const question = questions[currentQ];
  const questionId = question?.id;
  const hasRecording = Boolean(recordings[questionId]);
  const existingScore = voiceScores[questionId];
  const isLast = currentQ === questions.length - 1;
  const isScoringCurrent = scoringQuestionId === questionId;
  const isBusy = submitting || Boolean(scoringQuestionId);
  const canProceed = hasRecording && !isBusy;

  const score = Object.values(voiceScores).reduce((t, v) => t + Number(v || 0), 0);
  const progress = ((currentQ + 1) / questions.length) * 100;
  const completedCount = questions.filter((q) => recordings[q.id]).length;
  const videoCount = Object.values(recordings).filter(r => r?.isVideo).length;
  const upcomingQuestion = !isLast ? questions[currentQ + 1] : null;

  const handleReady = useCallback((videoEnabled) => {
    setEnableVideo(videoEnabled);
    setPhase("assess");
  }, []);

  const handleRecordingComplete = useCallback((blob, previewUrl, durationSeconds) => {
    const isVideo = blob.type.startsWith("video/");
    previewUrlsRef.current.add(previewUrl);
    setErrorMessage("");
    setRecordings(prev => {
      if (prev[questionId]?.previewUrl) {
        URL.revokeObjectURL(prev[questionId].previewUrl);
        previewUrlsRef.current.delete(prev[questionId].previewUrl);
      }
      return { ...prev, [questionId]: { blob, previewUrl, durationSeconds, isVideo } };
    });
    setVoiceScores(prev => { const n = { ...prev }; delete n[questionId]; return n; });
    setAudioFileIds(prev => { const n = { ...prev }; delete n[questionId]; return n; });
  }, [questionId]);

  const handleRecordingCleared = useCallback(() => {
    setErrorMessage("");
    setRecordings(prev => {
      if (prev[questionId]?.previewUrl) {
        URL.revokeObjectURL(prev[questionId].previewUrl);
        previewUrlsRef.current.delete(prev[questionId].previewUrl);
      }
      const n = { ...prev }; delete n[questionId]; return n;
    });
    setVoiceScores(prev => { const n = { ...prev }; delete n[questionId]; return n; });
    setAudioFileIds(prev => { const n = { ...prev }; delete n[questionId]; return n; });
  }, [questionId]);

  useEffect(() => {
    const previewUrls = previewUrlsRef.current;
    return () => {
      previewUrls.forEach((previewUrl) => {
        URL.revokeObjectURL(previewUrl);
      });
      previewUrls.clear();
    };
  }, []);

  const handleNext = async () => {
    if (!hasRecording || isBusy) return;
    setErrorMessage("");

    let qScore = Number(existingScore);
    let curAudioFileId = audioFileIds[questionId];

    if (!Number.isFinite(qScore) || !curAudioFileId) {
      setScoringQuestionId(questionId);
      try {
        const rec = recordings[questionId];
        if (!rec?.blob) throw new Error("Recording is required.");
        const uploaded = await uploadAudio(rec.blob, `q${questionId}.webm`);
        curAudioFileId = uploaded.fileId;
        const scored = await scoreQuestionAudio({ questionId, audioFileId: curAudioFileId, durationSec: rec.durationSeconds ?? null });
        qScore = clampScore3(scored.score);
        setLastLatencyMs(Number(scored.inferenceTimeMs ?? 0));
      } catch (err) {
        setErrorMessage(err.message || "Failed to score.");
        return;
      } finally {
        setScoringQuestionId(null);
      }
    }

    const nextScores = { ...voiceScores, [questionId]: clampScore3(qScore) };
    const nextAudioIds = { ...audioFileIds, [questionId]: curAudioFileId };
    setVoiceScores(nextScores);
    setAudioFileIds(nextAudioIds);

    if (isLast) {
      setSubmitting(true);
      try {
        const finalScore = Object.values(nextScores).reduce((t, v) => t + Number(v || 0), 0);
        const videoRecs = Object.entries(recordings).filter(([, r]) => r?.isVideo && r?.blob);
        const hasVideo = videoRecs.length > 0;

        const assessment = {
          userId: user?.id || null, userName: user?.name || "", email: user?.email || "", role: user?.role || "",
          answers: nextScores, audioFileIds: nextAudioIds,
          recordingMetadata: Object.fromEntries(Object.entries(recordings).map(([id, r]) => [id, { durationSeconds: r.durationSeconds ?? null, isVideo: r.isVideo || false }])),
          score: finalScore, severity: getSeverityLabel(finalScore),
          recordingCount: Object.keys(recordings).length,
          hasVideoRecordings: hasVideo, skipBackgroundInference: false,
          createdAt: new Date().toISOString(),
        };

        const saved = await saveAssessment(assessment);

        if (hasVideo) {
          try {
            const mmResults = await Promise.all(
              videoRecs.map(async ([qId, r]) => ({
                questionId: qId,
                result: await processVideoRecording(r.blob, `q${qId}.webm`, true),
              }))
            );
            saved.multimodalResult = mmResults[mmResults.length - 1]?.result || null;
            saved.multimodalResults = mmResults;
            saved.hasMultimodal = true;
          } catch (mmErr) {
            console.warn("Multimodal failed:", mmErr.message);
            saved.multimodalError = mmErr.message;
            saved.hasMultimodal = false;
          }
        }

        sessionStorage.setItem("latestAssessment", JSON.stringify(saved));
        navigate("/processing");
      } catch (err) {
        setErrorMessage(err.message || "Failed to save.");
      } finally {
        setSubmitting(false);
      }
      return;
    }
    setCurrentQ(prev => prev + 1);
  };

  const handlePrev = () => { if (currentQ > 0 && !isBusy) { setErrorMessage(""); setCurrentQ(prev => prev - 1); } };

  // ── Phase: Device Check ──
  if (phase === "check") return <EnhancedDeviceCheck onReady={handleReady} />;

  // ── Phase: Assessment ──
  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
      <div className="w-full max-w-[90rem] mx-auto animate-fade-in">
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.18em] uppercase font-semibold text-[#52B788] mb-3">PHQ-8 Assessment</p>
          <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
            {user?.name ? `Welcome, ${user.name}` : "PHQ-8 Screening"}
          </h1>
          <p className="mt-3 text-base text-[#777]">
            {enableVideo ? "Recording video & audio per question for trimodal analysis." : "Record your voice answer, then continue."}
          </p>
        </div>

        {/* Progress */}
        <div className="mb-8 max-w-4xl mx-auto">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-[#777] font-medium">Question {currentQ + 1} / {questions.length}</span>
            <div className="flex items-center gap-3">
              <span className="font-semibold text-[#2D6A4F]">Score {score}/24</span>
            </div>
          </div>
          <div className="w-full h-2.5 bg-[#D8F3DC] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-500 ease-out" style={{
              width: `${progress}%`,
              background: "linear-gradient(90deg, #52B788, #2D6A4F)",
            }} />
          </div>
        </div>

        <Card className="shadow-elevated p-8 md:p-10 max-w-4xl mx-auto">
          <div className="space-y-8">
            {/* Question */}
            <div className="text-left space-y-4">
              <div className="inline-flex items-center px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide uppercase"
                style={{ backgroundColor: "#D8F3DC", color: "#2D6A4F" }}>
                Question {currentQ + 1}
              </div>
              <h2 className="text-3xl font-semibold text-[#1B1B1B] leading-snug max-w-3xl">{question.text}</h2>
            </div>

            {/* Info */}
            <div className="rounded-2xl border border-[#E8E8E8] bg-[#FAFAF7] p-5">
              <p className="text-sm text-[#555]">{enableVideo
                ? "Speak naturally while looking at the camera. Your facial expressions and voice will be analyzed."
                : "Click Next to run model scoring from your recorded audio."}</p>
            </div>

            {/* Recorder */}
            <div className="rounded-2xl border border-[#DDEBE2] bg-white p-5 md:p-6">
              <div className="mb-5 grid gap-3 rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4 sm:grid-cols-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#6A766F]">Responses</p>
                  <p className="mt-1 text-sm font-bold text-[#1B1B1B]">{completedCount}/{questions.length}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#6A766F]">Question Score</p>
                  <p className="mt-1 text-sm font-bold text-[#2D6A4F]">
                    {Number.isFinite(Number(existingScore)) ? `${clampScore3(existingScore)}/3` : "Pending"}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-[#6A766F]">Video Captured</p>
                  <p className="mt-1 text-sm font-bold text-[#1B1B1B]">{enableVideo ? `${videoCount}/${questions.length}` : "Off"}</p>
                </div>
              </div>
              <p className="text-xs tracking-[0.16em] uppercase font-semibold mb-2" style={{ color: "#52B788" }}>
                {enableVideo ? "Video & Voice Recorder" : "Voice Recorder"}
              </p>
              {lastLatencyMs != null && <p className="text-xs text-[#6A766F] mb-4">Model response: {(lastLatencyMs / 1000).toFixed(2)}s</p>}

              <VoiceRecorder
                key={`${question.id}-${enableVideo}`}
                onRecordingComplete={handleRecordingComplete}
                onRecordingCleared={handleRecordingCleared}
                enableVideo={enableVideo}
              />
            </div>

            {errorMessage && (
              <div role="alert" className="rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-4 py-3">
                <p className="text-sm font-medium text-[#A94442]">{errorMessage}</p>
              </div>
            )}

            {upcomingQuestion && hasRecording && (
              <div className="rounded-xl border border-[#E8E8E8] bg-[#F8FBF9] px-4 py-4">
                <p className="text-xs uppercase tracking-[0.16em] font-semibold text-[#52B788] mb-2">Up Next</p>
                <p className="text-sm text-[#4A5550]">{upcomingQuestion.text}</p>
              </div>
            )}

            <div className="flex items-center justify-between gap-4">
              <Button variant="ghost" onClick={handlePrev} disabled={currentQ === 0 || isBusy}>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
                </svg>
                Previous
              </Button>
              <Button variant="primary" onClick={handleNext} disabled={!canProceed}>
                {isScoringCurrent ? "Scoring..." : submitting ? "Processing..." : isLast ? (enableVideo ? "Submit & Analyze" : "Submit Assessment") : "Next Question"}
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
