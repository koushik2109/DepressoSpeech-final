import { useCallback, useMemo, useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  BarChart,
  Bar,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
  ErrorBar,
} from "recharts";
import DepessionSpeedometer from "../components/DepessionSpeedometer.jsx";
import ChartPanel from "../components/ChartPanel.jsx";
import {
  getSeverityDescription,
  getSeverityLabel,
  PHQ8_QUESTIONS,
} from "../data/questionsData.js";
import {
  getCurrentUser,
  getAssessmentDetail,
  listAssessments,
  getMLDetails,
  invalidateCache,
} from "../services/api.js";

const severityGuidance = {
  Minimal: {
    explanation:
      "Your responses suggest low symptom frequency. Keep monitoring and maintain healthy routines.",
    suggestions: [
      "Maintain current routine and sleep consistency.",
      "Retake the PHQ-8 in 2 to 4 weeks to confirm stability.",
    ],
  },
  Mild: {
    explanation:
      "Your responses suggest mild symptoms. Early lifestyle support can prevent worsening.",
    suggestions: [
      "Add structured activity such as walking or light exercise.",
      "Schedule a check-in with a counselor if symptoms persist.",
    ],
  },
  Moderate: {
    explanation:
      "Your responses suggest a moderate symptom burden. Clinical follow-up is recommended.",
    suggestions: [
      "Book a mental health consultation in the near term.",
      "Track mood and sleep daily to discuss during review.",
    ],
  },
  "Moderately Severe": {
    explanation:
      "Your responses suggest high symptom burden. Prompt professional care is advised.",
    suggestions: [
      "Arrange a clinician appointment as soon as possible.",
      "Share this report with your doctor for quicker triage.",
    ],
  },
  Severe: {
    explanation:
      "Your responses suggest severe burden. Immediate professional support is strongly advised.",
    suggestions: [
      "Seek urgent professional support today.",
      "Avoid staying isolated and contact trusted support immediately.",
    ],
  },
};

const severityColors = {
  Minimal: "#52B788",
  Mild: "#95D5B2",
  Moderate: "#FBBF24",
  "Moderately Severe": "#FB923C",
  Severe: "#EF4444",
};

function readLatestAssessment() {
  try {
    return JSON.parse(sessionStorage.getItem("latestAssessment") || "null");
  } catch {
    return null;
  }
}

function normalizeAssessmentDetail(detail, fallback) {
  if (!detail) return fallback;
  const answers = Array.isArray(detail.answers)
    ? detail.answers.reduce((acc, item) => {
        acc[item.questionId] = item.score;
        return acc;
      }, {})
    : detail.answers;

  return {
    ...(fallback || {}),
    ...detail,
    answers,
    score: detail.score ?? fallback?.score,
    severity: detail.severity || fallback?.severity,
    recordingCount: detail.recordingCount ?? fallback?.recordingCount,
  };
}

export default function Results() {
  const [assessment, setAssessment] = useState(() => readLatestAssessment());
  const [allAssessments, setAllAssessments] = useState([]);
  const [mlDetails, setMlDetails] = useState(null);
  const [fallbackReportDate] = useState(() => Date.now());

  const score = assessment?.score ?? 0;
  const severity = assessment?.severity || getSeverityLabel(score);
  const severityColor = severityColors[severity] || "#2D6A4F";
  const assessmentAnswers = assessment ? assessment.answers : null;
  const answerMap = useMemo(() => {
    if (Array.isArray(assessmentAnswers)) {
      return assessmentAnswers.reduce((acc, item) => {
        acc[item.questionId] = item.score;
        return acc;
      }, {});
    }
    return assessmentAnswers || {};
  }, [assessmentAnswers]);
  const answerCount = Object.keys(answerMap).length;
  const user = getCurrentUser();

  const refreshAssessments = useCallback(
    () =>
      listAssessments()
        .then(setAllAssessments)
        .catch(() => setAllAssessments([])),
    [],
  );

  useEffect(() => {
    refreshAssessments();
  }, [refreshAssessments]);

  useEffect(() => {
    const assessmentId = assessment?.id;
    const assessmentStatus = assessment?.status;
    if (!assessmentId) return undefined;

    let stopped = false;
    let intervalId = null;

    const refreshCurrent = async () => {
      try {
        const detail = await getAssessmentDetail(assessmentId);
        if (stopped) return;
        setAssessment((current) => {
          const next = normalizeAssessmentDetail(detail, current);
          sessionStorage.setItem("latestAssessment", JSON.stringify(next));
          return next;
        });
        setMlDetails(detail.mlDetails ?? null);
        if (detail.status === "completed" || detail.status === "failed") {
          invalidateCache("GET:/assessments");
          refreshAssessments();
          if (intervalId) clearInterval(intervalId);
        }
      } catch {
        getMLDetails(assessmentId)
          .then((data) => !stopped && setMlDetails(data.mlDetails))
          .catch(() => !stopped && setMlDetails(null));
      }
    };

    refreshCurrent();
    if (assessmentStatus !== "completed" && assessmentStatus !== "failed") {
      intervalId = setInterval(refreshCurrent, 2000);
    }

    return () => {
      stopped = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [assessment?.id, assessment?.status, refreshAssessments]);

  const userAssessments = allAssessments
    .filter((item) => {
      const sameUser =
        user?.id && item.userId ? item.userId === user.id : false;
      const sameEmail =
        user?.email && item.email
          ? item.email.toLowerCase() === user.email.toLowerCase()
          : false;
      return sameUser || sameEmail;
    })
    .sort(
      (a, b) =>
        new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
    );

  const trendData = userAssessments.map((item, index) => ({
    session: `S${index + 1}`,
    score: item.score,
    mlScore: item.mlScore ?? null,
    severity: item.severity,
    date: new Date(item.createdAt).toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
    }),
  }));

  const guidance = severityGuidance[severity] || severityGuidance.Mild;

  const chartData = PHQ8_QUESTIONS.map((questionItem) => {
    const itemScore = Number(answerMap[questionItem.id] ?? 0);
    return {
      name: `Q${questionItem.id}`,
      value: Math.round((itemScore / 3) * 100),
      color:
        itemScore <= 1 ? "#52B788" : itemScore === 2 ? "#FBBF24" : "#FB923C",
    };
  });

  const severityData = {
    level: severity,
    score,
    description: getSeverityDescription(score),
  };

  const reportDate = new Date(assessment?.createdAt || fallbackReportDate);
  const formattedDate = reportDate.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2]">
      {/* ─── Header Banner ─── */}
      <div className="results-header">
        <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
            <div>
              <p className="text-xs font-semibold tracking-[0.18em] uppercase text-[#52B788] mb-3">
                Assessment Complete
              </p>
              <h1 className="results-page-title">Your Score Report</h1>
            </div>
            <div className="text-left md:text-right">
              <p className="text-xs text-[#B5B5B5] font-medium">Report date</p>
              <p className="text-sm font-semibold text-[#1B1B1B]">
                {formattedDate}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        {/* ─── Stat Cards ─── */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 -mt-8 mb-12 relative z-10">
          <div className="results-stat-card">
            <div
              className="results-stat-icon"
              style={{
                backgroundColor: `${severityColor}18`,
                color: severityColor,
              }}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <p className="text-xs font-medium text-[#777] mb-1">
              Severity Level
            </p>
            <p className="text-2xl font-bold" style={{ color: severityColor }}>
              {severityData.level}
            </p>
            <p className="text-xs text-[#B5B5B5]">
              Score: {severityData.score}/24
            </p>
          </div>
          <div className="results-stat-card">
            <div
              className="results-stat-icon"
              style={{ backgroundColor: "#D8F3DC", color: "#2D6A4F" }}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.172a2 2 0 011.414.586l5.828 5.828A2 2 0 0120 10.828V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <p className="text-xs font-medium text-[#777] mb-1">
              Questions Answered
            </p>
            <p className="text-2xl font-bold text-[#1B1B1B]">
              {answerCount} / 8
            </p>
            <p className="text-xs text-[#B5B5B5]">All PHQ-8 items</p>
          </div>
          <div className="results-stat-card">
            <div
              className="results-stat-icon"
              style={{ backgroundColor: "#D1FAE5", color: "#10B981" }}
            >
              <svg
                className="w-5 h-5"
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
            <p className="text-xs font-medium text-[#777] mb-1">Status</p>
            <p className="text-2xl font-bold text-[#10B981]">Completed</p>
            <p className="text-xs text-[#B5B5B5]">Ready for review</p>
          </div>
          <div className="results-stat-card">
            <div
              className="results-stat-icon"
              style={{ backgroundColor: "#D8F3DC", color: "#2D6A4F" }}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.964 0a9 9 0 10-11.964 0m11.964 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            </div>
            <p className="text-xs font-medium text-[#777] mb-1">Patient Name</p>
            <p className="text-2xl font-bold text-[#1B1B1B]">
              {assessment?.userName || "User"}
            </p>
            <p className="text-xs text-[#B5B5B5]">
              {assessment?.email || "not specified"}
            </p>
          </div>
        </div>

        {/* ─── Speedometer ─── */}
        <div className="results-section-card mb-10">
          <div className="mb-6">
            <h2 className="results-section-heading">Severity Meter</h2>
            <p className="text-sm text-[#777]">
              Visual representation of your PHQ-8 score on the 0–24 scale.
            </p>
          </div>
          <DepessionSpeedometer
            score={severityData.score}
            level={severityData.level}
            maxScore={24}
          />
        </div>

        {/* ─── Multimodal Results (when video was recorded) ─── */}
        {assessment?.hasMultimodal && assessment?.multimodalResult && (
          <div className="results-section-card mb-10">
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="results-section-heading">Multimodal Analysis</h2>
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-gradient-to-r from-[#EDE9FE] to-[#D8F3DC] text-xs font-bold text-[#7C3AED]">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                  Trimodal Fusion
                </span>
              </div>
              <p className="text-sm text-[#777]">
                Your assessment included video recording, enabling analysis across voice, facial expressions, and speech content.
              </p>
            </div>

            {/* Modality badges */}
            <div className="flex flex-wrap gap-2 mb-6">
              {(assessment.multimodalResult.modalities_used || []).map((mod) => (
                <span
                  key={mod}
                  className="px-3 py-1.5 rounded-full text-xs font-semibold"
                  style={{
                    backgroundColor: mod === "audio" ? "#D8F3DC" : mod === "video" ? "#EDE9FE" : "#FEF3C7",
                    color: mod === "audio" ? "#2D6A4F" : mod === "video" ? "#7C3AED" : "#D97706",
                  }}
                >
                  {mod === "audio" ? "🎙️" : mod === "video" ? "🎬" : "📝"} {mod.charAt(0).toUpperCase() + mod.slice(1)}
                </span>
              ))}
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="rounded-xl bg-[#F7F7F2] p-4 text-center">
                <p className="text-xs text-[#777] uppercase tracking-wider mb-1">Multimodal Score</p>
                <p className="text-2xl font-bold text-[#1B1B1B]">
                  {assessment.multimodalResult.phq8_score ?? "—"}
                </p>
              </div>
              <div className="rounded-xl bg-[#F7F7F2] p-4 text-center">
                <p className="text-xs text-[#777] uppercase tracking-wider mb-1">Confidence</p>
                <p className="text-2xl font-bold text-[#1B1B1B]">
                  {Math.round((assessment.multimodalResult.confidence || 0) * 100)}%
                </p>
              </div>
              <div className="rounded-xl bg-[#F7F7F2] p-4 text-center">
                <p className="text-xs text-[#777] uppercase tracking-wider mb-1">Processing</p>
                <p className="text-2xl font-bold text-[#1B1B1B]">
                  {(assessment.multimodalResult.processing_time_s || 0).toFixed(1)}s
                </p>
              </div>
            </div>

            {/* Modality contribution bars */}
            {assessment.multimodalResult.modality_contributions && (
              <div className="space-y-3">
                <p className="text-xs font-semibold text-[#777] uppercase tracking-wider">
                  Modality Contributions
                </p>
                {Object.entries(assessment.multimodalResult.modality_contributions).map(
                  ([mod, value]) => {
                    const colors = {
                      audio: { bar: "#2D6A4F", bg: "#D8F3DC", label: "Audio (Voice)" },
                      video: { bar: "#7C3AED", bg: "#EDE9FE", label: "Video (Face)" },
                      text: { bar: "#D97706", bg: "#FEF3C7", label: "Text (Speech)" },
                    };
                    const c = colors[mod] || colors.audio;
                    const pct = Math.round((value || 0) * 100);
                    return (
                      <div key={mod} className="flex items-center gap-3">
                        <span className="text-xs font-medium text-[#555] w-28">{c.label}</span>
                        <div className="flex-1 h-3 rounded-full overflow-hidden" style={{ backgroundColor: c.bg }}>
                          <div
                            className="h-full rounded-full transition-all duration-700"
                            style={{ width: `${pct}%`, backgroundColor: c.bar }}
                          />
                        </div>
                        <span className="text-xs font-bold text-[#1B1B1B] w-10 text-right">{pct}%</span>
                      </div>
                    );
                  },
                )}
              </div>
            )}
          </div>
        )}

        {/* Multimodal error notice */}
        {assessment?.hasVideoRecordings && !assessment?.hasMultimodal && assessment?.multimodalError && (
          <div className="rounded-xl border border-[#FEE2B3] bg-[#FFFBEB] px-5 py-4 mb-10">
            <div className="flex items-center gap-2 mb-1">
              <svg className="w-4 h-4 text-[#92400E]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <p className="text-sm font-semibold text-[#92400E]">Multimodal analysis unavailable</p>
            </div>
            <p className="text-xs text-[#B45309]">
              Video was recorded but multimodal processing failed: {assessment.multimodalError}. Audio-only results are shown above.
            </p>
          </div>
        )}

        {/* ─── Charts + Interpretation Grid ─── */}
        <div className="grid lg:grid-cols-3 gap-8 mb-10">
          <div className="lg:col-span-2 space-y-8">
            {/* Response chart */}
            <div className="results-section-card">
              <div className="mb-6">
                <h2 className="results-section-heading">Response Pattern</h2>
                <p className="text-sm text-[#777]">
                  How your answers were distributed across severity options.
                </p>
              </div>
              <ChartPanel data={chartData} title="" />
            </div>

            <div className="results-section-card">
              <h2 className="results-section-heading mb-2">Trend Over Time</h2>
              <p className="text-sm text-[#777] mb-6">
                Your PHQ-8 score across completed sessions.
              </p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={trendData}
                    margin={{ top: 10, right: 16, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#EAF3EE" />
                    <XAxis
                      dataKey="session"
                      tick={{ fill: "#777", fontSize: 12 }}
                      tickLine={false}
                      axisLine={{ stroke: "#E8E8E8" }}
                    />
                    <YAxis
                      domain={[0, 24]}
                      tick={{ fill: "#777", fontSize: 12 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload || !payload.length) return null;
                        const entry = payload[0].payload;
                        return (
                          <div className="rounded-xl border border-[#E8E8E8] bg-white px-3 py-2 shadow-md">
                            <p className="text-xs text-[#777]">{entry.date}</p>
                            <p className="text-sm font-semibold text-[#1B1B1B]">
                              Score {entry.score}/24
                            </p>
                            <p className="text-xs text-[#2D6A4F]">
                              {entry.severity}
                            </p>
                          </div>
                        );
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke="#2D6A4F"
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 0, fill: "#52B788" }}
                      activeDot={{ r: 6 }}
                      name="Self-Report"
                    />
                    {trendData.some((d) => d.mlScore != null) && (
                      <Line
                        type="monotone"
                        dataKey="mlScore"
                        stroke="#7C3AED"
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={{ r: 3, strokeWidth: 0, fill: "#7C3AED" }}
                        activeDot={{ r: 5 }}
                        name="ML Voice"
                        connectNulls
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {trendData.length < 2 && (
                <p className="text-xs text-[#B5B5B5] mt-4">
                  Complete more assessments to see a clearer personal trend.
                </p>
              )}
            </div>
          </div>

          {/* Interpretation panel */}
          <div className="space-y-8">
            <div className="results-section-card">
              <h3 className="text-lg font-bold text-[#1B1B1B] mb-6">
                Severity Explanation
              </h3>
              <div
                className="pl-4 border-l-4 rounded-sm"
                style={{ borderColor: severityColor }}
              >
                <p className="text-xs font-semibold tracking-[0.16em] text-[#52B788] uppercase mb-2">
                  Summary
                </p>
                <p className="text-sm text-[#555] leading-relaxed">
                  {guidance.explanation}
                </p>
              </div>
            </div>

            {assessment?.doctorRemarks && (
              <div className="results-section-card">
                <h3 className="text-lg font-bold text-[#1B1B1B] mb-4">
                  Doctor Remarks
                </h3>
                <p className="text-sm text-[#555] leading-relaxed">
                  {assessment.doctorRemarks}
                </p>
              </div>
            )}

            <div className="results-section-card">
              <h3 className="text-lg font-bold text-[#1B1B1B] mb-5">
                Suggestions
              </h3>
              <div className="space-y-3">
                {guidance.suggestions.map((item) => (
                  <div
                    key={item}
                    className="rounded-xl border border-[#E8E8E8] bg-white px-4 py-3"
                  >
                    <p className="text-sm text-[#555]">{item}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ─── ML Voice Analysis Section ─── */}
        {mlDetails && (
          <div className="mb-10">
            <div className="mb-6">
              <p className="text-xs font-semibold tracking-[0.18em] uppercase text-[#52B788] mb-2">
                Voice Analysis
              </p>
              <h2 className="text-2xl font-bold text-[#1B1B1B]">
                ML-Based Insights
              </h2>
              <p className="text-sm text-[#777] mt-1">
                Results from analysing your voice recordings with our depression
                detection model.
              </p>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* ML vs Self-Report Comparison */}
              <div className="results-section-card">
                <h3 className="text-lg font-bold text-[#1B1B1B] mb-4">
                  Score Comparison
                </h3>
                <p className="text-xs text-[#777] mb-4">
                  Self-report vs ML voice analysis
                </p>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: "Self-Report", value: score, fill: "#52B788" },
                        {
                          name: "ML Voice",
                          value:
                            mlDetails.phq8Score ??
                            assessment?.mlScore ??
                            0,
                          fill: "#2D6A4F",
                        },
                      ]}
                      margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#EAF3EE" />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "#777", fontSize: 11 }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 24]}
                        tick={{ fill: "#777", fontSize: 11 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                {mlDetails.confidenceStd > 0 && (
                  <p className="text-xs text-[#777] mt-2 text-center">
                    ML uncertainty: ±{mlDetails.confidenceStd.toFixed(2)} PHQ-8
                    points (CI: {mlDetails.ciLower?.toFixed(1)}–
                    {mlDetails.ciUpper?.toFixed(1)})
                  </p>
                )}
              </div>

              {/* Behavioral Radar Chart */}
              <div className="results-section-card">
                <h3 className="text-lg font-bold text-[#1B1B1B] mb-4">
                  Behavioral Profile
                </h3>
                <p className="text-xs text-[#777] mb-4">
                  Voice features extracted from recordings
                </p>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart
                      outerRadius={70}
                      data={(() => {
                        const b = mlDetails.behavioral || {};
                        const norm = (v, max) => Math.min(100, (v / max) * 100);
                        return [
                          {
                            feature: "Pitch",
                            value: norm(b.f0_mean || 0, 300),
                          },
                          {
                            feature: "Variation",
                            value: norm(b.f0_std || 0, 80),
                          },
                          {
                            feature: "Jitter",
                            value: norm(b.jitter || 0, 0.05),
                          },
                          {
                            feature: "Shimmer",
                            value: norm(b.shimmer || 0, 0.2),
                          },
                          {
                            feature: "Loudness",
                            value: norm(b.loudness_mean || 0, 1),
                          },
                          {
                            feature: "Duration",
                            value: norm(b.total_duration || 0, 120),
                          },
                        ];
                      })()}
                    >
                      <PolarGrid stroke="#E8E8E8" />
                      <PolarAngleAxis
                        dataKey="feature"
                        tick={{ fill: "#777", fontSize: 10 }}
                      />
                      <PolarRadiusAxis
                        domain={[0, 100]}
                        tick={false}
                        axisLine={false}
                      />
                      <Radar
                        dataKey="value"
                        stroke="#2D6A4F"
                        fill="#52B788"
                        fillOpacity={0.35}
                        strokeWidth={2}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Audio Quality */}
              <div className="results-section-card">
                <h3 className="text-lg font-bold text-[#1B1B1B] mb-4">
                  Audio Quality
                </h3>
                <p className="text-xs text-[#777] mb-6">
                  Recording clarity and signal quality
                </p>
                <div className="space-y-5">
                  {[
                    {
                      label: "Overall Quality",
                      value: mlDetails.audioQualityScore,
                      max: 1,
                      color:
                        mlDetails.audioQualityScore > 0.6
                          ? "#52B788"
                          : mlDetails.audioQualityScore > 0.3
                            ? "#FBBF24"
                            : "#EF4444",
                    },
                    {
                      label: "SNR",
                      value: Math.min(mlDetails.audioSnrDb || 0, 40) / 40,
                      max: 1,
                      suffix: `${(mlDetails.audioSnrDb || 0).toFixed(1)} dB`,
                      color: "#2D6A4F",
                    },
                    {
                      label: "Speech Detected",
                      value: mlDetails.audioSpeechProb,
                      max: 1,
                      color: "#52B788",
                    },
                  ].map((m) => (
                    <div key={m.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[#555] font-medium">
                          {m.label}
                        </span>
                        <span className="text-[#777]">
                          {m.suffix || `${((m.value || 0) * 100).toFixed(0)}%`}
                        </span>
                      </div>
                      <div className="h-2 bg-[#E8E8E8] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${(m.value || 0) * 100}%`,
                            backgroundColor: m.color,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {mlDetails.inferenceTimeMs && (
                  <p className="text-xs text-[#B5B5B5] mt-6">
                    Analysis completed in{" "}
                    {(mlDetails.inferenceTimeMs / 1000).toFixed(1)}s
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Voice analysis pending badge */}
        {!mlDetails && assessment?.status === "failed" && (
          <div className="mb-10 results-section-card text-center py-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#FEE2E2] text-[#991B1B] text-sm font-medium">
              Voice analysis could not be completed for this session.
            </div>
          </div>
        )}

        {!mlDetails &&
          assessment?.status !== "failed" &&
          (assessment?.recordingCount || 0) > 0 && (
            <div className="mb-10 results-section-card text-center py-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#FEF3C7] text-[#92400E] text-sm font-medium">
                <svg
                  className="w-4 h-4 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Voice analysis pending — results will appear here when ready
              </div>
            </div>
          )}

        {/* ─── Actions ─── */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link to="/assessment">
            <button className="results-btn-primary">
              Take Another Assessment
            </button>
          </Link>
          <Link to="/doctors">
            <button className="results-btn-primary">Find Doctor</button>
          </Link>
          <Link to="/">
            <button className="results-btn-outline">Return to Home</button>
          </Link>
        </div>
      </div>
    </div>
  );
}
