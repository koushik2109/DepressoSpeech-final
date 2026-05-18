import { useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import DepessionSpeedometer from "../components/DepessionSpeedometer.jsx";
import Loader from "../components/Loader.jsx";
import {
  getAssessmentDetail,
  getAudioBlobUrl,
  getCurrentUser,
  revokeBlobUrl,
} from "../services/api.js";

const severityTone = {
  Severe: "bg-red-50 text-red-700 border-red-200",
  "Moderately Severe": "bg-orange-50 text-orange-700 border-orange-200",
  Moderate: "bg-amber-50 text-amber-700 border-amber-200",
  Mild: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Minimal: "bg-green-50 text-green-700 border-green-200",
};

function formatDate(value) {
  if (!value) return "Not available";
  return new Date(value).toLocaleString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function Metric({ label, value }) {
  if (value == null || value === "") return null;
  return (
    <div className="rounded-xl border border-[#E8E8E8] bg-white px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
        {label}
      </p>
      <p className="mt-1 text-lg font-bold text-[#1B1B1B]">{value}</p>
    </div>
  );
}

function MediaPlayback({ fileId, isVideoHint = false }) {
  const [url, setUrl] = useState("");
  const [mimeType, setMimeType] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!fileId) return undefined;
    let revoked = false;
    let objectUrl = "";
    setLoading(true);

    getAudioBlobUrl(fileId)
      .then(({ url: nextUrl, blob }) => {
        if (revoked) {
          revokeBlobUrl(nextUrl);
          return;
        }
        objectUrl = nextUrl;
        setMimeType(blob.type || "");
        setUrl(nextUrl);
      })
      .catch((err) => setError(err.message || "Media unavailable"))
      .finally(() => setLoading(false));

    return () => {
      revoked = true;
      revokeBlobUrl(objectUrl);
    };
  }, [fileId]);

  if (!fileId) {
    return <p className="text-sm text-[#9AA49F]">No recording saved.</p>;
  }
  if (error) {
    return <p className="text-sm text-[#B45309]">{error}</p>;
  }
  if (loading || !url) {
    return <p className="text-sm text-[#6A766F]">Loading recording...</p>;
  }

  const isVideo = isVideoHint || mimeType.startsWith("video/");

  if (isVideo) {
    return (
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[#075985] flex items-center gap-1">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
          </svg>
          Video + Audio Recording
        </p>
        <video
          controls
          controlsList="nodownload"
          src={url}
          className="w-full rounded-lg"
          style={{ maxHeight: "240px" }}
        />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[#2D6A4F] flex items-center gap-1">
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
        </svg>
        Audio Recording
      </p>
      <audio
        controls
        controlsList="nodownload"
        src={url}
        className="h-11 w-full"
      />
    </div>
  );
}

export default function AssessmentDetail() {
  const { assessmentId } = useParams();
  const user = getCurrentUser();
  const [assessment, setAssessment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadAssessment = async () => {
      setLoading(true);
      try {
        const data = await getAssessmentDetail(assessmentId);
        if (!active) return;
        setAssessment(data);
        setError("");
      } catch (err) {
        if (active) setError(err.message || "Unable to load assessment");
      } finally {
        if (active) setLoading(false);
      }
    };

    loadAssessment();
    return () => {
      active = false;
    };
  }, [assessmentId]);

  const itemData = useMemo(
    () =>
      (assessment?.answers || []).map((answer) => ({
        name: `Q${answer.questionId}`,
        // Prefer real per-Q ML score from individual session analysis (mlScore)
        score: answer.mlScore != null ? Math.round(answer.mlScore) : (answer.score ?? 0),
      })),
    [assessment?.answers],
  );

  const mlModelDetails = assessment?.mlModelDetails;

  if (!user || user.role !== "patient") {
    return <Navigate to="/signin" replace />;
  }

  if (loading) {
    return (
      <div className="pt-28 min-h-screen flex items-center justify-center bg-[#F7F7F2]">
        <Loader size="lg" text="Loading report..." />
      </div>
    );
  }

  if (error || !assessment) {
    return (
      <div className="pt-28 min-h-screen px-4 bg-[#F7F7F2]">
        <div className="mx-auto max-w-3xl rounded-2xl border border-[#F3D5B5] bg-[#FFF8F0] p-8 text-center">
          <p className="text-lg font-semibold text-[#8A4B12]">
            {error || "Assessment not found"}
          </p>
          <Link
            to="/assessment-history"
            className="mt-5 inline-flex rounded-xl bg-[#1B3A2D] px-5 py-2.5 text-sm font-semibold text-white"
          >
            Back to History
          </Link>
        </div>
      </div>
    );
  }

  const mlDetails = assessment.mlDetails;
  const severityClass =
    severityTone[assessment.severity] ||
    "bg-gray-50 text-gray-700 border-gray-200";

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[88rem] space-y-8">
        <section className="rounded-2xl border border-[#D6E3DA] bg-white px-6 py-7 md:px-8">
          <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
            <div>
              <Link
                to="/assessment-history"
                className="text-sm font-semibold text-[#2D6A4F] hover:text-[#1B3A2D]"
              >
                Back to history
              </Link>
              <h1 className="mt-3 text-3xl font-bold text-[#1B1B1B] md:text-4xl">
                Assessment Report
              </h1>
              <p className="mt-2 text-base text-[#5F6B65]">
                Completed {formatDate(assessment.createdAt)}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <span className="rounded-full bg-[#ECF8F3] px-4 py-2 text-sm font-semibold text-[#1F7A66]">
                Score {assessment.score}/24
              </span>
              <span
                className={`rounded-full border px-4 py-2 text-sm font-semibold ${severityClass}`}
              >
                {assessment.severity}
              </span>


            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div className="rounded-2xl border border-[#E8E8E8] bg-white p-6">
            <h2 className="text-xl font-bold text-[#1B1B1B]">Severity Meter</h2>
            <p className="mt-1 text-sm text-[#6A766F]">
              Visual 0-24 PHQ-8 severity scale for this session.
            </p>
            <div className="mt-6">
              <DepessionSpeedometer
                score={assessment.score}
                level={assessment.severity}
                maxScore={24}
              />
            </div>
          </div>

          <div className="rounded-2xl border border-[#E8E8E8] bg-white p-6">
            <h2 className="text-xl font-bold text-[#1B1B1B]">
              Question Scores
            </h2>
            <p className="mt-1 text-sm text-[#6A766F]">
              {itemData.some((d) => d.score > 0)
                ? "Per-question ML estimate (each video independently analysed, scaled 0-3)."
                : "No per-question scores yet — run reeval after assessment."}
            </p>
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={itemData}
                  margin={{ top: 10, right: 12, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#EAF3EE" />
                  <XAxis dataKey="name" tick={{ fill: "#777", fontSize: 12 }} />
                  <YAxis
                    domain={[0, 3]}
                    allowDecimals={false}
                    tick={{ fill: "#777", fontSize: 12 }}
                  />
                  <Tooltip />
                  <Bar dataKey="score" fill="#2D6A4F" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </section>

        {assessment.doctorRemarks && (
          <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6">
            <h2 className="text-xl font-bold text-[#1B1B1B]">Doctor Remarks</h2>
            <p className="mt-3 text-sm leading-relaxed text-[#555]">
              {assessment.doctorRemarks}
            </p>
          </section>
        )}


        <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-xl font-bold text-[#1B1B1B]">Recorded Question Responses</h2>
            {assessment.hasVideoRecordings && (
              <span className="rounded-full bg-[#E8F3FF] px-3 py-1 text-xs font-semibold text-[#075985] flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
                Multimodal Session
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-[#6A766F] mb-6">
            {assessment.hasVideoRecordings
              ? "Video and audio recordings captured for trimodal analysis."
              : "Audio recordings for each PHQ-8 question."}
          </p>
          <div className="space-y-4">
            {(assessment.answers || []).map((answer) => (
              <article
                key={answer.questionId}
                className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-5"
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="max-w-3xl">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#52B788]">
                      Question {answer.questionId}
                    </p>
                    <h3 className="mt-2 text-lg font-semibold leading-snug text-[#1B1B1B]">
                      {answer.questionText}
                    </h3>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className="w-fit rounded-full bg-[#ECF8F3] px-4 py-2 text-sm font-bold text-[#1F7A66] ring-1 ring-[#D6E3DA]">
                        Score {answer.mlScore != null ? Math.round(answer.mlScore) : (answer.score ?? 0)}/3
                      </span>
                  </div>
                </div>
                <div className="mt-4 rounded-xl border border-[#E8E8E8] bg-white p-3">
                  <MediaPlayback fileId={answer.audioFileId} isVideoHint={Boolean(answer.isVideo)} />
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
