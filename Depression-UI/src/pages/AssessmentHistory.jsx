import { useState, useEffect } from "react";
import { Link, Navigate } from "react-router-dom";
import { getCurrentUser, listAssessments } from "../services/api.js";

const severityTone = {
  Severe: "bg-red-50 text-red-700 border-red-200",
  "Moderately Severe": "bg-orange-50 text-orange-700 border-orange-200",
  Moderate: "bg-amber-50 text-amber-700 border-amber-200",
  Mild: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Minimal: "bg-green-50 text-green-700 border-green-200",
};

export default function AssessmentHistory() {
  const user = getCurrentUser();
  const [assessments, setAssessments] = useState([]);

  useEffect(() => {
    listAssessments()
      .then((all) => {
        const filtered = all
          .filter((item) => {
            const sameUser =
              item.userId && user?.id ? item.userId === user.id : false;
            const sameEmail =
              item.email && user?.email
                ? item.email.toLowerCase() === user.email.toLowerCase()
                : false;
            return sameUser || sameEmail;
          })
          .sort(
            (a, b) =>
              new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
          );
        setAssessments(filtered);
      })
      .catch(() => setAssessments([]));
  }, [user?.email, user?.id]);

  if (!user || user.role !== "patient") {
    return <Navigate to="/signin" replace />;
  }

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-10 bg-[#F7F7F2]">
      <div className="w-full max-w-[88rem] mx-auto space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Assessment History
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Your Past Assessments
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                Track how your PHQ-8 scores changed over time. Tap any session
                for detailed ML metrics.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/assessment"
                className="inline-flex items-center rounded-xl bg-[#1B3A2D] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#2D6A4F] transition-colors"
              >
                Start Assessment
              </Link>
              <Link
                to="/"
                className="inline-flex items-center rounded-xl border border-[#D6E3DA] bg-white px-5 py-2.5 text-sm font-semibold text-[#1B1B1B] hover:bg-[#F4FAF6] transition-colors"
              >
                Back Home
              </Link>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          {assessments.length === 0 ? (
            <div className="text-center py-14">
              <p className="text-xl font-semibold text-[#1B1B1B] mb-2">
                No assessments yet
              </p>
              <p className="text-sm text-[#6A766F] mb-6">
                Complete your first PHQ-8 assessment to start building history.
              </p>
              <Link
                to="/assessment"
                className="inline-flex items-center rounded-xl bg-[#1B3A2D] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#2D6A4F] transition-colors"
              >
                Start First Assessment
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {assessments.map((item, index) => {
                const isCompleted =
                  item.status === "completed" ||
                  item.reportStatus === "available" ||
                  item.isReportReady;
                const hasMlScore = item.mlScore != null && !Number.isNaN(Number(item.mlScore));
                const mlScoreRounded = hasMlScore ? Math.round(Number(item.mlScore) * 10) / 10 : null;
                const hasVideo = Boolean(item.hasVideoRecordings);
                const hasRemarks = Boolean(item.doctorRemarks);
                return (
                  <article
                    key={item.id}
                    className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-5 py-5 transition-colors hover:border-[#B7E4C7] hover:bg-[#F3FBF7]"
                  >
                    {/* Row 1 — session label + action */}
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <p className="text-base font-semibold text-[#1B1B1B]">
                            Session #{assessments.length - index}
                          </p>
                          {/* Status badge */}
                          {!isCompleted && item.status === "failed" && (
                            <span className="rounded-full bg-[#FEE2E2] px-2.5 py-0.5 text-xs font-semibold text-[#991B1B]">Failed</span>
                          )}
                          {!isCompleted && item.status !== "failed" && (
                            <span className="rounded-full bg-[#FEF3C7] px-2.5 py-0.5 text-xs font-semibold text-[#92400E]">Processing</span>
                          )}
                          {isCompleted && (
                            <span className="rounded-full bg-[#D8F3DC] px-2.5 py-0.5 text-xs font-semibold text-[#2D6A4F]">Completed</span>
                          )}
                        </div>
                        <p className="mt-0.5 text-xs text-[#9AA49F]">
                          {new Date(item.createdAt).toLocaleString()}
                        </p>
                      </div>
                      {isCompleted ? (
                        <Link
                          to={`/assessment-history/${item.id}`}
                          className="self-start rounded-lg bg-[#1B3A2D] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#2D6A4F] whitespace-nowrap"
                        >
                          Open Report
                        </Link>
                      ) : (
                        <span className="self-start rounded-lg bg-[#E8E8E8] px-4 py-2 text-sm font-semibold text-[#9AA49F] whitespace-nowrap">
                          {item.status === "failed" ? "Unavailable" : "Processing…"}
                        </span>
                      )}
                    </div>

                    {/* Row 2 — score metrics grid */}
                    <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div className="rounded-lg border border-[#E8E8E8] bg-white px-3 py-2.5">
                        <p className="text-xs uppercase tracking-[0.12em] text-[#6A766F]">PHQ Score</p>
                        <p className="mt-1 text-lg font-bold text-[#1B1B1B]">
                          {item.score != null ? `${item.score}/24` : "—"}
                        </p>
                      </div>
                      <div className="rounded-lg border border-[#E8E8E8] bg-white px-3 py-2.5">
                        <p className="text-xs uppercase tracking-[0.12em] text-[#6A766F]">Severity</p>
                        <p className={`mt-1 text-sm font-bold truncate ${
                          item.severity === "Severe" ? "text-red-600"
                          : item.severity === "Moderately Severe" ? "text-orange-600"
                          : item.severity === "Moderate" ? "text-amber-600"
                          : item.severity === "Mild" ? "text-emerald-600"
                          : "text-green-600"
                        }`}>
                          {item.severity || "—"}
                        </p>
                      </div>
                      <div className="rounded-lg border border-[#E8E8E8] bg-white px-3 py-2.5">
                        <p className="text-xs uppercase tracking-[0.12em] text-[#6A766F]">ML Score</p>
                        <p className="mt-1 text-lg font-bold text-[#2D6A4F]">
                          {hasMlScore ? `${mlScoreRounded}/24` : "—"}
                        </p>
                      </div>
                      <div className="rounded-lg border border-[#E8E8E8] bg-white px-3 py-2.5">
                        <p className="text-xs uppercase tracking-[0.12em] text-[#6A766F]">Responses</p>
                        <p className="mt-1 text-lg font-bold text-[#1B1B1B]">
                          {item.recordingCount || 0}/8
                        </p>
                      </div>
                    </div>

                    {/* Row 3 — feature badges */}
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {hasVideo ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-[#E8F3FF] px-3 py-1 text-xs font-semibold text-[#075985]">
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                          </svg>
                          Video + Audio
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded-full bg-[#F0FDF4] px-3 py-1 text-xs font-semibold text-[#166534]">
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
                          </svg>
                          Audio Only
                        </span>
                      )}
                      {item.mlSeverity && (
                        <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${severityTone[item.mlSeverity] || "bg-gray-50 text-gray-700 border-gray-200"}`}>
                          ML: {item.mlSeverity}
                        </span>
                      )}
                      {hasRemarks && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-[#F5F3FF] px-3 py-1 text-xs font-semibold text-[#6D28D9]">
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                          </svg>
                          Doctor Remarks
                        </span>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
