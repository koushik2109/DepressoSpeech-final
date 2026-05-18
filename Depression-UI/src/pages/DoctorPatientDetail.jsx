import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Loader from "../components/Loader.jsx";
import {
  getCurrentUser,
  getDoctorPatientReports,
  getDoctorPatientTrends,
  updateDoctorAssignment,
} from "../services/api.js";

const severityTone = {
  Severe: "bg-red-50 text-red-700 border-red-200",
  "Moderately Severe": "bg-orange-50 text-orange-700 border-orange-200",
  Moderate: "bg-amber-50 text-amber-700 border-amber-200",
  Mild: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Minimal: "bg-green-50 text-green-700 border-green-200",
};

function formatIST(value) {
  if (!value) return "Not available";
  return new Date(value).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

function Metric({ label, value, hint }) {
  return (
    <div className="rounded-2xl border border-[#E8E8E8] bg-white p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
        {label}
      </p>
      <p className="mt-2 text-3xl font-bold text-[#1B1B1B]">{value}</p>
      {hint && <p className="mt-1 text-sm text-[#6A766F]">{hint}</p>}
    </div>
  );
}

export default function DoctorPatientDetail() {
  const { patientId } = useParams();
  const user = getCurrentUser();
  const [data, setData] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updating, setUpdating] = useState("");
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);

  const loadPatient = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [reports, trends] = await Promise.all([
        getDoctorPatientReports(patientId),
        getDoctorPatientTrends(patientId),
      ]);
      const trendPoints = trends?.patients?.[0]?.points || [];
      setData(reports);
      setTrendData(trendPoints);
    } catch (err) {
      setError(err.message || "Unable to load patient details.");
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    loadPatient();
  }, [loadPatient]);

  const latestItem = data?.items?.[0] || null;
  const assignment = data?.assignment || latestItem?.assignment || null;
  const patient = data?.patient || latestItem?.assessment?.patient || null;
  const metrics = data?.metrics || {};
  const history = useMemo(
    () =>
      [...(data?.items || [])].sort(
        (a, b) =>
          new Date(a.assessment?.createdAt || a.createdAt || 0).getTime() -
          new Date(b.assessment?.createdAt || b.createdAt || 0).getTime(),
      ),
    [data?.items],
  );

  const chartData = useMemo(() => {
    const source = trendData.length
      ? trendData
      : history.map((item, index) => ({
          sessionId: item.assessment?.id || item.id,
          session: `S${index + 1}`,
          timestamp: item.assessment?.createdAt,
          score: item.assessment?.score ?? 0,
          severity: item.assessment?.severity,
        }));

    return [...source]
      .sort(
        (a, b) =>
          new Date(a.timestamp || a.date || a.createdAt || 0).getTime() -
          new Date(b.timestamp || b.date || b.createdAt || 0).getTime(),
      )
      .map((item, index) => ({
        sessionId: item.sessionId || item.id || item.assessmentId || `S${index + 1}`,
        session: item.session || `S${index + 1}`,
        timestamp: item.timestamp || item.createdAt || item.date,
        score: item.score ?? 0,
        severity: item.severity,
        displayDate: formatIST(item.timestamp || item.createdAt || item.date),
      }));
  }, [history, trendData]);

  const latestAssignment = assignment;

  if (!user || user.role !== "doctor") {
    return <Navigate to="/login" replace />;
  }

  const handleAction = async (action) => {
    if (!latestAssignment?.id || updating) return;
    setUpdating(action);
    setMessage("");
    setIsError(false);
    try {
      await updateDoctorAssignment(latestAssignment.id, action);
      await loadPatient();
      setMessage(
        action === "accept"
          ? "Case accepted."
          : action === "reject"
            ? "Case rejected."
            : "Patient reassigned.",
      );
    } catch (err) {
      setIsError(true);
      setMessage(err.message || "Unable to update assignment.");
    } finally {
      setUpdating("");
    }
  };

  if (loading) {
    return (
      <div className="pt-28 min-h-screen flex items-center justify-center bg-[#F7F7F2]">
        <Loader size="lg" text="Loading patient history..." />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="pt-28 min-h-screen px-4 bg-[#F7F7F2]">
        <div className="mx-auto max-w-3xl rounded-2xl border border-[#F3D5B5] bg-[#FFF8F0] p-8 text-center">
          <p className="text-lg font-semibold text-[#8A4B12]">
            {error || "Patient history unavailable"}
          </p>
          <Link to="/doctor/dashboard">
            <Button className="mt-5">Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    );
  }

  const improvement =
    metrics.improvement != null
      ? metrics.improvement
      : history.length > 1
        ? (history[0]?.assessment?.score || 0) -
          (history[history.length - 1]?.assessment?.score || 0)
        : 0;

  const improvementLabel =
    improvement > 0
      ? `Improved by ${improvement}`
      : improvement < 0
        ? `Worsened by ${Math.abs(improvement)}`
        : "No change";

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[88rem] space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <Link
                to="/doctor/dashboard"
                className="text-sm font-bold text-[#2D6A4F]"
              >
                Back to dashboard
              </Link>
              <h1 className="mt-3 text-3xl font-bold text-[#1B1B1B] md:text-4xl">
                Patient History
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                {patient?.name || "Patient"} ·{" "}
                {patient?.email || "Email unavailable"}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full bg-[#ECF8F3] px-4 py-2 text-sm font-semibold text-[#1F7A66]">
                {metrics.totalSessions || history.length} sessions
              </span>
              <span className="rounded-full bg-[#EEF4FF] px-4 py-2 text-sm font-semibold text-[#3B5BDB]">
                {improvementLabel}
              </span>
            </div>
          </div>
        </section>

        {latestAssignment && (
          <Card className="p-6 md:p-7">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                  Current Assignment
                </p>
                <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                  Status:{" "}
                  <span className="capitalize">{latestAssignment.status}</span>
                </h2>
                <p className="mt-1 text-sm text-[#6A766F]">
                  Control the current patient routing from this page.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                {latestAssignment.status === "pending" && (
                  <>
                    <Button
                      size="sm"
                      onClick={() => handleAction("accept")}
                      disabled={!!updating}
                    >
                      Accept
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleAction("reject")}
                      disabled={!!updating}
                    >
                      Reject
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => handleAction("reassign")}
                      disabled={!!updating}
                    >
                      Reassign
                    </Button>
                  </>
                )}
              </div>
            </div>
            {message && (
              <p
                className={`mt-4 text-sm font-semibold ${isError ? "text-red-600" : "text-[#2D6A4F]"}`}
              >
                {message}
              </p>
            )}
          </Card>
        )}

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Metric
            label="Total Sessions"
            value={metrics.totalSessions || history.length || 0}
            hint="Assigned reports"
          />
          <Metric
            label="Latest Score"
            value={
              metrics.latestScore != null ? `${metrics.latestScore}/24` : "—"
            }
            hint={metrics.latestSeverity || "No severity data"}
          />
          <Metric
            label="Improvement"
            value={improvement > 0 ? `+${improvement}` : `${improvement}`}
            hint="Lower PHQ-8 is better"
          />
          <Metric
            label="Average Score"
            value={
              metrics.averageScore != null
                ? `${Number(metrics.averageScore).toFixed(1)}/24`
                : "—"
            }
            hint="Over the tracked history"
          />
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <Card className="p-6 md:p-7">
            <h2 className="text-xl font-bold text-[#1B1B1B]">
              Trend Over Time
            </h2>
            <p className="mt-1 text-sm text-[#6A766F]">
              Score progression across the patient history.
            </p>
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 16, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#EAF3EE" />
                  <XAxis
                    dataKey="displayDate"
                    tick={{ fill: "#777", fontSize: 12 }}
                  />
                  <YAxis
                    domain={[0, 24]}
                    tick={{ fill: "#777", fontSize: 12 }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null;
                      const point = payload[0].payload;
                      return (
                        <div className="rounded-xl border border-[#E8E8E8] bg-white px-3 py-2 shadow-md">
                          <p className="text-xs text-[#777]">{point.session}</p>
                          <p className="text-sm font-semibold text-[#1B1B1B]">
                            Score {point.score}/24
                          </p>
                          <p className="text-xs text-[#2D6A4F]">
                            {formatIST(point.timestamp)}
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
                    dot={{ r: 4, fill: "#52B788", strokeWidth: 0 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card className="p-6 md:p-7">
            <h2 className="text-xl font-bold text-[#1B1B1B]">
              Tracking Summary
            </h2>
            <div className="mt-5 space-y-3">
              <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                  Best Score
                </p>
                <p className="mt-1 text-lg font-bold text-[#1B1B1B]">
                  {metrics.bestScore != null ? `${metrics.bestScore}/24` : "—"}
                </p>
              </div>
              <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                  Worst Score
                </p>
                <p className="mt-1 text-lg font-bold text-[#1B1B1B]">
                  {metrics.worstScore != null
                    ? `${metrics.worstScore}/24`
                    : "—"}
                </p>
              </div>
              <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                  Latest Status
                </p>
                <p className="mt-1 text-lg font-bold text-[#1B1B1B] capitalize">
                  {metrics.latestStatus || "Unknown"}
                </p>
              </div>
            </div>
          </Card>
        </section>

        <section className="rounded-3xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Full History
              </p>
              <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                Patient Reports
              </h2>
            </div>
            {patient?.name && (
              <span className="text-sm text-[#6A766F]">{patient.name}</span>
            )}
          </div>

          <div className="mt-6 space-y-4">
            {history.length === 0 ? (
              <p className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-5 text-sm text-[#6A766F]">
                No reports found for this patient.
              </p>
            ) : (
              history.map((item) => {
                const assessment = item.assessment || {};
                const statusReady =
                  assessment.status === "completed" ||
                  assessment.reportStatus === "available" ||
                  assessment.isReportReady;
                return (
                  <article
                    key={item.id}
                    className="rounded-2xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-4"
                  >
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                      <div>
                        <p className="text-sm text-[#6A766F]">
                          {formatIST(assessment.createdAt)}
                        </p>
                        <p className="mt-1 text-lg font-bold text-[#1B1B1B]">
                          Score {assessment.score}/24
                        </p>
                        <p className="text-sm text-[#6A766F]">
                          {assessment.severity}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                            severityTone[assessment.severity] ||
                            "bg-gray-50 text-gray-700 border-gray-200"
                          }`}
                        >
                          {assessment.severity}
                        </span>
                        <span className="rounded-full bg-[#ECF8F3] px-3 py-1 text-xs font-semibold text-[#1F7A66]">
                          {statusReady ? "Completed" : "Preparing"}
                        </span>
                        {statusReady && (
                          <Link to={`/doctor/reports/${assessment.id}`}>
                            <Button size="sm" variant="ghost">
                              Open Report
                            </Button>
                          </Link>
                        )}
                      </div>
                    </div>
                    {assessment.doctorRemarks && (
                      <p className="mt-4 rounded-xl border border-[#E8E8E8] bg-white px-4 py-3 text-sm text-[#555]">
                        {assessment.doctorRemarks}
                      </p>
                    )}
                  </article>
                );
              })
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
