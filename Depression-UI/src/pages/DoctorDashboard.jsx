import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "../components/Card.jsx";
import Button from "../components/Button.jsx";
import {
  getCurrentUser,
  getDashboardSnapshot,
  listDoctorAssignments,
  logoutUser,
} from "../services/api.js";

const riskOrder = [
  "Severe",
  "Moderately Severe",
  "Moderate",
  "Mild",
  "Minimal",
];

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

function normalizeSnapshot(data) {
  return {
    assessments: Array.isArray(data?.assessments) ? data.assessments : [],
    users: Array.isArray(data?.users) ? data.users : [],
    alerts: Array.isArray(data?.alerts) ? data.alerts : [],
    patientCount: Number.isFinite(data?.patientCount) ? data.patientCount : 0,
    totals: data?.totals || {
      users: 0,
      doctors: 0,
      patients: 0,
      assessments: 0,
    },
  };
}

export default function DoctorDashboard() {
  const navigate = useNavigate();
  const doctor = getCurrentUser();
  const [snapshot, setSnapshot] = useState({ assessments: [], users: [] });
  const [assignments, setAssignments] = useState([]);

  const refreshDashboard = useCallback(async () => {
    try {
      const data = await getDashboardSnapshot();
      setSnapshot(normalizeSnapshot(data));
    } catch {
      setSnapshot(normalizeSnapshot());
    }
  }, []);

  const refreshAssignments = useCallback(async () => {
    try {
      setAssignments(await listDoctorAssignments());
    } catch {
      setAssignments([]);
    }
  }, []);

  useEffect(() => {
    let active = true;
    Promise.resolve().then(() => {
      if (!active) return;
      refreshDashboard();
      refreshAssignments();
    });
    return () => {
      active = false;
    };
  }, [refreshAssignments, refreshDashboard]);

  useEffect(() => {
    const handleRefresh = () => {
      refreshDashboard();
      refreshAssignments();
    };
    window.addEventListener("mindscope-session-updated", handleRefresh);
    return () => window.removeEventListener("mindscope-session-updated", handleRefresh);
  }, [refreshAssignments, refreshDashboard]);

  const recentAssessments = useMemo(() => {
    return [...(Array.isArray(snapshot.assessments) ? snapshot.assessments : [])]
      .sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
      .slice(0, 12);
  }, [snapshot.assessments]);

  const riskBreakdown = useMemo(() => {
    const assessments = Array.isArray(snapshot.assessments)
      ? snapshot.assessments
      : [];
    return riskOrder.map((level) => ({
      level,
      count: assessments.filter((item) => item.severity === level)
        .length,
    }));
  }, [snapshot.assessments]);

  const safeAssignments = Array.isArray(assignments) ? assignments : [];
  const activePatientCount = new Set(
    safeAssignments
      .filter((item) => ["accepted", "completed"].includes(item.status))
      .map((item) => item.patient?.id || item.patientId)
      .filter(Boolean),
  ).size;
  const patientCount =
    snapshot.patientCount || snapshot.totals?.patients || activePatientCount;
  const pendingCount = safeAssignments.filter(
    (item) => item.status === "pending",
  ).length;
  const pendingAssignments = safeAssignments.filter(
    (item) => item.status === "pending",
  );
  const activeCaseCount = safeAssignments.filter((item) =>
    ["accepted", "completed"].includes(item.status),
  ).length;

  const severeAlerts = useMemo(() => {
    const assessments = Array.isArray(snapshot.assessments)
      ? snapshot.assessments
      : [];
    return assessments
      .filter(
        (item) =>
          item.severity === "Severe" || item.severity === "Moderately Severe",
      )
      .sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
      .slice(0, 10);
  }, [snapshot.assessments]);

  const [selectedPatientKey, setSelectedPatientKey] = useState("");

  const patientCards = useMemo(() => {
    const groups = new Map();
    const safeAssignments = Array.isArray(assignments) ? assignments : [];
    const safeAssessments = Array.isArray(snapshot.assessments)
      ? snapshot.assessments
      : [];

    for (const assignment of safeAssignments) {
      const patient = assignment.patient || {};
      const patientId = patient.id || assignment.patientId;
      if (!patientId) continue;
      if (!groups.has(patientId)) {
        groups.set(patientId, {
          key: patientId,
          patientId,
          label: patient.name || patient.email || "Patient",
          email: patient.email || "",
          status: assignment.status,
          points: [],
        });
      }
      const group = groups.get(patientId);
      group.label = patient.name || group.label;
      group.email = patient.email || group.email;
      group.status = assignment.status;
    }

    for (const item of safeAssessments) {
      const patientId = item.userId;
      if (!patientId) continue;
      if (!groups.has(patientId)) {
        groups.set(patientId, {
          key: patientId,
          patientId,
          label: item.userName || item.email || "Patient",
          email: item.email || "",
          status: "",
          points: [],
        });
      }
      groups.get(patientId).points.push(item);
    }

    return [...groups.values()]
      .map((group) => {
        const ordered = [...group.points].sort(
          (a, b) =>
            new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
        );
        const latest = ordered[ordered.length - 1] || null;
        const improvement =
          ordered.length > 1 && latest
            ? (ordered[0]?.score || 0) - (latest.score || 0)
            : 0;
        return {
          ...group,
          points: ordered.map((item, index) => ({
            session: `S${index + 1}`,
            score: item.score,
            severity: item.severity,
            createdAt: item.createdAt,
            date: formatIST(item.createdAt),
          })),
          latest,
          improvement,
        };
      })
      .sort(
        (a, b) =>
          b.points.length - a.points.length ||
          String(a.label || "").localeCompare(String(b.label || "")),
      );
  }, [assignments, snapshot.assessments]);

  const activePatientTrend = useMemo(() => {
    if (!patientCards.length) return null;
    const explicit = patientCards.find((item) => item.key === selectedPatientKey);
    return explicit || patientCards[0];
  }, [patientCards, selectedPatientKey]);

  useEffect(() => {
    if (!patientCards.length) return;
    if (
      !selectedPatientKey ||
      !patientCards.some((item) => item.key === selectedPatientKey)
    ) {
      let active = true;
      const nextKey = patientCards[0].key;
      Promise.resolve().then(() => {
        if (active) setSelectedPatientKey(nextKey);
      });
      return () => {
        active = false;
      };
    }
    return undefined;
  }, [patientCards, selectedPatientKey]);

  if (!doctor || doctor.role !== "doctor") {
    return <Navigate to="/signin" replace />;
  }

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-10 bg-[#F7F7F2]">
      <div className="w-full max-w-[88rem] mx-auto space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Doctor Workspace
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Patient Monitoring Dashboard
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                Welcome back, Dr. {doctor.name}. Monitor latest assessments and
                identify high-risk patients quickly.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link to="/doctor/queue">
                <Button variant="secondary">Open Queue</Button>
              </Link>
              <Link to="/">
                <Button variant="outline">Home</Button>
              </Link>
              <Button
                onClick={() => {
                  logoutUser();
                  navigate("/signin");
                }}
              >
                Sign Out
              </Button>
            </div>
          </div>
        </section>

        <Card className="p-6 md:p-7">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Profile Settings
              </p>
              <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                Manage Your Profile
              </h2>
              <p className="mt-1 text-sm text-[#6A766F]">
                Update profile details from the dedicated profile page.
              </p>
            </div>
            <Link
              to="/profile"
              className="inline-flex items-center justify-center rounded-xl border-2 border-[#2D6A4F]/30 px-7 py-3.5 text-base font-semibold text-[#2D6A4F] hover:border-[#2D6A4F] hover:bg-[#D8F3DC]/50 transition-all"
            >
              Open Profile
            </Link>
          </div>
        </Card>

        <section className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <p className="text-sm text-[#6A766F]">Total Patients</p>
              <p className="mt-2 text-3xl font-bold text-[#1B1B1B]">{patientCount}</p>
            </Card>
          <Card>
            <p className="text-sm text-[#6A766F]">Total Assessments</p>
            <p className="mt-2 text-3xl font-bold text-[#1B1B1B]">
              {snapshot.assessments.length}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-[#6A766F]">High Risk Cases</p>
            <p className="mt-2 text-3xl font-bold text-[#B42318]">
              {
                snapshot.assessments.filter(
                  (item) =>
                    item.severity === "Severe" ||
                    item.severity === "Moderately Severe",
                ).length
              }
            </p>
          </Card>
          <Card>
            <p className="text-sm text-[#6A766F]">Low Risk Cases</p>
            <p className="mt-2 text-3xl font-bold text-[#027A48]">
              {
                snapshot.assessments.filter(
                  (item) =>
                    item.severity === "Minimal" || item.severity === "Mild",
                ).length
              }
            </p>
          </Card>
          <Card>
            <p className="text-sm text-[#6A766F]">Pending Requests</p>
            <p className="mt-2 text-3xl font-bold text-[#B54708]">
              {pendingCount}
            </p>
          </Card>
          <Card>
            <p className="text-sm text-[#6A766F]">Managed Cases</p>
            <p className="mt-2 text-3xl font-bold text-[#027A48]">
              {activeCaseCount}
            </p>
          </Card>
        </section>

        <Card className="p-6 md:p-7">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Patient Queue
              </p>
              <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                Pending Patients
              </h2>
              <p className="mt-1 text-sm text-[#6A766F]">
                Review only pending assignments from the queue page.
              </p>
            </div>
          </div>

          <div className="mt-5 space-y-3">
            {pendingAssignments.length === 0 ? (
              <p className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-5 text-sm text-[#6A766F]">
                No pending patients right now.
              </p>
            ) : (
              pendingAssignments
                .slice(0, 3)
                .map((assignment) => (
                <div
                  key={assignment.id}
                  className="rounded-xl border border-[#E8E8E8] bg-white px-4 py-4"
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="text-lg font-bold text-[#1B1B1B]">
                        {assignment.patient?.name || "Patient"}
                      </p>
                      <p className="text-sm text-[#6A766F]">
                        {assignment.patient?.email || "Email unavailable"}
                      </p>
                      <p className="text-sm text-[#6A766F]">
                        Status:{" "}
                        <span className="font-bold capitalize">
                          {assignment.status}
                        </span>
                      </p>
                      {assignment.assessment && (
                        <p className="text-xs text-[#9AA49F]">
                          Score {assignment.assessment.score}/24 ·{" "}
                          {assignment.assessment.severity}
                        </p>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {assignment.assessmentId &&
                        (assignment.assessment?.status === "completed" ||
                          assignment.assessment?.reportStatus === "available") && (
                          <Link to={`/doctor/reports/${assignment.assessmentId}`}>
                            <Button size="sm" variant="ghost">
                              Open Report
                            </Button>
                          </Link>
                        )}
                      {assignment.status === "pending" && (
                        <Link to="/doctor/queue">
                          <Button size="sm">Review Queue</Button>
                        </Link>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        <section className="space-y-4">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Patient Tracking
              </p>
              <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                Patient Cards
              </h2>
              <p className="mt-1 text-sm text-[#6A766F]">
                Open any patient to review the full history and progress.
              </p>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {patientCards.map((patient) => (
              <Link
                key={patient.key}
                to={`/doctor/patient/${patient.patientId || patient.key}`}
                className="rounded-2xl border border-[#E8E8E8] bg-white p-5 transition-all hover:border-[#B7E4C7] hover:bg-[#F7FBF8]"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-lg font-bold text-[#1B1B1B]">
                      {patient.label}
                    </p>
                    <p className="text-sm text-[#6A766F]">
                      {patient.points.length} sessions
                    </p>
                  </div>
                  <span className="rounded-full bg-[#ECF8F3] px-3 py-1 text-xs font-semibold text-[#1F7A66]">
                    Open
                  </span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-xl bg-[#FAFAF7] px-3 py-2">
                    <p className="text-xs text-[#6A766F]">Latest Score</p>
                    <p className="font-bold text-[#1B1B1B]">
                      {patient.latest?.score ?? "—"} / 24
                    </p>
                  </div>
                  <div className="rounded-xl bg-[#FAFAF7] px-3 py-2">
                    <p className="text-xs text-[#6A766F]">Improvement</p>
                    <p className="font-bold text-[#1B1B1B]">
                      {patient.improvement > 0
                        ? `+${patient.improvement}`
                        : patient.improvement < 0
                          ? `${patient.improvement}`
                          : "0"}
                    </p>
                  </div>
                </div>
                <p className="mt-4 text-xs text-[#6A766F]">
                  {patient.latest?.severity || "—"}
                </p>
              </Link>
            ))}
          </div>
        </section>

        <section className="grid xl:grid-cols-[1.1fr_1.9fr] gap-6">
          <Card className="p-6 md:p-7">
            <h2 className="text-xl font-semibold text-[#1B1B1B] mb-4">
              Risk Alerts
            </h2>
            <p className="text-sm text-[#6A766F] mb-4">
              Latest severe and moderately severe patient submissions.
            </p>
            <div className="space-y-3 max-h-[28rem] overflow-auto pr-1">
              {severeAlerts.length === 0 ? (
                <p className="text-sm text-[#6A766F]">
                  No high-risk alerts right now.
                </p>
              ) : (
                severeAlerts.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-xl border border-red-100 bg-red-50/40 px-4 py-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="font-semibold text-[#1B1B1B]">
                          {item.userName || item.email || "Unknown Patient"}
                        </p>
                        <p className="text-xs text-[#6A766F]">
                          {new Date(item.createdAt).toLocaleString()}
                        </p>
                      </div>
                      <span className="inline-flex items-center rounded-full bg-red-100 px-3 py-1 text-xs font-semibold text-red-700">
                        {item.severity} ({item.score}/24)
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>

          <Card className="p-6 md:p-7">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
              <div>
                <h2 className="text-xl font-semibold text-[#1B1B1B]">
                  Trend Per Patient
                </h2>
                <p className="text-sm text-[#6A766F]">
                  Track score movement over patient sessions.
                </p>
              </div>
              <select
                value={activePatientTrend?.key || ""}
                onChange={(event) => setSelectedPatientKey(event.target.value)}
                className="rounded-xl border border-[#D6E3DA] bg-white px-3 py-2 text-sm text-[#1B1B1B] outline-none focus:ring-2 focus:ring-[#52B788]/35"
                disabled={!patientCards.length}
              >
                {patientCards.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </div>

            {!activePatientTrend ? (
              <p className="text-sm text-[#6A766F]">
                No patient trend data available yet.
              </p>
            ) : (
              <>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={activePatientTrend.points}
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
                          if (!active || !payload || !payload.length)
                            return null;
                          const point = payload[0].payload;
                          return (
                            <div className="rounded-xl border border-[#E8E8E8] bg-white px-3 py-2 shadow-md">
                              <p className="text-xs text-[#777]">
                                {point.date}
                              </p>
                              <p className="text-sm font-semibold text-[#1B1B1B]">
                                Score {point.score}/24
                              </p>
                              <p className="text-xs text-[#2D6A4F]">
                                {point.severity}
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
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {(activePatientTrend.points || []).length < 2 && (
                  <p className="text-xs text-[#6A766F] mt-4">
                    This patient has one session. More assessments are needed
                    for trend confidence.
                  </p>
                )}
              </>
            )}
          </Card>
        </section>

        <section className="grid xl:grid-cols-[1fr_2fr] gap-6">
          <Card className="p-6 md:p-7">
            <h2 className="text-xl font-semibold text-[#1B1B1B] mb-4">
              Severity Distribution
            </h2>
            <div className="space-y-3">
              {riskBreakdown.map((item) => (
                <div
                  key={item.level}
                  className="flex items-center justify-between rounded-xl border border-[#E8E8E8] bg-white px-4 py-3"
                >
                  <span className="text-sm text-[#4C5852]">{item.level}</span>
                  <span className="text-base font-semibold text-[#1B1B1B]">
                    {item.count}
                  </span>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6 md:p-7">
            <h2 className="text-xl font-semibold text-[#1B1B1B] mb-4">
              Recent Patient Submissions
            </h2>
            <div className="space-y-3 max-h-[34rem] overflow-auto pr-1">
              {recentAssessments.length === 0 ? (
                <p className="text-sm text-[#6A766F]">
                  No patient assessments submitted yet.
                </p>
              ) : (
                recentAssessments.map((item) => (
                  <div
                    key={item.id}
                    className="rounded-xl border border-[#E8E8E8] bg-white px-4 py-4"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                      <div>
                        <p className="font-semibold text-[#1B1B1B]">
                          {item.userName || item.email || "Unknown Patient"}
                        </p>
                        <p className="text-sm text-[#6A766F]">
                          {new Date(item.createdAt).toLocaleString()}
                        </p>
                      </div>
                      <span
                        className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${severityTone[item.severity] || "bg-gray-50 text-gray-700 border-gray-200"}`}
                      >
                        {item.severity} ({item.score}/24)
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>
        </section>
      </div>
    </div>
  );
}
