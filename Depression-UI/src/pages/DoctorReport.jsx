import { useCallback, useEffect, useMemo, useState } from "react";
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
import Button from "../components/Button.jsx";
import DepessionSpeedometer from "../components/DepessionSpeedometer.jsx";
import Loader from "../components/Loader.jsx";
import {
  getAudioBlobUrl,
  getCurrentUser,
  getDoctorReport,
  revokeBlobUrl,
  updateDoctorReportRemarks,
  updateDoctorAssignment,
} from "../services/api.js";

function AudioPlayback({ fileId }) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!fileId) return undefined;
    let revoked = false;
    let objectUrl = "";

    getAudioBlobUrl(fileId)
      .then(({ url: nextUrl }) => {
        if (revoked) {
          revokeBlobUrl(nextUrl);
          return;
        }
        objectUrl = nextUrl;
        setUrl(nextUrl);
      })
      .catch((err) => setError(err.message || "Audio unavailable"));

    return () => {
      revoked = true;
      revokeBlobUrl(objectUrl);
    };
  }, [fileId]);

  if (!fileId) return <p className="text-sm text-[#9AA49F]">No audio saved.</p>;
  if (error) return <p className="text-sm text-[#B45309]">{error}</p>;
  if (!url) return <p className="text-sm text-[#6A766F]">Loading audio...</p>;
  return (
    <audio
      controls
      controlsList="nodownload"
      src={url}
      className="h-11 w-full"
    />
  );
}

export default function DoctorReport() {
  const { assessmentId } = useParams();
  const user = getCurrentUser();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [updating, setUpdating] = useState("");
  const [remarksDraft, setRemarksDraft] = useState("");
  const [savingRemarks, setSavingRemarks] = useState(false);

  const loadReport = useCallback(() => {
    setLoading(true);
    getDoctorReport(assessmentId)
      .then((data) => {
        setReport(data);
        setRemarksDraft(data?.assessment?.doctorRemarks || "");
        setError("");
      })
      .catch((err) => setError(err.message || "Unable to load report."))
      .finally(() => setLoading(false));
  }, [assessmentId]);

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  const chartData = useMemo(
    () =>
      (report?.assessment?.answers || []).map((answer) => ({
        name: `Q${answer.questionId}`,
        score: answer.score ?? 0,
      })),
    [report?.assessment?.answers],
  );

  if (!user || user.role !== "doctor") {
    return <Navigate to="/login" replace />;
  }

  const handleAction = async (action) => {
    setUpdating(action);
    setActionMessage("");
    try {
      const result = await updateDoctorAssignment(report.assignment.id, action);
      setActionMessage(
        action === "reassign" && result.reassigned
          ? "Patient reassigned to the next available doctor."
          : action === "reassign"
            ? "No alternate available doctor was found."
            : action === "accept"
              ? "Case accepted."
              : action === "complete"
                ? "Case completed."
                : "Case rejected.",
      );
      loadReport();
    } catch (err) {
      setActionMessage(err.message || "Unable to update case.");
    } finally {
      setUpdating("");
    }
  };

  const handleRemarksSave = async (event) => {
    event.preventDefault();
    setSavingRemarks(true);
    setActionMessage("");
    try {
      const result = await updateDoctorReportRemarks(
        assessmentId,
        remarksDraft,
      );
      setReport((current) =>
        current
          ? {
              ...current,
              assessment: {
                ...current.assessment,
                doctorRemarks: result.doctorRemarks || "",
              },
            }
          : current,
      );
      setActionMessage("Remarks saved.");
    } catch (err) {
      setActionMessage(err.message || "Unable to save remarks.");
    } finally {
      setSavingRemarks(false);
    }
  };

  if (loading) {
    return (
      <div className="pt-28 min-h-screen flex items-center justify-center bg-[#F7F7F2]">
        <Loader size="lg" text="Loading report..." />
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="pt-28 min-h-screen px-4 bg-[#F7F7F2]">
        <div className="mx-auto max-w-3xl rounded-2xl border border-[#F3D5B5] bg-[#FFF8F0] p-8 text-center">
          <p className="text-lg font-semibold text-[#8A4B12]">
            {error || "Report unavailable"}
          </p>
          <Link to="/doctor/dashboard">
            <Button className="mt-5">Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    );
  }

  const assessment = report.assessment;
  const assignment = report.assignment;
  const patient = assessment.patient || assignment.patient || {};

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[88rem] space-y-8">
        <section className="rounded-2xl border border-[#D6E3DA] bg-white px-6 py-7 md:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <Link
                to="/doctor/dashboard"
                className="text-sm font-bold text-[#2D6A4F]"
              >
                Back to dashboard
              </Link>
              <h1 className="mt-3 text-3xl font-bold text-[#1B1B1B] md:text-4xl">
                Patient Report
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                {patient.name} · {patient.email} · Status:{" "}
                <span className="font-bold capitalize">
                  {assignment.status}
                </span>
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {assignment.status === "pending" && (
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
                </>
              )}
              {assignment.status !== "rejected" &&
                assignment.status !== "completed" && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleAction("reassign")}
                    disabled={!!updating}
                  >
                    Reassign
                  </Button>
                )}
              {assignment.status === "accepted" && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleAction("complete")}
                  disabled={!!updating}
                >
                  Complete
                </Button>
              )}
            </div>
          </div>
          {actionMessage && (
            <p className="mt-4 text-sm font-semibold text-[#2D6A4F]">
              {actionMessage}
            </p>
          )}
        </section>

        <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.14em] text-[#52B788] font-semibold">
                Report Status
              </p>
              <h2 className="mt-2 text-xl font-bold text-[#1B1B1B]">
                {assessment.reportStatus === "available" || assessment.status === "completed"
                  ? "Completed"
                  : "Preparing"}
              </h2>
            </div>
            <span className="rounded-full bg-[#ECF8F3] px-4 py-2 text-sm font-semibold text-[#1F7A66]">
              Open Report
            </span>
          </div>

          <form onSubmit={handleRemarksSave} className="mt-6 space-y-4">
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-[#1B1B1B]">
                Doctor Remarks
              </span>
              <textarea
                value={remarksDraft}
                onChange={(event) => setRemarksDraft(event.target.value)}
                rows={5}
                className="w-full rounded-xl border border-[#D6E3DA] bg-white px-4 py-3 text-sm text-[#1B1B1B] outline-none transition-colors placeholder:text-[#9AA49F] focus:border-[#52B788] focus:ring-2 focus:ring-[#D8F3DC]"
                placeholder="Add clinical remarks for the patient"
              />
            </label>
            <div className="flex flex-wrap gap-3">
              <Button type="submit" disabled={savingRemarks}>
                {savingRemarks ? "Saving..." : "Save Remarks"}
              </Button>
            </div>
          </form>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div className="rounded-2xl border border-[#E8E8E8] bg-white p-6">
            <h2 className="text-xl font-bold text-[#1B1B1B]">Severity Meter</h2>
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
            <div className="mt-6 h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
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

        <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          <h2 className="text-xl font-bold text-[#1B1B1B]">
            Recorded Responses
          </h2>
          <div className="mt-6 space-y-4">
            {(assessment.answers || []).map((answer) => (
              <article
                key={answer.questionId}
                className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-5"
              >
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#52B788]">
                      Question {answer.questionId}
                    </p>
                    <h3 className="mt-2 text-lg font-semibold leading-snug text-[#1B1B1B]">
                      {answer.questionText}
                    </h3>
                  </div>
                  <span className="w-fit rounded-full bg-white px-4 py-2 text-sm font-bold text-[#1B3A2D] ring-1 ring-[#D6E3DA]">
                    Score {answer.score}/3
                  </span>
                </div>
                <div className="mt-4 rounded-xl border border-[#E8E8E8] bg-white p-3">
                  <AudioPlayback fileId={answer.audioFileId} />
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
