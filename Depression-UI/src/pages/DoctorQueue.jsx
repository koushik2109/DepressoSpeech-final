import { useCallback, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Loader from "../components/Loader.jsx";
import {
  getCurrentUser,
  listDoctorAssignments,
  updateDoctorAssignment,
} from "../services/api.js";

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

export default function DoctorQueue() {
  const user = getCurrentUser();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setItems((await listDoctorAssignments("pending")) || []);
    } catch (err) {
      setItems([]);
      setError(err.message || "Unable to load queue.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const handleRefresh = () => refresh();
    window.addEventListener("mindscope-session-updated", handleRefresh);
    return () =>
      window.removeEventListener("mindscope-session-updated", handleRefresh);
  }, [refresh]);

  if (!user || user.role !== "doctor") {
    return <Navigate to="/login" replace />;
  }

  const handleAction = async (assignmentId, action) => {
    if (busy) return;
    setBusy(true);
    setMessage("");
    setError("");
    try {
      await updateDoctorAssignment(assignmentId, action);
      await refresh();
      setMessage(
        action === "accept"
          ? "Accepted."
          : action === "reject"
            ? "Rejected."
            : "Reassigned.",
      );
    } catch (err) {
      setError(err.message || "Unable to update assignment.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[88rem] space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
            <div>
              <Link
                to="/doctor/dashboard"
                className="text-sm font-bold text-[#2D6A4F]"
              >
                Back to dashboard
              </Link>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Doctor Queue
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                Pending patients waiting for action.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <span className="rounded-full bg-[#ECF8F3] px-4 py-2 text-sm font-semibold text-[#1F7A66]">
                {items.length} pending
              </span>
            </div>
          </div>
        </section>

        {(message || error) && (
          <Card className="p-5">
            <p
              className={`text-sm font-semibold ${error ? "text-red-600" : "text-emerald-700"}`}
            >
              {error || message}
            </p>
          </Card>
        )}

        {loading ? (
          <Card className="p-8">
            <Loader size="md" text="Loading queue..." />
          </Card>
        ) : items.length === 0 ? (
          <Card className="p-8">
            <p className="text-sm text-[#6A766F]">No pending patients.</p>
          </Card>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            {items.map((assignment) => (
              <Card key={assignment.id} className="p-5">
                <div className="flex flex-col gap-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-lg font-bold text-[#1B1B1B]">
                        {assignment.patient?.name || "Patient"}
                      </p>
                      <p className="text-sm text-[#6A766F]">
                        {assignment.patient?.email || "Email unavailable"}
                      </p>
                      <p className="text-sm text-[#6A766F]">
                        Requested: {formatIST(assignment.createdAt)}
                      </p>
                    </div>
                    {(assignment.patient?.id || assignment.patientId) && (
                      <Link
                        to={`/doctor/patient/${assignment.patient?.id || assignment.patientId}`}
                        className="rounded-full bg-[#ECF8F3] px-3 py-1 text-xs font-semibold text-[#1F7A66]"
                      >
                        History
                      </Link>
                    )}
                  </div>

                  {assignment.assessment && (
                    <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                        Assessment
                      </p>
                      <p className="mt-1 text-sm font-semibold text-[#1B1B1B]">
                        Score {assignment.assessment.score}/24 ·{" "}
                        {assignment.assessment.severity}
                      </p>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleAction(assignment.id, "accept")}
                      disabled={busy}
                    >
                      Accept
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleAction(assignment.id, "reject")}
                      disabled={busy}
                    >
                      Reject
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => handleAction(assignment.id, "reassign")}
                      disabled={busy}
                    >
                      Reassign
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
