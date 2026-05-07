import { useCallback, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Loader from "../components/Loader.jsx";
import {
  getCurrentUser,
  getActiveConsultation,
  stopConsultation,
} from "../services/api.js";

export default function Consultation() {
  const user = getCurrentUser();
  const [consultation, setConsultation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [stopping, setStopping] = useState(false);

  const loadConsultation = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await getActiveConsultation();
      setConsultation(response.consultation);
    } catch (err) {
      setError(err.message || "Unable to load consultation.");
      setConsultation(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConsultation();
    // Poll for consultation updates every 5 seconds
    const interval = setInterval(loadConsultation, 5000);
    return () => clearInterval(interval);
  }, [loadConsultation]);

  const handleStopConsultation = async () => {
    if (!consultation) return;
    setStopping(true);
    setError("");
    try {
      await stopConsultation(consultation.id);
      setConsultation(null);
    } catch (err) {
      setError(err.message || "Unable to stop consultation.");
    } finally {
      setStopping(false);
    }
  };

  if (!user || user.role !== "patient") {
    return <Navigate to="/login" replace />;
  }

  const consultationStatus = {
    pending: { label: "Pending", color: "bg-amber-50 text-amber-700 border-amber-200" },
    active: { label: "Active", color: "bg-emerald-50 text-emerald-700 border-emerald-200" },
    stopped: { label: "Stopped", color: "bg-gray-50 text-gray-700 border-gray-200" },
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[60rem] space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Consultation
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Active Consultation
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                View your current doctor consultation details.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link to="/consultation-history">
                <Button variant="outline">History</Button>
              </Link>
              <Link to="/doctors">
                <Button variant="secondary">Find Doctors</Button>
              </Link>
            </div>
          </div>
        </section>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
            {error}
          </div>
        )}

        {loading ? (
          <Card className="p-8 text-center">
            <Loader />
            <p className="mt-4 text-[#6A766F]">Loading consultation...</p>
          </Card>
        ) : !consultation ? (
          <Card className="p-8 space-y-6">
            <div className="text-center">
              <p className="text-lg font-bold text-[#1B1B1B]">No Active Consultation</p>
              <p className="mt-2 text-[#6A766F]">
                You don't have an active consultation at the moment.
              </p>
            </div>
            <div className="flex justify-center">
              <Link to="/doctors">
                <Button>Find a Doctor</Button>
              </Link>
            </div>
          </Card>
        ) : (
          <>
            {/* Active Doctor Card */}
            <Card className="p-6 md:p-7 space-y-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                    Your Doctor
                  </p>
                  <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                    Dr. {consultation.doctor?.name || "Doctor"}
                  </h2>
                </div>
                <span
                  className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${
                    consultationStatus[consultation.status]?.color ||
                    "bg-gray-50 text-gray-700 border-gray-200"
                  }`}
                >
                  {consultationStatus[consultation.status]?.label || consultation.status}
                </span>
              </div>

              <div className="grid gap-3 text-sm text-[#4C5852]">
                <p>
                  <span className="font-semibold text-[#1B1B1B]">Email:</span>{" "}
                  {consultation.doctor?.email}
                </p>
                <p>
                  <span className="font-semibold text-[#1B1B1B]">Phone:</span>{" "}
                  {consultation.doctor?.phone}
                </p>
                {consultation.doctor?.specialization && (
                  <p>
                    <span className="font-semibold text-[#1B1B1B]">
                      Specialization:
                    </span>{" "}
                    {consultation.doctor.specialization}
                  </p>
                )}
                <p>
                  <span className="font-semibold text-[#1B1B1B]">Fee:</span>{" "}
                  {Number(consultation.doctor?.fee).toLocaleString("en-IN", {
                    style: "currency",
                    currency: "INR",
                    maximumFractionDigits: 0,
                  })}
                </p>
              </div>

              {consultation.assessment && (
                <div className="border-t border-[#E8E8E8] pt-4 space-y-3">
                  <p className="font-semibold text-[#1B1B1B]">Current Assessment</p>
                  <div className="rounded-lg bg-[#FAFAF7] p-4 space-y-2 text-sm">
                    <p>
                      <span className="font-semibold text-[#4C5852]">Score:</span>{" "}
                      {consultation.assessment.score}/24
                    </p>
                    <p>
                      <span className="font-semibold text-[#4C5852]">Severity:</span>{" "}
                      {consultation.assessment.severity}
                    </p>
                    <p>
                      <span className="font-semibold text-[#4C5852]">Date:</span>{" "}
                      {consultation.assessment.createdAt
                        ? new Date(consultation.assessment.createdAt).toLocaleDateString()
                        : "N/A"}
                    </p>
                  </div>
                </div>
              )}

              {consultation.status === "active" && (
                <div className="border-t border-[#E8E8E8] pt-4">
                  <Button
                    onClick={handleStopConsultation}
                    disabled={stopping}
                    className="w-full bg-red-600 text-white hover:bg-red-700"
                  >
                    {stopping ? "Stopping..." : "Stop Consultation"}
                  </Button>
                </div>
              )}
            </Card>

            {/* Consultation Timeline */}
            <Card className="p-6 md:p-7">
              <h3 className="text-lg font-bold text-[#1B1B1B] mb-4">
                Consultation Timeline
              </h3>
              <div className="space-y-3 text-sm text-[#6A766F]">
                <div className="flex gap-3">
                  <div className="flex flex-col items-center">
                    <div className="w-3 h-3 rounded-full bg-[#52B788]"></div>
                    <div className="w-0.5 h-8 bg-[#D6E3DA]"></div>
                  </div>
                  <div className="pb-8">
                    <p className="font-semibold text-[#1B1B1B]">Consultation Started</p>
                    <p>
                      {consultation.createdAt
                        ? new Date(consultation.createdAt).toLocaleString()
                        : "N/A"}
                    </p>
                  </div>
                </div>

                {consultation.startedAt && (
                  <div className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <div className="w-3 h-3 rounded-full bg-[#2D6A4F]"></div>
                      <div className="w-0.5 h-8 bg-[#D6E3DA]"></div>
                    </div>
                    <div className="pb-8">
                      <p className="font-semibold text-[#1B1B1B]">Doctor Accepted</p>
                      <p>
                        {new Date(consultation.startedAt).toLocaleString()}
                      </p>
                    </div>
                  </div>
                )}

                {consultation.endedAt && (
                  <div className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                    </div>
                    <div>
                      <p className="font-semibold text-[#1B1B1B]">Consultation Ended</p>
                      <p>
                        {new Date(consultation.endedAt).toLocaleString()}
                      </p>
                      {consultation.stopReason && (
                        <p className="mt-1 text-xs italic">{consultation.stopReason}</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
