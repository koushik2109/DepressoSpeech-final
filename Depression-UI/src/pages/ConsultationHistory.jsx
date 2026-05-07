import { useCallback, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Loader from "../components/Loader.jsx";
import { getCurrentUser, getConsultationHistory } from "../services/api.js";

export default function ConsultationHistory() {
  const user = getCurrentUser();
  const [consultations, setConsultations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await getConsultationHistory();
      setConsultations(response.items || []);
    } catch (err) {
      setError(err.message || "Unable to load consultation history.");
      setConsultations([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  if (!user || user.role !== "patient") {
    return <Navigate to="/login" replace />;
  }

  const statusTone = {
    stopped: "bg-gray-50 text-gray-700 border-gray-200",
    completed: "bg-emerald-50 text-emerald-700 border-emerald-200",
    rejected: "bg-red-50 text-red-700 border-red-200",
    cancelled: "bg-amber-50 text-amber-700 border-amber-200",
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[60rem] space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Consultation History
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Past Consultations
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                View your previous doctor consultations and their details.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link to="/consultation">
                <Button variant="outline">Active Consultation</Button>
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
            <p className="mt-4 text-[#6A766F]">Loading consultation history...</p>
          </Card>
        ) : consultations.length === 0 ? (
          <Card className="p-8 space-y-6">
            <div className="text-center">
              <p className="text-lg font-bold text-[#1B1B1B]">No Consultation History</p>
              <p className="mt-2 text-[#6A766F]">
                You don't have any past consultations yet.
              </p>
            </div>
            <div className="flex justify-center">
              <Link to="/doctors">
                <Button>Find a Doctor</Button>
              </Link>
            </div>
          </Card>
        ) : (
          <div className="space-y-4">
            {consultations.map((consultation) => (
              <Card key={consultation.id} className="p-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <h3 className="text-lg font-bold text-[#1B1B1B]">
                        Dr. {consultation.doctor?.name || "Doctor"}
                      </h3>
                      <span
                        className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${
                          statusTone[consultation.status] ||
                          "bg-gray-50 text-gray-700 border-gray-200"
                        }`}
                      >
                        {consultation.status}
                      </span>
                    </div>

                    <div className="grid gap-2 text-sm text-[#6A766F] md:grid-cols-2">
                      <p>
                        <span className="font-semibold text-[#4C5852]">Email:</span>{" "}
                        {consultation.doctor?.email}
                      </p>
                      <p>
                        <span className="font-semibold text-[#4C5852]">Phone:</span>{" "}
                        {consultation.doctor?.phone}
                      </p>
                      <p>
                        <span className="font-semibold text-[#4C5852]">Started:</span>{" "}
                        {consultation.createdAt
                          ? new Date(consultation.createdAt).toLocaleDateString()
                          : "N/A"}
                      </p>
                      <p>
                        <span className="font-semibold text-[#4C5852]">Ended:</span>{" "}
                        {consultation.endedAt
                          ? new Date(consultation.endedAt).toLocaleDateString()
                          : "N/A"}
                      </p>
                    </div>

                    {consultation.assessment && (
                      <div className="mt-3 pt-3 border-t border-[#E8E8E8]">
                        <p className="text-xs font-semibold text-[#4C5852] mb-2">
                          Assessment Score: {consultation.assessment.score}/24 ({consultation.assessment.severity})
                        </p>
                      </div>
                    )}

                    {consultation.stopReason && (
                      <div className="mt-3 pt-3 border-t border-[#E8E8E8]">
                        <p className="text-xs text-[#6A766F] italic">
                          <span className="font-semibold">Reason:</span> {consultation.stopReason}
                        </p>
                      </div>
                    )}
                  </div>

                  <Link to={`/assessment-history`}>
                    <Button variant="secondary" size="sm">
                      View Details
                    </Button>
                  </Link>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
