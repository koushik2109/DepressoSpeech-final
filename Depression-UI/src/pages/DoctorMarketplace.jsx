import { useEffect, useMemo, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Input from "../components/Input.jsx";
import {
  assignDoctor,
  getCurrentUser,
  listPatientAssignments,
  listDoctors,
  getActiveConsultation,
  getConsultationHistory,
} from "../services/api.js";

function readLatestAssessmentId() {
  try {
    return JSON.parse(sessionStorage.getItem("latestAssessment") || "{}")?.id || null;
  } catch {
    return null;
  }
}

export default function DoctorMarketplace() {
  const user = getCurrentUser();
  const [filters, setFilters] = useState({
    minFee: "",
    maxFee: "",
    availability: "available",
  });
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [assigningId, setAssigningId] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [assignments, setAssignments] = useState([]);
  const [activeConsultation, setActiveConsultation] = useState(null);
  const [checkingConsultation, setCheckingConsultation] = useState(true);
  const [consultationHistory, setConsultationHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const assessmentId = useMemo(() => readLatestAssessmentId(), []);
  const userId = user?.id;
  const userRole = user?.role;

  useEffect(() => {
    if (!userId || userRole !== "patient") return undefined;
    setCheckingConsultation(true);
    getActiveConsultation()
      .then((response) => setActiveConsultation(response.consultation))
      .catch(() => setActiveConsultation(null))
      .finally(() => setCheckingConsultation(false));
    return undefined;
  }, [userId, userRole]);

  useEffect(() => {
    let active = true;
    if (!userId || userRole !== "patient") return undefined;
    setLoading(true);
    setError("");

    listDoctors({
      minFee: filters.minFee,
      maxFee: filters.maxFee,
      isAvailable:
        filters.availability === "all"
          ? null
          : filters.availability === "available",
    })
      .then((items) => {
        if (!active) return;
        setDoctors([...items].sort((a, b) => Number(a.fee) - Number(b.fee)));
      })
      .catch((err) => active && setError(err.message || "Unable to load doctors."))
      .finally(() => active && setLoading(false));

    return () => {
      active = false;
    };
  }, [filters, userId, userRole]);

  useEffect(() => {
    if (!userId || userRole !== "patient") return undefined;
    listPatientAssignments()
      .then(setAssignments)
      .catch(() => setAssignments([]));
    setLoadingHistory(true);
    getConsultationHistory()
      .then((response) => setConsultationHistory(response.items || []))
      .catch(() => setConsultationHistory([]))
      .finally(() => setLoadingHistory(false));
    return undefined;
  }, [userId, userRole]);

  if (!user || user.role !== "patient") {
    return <Navigate to="/login" replace />;
  }

  const updateFilter = (field) => (event) => {
    setFilters((previous) => ({ ...previous, [field]: event.target.value }));
    setMessage("");
  };

  const handleAssign = async ({ doctorId = null, autoAssign = false }) => {
    setAssigningId(autoAssign ? "auto" : doctorId);
    setError("");
    setMessage("");
    try {
      const assignment = await assignDoctor({
        doctorId,
        autoAssign,
        assessmentId,
      });
      setMessage(`Request sent to Dr. ${assignment.doctor?.name || "Doctor"}. Contact details are emailed after acceptance.`);
      listPatientAssignments()
        .then(setAssignments)
        .catch(() => setAssignments([]));
    } catch (err) {
      setError(err.message || "Unable to assign doctor.");
    } finally {
      setAssigningId("");
    }
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-[88rem] space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Doctor Marketplace
              </p>
              <h1 className="mt-3 text-3xl font-bold text-[#1B1B1B] md:text-4xl">
                Choose a Doctor
              </h1>
              <p className="mt-2 max-w-2xl text-[#5F6B65]">
                Browse available doctors with complete contact profiles and assign one for follow-up.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                variant="secondary"
                onClick={() => handleAssign({ autoAssign: true })}
                disabled={assigningId === "auto" || loading}
              >
                {assigningId === "auto" ? "Assigning..." : "Auto Assign Lowest Fee"}
              </Button>
              <Link to="/assessment-history">
                <Button variant="outline">History</Button>
              </Link>
            </div>
          </div>
        </section>

        {!checkingConsultation && activeConsultation && (
          <Card className="p-5 md:p-6 border-l-4 border-l-[#2D6A4F] bg-[#F3FBF7]">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h3 className="text-lg font-bold text-[#1B1B1B]">
                  Active Consultation
                </h3>
                <p className="text-sm text-[#6A766F] mt-1">
                  You have an active consultation with Dr. {activeConsultation.doctor?.name}. 
                  The marketplace is unavailable while a consultation is ongoing.
                </p>
              </div>
              <Link to="/consultation">
                <Button>View Consultation</Button>
              </Link>
            </div>
          </Card>
        )}

        <Card className="p-5 md:p-6">
          <div className="grid gap-4 md:grid-cols-[1fr_1fr_220px] md:items-end">
            <Input
              label="Minimum Fee"
              id="minFee"
              type="number"
              placeholder="0"
              value={filters.minFee}
              onChange={updateFilter("minFee")}
            />
            <Input
              label="Maximum Fee"
              id="maxFee"
              type="number"
              placeholder="500"
              value={filters.maxFee}
              onChange={updateFilter("maxFee")}
            />
            <div className="space-y-1.5">
              <label
                htmlFor="availability"
                className="block text-sm font-semibold text-[#1B1B1B]/80"
              >
                Availability
              </label>
              <select
                id="availability"
                value={filters.availability}
                onChange={updateFilter("availability")}
                className="w-full rounded-xl border border-[#E8E8E8] bg-white/70 px-4 py-3.5 text-sm font-medium text-[#1B1B1B] outline-none transition-colors focus:border-[#2D6A4F] focus:ring-2 focus:ring-[#2D6A4F]/30"
              >
                <option value="available">Available only</option>
                <option value="all">All doctors</option>
              </select>
            </div>
          </div>
        </Card>

        {(message || error) && (
          <div
            className={`rounded-xl border px-4 py-3 text-sm font-semibold ${
              error
                ? "border-red-200 bg-red-50 text-red-700"
                : "border-emerald-200 bg-emerald-50 text-emerald-700"
            }`}
          >
            {error || message}
          </div>
        )}

        {consultationHistory.length > 0 && (
          <Card className="p-5 md:p-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between mb-4">
              <h2 className="text-xl font-bold text-[#1B1B1B]">Consultation History</h2>
              <Link to="/consultation-history">
                <Button variant="outline" size="sm">View All</Button>
              </Link>
            </div>
            <div className="space-y-3">
              {consultationHistory.slice(0, 5).map((consultation) => (
                <div key={consultation.id} className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] p-4">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex-1">
                      <p className="font-bold text-[#1B1B1B]">
                        Dr. {consultation.doctor?.name || "Doctor"}
                      </p>
                      <p className="text-sm text-[#6A766F]">
                        {consultation.assessment?.severity || 'No assessment'} (Score: {consultation.assessment?.score != null ? consultation.assessment.score : '—'}/24)
                      </p>
                      <p className="text-xs text-[#999] mt-1">
                        {new Date(consultation.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                    <span
                      className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold whitespace-nowrap ${
                        consultation.status === "active"
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : consultation.status === "stopped"
                            ? "bg-gray-50 text-gray-700 border-gray-200"
                            : "bg-blue-50 text-blue-700 border-blue-200"
                      }`}
                    >
                      {consultation.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {loading ? (
          <Card className="p-8 text-center text-[#6A766F]">Loading doctors...</Card>
        ) : activeConsultation ? (
          <Card className="p-8 text-center">
            <p className="text-lg font-bold text-[#1B1B1B]">Marketplace Unavailable</p>
            <p className="mt-1 text-sm text-[#6A766F]">
              You have an active consultation. Please complete or stop your current consultation to assign a new doctor.
            </p>
            <Link to="/consultation" className="mt-4 inline-block">
              <Button>View Your Consultation</Button>
            </Link>
          </Card>
        ) : doctors.length === 0 ? (
          <Card className="p-8 text-center">
            <p className="text-lg font-bold text-[#1B1B1B]">No matching doctors</p>
            <p className="mt-1 text-sm text-[#6A766F]">
              Adjust the fee range or availability filter.
            </p>
          </Card>
        ) : (
          <section className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {doctors.map((doctor) => (
              <Card key={doctor.id} className="flex h-full flex-col gap-5 p-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#52B788]">
                      Doctor
                    </p>
                    <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                      Dr. {doctor.name}
                    </h2>
                  </div>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-bold ${
                      doctor.isAvailable
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {doctor.isAvailable ? "Available" : "Unavailable"}
                  </span>
                </div>

                <div className="grid gap-3 text-sm text-[#4C5852]">
                  <p>
                    <span className="font-semibold text-[#1B1B1B]">Fee:</span>{" "}
                    {Number(doctor.fee).toLocaleString("en-IN", {
                      style: "currency",
                      currency: "INR",
                      maximumFractionDigits: 0,
                    })}
                  </p>
                  <p>
                    <span className="font-semibold text-[#1B1B1B]">Email:</span>{" "}
                    {doctor.email}
                  </p>
                  <p>
                    <span className="font-semibold text-[#1B1B1B]">Phone:</span>{" "}
                    {doctor.phone}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => handleAssign({ doctorId: doctor.id })}
                  disabled={!doctor.isAvailable || assigningId === doctor.id}
                  className={`mt-auto min-h-12 rounded-xl px-5 py-3 text-sm font-bold transition-colors ${
                    doctor.isAvailable
                      ? "bg-[#1B3A2D] text-white hover:bg-[#2D6A4F]"
                      : "cursor-not-allowed bg-[#E8E8E8] text-[#9AA49F]"
                  }`}
                >
                  {assigningId === doctor.id ? "Assigning..." : "Select Doctor"}
                </button>
              </Card>
            ))}
          </section>
        )}
      </div>
    </div>
  );
}
