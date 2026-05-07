import { useEffect, useMemo, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import Button from "../components/Button.jsx";
import Card from "../components/Card.jsx";
import Input from "../components/Input.jsx";
import {
  getCurrentUser,
  getDoctorProfile,
  getUserProfile,
  updateCurrentUser,
  updateDoctorProfile,
  updateUserProfile,
} from "../services/api.js";

export default function Profile() {
  const currentUser = getCurrentUser();
  const isDoctor = currentUser?.role === "doctor";
  const isPatient = currentUser?.role === "patient";
  const [profile, setProfile] = useState(null);
  const [form, setForm] = useState({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadProfile = async () => {
      setLoading(true);
      setError("");
      try {
        const data = isDoctor
          ? await getDoctorProfile()
          : await getUserProfile();
        if (!active) return;
        setProfile(data);
        setForm(
          isDoctor
            ? {
                email: data?.email || currentUser?.email || "",
                phone: data?.phone || "",
                fee: String(data?.fee ?? 100),
                isAvailable: Boolean(data?.isAvailable),
              }
            : {
                name: data?.name || currentUser?.name || "",
                age: data?.age == null ? "" : String(data.age),
                basicInfo: data?.basicInfo || "",
              },
        );
      } catch (err) {
        if (active) setError(err.message || "Unable to load profile.");
      } finally {
        if (active) setLoading(false);
      }
    };

    if (isDoctor || isPatient) loadProfile();
    return () => {
      active = false;
    };
  }, [currentUser?.email, currentUser?.name, isDoctor, isPatient]);

  const title = useMemo(
    () => (isDoctor ? "Doctor Profile" : "Patient Profile"),
    [isDoctor],
  );

  if (!currentUser || (!isDoctor && !isPatient)) {
    return <Navigate to="/login" replace />;
  }

  const updateField = (field) => (event) => {
    const value =
      field === "isAvailable" ? event.target.checked : event.target.value;
    setForm((previous) => ({ ...previous, [field]: value }));
    setMessage("");
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    setError("");

    try {
      if (isDoctor) {
        if (!form.email?.trim() || !form.phone?.trim() || form.fee === "") {
          throw new Error("Email, phone, and fee are required.");
        }
        const fee = Number(form.fee);
        if (!Number.isFinite(fee)) {
          throw new Error("Fee must be a valid number.");
        }
        const updated = await updateDoctorProfile({
          email: form.email.trim().toLowerCase(),
          phone: form.phone.trim(),
          fee,
          isAvailable: Boolean(form.isAvailable),
        });
        updateCurrentUser({ email: updated.email });
        setProfile(updated);
        setForm({
          email: updated?.email || "",
          phone: updated?.phone || "",
          fee: String(updated?.fee ?? 100),
          isAvailable: Boolean(updated?.isAvailable),
        });
        setMessage("Profile updated.");
      } else {
        if (!form.name?.trim()) {
          throw new Error("Name is required.");
        }
        const age =
          form.age === "" || form.age == null ? null : Number(form.age);
        if (age != null && !Number.isFinite(age)) {
          throw new Error("Age must be a valid number.");
        }
        const updated = await updateUserProfile({
          name: form.name.trim(),
          age,
          basicInfo: form.basicInfo?.trim() || "",
        });
        updateCurrentUser(updated);
        setProfile(updated);
        setForm({
          name: updated?.name || "",
          age: updated?.age == null ? "" : String(updated.age),
          basicInfo: updated?.basicInfo || "",
        });
        setMessage("Profile updated.");
      }
    } catch (err) {
      setError(err.message || "Unable to update profile.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen bg-[#F7F7F2] px-4 py-10">
      <div className="mx-auto max-w-5xl space-y-6">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Account Settings
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                {title}
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                Keep your profile details up to date.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                to={isDoctor ? "/doctor/dashboard" : "/assessment-history"}
                className="inline-flex items-center rounded-xl border border-[#D6E3DA] bg-white px-5 py-2.5 text-sm font-semibold text-[#1B1B1B] hover:bg-[#F4FAF6] transition-colors"
              >
                Back
              </Link>
            </div>
          </div>
        </section>

        {loading ? (
          <Card className="p-8">
            <p className="text-sm text-[#6A766F]">Loading profile...</p>
          </Card>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
            <Card className="p-6 md:p-7">
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Profile Summary
              </p>
              <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                {currentUser.name}
              </h2>
              <p className="mt-1 text-sm text-[#6A766F]">{currentUser.email}</p>
              <div className="mt-6 space-y-3">
                <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                    Role
                  </p>
                  <p className="mt-1 text-base font-semibold text-[#1B1B1B] capitalize">
                    {currentUser.role}
                  </p>
                </div>
                {isDoctor ? (
                  <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                      Managed Patients
                    </p>
                    <p className="mt-1 text-base font-semibold text-[#1B1B1B]">
                      {profile?.patientCount ?? 0}
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                        Age
                      </p>
                      <p className="mt-1 text-base font-semibold text-[#1B1B1B]">
                        {profile?.age ?? "—"}
                      </p>
                    </div>
                    <div className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-3">
                      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6A766F]">
                        Bio
                      </p>
                      <p className="mt-1 text-sm text-[#5F6B65]">
                        {profile?.basicInfo ||
                          "No additional information provided."}
                      </p>
                    </div>
                  </>
                )}
              </div>
            </Card>

            <Card className="p-6 md:p-7">
              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                    Update Profile
                  </p>
                  <h2 className="mt-2 text-2xl font-bold text-[#1B1B1B]">
                    Edit Details
                  </h2>
                </div>

                {isDoctor ? (
                  <>
                    <Input
                      label="Email"
                      id="doctorProfileEmail"
                      type="email"
                      value={form.email || ""}
                      onChange={updateField("email")}
                      required
                    />
                    <Input
                      label="Phone"
                      id="doctorProfilePhone"
                      value={form.phone || ""}
                      onChange={updateField("phone")}
                      required
                    />
                    <Input
                      label="Fee"
                      id="doctorProfileFee"
                      type="number"
                      value={form.fee || ""}
                      onChange={updateField("fee")}
                      required
                    />
                    <label className="inline-flex min-h-12 items-center gap-3 rounded-xl border border-[#D6E3DA] bg-white px-4 py-3 text-sm font-bold text-[#1B1B1B]">
                      <input
                        type="checkbox"
                        checked={Boolean(form.isAvailable)}
                        onChange={updateField("isAvailable")}
                        className="h-5 w-5 accent-[#2D6A4F]"
                      />
                      Available
                    </label>
                  </>
                ) : (
                  <>
                    <Input
                      label="Name"
                      id="patientProfileName"
                      value={form.name || ""}
                      onChange={updateField("name")}
                      required
                    />
                    <Input
                      label="Age"
                      id="patientProfileAge"
                      type="number"
                      value={form.age || ""}
                      onChange={updateField("age")}
                    />
                    <label className="block">
                      <span className="mb-2 block text-sm font-semibold text-[#1B1B1B]">
                        Basic Info
                      </span>
                      <textarea
                        value={form.basicInfo || ""}
                        onChange={updateField("basicInfo")}
                        rows={5}
                        className="w-full rounded-xl border border-[#D6E3DA] bg-white px-4 py-3 text-sm text-[#1B1B1B] outline-none transition-colors placeholder:text-[#9AA49F] focus:border-[#52B788] focus:ring-2 focus:ring-[#D8F3DC]"
                        placeholder="Tell us a little about yourself"
                      />
                    </label>
                  </>
                )}

                {(message || error) && (
                  <p
                    className={`text-sm font-semibold ${error ? "text-red-600" : "text-emerald-700"}`}
                  >
                    {error || message}
                  </p>
                )}

                <Button type="submit" disabled={saving}>
                  {saving ? "Saving..." : "Update Profile"}
                </Button>
              </form>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
