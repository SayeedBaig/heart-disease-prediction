import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { UserCheck, CheckCircle2, ArrowRight, ArrowLeft } from "lucide-react";
import Navbar from "../components/Navbar";
import { getAvailableDoctors, bookAppointment } from "../services/portalService";

export default function Appointments() {
  const navigate = useNavigate();

  // Workflow steps: 1. Choose Doctor -> 2. Select Date -> 3. Select Time -> 4. Submit
  const [step, setStep] = useState(1);
  const [doctors, setDoctors] = useState([]);
  const [loadingDoctors, setLoadingDoctors] = useState(true);
  const [doctorsError, setDoctorsError] = useState("");

  const [form, setForm] = useState({
    doctorId: "",
    doctorName: "",
    date: "",
    time: "09:00",
    reason: "",
    symptoms: ""
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    getAvailableDoctors()
      .then((data) => {
        setDoctors(data || []);
      })
      .catch((requestError) => setDoctorsError(requestError.response?.data?.detail || "Unable to load available doctors. Please try again."))
      .finally(() => setLoadingDoctors(false));
  }, []);

  const handleSelectDoctor = (doc) => {
    setForm(prev => ({
      ...prev,
      doctorId: doc.doctor_id,
      doctorName: doc.full_name
    }));
    setStep(2);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!form.doctorId || !form.date || !form.time) {
      setErrorMsg("Please complete all required appointment fields.");
      return;
    }

    setErrorMsg("");
    setIsSubmitting(true);

    try {
      await bookAppointment({
        doctorId: form.doctorId,
        date: form.date,
        time: form.time,
        reason: form.reason || "General cardiac follow-up",
        symptoms: form.symptoms || "None"
      });

      setStatusMessage("Your appointment request has been submitted to Dr. " + form.doctorName + ". You will receive confirmation once accepted.");
      setStep(5); // Success step
    } catch (requestError) {
      setErrorMsg(requestError.response?.data?.detail || "Your appointment could not be saved. Please sign in again and retry.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="cardio-shell">
      <Navbar breadcrumb="Book Appointment" />

      <main className="cardio-container py-8 flex-1 w-full max-w-5xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)]">
          <div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Specialist Consultation
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Book Doctor Appointment
            </h1>
          </div>

          <div className="flex items-center gap-2 text-xs font-semibold text-[var(--text-muted)] mt-2 sm:mt-0">
            <span>Step {Math.min(step, 4)} of 4</span>
          </div>
        </div>

        {/* Workflow Progress Bar */}
        {step <= 4 && (
          <div className="flex items-center justify-between mb-8 text-xs font-medium border-b border-[var(--border-color)] pb-4">
            <div className={`flex items-center gap-2 ${step >= 1 ? "text-[var(--accent-melanzane)] font-bold" : "text-[var(--text-muted)]"}`}>
              <span className="w-5 h-5 rounded-full bg-[var(--accent-melanzane-light)] flex items-center justify-center text-[10px]">1</span>
              1. Choose Doctor
            </div>
            <div className={`flex items-center gap-2 ${step >= 2 ? "text-[var(--accent-melanzane)] font-bold" : "text-[var(--text-muted)]"}`}>
              <span className="w-5 h-5 rounded-full bg-[var(--accent-melanzane-light)] flex items-center justify-center text-[10px]">2</span>
              2. Select Date
            </div>
            <div className={`flex items-center gap-2 ${step >= 3 ? "text-[var(--accent-melanzane)] font-bold" : "text-[var(--text-muted)]"}`}>
              <span className="w-5 h-5 rounded-full bg-[var(--accent-melanzane-light)] flex items-center justify-center text-[10px]">3</span>
              3. Select Time
            </div>
            <div className={`flex items-center gap-2 ${step >= 4 ? "text-[var(--accent-melanzane)] font-bold" : "text-[var(--text-muted)]"}`}>
              <span className="w-5 h-5 rounded-full bg-[var(--accent-melanzane-light)] flex items-center justify-center text-[10px]">4</span>
              4. Submit
            </div>
          </div>
        )}

        {/* STEP 1: CHOOSE DOCTOR */}
        {step === 1 && (
          <div className="space-y-6">
            <h2 className="section-title text-sm font-semibold text-[var(--text-primary)]">
              Select a Certified Cardiologist
            </h2>

            {loadingDoctors ? (
              <div className="text-xs text-[var(--text-muted)] py-8 text-center">Loading available specialists...</div>
            ) : doctorsError ? (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center text-xs font-semibold text-red-600">{doctorsError}</div>
            ) : doctors.length === 0 ? (
              <div className="rounded-xl border border-[var(--border-color)] bg-[var(--bg-secondary)] p-4 text-center text-xs text-[var(--text-muted)]">No doctors are available for appointment booking yet.</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {doctors.map((doc) => (
                  <div
                    key={doc.doctor_id}
                    onClick={() => handleSelectDoctor(doc)}
                    className="cardio-card-interactive p-5 flex flex-col justify-between group"
                  >
                    <div>
                      <div className="w-10 h-10 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-3">
                        <UserCheck size={20} />
                      </div>
                      <h3 className="text-sm font-bold text-[var(--text-primary)]">
                        {doc.full_name}
                      </h3>
                      <p className="caption-small text-[var(--accent-melanzane)] font-medium mt-0.5">
                        {doc.specialization}
                      </p>
                      <p className="caption-small text-[var(--text-muted)] mt-1">
                        {doc.hospital}
                      </p>
                    </div>

                    <button className="btn-primary text-xs py-2 px-3 mt-4 w-full flex items-center justify-center gap-1">
                      Select Doctor <ArrowRight size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* STEP 2: SELECT DATE */}
        {step === 2 && (
          <div className="cardio-card p-6 space-y-6 max-w-xl mx-auto">
            <div className="flex items-center justify-between">
              <h2 className="section-title text-sm font-semibold text-[var(--text-primary)]">
                Selected Doctor: {form.doctorName}
              </h2>
              <button onClick={() => setStep(1)} className="btn-minimal text-xs">
                <ArrowLeft size={14} /> Change
              </button>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-2">
                Preferred Date *
              </label>
              <input
                type="date"
                value={form.date}
                min={new Date().toISOString().split("T")[0]}
                onChange={(e) => setForm(prev => ({ ...prev, date: e.target.value }))}
                className="cardio-input text-xs"
              />
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t border-[var(--border-color)]">
              <button onClick={() => setStep(1)} className="btn-secondary text-xs py-2 px-4">
                Back
              </button>
              <button
                disabled={!form.date}
                onClick={() => setStep(3)}
                className="btn-primary text-xs py-2 px-5 disabled:opacity-50"
              >
                Next: Select Time
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: SELECT TIME */}
        {step === 3 && (
          <div className="cardio-card p-6 space-y-6 max-w-xl mx-auto">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="section-title text-sm font-semibold text-[var(--text-primary)]">
                  Doctor: {form.doctorName}
                </h2>
                <p className="caption-small">Date: {form.date}</p>
              </div>
              <button onClick={() => setStep(2)} className="btn-minimal text-xs">
                <ArrowLeft size={14} /> Change Date
              </button>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-2">
                Preferred Time Slot *
              </label>
              <div className="grid grid-cols-3 gap-3">
                {["09:00", "10:30", "11:45", "14:00", "15:30", "16:45"].map((t) => (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setForm(prev => ({ ...prev, time: t }))}
                    className={`py-2.5 px-3 rounded-xl border text-xs font-semibold transition-all ${
                      form.time === t
                        ? "bg-[#39062B] text-white border-[#39062B]"
                        : "bg-[var(--input-bg)] text-[var(--text-primary)] border-[var(--input-border)] hover:border-[var(--accent-melanzane-border)]"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t border-[var(--border-color)]">
              <button onClick={() => setStep(2)} className="btn-secondary text-xs py-2 px-4">
                Back
              </button>
              <button
                onClick={() => setStep(4)}
                className="btn-primary text-xs py-2 px-5"
              >
                Next: Review & Submit
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: SUBMIT */}
        {step === 4 && (
          <div className="cardio-card p-6 space-y-6 max-w-xl mx-auto">
            <h2 className="section-title text-sm font-semibold text-[var(--text-primary)]">
              Review & Confirm Request
            </h2>

            <div className="p-4 rounded-xl bg-[var(--bg-secondary)] space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="caption-small">Doctor:</span>
                <span className="font-bold text-[var(--text-primary)]">{form.doctorName}</span>
              </div>
              <div className="flex justify-between">
                <span className="caption-small">Date & Time:</span>
                <span className="font-bold text-[var(--text-primary)]">{form.date} at {form.time}</span>
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Reason for Visit (Optional)
              </label>
              <input
                type="text"
                placeholder="e.g. Heart health check, chest tightness follow-up..."
                value={form.reason}
                onChange={(e) => setForm(prev => ({ ...prev, reason: e.target.value }))}
                className="cardio-input text-xs"
              />
            </div>

            {errorMsg && (
              <p className="text-xs text-red-500 font-semibold">{errorMsg}</p>
            )}

            <div className="flex justify-end gap-3 pt-4 border-t border-[var(--border-color)]">
              <button onClick={() => setStep(3)} className="btn-secondary text-xs py-2 px-4">
                Back
              </button>
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="btn-primary text-xs py-2.5 px-6"
              >
                {isSubmitting ? "Submitting Request..." : "Submit Request"}
              </button>
            </div>
          </div>
        )}

        {/* STEP 5: CONFIRMATION SUCCESS */}
        {step === 5 && (
          <div className="cardio-card p-8 text-center max-w-lg mx-auto space-y-4">
            <div className="w-16 h-16 rounded-full bg-emerald-500/10 text-emerald-500 flex items-center justify-center mx-auto">
              <CheckCircle2 size={36} />
            </div>

            <h2 className="section-title text-base font-bold text-[var(--text-primary)]">
              Request Submitted
            </h2>

            <p className="body-regular text-xs leading-relaxed">
              {statusMessage}
            </p>

            <div className="pt-4 border-t border-[var(--border-color)] flex justify-center gap-3">
              <button
                onClick={() => navigate("/patient/dashboard")}
                className="btn-primary text-xs py-2.5 px-6"
              >
                Return to Patient Dashboard
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
