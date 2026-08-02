import { useState } from "react";
import Navbar from "../components/Navbar";
import api from "../services/api";

function DoctorNotes() {
  const [diagnosisId, setDiagnosisId] = useState("");
  const [notes, setNotes] = useState("");
  const [prescription, setPrescription] = useState("");
  const [advice, setAdvice] = useState("");
  const [followUp, setFollowUp] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleSaveNotes = async () => {
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      const response = await api.post("/notes/", {
        diagnosis_id: diagnosisId,
        notes: notes,
        prescription: prescription,
        advice: advice,
        follow_up: followUp,
      });

      console.log(response.data);

      setSuccess("Doctor notes saved successfully!");

      // Clear the form after successful save
      setDiagnosisId("");
      setNotes("");
      setPrescription("");
      setAdvice("");
      setFollowUp("");
    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail || "Failed to save notes."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cardio-shell">
      <Navbar breadcrumb="Clinical Observations" />

      <main className="cardio-container py-8 flex-1 w-full max-w-4xl">
        <div className="cardio-card p-8">
          <div className="border-b border-[var(--border-color)] pb-4 mb-6">
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Clinician Portal
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Doctor Clinical Notes
            </h1>
            <p className="body-regular text-xs mt-1">
              Add observations, prescriptions, and follow-up guidance to patient records.
            </p>
          </div>

          <div className="space-y-5">
            {/* Diagnosis ID */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Diagnosis ID *
              </label>
              <input
                type="text"
                value={diagnosisId}
                onChange={(e) => setDiagnosisId(e.target.value)}
                className="cardio-input text-xs"
                placeholder="Enter Diagnosis ID (e.g. DIAG-9021)"
              />
            </div>

            {/* Notes */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Clinical Observations
              </label>
              <textarea
                rows="4"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="cardio-input text-xs"
                placeholder="Enter detailed physician findings..."
              />
            </div>

            {/* Prescription */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Prescription & Medication Plan
              </label>
              <textarea
                rows="3"
                value={prescription}
                onChange={(e) => setPrescription(e.target.value)}
                className="cardio-input text-xs"
                placeholder="Enter prescribed medications and dosage..."
              />
            </div>

            {/* Advice */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Lifestyle & Dietary Advice
              </label>
              <textarea
                rows="3"
                value={advice}
                onChange={(e) => setAdvice(e.target.value)}
                className="cardio-input text-xs"
                placeholder="Lifestyle, dietary, and exercise recommendations..."
              />
            </div>

            {/* Follow Up */}
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Follow-Up Schedule
              </label>
              <input
                type="text"
                value={followUp}
                onChange={(e) => setFollowUp(e.target.value)}
                className="cardio-input text-xs"
                placeholder="e.g. Re-evaluate blood pressure in 30 days"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-semibold">
                {error}
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold">
                {success}
              </div>
            )}

            {/* Save Button */}
            <div className="pt-2">
              <button
                onClick={handleSaveNotes}
                disabled={loading}
                className="btn-primary w-full py-3 text-xs font-semibold rounded-xl"
              >
                {loading ? "Saving Notes..." : "Save Doctor Notes"}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default DoctorNotes;