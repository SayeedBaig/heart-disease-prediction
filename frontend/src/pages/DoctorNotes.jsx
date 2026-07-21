import { useState } from "react";
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
    <div className="min-h-screen bg-slate-100 p-10">
      <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-lg p-8">

        <h1 className="text-3xl font-bold text-slate-800">
          Doctor Notes
        </h1>

        <p className="text-slate-500 mt-2 mb-8">
          Add clinical observations and recommendations.
        </p>

        <div className="space-y-6">

          {/* Diagnosis ID */}

          <div>
            <label className="block font-medium mb-2">
              Diagnosis ID
            </label>

            <input
              type="text"
              value={diagnosisId}
              onChange={(e) => setDiagnosisId(e.target.value)}
              className="w-full border rounded-xl p-3"
              placeholder="Enter Diagnosis ID"
            />
          </div>

          {/* Notes */}

          <div>
            <label className="block font-medium mb-2">
              Clinical Notes
            </label>

            <textarea
              rows="4"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full border rounded-xl p-3"
              placeholder="Enter doctor observations"
            />
          </div>

          {/* Prescription */}

          <div>
            <label className="block font-medium mb-2">
              Prescription
            </label>

            <textarea
              rows="3"
              value={prescription}
              onChange={(e) => setPrescription(e.target.value)}
              className="w-full border rounded-xl p-3"
              placeholder="Enter prescription"
            />
          </div>

          {/* Advice */}

          <div>
            <label className="block font-medium mb-2">
              Advice
            </label>

            <textarea
              rows="3"
              value={advice}
              onChange={(e) => setAdvice(e.target.value)}
              className="w-full border rounded-xl p-3"
              placeholder="Lifestyle and medical advice"
            />
          </div>

          {/* Follow Up */}

          <div>
            <label className="block font-medium mb-2">
              Follow Up
            </label>

            <input
              type="text"
              value={followUp}
              onChange={(e) => setFollowUp(e.target.value)}
              className="w-full border rounded-xl p-3"
              placeholder="Example: Review after 30 days"
            />
          </div>

          {/* Error Message */}

          {error && (
            <div className="bg-red-100 border border-red-300 text-red-700 rounded-xl p-3">
              {error}
            </div>
          )}

          {/* Success Message */}

          {success && (
            <div className="bg-green-100 border border-green-300 text-green-700 rounded-xl p-3">
              {success}
            </div>
          )}

          {/* Save Button */}

          <button
            onClick={handleSaveNotes}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition disabled:opacity-60"
          >
            {loading ? "Saving..." : "Save Notes"}
          </button>

        </div>

      </div>
    </div>
  );
}

export default DoctorNotes;