import { useEffect, useState } from "react";
import api from "../services/api";

import Navbar from "../components/Navbar";
import DoctorReportCard from "../components/DoctorReportCard";
import PatientReportCard from "../components/PatientReportCard";

function Reports() {
  const [doctorReport, setDoctorReport] = useState(null);
  const [patientReport, setPatientReport] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // NEW
  const [activeTab, setActiveTab] = useState("doctor");

  useEffect(() => {
    const loadReports = async () => {
      try {
        const predictionId = localStorage.getItem("prediction_id");

if (!predictionId) {
  setError("Prediction ID not found.");
  setLoading(false);
  return;
}

const [doctor, patient] = await Promise.all([
  api.get(`/reports/${predictionId}/doctor`),

  api.get(`/reports/${predictionId}/patient`),
]);

        setDoctorReport(doctor.data);
        setPatientReport(patient.data);

        
      } catch (err) {
        console.error(err);
        setError("Unable to load reports. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  if (loading) {
    return (
      <>
        <Navbar />

        <div className="min-h-screen flex items-center justify-center bg-gray-100">
          <h2 className="text-3xl font-semibold text-blue-700 animate-pulse">
            Loading Reports...
          </h2>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Navbar />

        <div className="min-h-screen flex items-center justify-center bg-gray-100">
          <h2 className="text-xl text-red-600 font-semibold">
            {error}
          </h2>
        </div>
      </>
    );
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 py-10">
        <div className="max-w-7xl mx-auto px-6">

          <div className="text-center mb-10">
  <h1 className="text-5xl font-extrabold text-slate-900">
    AI Diagnosis Reports
  </h1>

  <p className="mt-3 text-lg text-slate-500">
    View and manage detailed reports generated from the AI prediction system.
  </p>
</div>

<div className="flex flex-wrap justify-center gap-4 mb-8">
  <button
    onClick={() => window.print()}
    className="px-5 py-3 bg-slate-800 text-white rounded-xl hover:bg-slate-900 transition"
  >
    🖨 Print Report
  </button>

  <button
    className="px-5 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition"
  >
    📄 Download PDF
  </button>

  <button
    className="px-5 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition"
  >
    📧 Email Report
  </button>

  <button
    onClick={() => window.location.href = "/diagnose"}
    className="px-5 py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition"
  >
    🔄 New Diagnosis
  </button>
</div>

          {/* Tabs */}

          <div className="flex justify-center gap-4 mb-8">

            <button
              onClick={() => setActiveTab("doctor")}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                activeTab === "doctor"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "bg-white text-gray-700 border hover:bg-blue-50"
              }`}
            >
              👨‍⚕️ Doctor Report
            </button>

            <button
              onClick={() => setActiveTab("patient")}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                activeTab === "patient"
                  ? "bg-green-600 text-white shadow-lg"
                  : "bg-white text-gray-700 border hover:bg-green-50"
              }`}
            >
              🩺 Patient Report
            </button>

          </div>

          {/* Selected Report */}

          {activeTab === "doctor" ? (
            <DoctorReportCard report={doctorReport} />
          ) : (
            <PatientReportCard report={patientReport} />
          )}

        </div>
      </div>
    </>
  );
}

export default Reports;