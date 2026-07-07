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

  useEffect(() => {
    const loadReports = async () => {
      try {
        const [doctor, patient] = await Promise.all([
          api.get("/reports/doctor"),
          api.get("/reports/patient"),
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

          <h1 className="text-5xl font-bold text-blue-800 mb-10 text-center">
            Reports
          </h1>

          <DoctorReportCard report={doctorReport} />

          <div className="my-8" />

          <PatientReportCard report={patientReport} />

        </div>
      </div>
    </>
  );
}

export default Reports;