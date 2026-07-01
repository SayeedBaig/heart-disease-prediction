import { useEffect, useState } from "react";
import api from "../services/api";

import DoctorReportCard from "../components/DoctorReportCard";
import PatientReportCard from "../components/PatientReportCard";

function Reports() {
  const [doctorReport, setDoctorReport] = useState(null);
  const [patientReport, setPatientReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadReports = async () => {
      try {
        const [doctor, patient] = await Promise.all([
          api.get("/reports/doctor"),
          api.get("/reports/patient"),
        ]);

        setDoctorReport(doctor.data);
        setPatientReport(patient.data);
      } catch (error) {
        console.error("Failed to load reports:", error);
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto mt-10">
        <h2 className="text-2xl font-bold">
          Loading Reports...
        </h2>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto py-10 px-6">
      <h1 className="text-5xl font-bold text-blue-800 mb-10">
        Reports
      </h1>

      <DoctorReportCard report={doctorReport} />

      <PatientReportCard report={patientReport} />
    </div>
  );
}

export default Reports;