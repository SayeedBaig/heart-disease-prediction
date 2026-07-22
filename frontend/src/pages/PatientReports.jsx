import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function PatientReports() {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("patient") || "{}");
  const [loading, setLoading] = useState(Boolean(patient.patient_id));
  const [error, setError] = useState("");
  const [reports, setReports] = useState([]);

  useEffect(() => {
    if (!patient.patient_id) {
      return undefined;
    }

    const loadReports = async () => {
      try {
        const response = await api.get(
          `/patients/${patient.patient_id}/predictions`
        );
        setReports(response.data.predictions || []);
        setError("");
      } catch (requestError) {
        setError(
          requestError.response?.data?.detail || "Unable to load your reports."
        );
      } finally {
        setLoading(false);
      }
    };

    loadReports();
    const intervalId = window.setInterval(loadReports, 15000);
    return () => window.clearInterval(intervalId);
  }, [patient.patient_id]);

  const displayError = patient.patient_id
    ? error
    : "Please sign in to view your reports.";

  return (
    <div className="min-h-screen bg-slate-100 p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-800">
            My Health Reports
          </h1>
          <p className="text-slate-500 mt-2">
            View all your heart disease prediction reports.
          </p>
        </div>

        {/* Summary Card */}
        <div className="bg-white rounded-xl shadow p-6 mb-8">
          <h2 className="text-lg font-semibold text-slate-700">
            Total Reports
          </h2>

          <p className="text-4xl font-bold text-blue-600 mt-2">
            {reports.length}
          </p>
        </div>

        {/* Loading */}
        {loading && (
          <div className="bg-white rounded-xl shadow p-8 text-center">
            <p className="text-gray-500">Loading reports...</p>
          </div>
        )}

        {/* Error */}
        {!loading && displayError && (
          <div className="bg-red-100 border border-red-300 text-red-600 rounded-xl p-4 mb-6">
            {displayError}
          </div>
        )}

        {/* Empty State */}
        {!loading && !displayError && reports.length === 0 && (
          <div className="bg-white rounded-xl shadow p-12 text-center">
            <h2 className="text-xl font-semibold text-slate-700">
              No Reports Available
            </h2>

            <p className="text-slate-500 mt-2">
              Your prediction reports will appear here after your doctor completes a diagnosis.
            </p>
          </div>
        )}

        {/* Reports Table */}
        {!loading && !displayError && reports.length > 0 && (
          <div className="bg-white rounded-xl shadow overflow-hidden">

            <table className="min-w-full">

              <thead className="bg-slate-200">

                <tr>
                  <th className="px-6 py-4 text-left">Date</th>
                  <th className="px-6 py-4 text-left">Risk</th>
                  <th className="px-6 py-4 text-left">Risk %</th>
                  <th className="px-6 py-4 text-left">Clinical</th>
                  <th className="px-6 py-4 text-left">ECG</th>
                  <th className="px-6 py-4 text-left">Echo</th>
                  <th className="px-6 py-4 text-center">Action</th>
                </tr>

              </thead>

              <tbody>

                {reports.map((report) => (

                  <tr
                    key={report.prediction_id}
                    className="border-t hover:bg-slate-50"
                  >

                    <td className="px-6 py-4">
                      {new Date(report.created_at).toLocaleDateString()}
                    </td>

                    <td className="px-6 py-4">
                      {report.risk_level}
                    </td>

                    <td className="px-6 py-4">
                      {report.risk_percentage}%
                    </td>

                    <td className="px-6 py-4">
                      {report.clinical_level}
                    </td>

                    <td className="px-6 py-4">
                      {report.ecg_level}
                    </td>

                    <td className="px-6 py-4">
                      {report.echo_level}
                    </td>

                    <td className="px-6 py-4 text-center">
                      <button
                        onClick={() => {
                          localStorage.setItem(
                            "prediction_id", String(report.prediction_id)
                          );
                          navigate("/reports");
                        }}
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                      >
                        View
                      </button>
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>
        )}

      </div>
    </div>
  );
}

export default PatientReports;
