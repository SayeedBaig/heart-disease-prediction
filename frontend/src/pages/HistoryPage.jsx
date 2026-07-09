import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import api from "../services/api";

function HistoryPage() {
  const patientId = localStorage.getItem("patient_id");

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!patientId) {
        setLoading(false);
        return;
      }

      try {
        const response = await api.get(
          `/patients/${patientId}/predictions`
        );

        setHistory(response.data.predictions || []);
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [patientId]);

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-slate-100 p-10">
        <div className="max-w-5xl mx-auto">

          <h1 className="text-4xl font-bold text-blue-900 mb-8">
            Prediction History
          </h1>

          {loading ? (
            <div className="bg-white rounded-xl shadow p-8">
              <p className="text-lg">Loading...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="bg-white rounded-xl shadow p-8">
              <p className="text-lg text-gray-600">
                No predictions available.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {history.map((item) => (
                <div
                  key={item.prediction_id}
                  className="bg-white rounded-xl shadow-lg p-6"
                >
                  <h2 className="text-2xl font-bold text-blue-700">
                    {item.risk_level} Risk
                  </h2>

                  <p className="mt-3">
                    <strong>Risk Percentage:</strong>{" "}
                    {item.risk_percentage}%
                  </p>

                  <p>
                    <strong>Clinical:</strong>{" "}
                    {item.clinical_level}
                  </p>

                  <p>
                    <strong>ECG:</strong>{" "}
                    {item.ecg_level}
                  </p>

                  <p>
                    <strong>Echo:</strong>{" "}
                    {item.echo_level}
                  </p>

                  <p className="text-gray-500 mt-4">
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
              ))}
            </div>
          )}

        </div>
      </div>
    </>
  );
}

export default HistoryPage;