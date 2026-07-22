import { Link, Navigate, useLocation } from "react-router-dom";

import Navbar from "../components/Navbar";
import ResultCard from "../components/ResultCard";

function ResultsPage() {
  const location = useLocation();

  const result = location.state?.result ?? JSON.parse(
    localStorage.getItem("latest_prediction_result") || "null"
  );

  if (!result) {
    return <Navigate to="/diagnose" replace />;
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-cyan-100 py-12">

       <div className="max-w-7xl mx-auto px-6">

          {/* Page Title */}

          <div className="text-center mb-12">

  <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-green-100 mb-5">
  </div>

  <h1 className="text-5xl font-extrabold text-slate-900">
    Heart Disease Prediction
  </h1>

  <p className="text-gray-600 mt-4 text-xl">
    Your AI-powered cardiovascular assessment has been generated successfully.
  </p>

</div>

          {/* Result */}

         <div className="bg-white rounded-3xl shadow-xl p-6">
  <ResultCard result={result} />
</div>

          {/* Action Buttons */}

         <div className="grid md:grid-cols-3 gap-6 mt-12">
            <Link
              to="/reports"
             className="bg-blue-600 hover:bg-blue-700 text-white text-center py-4 rounded-2xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
              View Reports
            </Link>

            <Link
              to="/appointment-management"
              className="bg-green-600 hover:bg-green-700 text-white text-center py-4 rounded-2xl font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
            >
              New Diagnosis
            </Link>

            <Link
              to="/"
              className="border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-center py-4 rounded-2xl font-semibold shadow-md hover:shadow-lg transition-all duration-300"    >
              Back Home
            </Link>

          </div>

        </div>
        <div className="bg-white rounded-3xl shadow-xl p-8 mt-12">

  <h2 className="text-3xl font-bold text-slate-800 mb-6">
    Recommended Next Steps
  </h2>

  <div className="grid md:grid-cols-2 gap-5">

    <div className="bg-green-50 border-l-4 border-green-500 rounded-2xl p-5">
      Maintain regular physical activity.
    </div>

    <div className="bg-blue-50 border-l-4 border-blue-500 rounded-2xl p-5">
      Follow a balanced and heart-healthy diet.
    </div>

    <div className="bg-yellow-50 border-l-4 border-yellow-500 rounded-2xl p-5">
      Schedule regular health check-ups.
    </div>

    <div className="bg-purple-50 border-l-4 border-purple-500 rounded-2xl p-5">
      Review your detailed Doctor and Patient reports.
    </div>

  </div>

</div>

      </div>
    </>
  );
}

export default ResultsPage;
