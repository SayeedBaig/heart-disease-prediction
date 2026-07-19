import { Link, Navigate, useLocation } from "react-router-dom";

import Navbar from "../components/Navbar";
import ResultCard from "../components/ResultCard";

function ResultsPage() {
  const location = useLocation();

  const result = location.state?.result ?? null;

  if (!result) {
    return <Navigate to="/diagnose" replace />;
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 py-12">

        <div className="max-w-6xl mx-auto px-6">

          {/* Page Title */}

          <div className="text-center mb-10">

            <h1 className="text-5xl font-bold text-slate-900">
              AI Prediction Result
            </h1>

            <p className="text-gray-600 mt-3 text-lg">
              Your cardiovascular risk assessment has been generated successfully.
            </p>

          </div>

          {/* Result */}

          <ResultCard result={result} />

          {/* Action Buttons */}

          <div className="flex flex-wrap justify-center gap-5 mt-10">

            <Link
              to="/reports"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl shadow-lg transition"
            >
              📄 View Reports
            </Link>

            <Link
              to="/register"
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-xl shadow-lg transition"
            >
              🔄 New Diagnosis
            </Link>

            <Link
              to="/"
              className="border border-slate-300 hover:bg-slate-100 text-slate-700 px-8 py-3 rounded-xl shadow-sm transition"
            >
              🏠 Back Home
            </Link>

          </div>

        </div>

      </div>
    </>
  );
}

export default ResultsPage;