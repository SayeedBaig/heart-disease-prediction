import { Link, Navigate, useLocation } from "react-router-dom";

import Navbar from "../components/Navbar";
import ResultCard from "../components/ResultCard";

function ResultsPage() {
  const location = useLocation();

  const result = location.state?.result ?? null;

  // Prevent direct access without prediction
  if (!result) {
    return <Navigate to="/diagnose" replace />;
  }

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 py-10">

        <div className="max-w-6xl mx-auto px-6">

          <ResultCard result={result} />

          <div className="flex justify-center mt-10">
            <Link
              to="/reports"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg shadow-lg transition"
            >
              📄 View Reports
            </Link>
          </div>

        </div>

      </div>
    </>
  );
}

export default ResultsPage;