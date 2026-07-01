import { Link, useLocation, Navigate } from "react-router-dom";
import ResultCard from "../components/ResultCard";

function ResultsPage() {
  const location = useLocation();

  const result = location.state?.result;

  // If user directly opens /results without a prediction
  if (!result) {
    return <Navigate to="/diagnose" replace />;
  }

  return (
    <div className="min-h-screen bg-gray-100 py-10">

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
  );
}

export default ResultsPage;