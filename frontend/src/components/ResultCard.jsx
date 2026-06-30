import RiskBadge from "./RiskBadge";
import ConfidenceBar from "./ConfidenceBar";
import ExplanationCard from "./ExplanationCard";
import TwinSimulationCard from "./TwinSimulationCard";
import RecommendationCard from "./RecommendationCard";

function ResultCard({ result }) {
  const prediction = result.prediction;

  return (
    <div className="max-w-6xl mx-auto mt-10 bg-white shadow-lg rounded-xl p-6 md:p-8">

      {/* Title */}

      <h2 className="text-4xl font-bold text-blue-900 mb-8">
        Heart Disease Prediction Report
      </h2>

      {/* ================= Final Prediction ================= */}

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mb-8 shadow-sm">

        <h3 className="text-2xl font-bold text-blue-800 mb-6">
          🫀 Final Prediction
        </h3>

        <div className="flex flex-col md:flex-row justify-between items-center">

          <div className="flex flex-col justify-center">
            <p className="text-lg font-semibold mb-3">
              Risk Level
            </p>

            <RiskBadge level={prediction.fusion.final_level} />
          </div>

          <div className="text-center mt-6 md:mt-0">
            <p className="text-lg font-semibold">
              Risk Percentage
            </p>

            <p className="text-5xl font-extrabold text-blue-700">
              {prediction.fusion.risk_percentage}%
            </p>
          </div>

        </div>

      </div>

      {/* ================= Clinical ================= */}

      <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">

        <h3 className="text-2xl font-bold text-gray-800 mb-4">
          🩺 Clinical Analysis
        </h3>

        <p className="flex items-center gap-2 mb-4">
          <strong>Risk:</strong>
          <RiskBadge level={prediction.clinical.level} />
        </p>

        <ConfidenceBar score={prediction.clinical.score} />

        <p className="mt-4 leading-8 text-gray-700">
          {prediction.clinical.reason}
        </p>

      </div>

      {/* ================= ECG ================= */}

      <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">

        <h3 className="text-2xl font-bold text-gray-800 mb-4">
          📈 ECG Analysis
        </h3>

        <p className="flex items-center gap-2 mb-4">
          <strong>Risk:</strong>
          <RiskBadge level={prediction.ecg.level} />
        </p>

        <ConfidenceBar score={prediction.ecg.score} />

        <p className="mt-4 leading-8 text-gray-700">
          {prediction.ecg.reason}
        </p>

      </div>

      {/* ================= Echo ================= */}

      <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">

        <h3 className="text-2xl font-bold text-gray-800 mb-4">
          🫁 Echo Analysis
        </h3>

        <p className="flex items-center gap-2 mb-4">
          <strong>Risk:</strong>
          <RiskBadge level={prediction.echo.level} />
        </p>

        <ConfidenceBar score={prediction.echo.score} />

        <p className="mt-4 leading-8 text-gray-700">
          {prediction.echo.reason}
        </p>

      </div>

      <RecommendationCard prediction={prediction} />

      

      <ExplanationCard explanation={result.explanation} />

      <TwinSimulationCard digitalTwin={result.digital_twin} />
    </div>
  );
}

export default ResultCard;