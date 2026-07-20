import {
  HeartPulse,
  Activity,
  Stethoscope,

} from "lucide-react";

import RiskBadge from "./RiskBadge";
import ConfidenceBar from "./ConfidenceBar";
import ExplanationCard from "./ExplanationCard";
import TwinSimulationCard from "./TwinSimulationCard";
import RecommendationCard from "./RecommendationCard";

function ResultCard({ result }) {
  const prediction = result.prediction;

  return (
    <div className="max-w-7xl mx-auto mt-10">

      {/* ================= HEADER ================= */}

      <div className="bg-white rounded-3xl border border-slate-200 shadow-xl p-8 mb-10">

        <div className="flex items-center gap-4 mb-8">

          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg">
            <HeartPulse className="text-white" size={34} />
          </div>

          <div>

            <h2 className="text-4xl font-bold text-slate-900">
              Heart Disease Prediction Report
            </h2>

            <p className="text-slate-500 mt-2">
              AI-powered cardiovascular risk assessment
            </p>

          </div>

        </div>

        {/* Summary Cards */}

        <div className="grid md:grid-cols-3 gap-6">

          <div className="rounded-2xl bg-blue-50 border border-blue-200 p-6">

            <p className="text-slate-500 mb-2">
              Overall Risk
            </p>

            <RiskBadge level={prediction.fusion.final_level} />

          </div>

          <div className="rounded-2xl bg-green-50 border border-green-200 p-6">

            <p className="text-slate-500 mb-2">
              Risk Percentage
            </p>

            <h3 className="text-4xl font-bold text-green-600">
              {prediction.fusion.risk_percentage}%
            </h3>

          </div>

          <div className="rounded-2xl bg-purple-50 border border-purple-200 p-6">

            <p className="text-slate-500 mb-2">
              AI Modules
            </p>

            <h3 className="text-2xl font-bold text-purple-700">
              Clinical + ECG + Echo
            </h3>

          </div>

        </div>

      </div>

      {/* ================= AI MODULES ================= */}

      <div className="grid lg:grid-cols-3 gap-8 mb-10">

        {/* Clinical */}

        <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-6 hover:shadow-xl transition">

          <div className="flex items-center gap-3 mb-5">

            <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">
              <Stethoscope className="text-blue-600" />
            </div>

            <h3 className="text-2xl font-bold">
              Clinical
            </h3>

          </div>

          <RiskBadge level={prediction.clinical.level} />

          <ConfidenceBar score={prediction.clinical.score} />

          <p className="mt-5 leading-8 text-slate-700">
            {prediction.clinical.reason}
          </p>

        </div>

        {/* ECG */}

        <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-6 hover:shadow-xl transition">

          <div className="flex items-center gap-3 mb-5">

            <div className="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center">
              <Activity className="text-red-600" />
            </div>

            <h3 className="text-2xl font-bold">
              ECG
            </h3>

          </div>

          <RiskBadge level={prediction.ecg.level} />

          <ConfidenceBar score={prediction.ecg.score} />

          <p className="mt-5 leading-8 text-slate-700">
            {prediction.ecg.reason}
          </p>

        </div>

        {/* Echo */}

        <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-6 hover:shadow-xl transition">

          <div className="flex items-center gap-3 mb-5">

            <div className="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center">
              <HeartPulse className="text-green-600" />
            </div>

            <h3 className="text-2xl font-bold">
              Echo
            </h3>

          </div>

          <RiskBadge level={prediction.echo.level} />

          <ConfidenceBar score={prediction.echo.score} />

          <p className="mt-5 leading-8 text-slate-700">
            {prediction.echo.reason}
          </p>

        </div>

      </div>

      {/* ================= AI RECOMMENDATION ================= */}

      <RecommendationCard prediction={prediction} />

      {/* ================= MEDICAL EXPLANATION ================= */}

      <ExplanationCard explanation={result.explanation} />

      {/* ================= DIGITAL TWIN ================= */}

      <TwinSimulationCard digitalTwin={result.digital_twin} />

    </div>
  );
}

export default ResultCard;