import { Link, useLocation } from "react-router-dom";
import { FileText, Layers, Calendar, ShieldCheck } from "lucide-react";
import Navbar from "../components/Navbar";

export default function ResultsPage() {
  const location = useLocation();

  const prediction = location.state?.result || JSON.parse(
    localStorage.getItem("cardio-prediction") || "null"
  ) || {
    risk_percentage: 28,
    final_level: "Moderate",
    confidence: 89,
    recommendations: [
      "Schedule follow-up assessment with a cardiologist",
      "Maintain active lifestyle and monitor blood pressure weekly",
      "Follow low-sodium diet and reduce saturated fat intake"
    ]
  };

  const riskPct = prediction?.fusion?.risk_percentage || prediction?.risk_percentage || 28;
  const level = prediction?.fusion?.final_level || prediction?.final_level || "Moderate";
  const confidence = prediction?.fusion?.confidence_percentage || prediction?.confidence || 88;
  const recs = prediction?.fusion?.lifestyle_recommendations || prediction?.recommendations || [
    "Schedule follow-up assessment with a cardiologist",
    "Maintain active lifestyle and monitor blood pressure weekly",
    "Follow low-sodium diet and reduce saturated fat intake"
  ];

  const levelColor = level === "Low" ? "#10b981" : level === "Moderate" ? "#f59e0b" : "#ef4444";

  return (
    <>
      <div>
        <div className="text-center mb-10">
          <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
            AI Screening Result
          </span>
          <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
            Cardiovascular Risk Assessment
          </h1>
        </div>

        {/* Minimal Prediction Summary Card */}
        <div className="cardio-card p-8 mb-8 text-center">
          <div className="inline-flex items-center justify-center w-28 h-28 rounded-full mb-4 relative"
            style={{ backgroundColor: `${levelColor}12`, border: `2px solid ${levelColor}30` }}>
            <span className="text-4xl font-extrabold text-[var(--text-primary)]">
              {riskPct}%
            </span>
          </div>

          <div className="mb-6">
            <span className="px-4 py-1 rounded-full text-xs font-bold uppercase tracking-wider"
              style={{ backgroundColor: `${levelColor}18`, color: levelColor }}>
              {level} Risk Profile
            </span>
          </div>

          <div className="flex items-center justify-center gap-2 text-xs text-[var(--text-muted)] font-medium mb-6">
            <ShieldCheck size={16} className="text-[var(--accent-melanzane)]" />
            AI Confidence Score: <strong className="text-[var(--text-primary)]">{confidence}%</strong>
          </div>

          {/* Recommendations List */}
          <div className="border-t border-[var(--border-color)] pt-6 text-left">
            <h3 className="section-title text-sm font-semibold text-[var(--text-primary)] mb-3">
              Clinical & Lifestyle Recommendations
            </h3>
            <ul className="space-y-2.5 text-xs text-[var(--text-secondary)]">
              {recs.map((item, idx) => (
                <li key={idx} className="flex items-start gap-2.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)] mt-1.5 flex-shrink-0" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* 3 Explicit Action Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Button 1: View Report */}
          <Link
            to="/patient/reports"
            className="btn-primary py-3.5 px-4 text-xs font-semibold rounded-xl text-center flex items-center justify-center gap-2"
          >
            <FileText size={16} />
            View Report
          </Link>

          {/* Button 2: Digital Twin */}
          <Link
            to="/patient/digital-twin"
            className="btn-secondary py-3.5 px-4 text-xs font-semibold rounded-xl text-center flex items-center justify-center gap-2"
          >
            <Layers size={16} />
            Digital Twin
          </Link>

          {/* Button 3: Book Appointment */}
          <Link
            to="/patient/appointments"
            className="btn-secondary py-3.5 px-4 text-xs font-semibold rounded-xl text-center flex items-center justify-center gap-2"
          >
            <Calendar size={16} />
            Book Appointment
          </Link>
        </div>
      </div>
    </>
  );
}
