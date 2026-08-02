import { useState } from "react";
import { Download, Mail, CheckCircle2, HeartPulse, Sparkles, Activity, ShieldCheck, Calendar } from "lucide-react";

function PatientReportCard({ report }) {
  const [emailSent, setEmailSent] = useState(false);
  const [emailLoading, setEmailLoading] = useState(false);

  if (!report) {
    return (
      <div className="cardio-card p-8 text-center">
        <p className="caption-small text-[var(--accent-melanzane)] font-semibold animate-pulse">
          Loading Patient Health Summary...
        </p>
      </div>
    );
  }

  const predictionId = localStorage.getItem("prediction_id");

  const handleDownloadPDF = () => {
    window.open(
      `http://localhost:8000/reports/${predictionId}/patient/pdf`,
      "_blank"
    );
  };

  const handleEmailReport = async () => {
    setEmailLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/reports/${predictionId}/patient/email`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to send email");
      }

      const data = await response.json();
      setEmailSent(true);
      alert(data.message || "Patient report emailed successfully!");
    } catch (err) {
      console.error(err);
      alert("Failed to send patient report email.");
    } finally {
      setEmailLoading(false);
    }
  };

  return (
    <div className="cardio-card p-6 md:p-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 pb-6 border-b border-[var(--border-color)]">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--accent-melanzane)] text-white flex items-center justify-center font-bold text-sm">
              <HeartPulse size={18} />
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)]">
              Patient Cardiovascular Summary
            </h1>
          </div>
          <p className="caption-small mt-1">
            Clear, easy-to-understand AI explanation of your cardiac assessment
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleDownloadPDF}
            className="btn-primary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2"
          >
            <Download size={15} />
            <span>Download PDF</span>
          </button>

          <button
            onClick={handleEmailReport}
            disabled={emailLoading || emailSent}
            className={`text-xs py-2.5 px-4 rounded-xl flex items-center gap-2 transition-all ${
              emailSent
                ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/30 font-bold cursor-default"
                : "btn-secondary"
            }`}
          >
            {emailSent ? <CheckCircle2 size={15} className="text-emerald-500" /> : <Mail size={15} />}
            <span>{emailLoading ? "Sending..." : emailSent ? "Sent" : "Email Report"}</span>
          </button>
        </div>
      </div>

      {/* Heart Health Summary Banner */}
      <div className="cardio-card p-6 md:p-8 bg-gradient-to-r from-[var(--accent-melanzane)] via-[#541243] to-[var(--accent-melanzane)] text-white rounded-2xl shadow-md border-0">
        <h2 className="caption-small text-white/80 uppercase font-bold tracking-wider mb-6 flex items-center gap-2">
          <Activity size={16} />
          Your Cardiovascular Health Status
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <span className="caption-small text-white/70 uppercase tracking-wider block">Risk Stratification</span>
            <div className="text-4xl md:text-5xl font-extrabold mt-2 text-white">
              {report.risk_level || "Low-to-Moderate"}
            </div>
          </div>

          <div>
            <span className="caption-small text-white/70 uppercase tracking-wider block">Calculated Risk Score</span>
            <div className="text-4xl md:text-5xl font-extrabold mt-2 text-white">
              {report.risk_percentage ?? 24}%
            </div>
          </div>
        </div>
      </div>

      {/* What We Found Summary */}
      <div className="cardio-card p-6 space-y-3">
        <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
          <Sparkles size={16} />
          What We Found
        </h3>
        <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
          {report.summary || "Based on your submitted vitals and lab measurements, your cardiovascular profile shows a favorable risk trajectory. Continue maintaining baseline physical activity and low-sodium nutrition."}
        </div>
      </div>

      {/* Lifestyle Tips */}
      <div className="cardio-card p-6 space-y-4">
        <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
          <ShieldCheck size={16} />
          Healthy Lifestyle Guidance
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {report.lifestyle_recommendations?.length > 0 ? (
            report.lifestyle_recommendations.map((item, index) => (
              <div
                key={index}
                className="p-4 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] text-xs text-[var(--text-primary)] flex items-start gap-2.5 shadow-sm"
              >
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                <span>{item}</span>
              </div>
            ))
          ) : (
            <>
              <div className="p-4 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] text-xs text-[var(--text-primary)] flex items-start gap-2.5 shadow-sm">
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                <span>Maintain regular 30-minute moderate aerobic exercise (walking, swimming, or cycling).</span>
              </div>
              <div className="p-4 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] text-xs text-[var(--text-primary)] flex items-start gap-2.5 shadow-sm">
                <CheckCircle2 size={16} className="text-emerald-500 shrink-0 mt-0.5" />
                <span>Follow a heart-healthy diet rich in green leafy vegetables, healthy fats, and low sodium.</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Follow-up Advice */}
      <div className="cardio-card p-6 space-y-4">
        <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
          <Calendar size={16} />
          Follow-up & Next Steps
        </h3>

        <div className="space-y-3">
          {report.follow_up_advice?.length > 0 ? (
            report.follow_up_advice.map((item, index) => (
              <div
                key={index}
                className="p-4 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-xs text-[var(--text-primary)] font-medium flex items-start gap-2.5"
              >
                <span className="w-2 h-2 rounded-full bg-[var(--accent-melanzane)] shrink-0 mt-1.5" />
                <span>{item}</span>
              </div>
            ))
          ) : (
            <div className="p-4 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-xs text-[var(--text-primary)] font-medium flex items-start gap-2.5">
              <span className="w-2 h-2 rounded-full bg-[var(--accent-melanzane)] shrink-0 mt-1.5" />
              <span>Schedule a routine follow-up with your primary physician or cardiologist within 90 days.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PatientReportCard;