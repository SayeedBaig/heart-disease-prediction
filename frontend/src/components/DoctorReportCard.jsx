
import { useState } from "react";
import { FileText, Download, Mail, Activity, ShieldCheck, CheckCircle2 } from "lucide-react";

function DoctorReportCard({ report }) {
  const [emailSent, setEmailSent] = useState(false);
  const [emailLoading, setEmailLoading] = useState(false);

  if (!report) {
    return (
      <div className="cardio-card p-8 text-center">
        <p className="caption-small text-[var(--accent-melanzane)] font-semibold animate-pulse">
          Loading Doctor Report...
        </p>
      </div>
    );
  }

  const predictionId = localStorage.getItem("prediction_id");

  const handleDownloadPDF = () => {
    window.open(
      `http://localhost:8000/reports/${predictionId}/doctor/pdf`,
      "_blank"
    );
  };

  const handleEmailReport = async () => {
    setEmailLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/reports/${predictionId}/doctor/email`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to send email");
      }

      const data = await response.json();
      setEmailSent(true);
      alert(data.message || "Doctor report emailed successfully!");
    } catch (error) {
      console.error(error);
      alert("Failed to send doctor report email.");
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
              <FileText size={18} />
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)]">
              Doctor Clinical Report
            </h1>
          </div>
          <p className="caption-small mt-1">
            Comprehensive AI-generated multimodal medical assessment
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

      {/* Overall AI Prediction Banner */}
      <div className="cardio-card p-6 md:p-8 bg-gradient-to-r from-[var(--accent-melanzane)] via-[#541243] to-[var(--accent-melanzane)] text-white rounded-2xl shadow-md border-0">
        <h2 className="caption-small text-white/80 uppercase font-bold tracking-wider mb-6">
          Overall Multimodal AI Prediction
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div>
            <p className="caption-small text-white/70 uppercase">Risk Level</p>
            <h3 className="text-3xl md:text-4xl font-extrabold mt-1 text-white">
              {report.final_prediction?.final_level || "N/A"}
            </h3>
          </div>

          <div>
            <p className="caption-small text-white/70 uppercase">Risk Score</p>
            <h3 className="text-3xl md:text-4xl font-extrabold mt-1 text-white">
              {report.final_prediction?.risk_percentage || "N/A"}%
            </h3>
          </div>

          <div>
            <p className="caption-small text-white/70 uppercase">Recommendation</p>
            <h3 className="text-xl md:text-2xl font-bold mt-1 text-white">
              Routine Check-up
            </h3>
          </div>
        </div>
      </div>

      {/* Analysis Results */}
      <div>
        <h2 className="section-title text-sm font-bold text-[var(--text-primary)] mb-4 flex items-center gap-2">
          <Activity size={18} className="text-[var(--accent-melanzane)]" />
          Multimodal Analysis Breakdown
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Clinical */}
          <div className="cardio-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">Clinical Vitals</h3>
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] border border-[var(--accent-melanzane-border)]">
                  {report.clinical_analysis?.level || "N/A"}
                </span>
              </div>

              <div className="mb-2 flex justify-between text-xs caption-small">
                <span>Confidence</span>
                <span className="font-bold text-[var(--text-primary)]">
                  {((report.clinical_analysis?.score || 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="w-full bg-[var(--bg-secondary)] rounded-full h-2 overflow-hidden mb-4">
                <div
                  className="bg-[var(--accent-melanzane)] h-2 rounded-full transition-all"
                  style={{ width: `${(report.clinical_analysis?.score || 0) * 100}%` }}
                />
              </div>

              <p className="body-regular text-xs leading-relaxed">
                {report.clinical_analysis?.reason || "No clinical explanation available."}
              </p>
            </div>
          </div>

          {/* ECG */}
          <div className="cardio-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">ECG Waveform</h3>
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                  {report.ecg_analysis?.level || "N/A"}
                </span>
              </div>

              <div className="mb-2 flex justify-between text-xs caption-small">
                <span>Confidence</span>
                <span className="font-bold text-[var(--text-primary)]">
                  {((report.ecg_analysis?.score || 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="w-full bg-[var(--bg-secondary)] rounded-full h-2 overflow-hidden mb-4">
                <div
                  className="bg-emerald-500 h-2 rounded-full transition-all"
                  style={{ width: `${(report.ecg_analysis?.score || 0) * 100}%` }}
                />
              </div>

              <p className="body-regular text-xs leading-relaxed">
                {report.ecg_analysis?.reason || "No ECG explanation available."}
              </p>
            </div>
          </div>

          {/* Echo */}
          <div className="cardio-card p-6 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-bold text-[var(--text-primary)]">Echocardiogram</h3>
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-purple-500/10 text-purple-500 border border-purple-500/20">
                  {report.echo_analysis?.level || "N/A"}
                </span>
              </div>

              <div className="mb-2 flex justify-between text-xs caption-small">
                <span>Confidence</span>
                <span className="font-bold text-[var(--text-primary)]">
                  {((report.echo_analysis?.score || 0) * 100).toFixed(1)}%
                </span>
              </div>

              <div className="w-full bg-[var(--bg-secondary)] rounded-full h-2 overflow-hidden mb-4">
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all"
                  style={{ width: `${(report.echo_analysis?.score || 0) * 100}%` }}
                />
              </div>

              <p className="body-regular text-xs leading-relaxed">
                {report.echo_analysis?.reason || "No Echo explanation available."}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* AI Recommendation */}
      <div className="cardio-card p-6 bg-[var(--accent-melanzane-light)] border border-[var(--accent-melanzane-border)]">
        <h2 className="section-title text-sm font-bold text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
          <ShieldCheck size={18} />
          AI Diagnostic Guidance
        </h2>

        <p className="body-regular text-xs leading-relaxed text-[var(--text-primary)] font-medium">
          {report.ai_recommendation?.explanation || "No AI recommendation available."}
        </p>

        {report.ai_recommendation?.details?.length > 0 && (
          <ul className="mt-4 space-y-2">
            {report.ai_recommendation.details.map((item, index) => (
              <li key={index} className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)] shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Medical Explanation */}
      <div className="cardio-card p-6 space-y-4">
        <h2 className="section-title text-sm font-bold text-[var(--text-primary)]">
          Medical & Clinical Explanation
        </h2>

        <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs text-[var(--text-secondary)] space-y-2">
          <h3 className="font-bold text-[var(--text-primary)]">Clinical Summary</h3>
          <p>{report.medical_explanation?.summary || "N/A"}</p>
        </div>

        <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs text-[var(--text-secondary)] space-y-2">
          <h3 className="font-bold text-[var(--text-primary)]">Detailed Diagnostic Reasoning</h3>
          <p>{report.medical_explanation?.details || "N/A"}</p>
        </div>
      </div>

      {/* Digital Twin */}
      <div className="cardio-card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="section-title text-sm font-bold text-[var(--text-primary)]">
            Digital Twin Simulations
          </h2>
          <span className="caption-small">
            Baseline Risk: <strong>{report.digital_twin?.baseline_risk != null ? `${(report.digital_twin.baseline_risk * 100).toFixed(1)}%` : "N/A"}</strong>
          </span>
        </div>

        {report.digital_twin?.simulations?.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-[var(--bg-secondary)] border-b border-[var(--border-color)] text-[var(--text-muted)] uppercase font-semibold">
                  <th className="p-3">Scenario</th>
                  <th className="p-3">Risk Level</th>
                  <th className="p-3">Improvement</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-color)]">
                {report.digital_twin.simulations.map((sim, index) => (
                  <tr key={index} className="hover:bg-[var(--card-hover)]">
                    <td className="p-3 font-semibold text-[var(--text-primary)]">{sim.scenario}</td>
                    <td className="p-3 text-[var(--text-secondary)]">{sim.risk}</td>
                    <td className="p-3 font-bold text-emerald-500">{sim.change}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs text-[var(--text-muted)] text-center">
            No specific Digital Twin scenario simulations logged for this report.
          </div>
        )}
      </div>
    </div>
  );
}

export default DoctorReportCard;