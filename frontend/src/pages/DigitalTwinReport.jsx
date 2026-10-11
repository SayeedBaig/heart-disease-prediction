import { useMemo, useState } from "react";
import {
  Download,
  FileText,
  Sparkles,
  Activity,
  User,
  HeartPulse,
  BarChart3,
  CheckCircle2,
  Calendar,
  ShieldCheck,
  AlertTriangle,
  Info,
} from "lucide-react";
import Navbar from "../components/Navbar";

export default function DigitalTwinReport({ onBack }) {
  const [showSummary, setShowSummary] = useState(false);

  const report = useMemo(() => {
    try {
      return JSON.parse(localStorage.getItem("cardio-digital-twin-report") || "null");
    } catch {
      return null;
    }
  }, []);

  const clinicalAlerts = useMemo(() => {
    if (Array.isArray(report?.clinical_alerts) && report.clinical_alerts.length > 0) {
      return report.clinical_alerts;
    }
    const bpMetric = (report?.metrics || []).find(([k]) => String(k).toLowerCase().includes("blood pressure"));
    if (bpMetric && bpMetric[1]) {
      const match = String(bpMetric[1]).match(/(\d+)\s*\/\s*(\d+)/);
      if (match) {
        const sys = Number(match[1]);
        const dia = Number(match[2]);
        const fallback = [];
        if (sys < 90) {
          fallback.push({
            alert_id: "LOW_SYSTOLIC_BP",
            parameter: "Systolic Blood Pressure",
            measured_value: sys,
            unit: "mmHg",
            category: "Blood Pressure",
            severity: "warning",
            threshold_rule: "Systolic BP < 90 mmHg (adult screening reference)",
            explanation: `This simulated scenario contains a systolic blood pressure of ${sys} mmHg, which is below the configured adult screening threshold of 90 mmHg.`,
            recommendation: "Do not interpret this hypothetical reading as an actual clinical finding. If a real measurement returns a similar value, consult a physician. Do not alter medication without medical advice.",
            is_hypothetical: true,
          });
        }
        if (dia < 60) {
          fallback.push({
            alert_id: "LOW_DIASTOLIC_BP",
            parameter: "Diastolic Blood Pressure",
            measured_value: dia,
            unit: "mmHg",
            category: "Blood Pressure",
            severity: "warning",
            threshold_rule: "Diastolic BP < 60 mmHg (adult screening reference)",
            explanation: `This simulated scenario contains a diastolic blood pressure of ${dia} mmHg, which is below the configured adult screening threshold of 60 mmHg.`,
            recommendation: "Do not interpret this hypothetical reading as an actual clinical finding. Consult a physician if persistent.",
            is_hypothetical: true,
          });
        }
        if (sys >= 140) {
          fallback.push({
            alert_id: "HIGH_SYSTOLIC_BP",
            parameter: "Systolic Blood Pressure",
            measured_value: sys,
            unit: "mmHg",
            category: "Blood Pressure",
            severity: "warning",
            threshold_rule: "Systolic BP ≥ 140 mmHg (adult screening reference)",
            explanation: `This simulated scenario contains an elevated systolic blood pressure of ${sys} mmHg (≥ 140 mmHg).`,
            recommendation: "Discuss this simulated reading with a healthcare professional.",
            is_hypothetical: true,
          });
        }
        if (dia >= 90) {
          fallback.push({
            alert_id: "HIGH_DIASTOLIC_BP",
            parameter: "Diastolic Blood Pressure",
            measured_value: dia,
            unit: "mmHg",
            category: "Blood Pressure",
            severity: "warning",
            threshold_rule: "Diastolic BP ≥ 90 mmHg (adult screening reference)",
            explanation: `This simulated scenario contains an elevated diastolic blood pressure of ${dia} mmHg (≥ 90 mmHg).`,
            recommendation: "Discuss this simulated reading with a healthcare professional.",
            is_hypothetical: true,
          });
        }
        return fallback;
      }
    }
    return [];
  }, [report]);

  const handleDownload = () => window.print();

  if (!report) {
    return (
      <div>
        <main className="cardio-container flex flex-1 items-center justify-center">
          <div className="cardio-card max-w-md p-8 text-center space-y-4">
            <div className="w-14 h-14 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mx-auto">
              <FileText size={28} />
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)]">No Simulation Report</h1>
            <p className="body-regular text-xs leading-relaxed">
              Run the Digital Twin simulation suite and click <strong>"Generate Report"</strong> to create an official simulation summary.
            </p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <>
      <div>

        {/* ── Header ─────────────────────────────────────────────── */}
        <div className="no-print flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-6 border-b border-[var(--border-color)]">
          <div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-[var(--accent-melanzane)] text-white flex items-center justify-center">
                <Activity size={18} />
              </div>
              <h1 className="h2-semibold text-[var(--text-primary)]">
                Digital Twin Simulation Report
              </h1>
            </div>
            <p className="caption-small mt-1">
              AI-generated longitudinal cardiovascular risk projection from parameter simulation
            </p>
          </div>

          <div className="flex items-center gap-3 no-print">
            <button
              onClick={() => setShowSummary((v) => !v)}
              className="btn-secondary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2"
            >
              <Sparkles size={14} />
              <span>{showSummary ? "Hide Summary" : "AI Summary"}</span>
            </button>
            <button
              onClick={handleDownload}
              className="btn-primary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2"
            >
              <Download size={14} />
              <span>Download Report</span>
            </button>
          </div>
        </div>

        {/* ── AI Executive Summary (toggle) ─────────────────────── */}
        {showSummary && (
          <div className="no-print cardio-card p-5 border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
              <Sparkles size={14} />
              Executive AI Summary
            </h3>
            <p className="text-xs leading-relaxed text-[var(--text-primary)]">{report.summary}</p>
          </div>
        )}

        {/* ── Risk Banner ────────────────────────────────────────── */}
        <div className="cardio-card p-6 md:p-8 bg-gradient-to-r from-[var(--accent-melanzane)] via-[#541243] to-[var(--accent-melanzane)] text-white rounded-2xl shadow-md border-0">
          <h2 className="caption-small text-white/80 uppercase font-bold tracking-wider mb-6 flex items-center gap-2">
            <HeartPulse size={16} />
            Simulated Cardiovascular Risk Profile
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <span className="caption-small text-white/70 uppercase tracking-wider block">Risk Stratification</span>
              <div className="text-3xl md:text-4xl font-extrabold mt-2 text-white">
                {report.risk?.level ?? "N/A"}
              </div>
            </div>

            <div>
              <span className="caption-small text-white/70 uppercase tracking-wider block">Projected 10-Year Risk</span>
              <div className="text-3xl md:text-4xl font-extrabold mt-2 text-white">
                {report.risk?.score ?? "N/A"}%
              </div>
            </div>

            <div>
              <span className="caption-small text-white/70 uppercase tracking-wider block">Simulation Status</span>
              <div className="text-xl md:text-2xl font-bold mt-2 text-white flex items-center gap-2">
                <CheckCircle2 size={20} className="text-emerald-400" />
                Completed
              </div>
            </div>
          </div>
        </div>

        {/* ── Patient Credentials ────────────────────────────────── */}
        <div className="cardio-card p-6 space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
            <User size={15} />
            Patient Credentials
          </h3>

          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              { label: "Patient ID", value: report.patient?.id ?? "—" },
              { label: "Full Name", value: report.patient?.name ?? "—" },
              {
                label: "Generated",
                value: report.generatedAt
                  ? new Date(report.generatedAt).toLocaleString()
                  : "—",
              },
              { label: "Simulation Status", value: "Completed & Verified", accent: true },
            ].map(({ label, value, accent }) => (
              <div
                key={label}
                className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs space-y-1"
              >
                <dt className="caption-small text-[var(--text-muted)] font-semibold uppercase tracking-wider">
                  {label}
                </dt>
                <dd
                  className={`font-bold ${
                    accent ? "text-emerald-500" : "text-[var(--text-primary)]"
                  }`}
                >
                  {value}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        {/* ── Parameter Profile ──────────────────────────────────── */}
        <div className="cardio-card p-6 space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
            <BarChart3 size={15} />
            Digital Twin Parameter Profile
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(report.metrics ?? []).map(([label, value]) => (
              <div
                key={label}
                className="flex items-center justify-between p-3.5 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] text-xs"
              >
                <span className="text-[var(--text-muted)] font-medium">{label}</span>
                <span className="font-bold text-[var(--text-primary)]">{value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Clinical Alerts & Screening Warnings ─────────────────── */}
        <div className="cardio-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
              <AlertTriangle size={15} className={clinicalAlerts.length > 0 ? "text-amber-500" : "text-emerald-500"} />
              Clinical Alerts &amp; Screening Warnings
            </h3>
            <span className="text-[10px] font-bold uppercase px-2.5 py-1 rounded-full bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border-color)]">
              {clinicalAlerts.length} Alert{clinicalAlerts.length !== 1 ? "s" : ""}
            </span>
          </div>

          {clinicalAlerts.length > 0 ? (
            <div className="space-y-3">
              {clinicalAlerts.map((alt, idx) => (
                <div
                  key={idx}
                  className="p-4 rounded-xl border border-amber-300 bg-amber-500/10 space-y-2 text-xs"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-amber-950 text-sm">{alt.parameter}</span>
                      <span className="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded bg-amber-200 text-amber-900">
                        {alt.severity || "Warning"}
                      </span>
                      <span className="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-300">
                        Hypothetical Scenario
                      </span>
                    </div>
                    <span className="font-mono font-bold text-amber-900 bg-amber-200/80 px-2.5 py-1 rounded">
                      {alt.measured_value} {alt.unit}
                    </span>
                  </div>

                  <div className="text-[11px] text-amber-900 font-semibold">
                    Rule Triggered: <span className="font-mono text-amber-800">{alt.threshold_rule}</span>
                  </div>

                  <p className="text-[var(--text-secondary)] leading-relaxed">
                    {alt.explanation}
                  </p>

                  <div className="p-3 rounded-lg bg-white/80 border border-amber-200 text-amber-950 font-medium leading-relaxed">
                    <strong>Clinical Advice:</strong> {alt.recommendation}
                  </div>
                </div>
              ))}

              <div className="p-3.5 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[11px] text-[var(--text-muted)] space-y-1">
                <div className="font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
                  <Info size={13} />
                  <span>Screening Reference Information &amp; Clinical Context</span>
                </div>
                <p>
                  Threshold source: Adult population screening references (ESH/ESC and AHA/ACC guidelines: Low BP &lt; 90/60 mmHg, High BP ≥ 140/90 mmHg). A single simulated reading does not constitute a clinical diagnosis. Do not alter prescribed medications without consulting your treating physician. Alerts are independent of the machine-learning cardiovascular risk projection.
                </p>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs flex items-center gap-3 text-emerald-700">
              <CheckCircle2 size={18} className="shrink-0 text-emerald-600" />
              <div>
                <div className="font-bold">No Clinical Screening Alerts Triggered</div>
                <div className="text-[11px] text-emerald-600/90 mt-0.5">
                  All simulated vital sign parameters fall within standard adult screening reference ranges (Systolic 90–139 mmHg, Diastolic 60–89 mmHg).
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── AI Clinical Recommendations ────────────────────────── */}
        <div className="cardio-card p-6 space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
            <ShieldCheck size={15} />
            AI Clinical Recommendations
          </h3>

          {report.summary && (
            <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs leading-relaxed text-[var(--text-secondary)]">
              {report.summary}
            </div>
          )}

          <ul className="space-y-3">
            {(report.recommendations ?? []).map((item, idx) => (
              <li
                key={idx}
                className="flex items-start gap-3 p-4 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-xs text-[var(--text-primary)] font-medium"
              >
                <CheckCircle2 size={15} className="text-[var(--accent-melanzane)] shrink-0 mt-0.5" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* ── Footer ─────────────────────────────────────────────── */}
        <div className="cardio-card p-5 flex items-center justify-between text-xs text-[var(--text-muted)]">
          <div className="flex items-center gap-2">
            <Calendar size={14} />
            <span>
              Report generated: {report.generatedAt ? new Date(report.generatedAt).toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" }) : "N/A"}
            </span>
          </div>
          <div className="italic text-right hidden sm:block">
            CardioAI Intelligence Engine · Confidential Medical Decision Support
          </div>
        </div>

      </div>
    </>
  );
}
