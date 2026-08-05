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
