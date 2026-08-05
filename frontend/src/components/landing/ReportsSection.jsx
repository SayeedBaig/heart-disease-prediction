/**
 * ReportsSection — two-column showcase with AI report preview.
 * Left: copy + CTA. Right: report card with metrics.
 * Preserves: /get-started link, reportMetrics data.
 */
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import Section from "../ui/Section";
import Container from "../ui/Container";
import Button from "../ui/Button";
import { fadeUp } from "../ui/animations";
import { reportMetrics } from "./landingData";

export default function ReportsSection() {
  return (
    <Section id="reports-preview" bg="secondary" bordered>
      <Container>
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-2">
          {/* Left: copy */}
          <motion.div {...fadeUp} className="space-y-5">
            <span className="text-xs font-bold uppercase tracking-widest text-[var(--accent-melanzane)]">
              AI Clinical Reports Preview
            </span>
            <h2
              className="text-3xl font-extrabold leading-tight tracking-tight text-[var(--text-primary)] md:text-4xl"
              style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
            >
              Explainable reports designed for patients and clinicians
            </h2>
            <p className="text-base leading-relaxed text-[var(--text-secondary)]">
              Generate standardized, print-ready clinical summaries with transparent risk breakdowns, confidence metrics, ECG interpretations, and specialist recommendations.
            </p>
            <div className="pt-2">
              <Button to="/get-started" variant="secondary">
                View Sample Report
                <ArrowRight size={16} />
              </Button>
            </div>
          </motion.div>

          {/* Right: report card */}
          <motion.div {...fadeUp}>
            <div className="rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] p-8 shadow-[var(--shadow-elevated)]">
              {/* Header */}
              <div className="mb-6 border-b border-[var(--border-color)] pb-5">
                <div className="flex items-center gap-3.5">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-melanzane)] text-xs font-black text-white shadow-sm">
                    AI
                  </div>
                  <div>
                    <h3 className="text-lg font-extrabold text-[var(--text-primary)]">
                      CardioAI Medical Report
                    </h3>
                    <p className="text-xs text-[var(--text-muted)]">
                      Multimodal Diagnostic Summary & Risk Profile
                    </p>
                  </div>
                </div>
              </div>

              {/* Metrics grid */}
              <div className="mb-6 grid gap-4 sm:grid-cols-3">
                {reportMetrics.map(({ label, value }) => (
                  <div
                    key={label}
                    className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-secondary)] p-4 text-center"
                  >
                    <p className="text-[11px] font-bold uppercase tracking-wide text-[var(--text-muted)]">
                      {label}
                    </p>
                    <p className="mt-1 text-xl font-extrabold text-[var(--text-primary)]">
                      {value}
                    </p>
                  </div>
                ))}
              </div>

              {/* Summary text */}
              <div className="space-y-4 text-xs leading-relaxed sm:text-sm">
                <p className="text-[var(--text-secondary)]">
                  Screening findings indicate a low-to-moderate cardiovascular risk trajectory. Routine monitoring and preventive lifestyle modifications are recommended.
                </p>
                <div className="rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] p-4 text-xs font-semibold leading-relaxed text-[var(--accent-melanzane)]">
                  Report Dossier Includes: Clinical Summary, 12-Lead ECG Findings, Echocardiogram Motion Scans, and Specialist Notes.
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </Container>
    </Section>
  );
}