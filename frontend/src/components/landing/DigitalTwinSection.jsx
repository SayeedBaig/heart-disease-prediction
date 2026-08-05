/**
 * DigitalTwinSection — two-column showcase with simulation dashboard.
 * Left: heading + CTA. Right: live metric bars.
 * Preserves: /get-started link, twinMetrics data.
 */
import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import Section from "../ui/Section";
import Container from "../ui/Container";
import Button from "../ui/Button";
import { fadeUp } from "../ui/animations";
import { twinMetrics } from "./landingData";

export default function DigitalTwinSection() {
  return (
    <Section id="digital-twin" bg="primary" className="scroll-mt-20">
      <Container>
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-2">
          {/* Left: copy */}
          <motion.div {...fadeUp} className="space-y-5">
            <span className="text-xs font-bold uppercase tracking-widest text-[var(--accent-melanzane)]">
              Digital Twin Preview
            </span>
            <h2
              className="text-3xl font-extrabold leading-tight tracking-tight text-[var(--text-primary)] md:text-4xl"
              style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
            >
              Simulate future cardiac risk before it becomes critical
            </h2>
            <p className="text-base leading-relaxed text-[var(--text-secondary)]">
              Our dynamic physiological simulation engine models how targeted risk interventions—such as blood pressure management or lifestyle adjustments—impact 10-year cardiac trajectories in real time.
            </p>
            <div className="pt-2">
              <Button to="/get-started">
                Launch Digital Twin Simulation
                <ArrowRight size={16} />
              </Button>
            </div>
          </motion.div>

          {/* Right: simulation dashboard */}
          <motion.div {...fadeUp}>
            <div className="rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] p-8 shadow-[var(--shadow-elevated)]">
              <div className="mb-8 flex items-center justify-between gap-4 border-b border-[var(--border-color)] pb-5">
                <div>
                  <p className="text-xs font-bold uppercase tracking-wide text-[var(--accent-melanzane)]">
                    Live Simulation Dashboard
                  </p>
                  <h3 className="mt-1 text-xl font-extrabold text-[var(--text-primary)]">
                    10-Year Risk Trajectory
                  </h3>
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)]">
                  <Sparkles size={22} />
                </div>
              </div>

              <div className="space-y-6">
                {twinMetrics.map(({ label, value, percent }) => (
                  <div key={label} className="space-y-2">
                    <div className="flex items-center justify-between text-xs font-bold">
                      <span className="text-[var(--text-primary)]">{label}</span>
                      <span className="text-[var(--accent-melanzane)]">{value}</span>
                    </div>
                    <div className="h-3 w-full overflow-hidden rounded-full border border-[var(--border-subtle)] bg-[var(--bg-secondary)]">
                      <div
                        className="h-full rounded-full bg-[var(--accent-melanzane)] transition-all duration-500"
                        style={{ width: percent }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </Container>
    </Section>
  );
}