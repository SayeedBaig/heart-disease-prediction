/**
 * Hero — top-of-page showcase.
 * Badge → Headline → Subtitle → CTAs → Showcase card.
 * Preserves: /get-started link, scroll-to-capabilities, heart image,
 * live risk meter, and stream indicators.
 */
import { motion } from "framer-motion";
import { ArrowRight, ChevronDown, CheckCircle2, HeartPulse, ShieldCheck } from "lucide-react";
import Container from "../ui/Container";
import Button from "../ui/Button";
import Badge from "../ui/Badge";
import { heroStreams } from "./landingData";
import heart from "../../assets/heart.png";

export default function Hero() {
  const scrollToCapabilities = () => {
    document.getElementById("capabilities")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  return (
    <section
      className="relative overflow-hidden border-b border-[var(--border-color)] bg-[var(--bg-primary)]"
      style={{ paddingTop: "var(--section-pad-y)", paddingBottom: "var(--section-pad-y)" }}
    >
      {/* Ambient glow */}
      <div className="pointer-events-none absolute left-1/2 top-1/4 -z-10 h-[350px] w-[650px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--accent-melanzane)]/10 blur-[130px]" />

      <Container className="flex flex-col items-center text-center">
        {/* Eyebrow badge */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-6"
        >
          <Badge>
            <HeartPulse size={14} />
            AI-Powered Cardiovascular Diagnostics
          </Badge>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="max-w-3xl text-3xl font-extrabold leading-[1.18] tracking-tight text-[var(--text-primary)] sm:text-5xl lg:text-6xl"
          style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
        >
          Predict Heart Disease with{" "}
          <span className="text-[var(--accent-melanzane)]">
            Multimodal AI Precision
          </span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.18 }}
          className="mt-6 max-w-2xl text-base leading-relaxed text-[var(--text-secondary)] sm:text-lg"
        >
          Synthesize clinical vitals, 12-lead ECG waveforms, and echocardiography signals into transparent, explainable cardiovascular risk assessments.
        </motion.p>

        {/* CTA buttons */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.24 }}
          className="relative z-20 mt-8 flex flex-wrap items-center justify-center gap-4"
        >
          <Button to="/get-started" size="lg">
            Get Started
            <ArrowRight size={16} />
          </Button>
          <Button variant="secondary" size="lg" onClick={scrollToCapabilities}>
            Explore Capabilities
            <ChevronDown size={16} />
          </Button>
        </motion.div>

        {/* Showcase card */}
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.32 }}
          className="relative z-10 mt-14 w-full max-w-4xl"
        >
          <div className="rounded-3xl border border-[var(--border-color)] bg-[var(--card-bg)] p-6 shadow-2xl sm:p-8 md:p-10">
            <div className="grid items-center gap-8 md:grid-cols-2">
              {/* Left: heart visual */}
              <div className="relative flex min-h-[280px] items-center justify-center overflow-hidden rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-secondary)] p-6 sm:min-h-[320px]">
                <img
                  src={heart}
                  alt="CardioAI heart intelligence visualization"
                  className="max-h-[260px] w-full object-contain drop-shadow-xl transition-transform duration-500 hover:scale-105 sm:max-h-[290px]"
                />
              </div>

              {/* Right: live diagnostic summary */}
              <div className="space-y-4 text-left">
                {/* Risk meter */}
                <div className="rounded-2xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] p-5 shadow-sm">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-[11px] font-bold uppercase tracking-wider text-[var(--accent-melanzane)]">
                        Live Risk Stratification
                      </p>
                      <p className="mt-1 text-3xl font-extrabold text-[var(--text-primary)]">
                        24% Risk Score
                      </p>
                    </div>
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--card-bg)] text-[var(--accent-melanzane)] shadow-sm">
                      <ShieldCheck size={26} />
                    </div>
                  </div>
                  <div className="mt-4 h-2.5 w-full overflow-hidden rounded-full border border-[var(--border-subtle)] bg-[var(--card-bg)]">
                    <div className="h-full w-[24%] rounded-full bg-gradient-to-r from-emerald-500 via-[var(--accent-melanzane)] to-[var(--accent-melanzane)]" />
                  </div>
                </div>

                {/* Stream indicators */}
                {heroStreams.map(({ title, label }) => (
                  <div
                    key={title}
                    className="flex items-start gap-3.5 rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] p-4 shadow-sm transition-all hover:border-[var(--accent-melanzane-border)]"
                  >
                    <CheckCircle2 className="mt-0.5 shrink-0 text-emerald-500" size={18} />
                    <div>
                      <p className="text-sm font-bold leading-none text-[var(--text-primary)]">{title}</p>
                      <p className="mt-1.5 text-xs font-normal leading-relaxed text-[var(--text-secondary)]">{label}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </Container>
    </section>
  );
}