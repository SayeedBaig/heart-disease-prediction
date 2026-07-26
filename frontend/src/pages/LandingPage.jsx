import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, HeartPulse, Activity, Shield } from "lucide-react";
import Navbar from "../components/Navbar";

export default function LandingPage() {
  return (
    <div className="cardio-shell overflow-x-hidden flex flex-col min-h-screen">
      <Navbar />

      {/* HERO SECTION */}
      <main className="flex-1 flex flex-col justify-center items-center">
        <section className="relative cardio-container flex w-full flex-col items-center py-20 md:py-28 text-center max-w-5xl mx-auto">
          {/* Eyebrow label */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mb-6 inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] text-xs font-semibold"
          >
            <HeartPulse size={14} />
            <span>AI-Powered Cardiovascular Diagnostics</span>
          </motion.div>

          {/* Large Title */}
          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="max-w-4xl text-[var(--text-primary)] text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[1.08]"
            style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
          >
            Predict Heart Disease with{" "}
            <br className="hidden sm:block" />
            <span className="text-[var(--accent-melanzane)]">Multimodal AI Precision</span>
          </motion.h1>

          {/* Subheading */}
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-6 text-sm md:text-lg text-[var(--text-secondary)] max-w-2xl font-normal leading-relaxed"
          >
            Combine clinical metrics, ECG waveforms, and echocardiography signals into explainable cardiovascular risk assessments and interactive Digital Twin simulations.
          </motion.p>

          {/* Primary & Secondary CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <Link
              to="/get-started"
              className="btn-primary py-3.5 px-8 text-sm rounded-xl font-semibold flex items-center gap-2 shadow-lg hover:shadow-xl"
            >
              <span>Get Started</span>
              <ArrowRight size={16} />
            </Link>

            <Link
              to="/explore-capabilities"
              className="btn-secondary py-3.5 px-7 text-sm rounded-xl font-semibold flex items-center gap-2"
            >
              <span>Explore Capabilities</span>
              <ArrowRight size={14} />
            </Link>
          </motion.div>

          {/* Stats strip */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="mt-16 flex flex-wrap items-center justify-center gap-8 text-center"
          >
            {[
              { icon: Activity, value: "3-Modal", label: "AI Analysis" },
              { icon: HeartPulse, value: "Real-time", label: "Risk Assessment" },
              { icon: Shield, value: "HIPAA", label: "Data Security" },
            ].map(({ icon: Icon, value, label }) => (
              <div key={label} className="flex flex-col items-center gap-1">
                <div className="w-9 h-9 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-1">
                  <Icon size={18} />
                </div>
                <span className="text-base font-bold text-[var(--text-primary)]">{value}</span>
                <span className="text-xs text-[var(--text-muted)]">{label}</span>
              </div>
            ))}
          </motion.div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="py-6 border-t border-[var(--border-color)] text-xs text-[var(--text-muted)] bg-[var(--card-bg)]">
        <div className="cardio-container max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 font-semibold text-[var(--text-secondary)] text-sm">
            <div className="w-6 h-6 rounded-md bg-[#39062B] text-white flex items-center justify-center text-[10px] font-black">
              AI
            </div>
            <span>CardioAI Intelligence Platform</span>
          </div>

          <div className="text-center md:text-right text-[var(--text-muted)]">
            © {new Date().getFullYear()} CardioAI. All rights reserved. · For clinical decision support only.
          </div>
        </div>
      </footer>
    </div>
  );
}

