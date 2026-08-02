import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, HeartPulse, Activity, ShieldCheck } from "lucide-react";
import Navbar from "../components/Navbar";
import ChatbotWidget from "../components/ChatbotWidget";
import heart from "../assets/heart.png";

export default function LandingPage() {
  return (
    <div className="cardio-shell overflow-x-hidden flex flex-col min-h-screen">
      <Navbar />

      {/* HERO SECTION - dark gradient with heart artwork */}
      <section
        id="platform"
        className="relative overflow-hidden bg-gradient-to-br from-[#0b1230] via-[#101a45] to-[#152559] flex-1 flex flex-col justify-center"
      >
        {/* Decorative heart artwork, muted into the background */}
        <img
          src={heart}
          alt=""
          aria-hidden="true"
          className="pointer-events-none select-none absolute right-[-6%] top-1/2 -translate-y-1/2 w-[520px] md:w-[640px] opacity-15 mix-blend-screen"
        />
        {/* Soft radial glow */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,_rgba(76,125,255,0.25),_transparent_55%)]" />

        <div className="relative cardio-container flex w-full flex-col items-center pt-36 md:pt-44 pb-16 md:pb-20 text-center max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="mb-6 inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-white/15 bg-white/10 backdrop-blur text-white text-xs font-semibold"
          >
            <HeartPulse size={14} />
            <span>AI-Powered Cardiovascular Diagnostics</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="max-w-4xl text-white text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight leading-[1.08]"
            style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}
          >
            Predict Heart Disease with{" "}
            <br className="hidden sm:block" />
            <span className="text-[#7fa2ff]">Multimodal AI Precision</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-6 text-sm md:text-lg text-white/70 max-w-2xl font-normal leading-relaxed"
          >
            Combine clinical metrics, ECG waveforms, and echocardiography signals into explainable cardiovascular risk assessments and interactive Digital Twin simulations.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-10 flex flex-wrap items-center justify-center gap-4"
          >
            <Link
              to="/get-started"
              className="btn-primary py-3.5 px-8 text-sm rounded-full font-semibold flex items-center gap-2 shadow-lg hover:shadow-xl min-w-[180px] justify-center"
            >
              <span>Get Started</span>
              <ArrowRight size={16} />
            </Link>

            <Link
              to="/explore-capabilities"
              className="btn-primary py-3.5 px-8 text-sm rounded-full font-semibold flex items-center gap-2 shadow-lg hover:shadow-xl min-w-[180px] justify-center"
            >
              <span>Explore Capabilities</span>
              <ArrowRight size={16} />
            </Link>
          </motion.div>

          {/* Rich stat cards - icon + title on one row, description below */}
          <motion.div
            id="capabilities"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            style={{ marginTop: "50px" }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full max-w-4xl scroll-mt-24"
          >
            {[
              {
                icon: Activity,
                title: "3-Modal Analysis",
                desc: "Harness clinical data, ECG, & echo signals for precision.",
              },
              {
                icon: HeartPulse,
                title: "Real-time Risk Assessment",
                desc: "Instant, explainable insights & predictive modeling.",
              },
              {
                icon: ShieldCheck,
                title: "HIPAA & Data Security",
                desc: "End-to-end protection and responsible data handling.",
              },
            ].map(({ icon: Icon, title, desc }) => (
              <div
                key={title}
                style={{ padding: "15px", maxWidth: "380px" }}
                className="text-left rounded-[24px] border border-white/15 bg-white/[0.06] backdrop-blur w-full mx-auto flex flex-col gap-3 transition-all duration-300 hover:bg-white/[0.09] hover:border-white/25"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 shrink-0 rounded-2xl bg-white/10 text-[#7fa2ff] flex items-center justify-center">
                    <Icon size={18} />
                  </div>
                  <h3 className="text-sm font-bold text-white">{title}</h3>
                </div>
                <p className="text-xs text-white/60 leading-relaxed">{desc}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ABOUT SECTION */}
      <section id="about" className="scroll-mt-20 py-16 md:py-20 bg-[var(--bg-secondary)] border-b border-[var(--border-color)]">
        <div className="cardio-container flex flex-col items-center text-center">
          <span className="caption-small uppercase tracking-widest text-[var(--accent-primary)] font-bold">
            About Us
          </span>
          <h2 className="h2-semibold text-[var(--text-primary)] mt-2 text-lg md:text-xl font-extrabold tracking-tight max-w-2xl">
            Built for Explainable, Trustworthy Cardiac AI
          </h2>
          <p className="body-regular mt-4 text-sm md:text-base text-[var(--text-secondary)] leading-relaxed max-w-2xl">
            CardioAI brings together clinical data, ECG signal processing, and echocardiography analysis into a single multimodal platform, giving patients and clinicians clear, explainable insight into cardiovascular risk.
          </p>
        </div>
      </section>

      {/* FOOTER / CONTACT */}
      <footer id="contact" className="scroll-mt-20 py-6 border-t border-[var(--border-color)] text-xs text-[var(--text-muted)] bg-[var(--card-bg)]">
        <div className="cardio-container max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 font-semibold text-[var(--text-secondary)] text-sm">
            <div className="w-6 h-6 rounded-md bg-[var(--accent-primary)] text-white flex items-center justify-center text-[10px] font-black">
              AI
            </div>
            <span>CardioAI Intelligence Platform</span>
          </div>

          <div className="text-center md:text-right text-[var(--text-muted)]">
            &copy; {new Date().getFullYear()} CardioAI. All rights reserved. &middot; For clinical decision support only.
          </div>
        </div>
      </footer>

      <ChatbotWidget mode="public" />
    </div>
  );
}
