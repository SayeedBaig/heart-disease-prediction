import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Stethoscope,
  Activity,
  HeartPulse,
  Brain,
  ShieldCheck,
  FileText,
  UserCheck,
  Calendar,
  Layers,
  ArrowRight,
  ArrowLeft,
} from "lucide-react";

import Navbar from "../components/Navbar";
import Workflow from "../components/Workflow";

export default function ExploreCapabilities() {
  const navigate = useNavigate();

  return (
    <div className="cardio-shell overflow-x-hidden">
      <Navbar onBack={() => navigate("/")} backLabel="Home" breadcrumb="Platform Capabilities" />

      <main className="cardio-container py-10 md:py-16 max-w-7xl mx-auto w-full flex-1">
        {/* Back Button Banner */}
        <div className="mb-8 flex items-center justify-between border-b border-[var(--border-color)] pb-4">
          <button
            onClick={() => navigate("/")}
            className="btn-secondary py-2.5 px-4 text-xs font-semibold rounded-xl flex items-center gap-2 transition-all hover:border-[var(--accent-melanzane-border)]"
          >
            <ArrowLeft size={16} />
            <span>Back to Home</span>
          </button>

          <span className="caption-small text-[var(--accent-melanzane)] font-bold uppercase tracking-wider">
            CardioAI Platform Capabilities
          </span>
        </div>

        {/* Hero Title for Capabilities */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <span className="caption-small uppercase tracking-widest text-[var(--accent-melanzane)] font-bold">
            Platform Capabilities & Architecture
          </span>
          <h1 className="h1-large text-[var(--text-primary)] mt-2 text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight">
            Comprehensive Cardiac Intelligence Suite
          </h1>
          <p className="body-regular mt-4 text-sm md:text-base text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
            Explore our multimodal neural architecture, clinical workflow, and interactive Digital Twin simulation engine.
          </p>
        </div>

        {/* Clean Statistics Row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full max-w-4xl mx-auto mb-16">
          <div className="cardio-card p-6 text-center rounded-2xl">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-melanzane)]">96%</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">Prediction Confidence</p>
          </div>

          <div className="cardio-card p-6 text-center rounded-2xl">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-melanzane)]">3-in-1</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">Clinical • ECG • Echo Fusion</p>
          </div>

          <div className="cardio-card p-6 text-center rounded-2xl">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-melanzane)]">Explainable</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">AI Clinical Summaries</p>
          </div>
        </div>

        {/* ABOUT SECTION */}
        <section className="py-12 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-3xl p-8 md:p-12 mb-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <span className="caption-small uppercase tracking-widest text-[var(--accent-melanzane)] font-bold">
              About CardioAI
            </span>
            <h2 className="h2-semibold text-[var(--text-primary)] mt-2 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight">
              Next-Generation Cardiovascular Intelligence
            </h2>
            <p className="body-regular mt-4 text-sm md:text-base text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
              CardioAI bridges modern diagnostic data with advanced artificial intelligence to give patients and physicians unprecedented clarity on cardiac health.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
            {/* About 1: AI Prediction */}
            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-2xl hover:border-[var(--accent-melanzane-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5">
                <Brain size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Multimodal AI</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Machine learning models evaluate clinical vitals, lab biomarkers, ECG waves, and echo imagery simultaneously.
              </p>
            </div>

            {/* About 2: Clinical Decision Support */}
            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-2xl hover:border-[var(--accent-melanzane-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5">
                <ShieldCheck size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Decision Support</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Provides actionable risk stratification and confidence metrics to assist cardiologists in timely intervention.
              </p>
            </div>

            {/* About 3: Personalized Healthcare */}
            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-2xl hover:border-[var(--accent-melanzane-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5">
                <UserCheck size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Personalized Care</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Tailors treatment guidance, lifestyle targets, and longitudinal tracking to each unique patient profile.
              </p>
            </div>

            {/* About 4: Digital Twin */}
            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-2xl hover:border-[var(--accent-melanzane-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5">
                <Layers size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Digital Twin</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Interactive real-time cardiovascular simulation allowing users to model physiological parameter adjustments live.
              </p>
            </div>
          </div>
        </section>

        {/* WORKFLOW COMPONENT */}
        <div className="mb-16">
          <Workflow />
        </div>

        {/* FEATURES SECTION */}
        <section className="mb-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <span className="caption-small uppercase tracking-widest text-[var(--accent-melanzane)] font-bold">
              Platform Capabilities
            </span>
            <h2 className="h2-semibold text-[var(--text-primary)] mt-2 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight">
              Comprehensive Cardiac Feature Suite
            </h2>
            <p className="body-regular mt-4 text-sm md:text-base text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
              Designed for clarity, clinical rigor, and seamless interaction across every stage of cardiac care.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            {/* Card 1 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Stethoscope size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                Clinical Assessment
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Evaluates vitals, blood pressure, cholesterol, fasting glucose, and lifestyle metrics for risk stratification.
              </p>
            </div>

            {/* Card 2 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Activity size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                ECG Waveform Analysis
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Deep learning wave classification for automated detection of rhythmic and ischemic cardiac patterns.
              </p>
            </div>

            {/* Card 3 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <HeartPulse size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                Echocardiogram Imaging
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Evaluates ventricular wall motion and structural ejection metrics via AI image processing.
              </p>
            </div>

            {/* Card 4 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Layers size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                Digital Twin Simulation
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Simulate 14 physiological parameter adjustments and visualize 10-year risk trajectories live.
              </p>
            </div>

            {/* Card 5 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <FileText size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                Standardized Reports
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Generate standardized, print-ready clinical summaries with detailed diagnostic findings.
              </p>
            </div>

            {/* Card 6 */}
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-2xl group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Calendar size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                Specialist Consultation
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Schedule consultations and share digital screening reports directly with certified cardiologists.
              </p>
            </div>
          </div>
        </section>

        {/* CTA BANNER */}
        <section className="mb-12">
          <div className="cardio-card p-8 md:p-12 text-center rounded-3xl bg-gradient-to-r from-[var(--accent-melanzane-light)] via-[var(--card-bg)] to-[var(--accent-melanzane-light)] border border-[var(--accent-melanzane-border)] shadow-md">
            <h2 className="text-2xl md:text-3xl font-extrabold text-[var(--text-primary)] tracking-tight">
              Ready to Evaluate Cardiovascular Health?
            </h2>
            <p className="mt-3 text-sm md:text-base text-[var(--text-secondary)] max-w-xl mx-auto leading-relaxed">
              Access the patient or clinician portal to experience CardioAI multimodal risk prediction.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <Link
                to="/get-started"
                className="btn-primary py-3.5 px-8 text-sm font-semibold rounded-xl flex items-center gap-2 shadow-md hover:shadow-lg transition-all"
              >
                <span>Get Started Now</span>
                <ArrowRight size={16} />
              </Link>
              <button
                onClick={() => navigate("/")}
                className="btn-secondary py-3.5 px-6 text-sm font-semibold rounded-xl flex items-center gap-2 transition-all"
              >
                <ArrowLeft size={16} />
                <span>Return to Home</span>
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <footer className="py-8 border-t border-[var(--border-color)] text-xs text-[var(--text-muted)] bg-[var(--card-bg)]">
        <div className="cardio-container max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5 font-bold text-[var(--text-primary)] text-sm">
            <div className="w-7 h-7 rounded-lg bg-[#39062B] text-white flex items-center justify-center text-xs font-black">
              AI
            </div>
            <span>CardioAI Intelligence</span>
          </div>

          <div className="text-center md:text-right text-[var(--text-secondary)]">
            © {new Date().getFullYear()} CardioAI Platform. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
