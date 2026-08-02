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
            className="btn-secondary py-2.5 px-4 text-xs font-semibold rounded-full flex items-center gap-2 transition-all hover:border-[var(--accent-primary-border)]"
          >
            <ArrowLeft size={16} />
            <span>Back to Home</span>
          </button>

          <span className="caption-small text-[var(--accent-primary)] font-bold uppercase tracking-wider">
            CardioAI Platform Capabilities
          </span>
        </div>

        {/* Hero Title for Capabilities */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <span className="caption-small uppercase tracking-widest text-[var(--accent-primary)] font-bold">
            Platform Capabilities & Architecture
          </span>
          <h1 className="h1-large text-[var(--text-primary)] mt-2 font-extrabold tracking-tight whitespace-nowrap text-3xl sm:text-4xl md:text-5xl">
            Comprehensive Cardiac Intelligence Suite
          </h1>
          <p className="body-regular mt-4 text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
            Explore our multimodal neural architecture, clinical workflow, and interactive Digital Twin simulation engine.
          </p>
        </div>

        {/* Clean Statistics Row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full max-w-4xl mx-auto mb-16">
          <div className="cardio-card p-8 text-center rounded-[28px]">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-primary)]">96%</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">Prediction Confidence</p>
          </div>

          <div className="cardio-card p-8 text-center rounded-[28px]">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-primary)]">3-in-1</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">Clinical &middot; ECG &middot; Echo Fusion</p>
          </div>

          <div className="cardio-card p-8 text-center rounded-[28px]">
            <h3 className="text-3xl md:text-4xl font-extrabold text-[var(--accent-primary)]">Explainable</h3>
            <p className="caption-small mt-1 text-[var(--text-secondary)]">AI Clinical Summaries</p>
          </div>
        </div>

        {/* ABOUT SECTION */}
        <section className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[28px] p-8 md:p-12 mb-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <span className="caption-small uppercase tracking-widest text-[var(--accent-primary)] font-bold">
              About CardioAI
            </span>
            <h2 className="h2-semibold text-[var(--text-primary)] mt-2 font-extrabold tracking-tight">
              Next-Generation Cardiovascular Intelligence
            </h2>
            <p className="body-regular mt-4 text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
              CardioAI bridges modern diagnostic data with advanced artificial intelligence to give patients and physicians unprecedented clarity on cardiac health.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-[28px] hover:border-[var(--accent-primary-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5">
                <Brain size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Multimodal AI</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Machine learning models evaluate clinical vitals, lab biomarkers, ECG waves, and echo imagery simultaneously.
              </p>
            </div>

            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-[28px] hover:border-[var(--accent-primary-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5">
                <ShieldCheck size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Decision Support</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Provides actionable risk stratification and confidence metrics to assist cardiologists in timely intervention.
              </p>
            </div>

            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-[28px] hover:border-[var(--accent-primary-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5">
                <UserCheck size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2">Personalized Care</h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Tailors treatment guidance, lifestyle targets, and longitudinal tracking to each unique patient profile.
              </p>
            </div>

            <div className="cardio-card p-7 flex flex-col items-start h-full rounded-[28px] hover:border-[var(--accent-primary-border)] transition-all duration-300">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5">
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

        {/* FEATURES SECTION - now matches About section's card-wrapper treatment */}
        <section className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[28px] p-8 md:p-12 mb-16">
          <div className="text-center max-w-3xl mx-auto mb-12">
            <span className="caption-small uppercase tracking-widest text-[var(--accent-primary)] font-bold">
              Platform Capabilities
            </span>
            <h2 className="h2-semibold text-[var(--text-primary)] mt-2 font-extrabold tracking-tight">
              Comprehensive Cardiac Feature Suite
            </h2>
            <p className="body-regular mt-4 text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto">
              Designed for clarity, clinical rigor, and seamless interaction across every stage of cardiac care.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Stethoscope size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Clinical Assessment
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Evaluates vitals, blood pressure, cholesterol, fasting glucose, and lifestyle metrics for risk stratification.
              </p>
            </div>

            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Activity size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                ECG Waveform Analysis
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Deep learning wave classification for automated detection of rhythmic and ischemic cardiac patterns.
              </p>
            </div>

            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <HeartPulse size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Echocardiogram Imaging
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Evaluates ventricular wall motion and structural ejection metrics via AI image processing.
              </p>
            </div>

            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Layers size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Digital Twin Simulation
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Simulate 14 physiological parameter adjustments and visualize 10-year risk trajectories live.
              </p>
            </div>

            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <FileText size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Standardized Reports
              </h3>
              <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                Generate standardized, print-ready clinical summaries with detailed diagnostic findings.
              </p>
            </div>

            <div className="cardio-card-interactive p-7 flex flex-col items-start rounded-[28px] group transition-all duration-300 hover:shadow-xl">
              <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mb-5 transition-transform group-hover:scale-105">
                <Calendar size={24} />
              </div>
              <h3 className="text-base font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
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
          <div className="cardio-card p-8 md:p-12 text-center rounded-[28px] bg-gradient-to-r from-[var(--accent-primary-light)] via-[var(--card-bg)] to-[var(--accent-primary-light)] border border-[var(--accent-primary-border)] shadow-md">
            <h2 className="text-2xl md:text-3xl font-extrabold text-[var(--text-primary)] tracking-tight">
              Ready to Evaluate Cardiovascular Health?
            </h2>
            <p className="mt-3 text-sm md:text-base text-[var(--text-secondary)] max-w-xl mx-auto leading-relaxed">
              Access the patient or clinician portal to experience CardioAI multimodal risk prediction.
            </p>
            <div className="mt-8 flex flex-wrap justify-center gap-4">
              <Link
                to="/get-started"
                className="btn-primary py-3.5 px-8 text-sm font-semibold rounded-full flex items-center gap-2 shadow-md hover:shadow-lg transition-all"
              >
                <span>Get Started Now</span>
                <ArrowRight size={16} />
              </Link>
              <button
                onClick={() => navigate("/")}
                className="btn-secondary py-3.5 px-6 text-sm font-semibold rounded-full flex items-center gap-2 transition-all"
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
            <div className="w-7 h-7 rounded-lg bg-[var(--accent-primary)] text-white flex items-center justify-center text-xs font-black">
              AI
            </div>
            <span>CardioAI Intelligence</span>
          </div>

          <div className="text-center md:text-right text-[var(--text-secondary)]">
            &copy; {new Date().getFullYear()} CardioAI Platform. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
