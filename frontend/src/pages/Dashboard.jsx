import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import PageBackground from "../components/PageBackground";

import {
  User,
  HeartPulse,
  FileText,
  History,
  Activity,
} from "lucide-react";

function Dashboard() {
  const patientId = localStorage.getItem("patient_id");
  const patientName = localStorage.getItem("patient_name");
  const patientEmail = localStorage.getItem("patient_email");

  return (
    <div className="cardio-shell">
      <Navbar />

      <main className="cardio-container flex-1 relative overflow-hidden">
        <PageBackground />

        <div className="relative z-10 w-full">

          {/* ================= Welcome Card ================= */}
          <div className="relative overflow-hidden rounded-2xl bg-[var(--accent-melanzane)] border border-[var(--accent-melanzane-border)] shadow-xl p-8 md:p-10 text-white">
            <Activity className="absolute right-6 top-1/2 -translate-y-1/2 w-48 h-48 text-white/10 pointer-events-none" />

            <div className="relative z-10 space-y-2">
              <span className="text-xs uppercase font-bold tracking-widest text-white/70">
                Patient Workspace
              </span>
              <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
                Welcome 👋 {patientName || "Patient"}
              </h1>

              <div className="flex flex-wrap gap-x-6 gap-y-1 pt-3 text-xs md:text-sm text-white/80 font-medium">
                <p>
                  Patient ID: <span className="font-bold text-white ml-1">{patientId || "Not Available"}</span>
                </p>
                <p>
                  Email: <span className="font-bold text-white ml-1">{patientEmail || "Not Available"}</span>
                </p>
              </div>
            </div>
          </div>

          {/* ================= Dashboard Cards ================= */}
          <div className="grid md:grid-cols-3 gap-6 mt-8">
            {/* Patient Profile */}
            <div className="cardio-card p-6 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-4">
                  <User size={24} />
                </div>
                <h2 className="text-lg font-bold text-[var(--text-primary)]">
                  Patient Profile
                </h2>
                <p className="caption-small text-[var(--text-secondary)] mt-2">
                  Registered patient account active.
                </p>
              </div>
            </div>

            {/* Latest Prediction */}
            <div className="cardio-card p-6 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-red-500/10 text-red-500 flex items-center justify-center mb-4">
                  <HeartPulse size={24} />
                </div>
                <h2 className="text-lg font-bold text-[var(--text-primary)]">
                  Latest Prediction
                </h2>
                <p className="caption-small text-[var(--text-secondary)] mt-2">
                  No prediction available yet.
                </p>
              </div>
            </div>

            {/* History */}
            <div className="cardio-card p-6 flex flex-col justify-between">
              <div>
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center mb-4">
                  <History size={24} />
                </div>
                <h2 className="text-lg font-bold text-[var(--text-primary)]">
                  Prediction History
                </h2>
                <p className="caption-small text-[var(--text-secondary)] mt-2">
                  View all previous predictions and clinical assessments.
                </p>
              </div>
            </div>
          </div>

          {/* ================= Quick Actions ================= */}
          <div className="cardio-card p-8 mt-8">
            <h2 className="h2-semibold text-xl md:text-2xl mb-6 text-[var(--text-primary)]">
              Quick Actions
            </h2>

            <div className="grid md:grid-cols-3 gap-6">
              {/* Diagnosis */}
              <Link
                to="/patient/intake"
                className="btn-primary flex-col items-start p-6 text-left h-auto gap-3 rounded-2xl shadow-md hover:shadow-lg"
              >
                <HeartPulse size={32} />
                <div>
                  <h3 className="text-base font-bold">Start AI Diagnosis</h3>
                  <p className="caption-small text-white/80 mt-1">
                    Begin a new multi-modal cardiovascular risk assessment.
                  </p>
                </div>
              </Link>

              {/* Reports */}
              <Link
                to="/reports"
                className="cardio-card-interactive p-6 flex-col items-start text-left h-auto gap-3 rounded-2xl"
              >
                <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center">
                  <FileText size={24} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[var(--text-primary)]">Clinical Reports</h3>
                  <p className="caption-small text-[var(--text-secondary)] mt-1">
                    Download and print generated risk reports.
                  </p>
                </div>
              </Link>

              {/* History */}
              <Link
                to="/digital-twin"
                className="cardio-card-interactive p-6 flex-col items-start text-left h-auto gap-3 rounded-2xl"
              >
                <div className="w-10 h-10 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center">
                  <History size={24} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[var(--text-primary)]">Digital Twin</h3>
                  <p className="caption-small text-[var(--text-secondary)] mt-1">
                    Simulate lifestyle & therapeutic parameter shifts.
                  </p>
                </div>
              </Link>
            </div>
          </div>

        </div>

      </main>
    </div>
  );
}

export default Dashboard;