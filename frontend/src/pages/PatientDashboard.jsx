import { useNavigate } from "react-router-dom";
import {
  UploadCloud,
  FileText,
  Activity,
  Calendar,
  User,
  LogOut,
  ArrowRight,
} from "lucide-react";
import Navbar from "../components/Navbar";

export default function PatientDashboard({ onNavigate, onLogout }) {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");

  const handleAction = (key, route) => {
    if (onNavigate) {
      onNavigate(key);
    }
    if (route) {
      navigate(route);
    }
  };

  const handleLogoutAction = () => {
    if (onLogout) {
      onLogout();
    } else {
      localStorage.removeItem("access_token");
      localStorage.removeItem("cardio-patient");
      navigate("/");
    }
  };

  return (
    <div className="cardio-shell overflow-x-hidden">
      <Navbar />

      <main className="cardio-container py-8 md:py-10 w-full flex-1 max-w-[1400px] mx-auto space-y-12">
        {/* HERO SECTION - Compact height, full width */}
        <section className="cardio-card p-6 md:p-8 relative overflow-hidden bg-gradient-to-r from-[var(--card-bg)] via-[var(--card-hover)] to-[var(--accent-melanzane-light)] border border-[var(--border-color)] rounded-2xl shadow-sm">
          <div className="relative z-10 max-w-3xl">
            <h1 className="h1-large text-[var(--text-primary)] text-3xl md:text-4xl font-extrabold tracking-tight">
              Welcome back, {patient.full_name || patient.name || "Patient"}
            </h1>
            <p className="body-regular text-xs md:text-sm mt-2 text-[var(--text-secondary)] leading-relaxed">
              Choose a workspace below to upload medical data, review reports, or simulate future cardiovascular outcomes.
            </p>
          </div>
        </section>

        {/* SECTION 1: CORE DIAGNOSTIC MODULES */}
        <section className="space-y-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Core Diagnostic Modules
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 1: Upload Medical Data */}
            <div
              onClick={() => handleAction("intake", "/patient/intake")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <UploadCloud size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Upload Medical Data
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Submit clinical vitals, ECG waveforms, and echo scans for rapid AI assessment.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Start Assessment</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>

            {/* Card 2: View Reports */}
            <div
              onClick={() => handleAction("report", "/patient/reports")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <FileText size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  View Reports
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Access, download, and review your official diagnostic summaries and screening history.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Open Reports</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>

            {/* Card 3: Digital Twin */}
            <div
              onClick={() => handleAction("twin", "/patient/digital-twin")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane)] text-white flex items-center justify-center shadow-md transition-transform group-hover:scale-110">
                  <Activity size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Digital Twin
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Interactive parameter simulation to model real-time cardiovascular risk trajectories.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Launch Simulation</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2: PATIENT SERVICES */}
        <section className="space-y-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Patient Services
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 4: Book Appointment */}
            <div
              onClick={() => handleAction("appointment", "/patient/appointments")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <Calendar size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Book Appointment
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Schedule a consultation with a certified cardiac specialist or view pending requests.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Book Consultation</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>

            {/* Card 5: Personal Profile */}
            <div
              onClick={() => handleAction("profile", "/patient/profile")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <User size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Personal Profile
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Manage personal identification details, contact settings, and baseline health indicators.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Edit Settings</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 3: ACCOUNT */}
        <section className="space-y-4 pb-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Account
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 6: Sign Out */}
            <div
              onClick={handleLogoutAction}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-red-300 opacity-95 hover:opacity-100"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-red-500/10 text-red-500 flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <LogOut size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-red-500 transition-colors">
                  Sign Out
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Sign out safely from your CardioAI patient workspace session.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-red-500 mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Sign Out</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

