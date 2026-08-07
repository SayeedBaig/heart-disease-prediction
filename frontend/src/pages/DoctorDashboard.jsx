import { useNavigate } from "react-router-dom";
import {
  Users,
  Calendar,
  FileText,
  User,
  LogOut,
  ArrowRight,
  Stethoscope,
  Building2,
  CheckCircle2,
} from "lucide-react";
import Navbar from "../components/Navbar";
import ChatbotWidget from "../components/ChatbotWidget";

export default function DoctorDashboard({ doctor, onNavigate, onLogout }) {
  const navigate = useNavigate();
  const currentDoctor = doctor || JSON.parse(localStorage.getItem("cardio-doctor") || "{}");

  const handleAction = (key, route) => {
    if (onNavigate) {
      onNavigate(key);
    } else if (route) {
      navigate(route);
    }
  };

  const handleLogoutAction = () => {
    if (onLogout) {
      onLogout();
    } else {
      localStorage.removeItem("doctor_access_token");
      localStorage.removeItem("cardio-doctor");
      navigate("/doctor/login");
    }
  };

  const doctorName = currentDoctor.full_name || currentDoctor.name || "Sarah Wilson";
  const hospitalName = currentDoctor.hospital || "Apollo Hospitals";
  const specialization = currentDoctor.specialization || "Cardiology";

  return (
    <div className="cardio-shell overflow-x-hidden">
      <Navbar />

      <main className="cardio-container py-8 md:py-10 w-full flex-1 max-w-[1400px] mx-auto space-y-12">
        {/* HERO SECTION - Compact height, clean whitespace */}
        <section className="cardio-card p-6 md:p-8 relative overflow-hidden bg-gradient-to-r from-[var(--card-bg)] via-[var(--card-hover)] to-[var(--accent-melanzane-light)] border border-[var(--border-color)] rounded-2xl shadow-sm">
          <div className="relative z-10 max-w-3xl">
            <h1 className="h1-large text-[var(--text-primary)] text-3xl md:text-4xl font-extrabold tracking-tight">
              Welcome, Dr. {doctorName}
            </h1>
            <p className="body-regular text-xs md:text-sm mt-2 text-[var(--text-secondary)] font-medium">
              {hospitalName} • {specialization} • Consultant
            </p>
          </div>
        </section>

        {/* SECTION 1: PATIENT MANAGEMENT */}
        <section className="space-y-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Patient Management
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 1: Patient Records */}
            <div
              onClick={() => handleAction("patients", "/doctor/patients")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <Users size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Patient Records
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Access patient health records, diagnostic histories, vitals, and clinical charts.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>View Patient Registry</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>

            {/* Card 2: Consultation Schedule */}
            <div
              onClick={() => handleAction("appointments", "/doctor/appointments")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <Calendar size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Consultation Schedule
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Review upcoming diagnostic appointments, tele-consultations, and pending requests.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Open Schedule</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 2: ACCOUNT */}
        <section className="space-y-4 pb-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            Account
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Card 4: Doctor Profile */}
            <div
              onClick={() => handleAction("profile", "/doctor/profile")}
              className="cardio-card-interactive p-6 md:p-7 flex flex-col justify-between min-h-[220px] rounded-2xl group transition-all duration-300 hover:shadow-xl hover:border-[var(--accent-melanzane-border)]"
            >
              <div>
                <div className="w-12 h-12 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mb-5 transition-transform group-hover:scale-110">
                  <User size={24} />
                </div>
                <h3 className="text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-melanzane)] transition-colors">
                  Doctor Profile
                </h3>
                <p className="body-regular text-xs md:text-sm leading-relaxed">
                  Manage practitioner credentials, hospital department settings, and consultation availability.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-[var(--accent-melanzane)] mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>View Profile</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>

            {/* Card 5: Sign Out */}
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
                  Sign out safely from the CardioAI clinician workspace session.
                </p>
              </div>
              <div className="flex items-center text-xs font-semibold text-red-500 mt-6 pt-4 border-t border-[var(--border-color)]">
                <span>Sign Out</span>
                <ArrowRight size={14} className="ml-1.5 transition-transform group-hover:translate-x-1.5" />
              </div>
            </div>
          </div>
        </section>

        {/* FUTURE READY: AI WORKSPACE (Hidden until modules implemented) */}
        {/*
        <section className="hidden space-y-4">
          <h2 className="text-xs md:text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
            AI Workspace (Future Release)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>ECG Analysis</div>
            <div>Echo Analysis</div>
            <div>Clinical AI Assistant</div>
            <div>Digital Twin Review</div>
          </div>
        </section>
        */}
      <ChatbotWidget mode="doctor" />
      </main>
    </div>
  );
}

