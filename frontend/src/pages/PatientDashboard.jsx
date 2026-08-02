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
import ChatbotWidget from "../components/ChatbotWidget";
import heart from "../assets/heart.png";

export default function PatientDashboard({ onNavigate, onLogout }) {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const displayName = patient.full_name || patient.name || "Patient";

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

  const cards = [
    {
      key: "profile",
      route: "/patient/profile",
      icon: User,
      title: "Personal Profile",
      desc: "Manage personal details and baseline health indicators.",
      cta: "Edit Settings",
    },
    {
      key: "intake",
      route: "/patient/intake",
      icon: UploadCloud,
      title: "Upload Medical Data",
      desc: "Submit clinical vitals, ECG waveforms, and echo scans for rapid AI assessment.",
      cta: "Start Assessment",
    },
    {
      key: "report",
      route: "/patient/reports",
      icon: FileText,
      title: "View Reports",
      desc: "Access, download, and review your official diagnostic summaries and screening history.",
      cta: "Open Reports",
    },
    {
      key: "appointment",
      route: "/patient/appointments",
      icon: Calendar,
      title: "Book Appointment",
      desc: "Schedule a consultation with a certified cardiac specialist.",
      cta: "Book Consultation",
    },
    {
      key: "twin",
      route: "/patient/digital-twin",
      icon: Activity,
      title: "Digital Twin",
      desc: "Interactive parameter simulation to model real-time cardiovascular risk trajectories.",
      cta: "Launch Simulation",
    },
    {
      key: "signout",
      icon: LogOut,
      title: "Sign Out",
      desc: "Sign out safely from your CardioAI patient workspace session.",
      cta: "Sign Out",
      isSignOut: true,
    },
  ];

  return (
    <div className="cardio-shell overflow-x-hidden relative">
      <Navbar />

      <img
        src={heart}
        alt=""
        aria-hidden="true"
        className="pointer-events-none select-none fixed right-[-5%] top-1/3 -translate-y-1/2 w-[420px] md:w-[560px] opacity-[0.06] z-0"
      />

      <main className="relative z-10 cardio-container py-8 md:py-10 w-full flex-1 max-w-[1400px] mx-auto space-y-8">
        {/* HERO SECTION */}
        <section className="pt-2 pb-2">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-[var(--accent-primary)] mb-2">
            Overview
          </p>
          <h1 className="text-[var(--text-primary)] text-4xl md:text-5xl font-extrabold tracking-tight leading-[1.05]">
            Welcome back, {displayName}
          </h1>
          <p className="text-sm md:text-base mt-3 text-[var(--text-secondary)] leading-relaxed max-w-xl">
            Choose a workspace below to upload medical data, review reports, or simulate future cardiovascular outcomes.
          </p>
        </section>

        {/* SINGLE UNIFIED GRID - all cards together, equal treatment */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {cards.map(({ key, route, icon: Icon, title, desc, cta, isSignOut }) => (
            <div
              key={key}
              onClick={() => (isSignOut ? handleLogoutAction() : handleAction(key, route))}
              className={`group cursor-pointer cardio-card rounded-2xl p-6 flex flex-col justify-between min-h-[200px] border transition-all duration-300 hover:shadow-lg ${
                isSignOut
                  ? "border-[var(--border-color)] hover:border-red-300"
                  : "border-[var(--border-color)] hover:border-[var(--accent-primary-border)]"
              }`}
            >
              <div>
                <div
                  className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110 ${
                    isSignOut
                      ? "bg-red-500/10 text-red-500"
                      : "bg-[var(--accent-primary-light)] text-[var(--accent-primary)]"
                  }`}
                >
                  <Icon size={20} />
                </div>
                <h3
                  className={`text-sm font-bold mb-1.5 transition-colors ${
                    isSignOut ? "text-[var(--text-primary)] group-hover:text-red-500" : "text-[var(--text-primary)] group-hover:text-[var(--accent-primary)]"
                  }`}
                >
                  {title}
                </h3>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                  {desc}
                </p>
              </div>
              <div
                className={`flex items-center text-xs font-semibold mt-4 pt-3 border-t border-[var(--border-color)] ${
                  isSignOut ? "text-red-500" : "text-[var(--accent-primary)]"
                }`}
              >
                <span>{cta}</span>
                <ArrowRight size={13} className="ml-1.5 transition-transform group-hover:translate-x-1" />
              </div>
            </div>
          ))}
        </section>
      </main>

      <ChatbotWidget mode="patient" />
    </div>
  );
}
