import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Stethoscope,
  HeartPulse,
  LogIn,
  UserPlus,
  ArrowRight,
  Activity,
} from "lucide-react";
import Navbar from "../components/Navbar";

export default function RoleSelection() {
  const navigate = useNavigate();

  return (
    <div className="cardio-shell">
      <Navbar breadcrumb="Portal Selection" />

      <main className="flex-1 cardio-container flex flex-col justify-center items-center py-10 my-auto w-full">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center max-w-3xl mx-auto mb-10"
        >


          <h1 className="h1-large text-[var(--text-primary)] tracking-tight mb-3">
            Select Your Portal
          </h1>

          <p className="body-regular text-sm md:text-base text-[var(--text-secondary)] leading-relaxed">
            Welcome to CardioAI. Choose your workspace below to sign in or register for a new account.
          </p>
        </motion.div>

        {/* Portal Cards Grid - Significantly Wider (80-90% width) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-6xl">
          {/* Doctor Portal Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="cardio-card p-8 flex flex-col justify-between hover:border-[var(--accent-primary-border)] transition-all duration-300 shadow-md hover:shadow-xl rounded-[28px] group"
          >
            <div>
              <div className="flex items-center justify-between mb-6">
                <div className="w-14 h-14 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center transition-transform group-hover:scale-105">
                  <Stethoscope size={28} />
                </div>
              </div>

              <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Doctor Portal
              </h2>

              <p className="body-regular text-xs md:text-sm mb-8 leading-relaxed">
                Comprehensive workspace for medical practitioners to manage patient cohorts, analyze ECG/Echo diagnostics, and issue signed clinical assessments.
              </p>
            </div>

            {/* Doctor Actions */}
            <div className="space-y-3 pt-4 border-t border-[var(--border-color)]">
              <button
                onClick={() => navigate("/doctor/login")}
                className="btn-primary w-full py-3.5 text-sm font-semibold rounded-full flex items-center justify-center gap-2 shadow-md hover:shadow-lg transition-all"
              >
                <LogIn size={18} />
                Doctor Login
              </button>

              <button
                onClick={() => navigate("/doctor/register")}
                className="btn-secondary w-full py-3.5 text-sm font-semibold rounded-full flex items-center justify-center gap-2 transition-all"
              >
                <UserPlus size={18} />
                Doctor Sign Up
              </button>
            </div>
          </motion.div>

          {/* Patient Portal Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="cardio-card p-8 flex flex-col justify-between hover:border-[var(--accent-primary-border)] transition-all duration-300 shadow-md hover:shadow-xl rounded-[28px] group"
          >
            <div>
              <div className="flex items-center justify-between mb-6">
                <div className="w-14 h-14 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center transition-transform group-hover:scale-105">
                  <HeartPulse size={28} />
                </div>
              </div>

              <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                Patient Portal
              </h2>

              <p className="body-regular text-xs md:text-sm mb-8 leading-relaxed">
                Personalized cardiovascular health suite to evaluate heart disease risk, explore Digital Twin simulations, view reports, and book doctor consultations.
              </p>
            </div>

            {/* Patient Actions */}
            <div className="space-y-3 pt-4 border-t border-[var(--border-color)]">
              <button
                onClick={() => navigate("/patient/login")}
                className="btn-primary w-full py-3.5 text-sm font-semibold rounded-full flex items-center justify-center gap-2 shadow-md hover:shadow-lg transition-all"
              >
                <LogIn size={18} />
                Patient Login
              </button>

              <button
                onClick={() => navigate("/patient/signup")}
                className="btn-secondary w-full py-3.5 text-sm font-semibold rounded-full flex items-center justify-center gap-2 transition-all"
              >
                <UserPlus size={18} />
                Patient Sign Up
              </button>
            </div>
          </motion.div>
        </div>


      </main>
    </div>
  );
}
