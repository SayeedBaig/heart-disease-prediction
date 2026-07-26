import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  HeartPulse,
  ArrowRight,
  ShieldCheck,
} from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";
import { clearPatientSession } from "../services/portalService";

function PatientLogin() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    clearPatientSession();

    try {
      // Step 1: Login
      const loginResponse = await api.post("/patients/login", {
        email,
        password,
      });

      const token = loginResponse.data.access_token;
      localStorage.setItem("access_token", token);

      // Step 2: Fetch logged-in patient profile
      const profileResponse = await api.get("/patients/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const patientData = profileResponse.data;

      localStorage.setItem("patient", JSON.stringify(patientData));
      localStorage.setItem("cardio-patient", JSON.stringify(patientData));

      navigate("/patient/dashboard");
    } catch (err) {
      console.error("Patient Login error:", err);
      setError(err.response?.data?.detail || "Unable to sign in. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cardio-shell">
      <Navbar onBack={() => navigate("/get-started")} backLabel="Portals" />

      <main className="cardio-container flex-1 flex flex-col items-center justify-center py-6 sm:py-10 my-auto w-full max-w-md">
        <div className="cardio-card p-6 sm:p-8 w-full shadow-lg rounded-2xl">
          <div className="text-center mb-6">
            <div className="w-12 h-12 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mx-auto mb-3">
              <HeartPulse size={24} />
            </div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Patient Portal Access
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Patient Login
            </h1>
            <p className="body-regular text-xs mt-1">
              Log in to access your cardiovascular health suite
            </p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="cardio-input text-xs !pl-11"
                  placeholder="patient@example.com"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="cardio-input text-xs !pl-11 !pr-11"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-medium">
                {error}
              </div>
            )}

            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary w-full py-3.5 text-xs font-semibold rounded-xl flex items-center justify-center gap-2 shadow-md"
              >
                {loading ? "Logging in..." : "Log In to Patient Workspace"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="text-center pt-2 text-xs text-[var(--text-secondary)]">
              Don't have an account?{" "}
              <Link to="/patient/signup" className="font-semibold text-[var(--accent-melanzane)] hover:underline">
                Register here
              </Link>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default PatientLogin;

