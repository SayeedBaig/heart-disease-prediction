import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  HeartPulse,
  ArrowRight,
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
    <div className="auth-page-wrapper">
      <Navbar onBack={() => navigate("/get-started")} backLabel="Portals" />

      <main className="auth-content-container">
        <div className="auth-card">
          <div className="auth-logo-badge">
            <HeartPulse size={24} />
          </div>

          <div className="auth-header">
            <span className="auth-eyebrow">Patient Portal</span>
            <h1 className="auth-title">Patient Login</h1>
            <p className="auth-subtitle">
              Log in to access your cardiovascular health suite
            </p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div className="auth-form-group">
              <label className="auth-label">Email Address</label>
              <div className="auth-input-wrapper">
                <Mail className="auth-input-icon" size={18} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="auth-input"
                  placeholder="patient@example.com"
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Password</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" size={18} />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="auth-input !pr-11"
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

            <div className="pt-1">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary auth-btn-primary flex items-center justify-center gap-2"
              >
                {loading ? "Logging in..." : "Log In to Patient Workspace"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="auth-footer">
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
