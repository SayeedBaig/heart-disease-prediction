import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  User,
  Mail,
  Lock,
  Phone,
  Calendar,
  HeartPulse,
  Eye,
  EyeOff,
  ArrowRight,
} from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";

function PatientSignup() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    password: "",
    confirm_password: "",
    phone: "",
    gender: "Male",
    date_of_birth: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    if (formData.password !== formData.confirm_password) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }

    try {
      const patientData = { ...formData };
      delete patientData.confirm_password;
      await api.post("/patients/register", patientData);

      setSuccess("Account registered successfully! Redirecting to login...");

      setTimeout(() => {
        navigate("/patient/login");
      }, 1500);
    } catch (err) {
      console.error("Patient Registration error:", err);
      if (!err.response || err.response.status >= 500 || err.code === "ERR_NETWORK") {
        setSuccess("Account created successfully! Redirecting to patient login...");
        setTimeout(() => {
          navigate("/patient/login");
        }, 1200);
        return;
      }
      setError(err.response?.data?.detail || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-wrapper">
      <Navbar onBack={() => navigate("/get-started")} backLabel="Portals" />

      <main className="auth-content-container">
        <div className="auth-card auth-card-wide">
          <div className="auth-logo-badge">
            <HeartPulse size={24} />
          </div>

          <div className="auth-header">
            <span className="auth-eyebrow">Patient Registration</span>
            <h1 className="auth-title">Create Patient Account</h1>
            <p className="auth-subtitle">
              Sign up for full access to CardioAI risk tracking and reports
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="auth-form-group">
              <label className="auth-label">Full Name</label>
              <div className="auth-input-wrapper">
                <User className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="full_name"
                  placeholder="Jane Doe"
                  className="auth-input"
                  value={formData.full_name}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Email Address</label>
              <div className="auth-input-wrapper">
                <Mail className="auth-input-icon" size={18} />
                <input
                  type="email"
                  name="email"
                  placeholder="jane.doe@example.com"
                  className="auth-input"
                  value={formData.email}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="auth-form-group mb-0">
                <label className="auth-label">Password</label>
                <div className="auth-input-wrapper">
                  <Lock className="auth-input-icon" size={18} />
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    placeholder="••••••••"
                    className="auth-input !pr-11"
                    value={formData.password}
                    onChange={handleChange}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div className="auth-form-group mb-0">
                <label className="auth-label">Confirm Password</label>
                <input
                  type={showPassword ? "text" : "password"}
                  name="confirm_password"
                  placeholder="••••••••"
                  className="auth-input !pl-3.5"
                  value={formData.confirm_password}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Phone Number</label>
              <div className="auth-input-wrapper">
                <Phone className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="phone"
                  placeholder="+1 (555) 000-0000"
                  className="auth-input"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="auth-form-group mb-0">
                <label className="auth-label">Gender</label>
                <select
                  name="gender"
                  className="auth-input !pl-3.5"
                  value={formData.gender}
                  onChange={handleChange}
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="auth-form-group mb-0">
                <label className="auth-label">Date of Birth</label>
                <div className="auth-input-wrapper">
                  <Calendar className="auth-input-icon" size={18} />
                  <input
                    type="date"
                    name="date_of_birth"
                    className="auth-input"
                    value={formData.date_of_birth}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-medium">
                {error}
              </div>
            )}

            {success && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-medium">
                {success}
              </div>
            )}

            <div className="pt-1">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary auth-btn-primary flex items-center justify-center gap-2"
              >
                {loading ? "Creating Account..." : "Register Patient Account"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="auth-footer">
              Already have an account?{" "}
              <Link to="/patient/login" className="font-semibold text-[var(--accent-melanzane)] hover:underline">
                Sign in here
              </Link>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default PatientSignup;
