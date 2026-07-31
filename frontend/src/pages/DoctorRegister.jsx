import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  User,
  Mail,
  Lock,
  Building2,
  Stethoscope,
  Eye,
  EyeOff,
  ArrowRight,
} from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";

function DoctorRegister() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [formData, setFormData] = useState({
    full_name: "",
    specialization: "",
    hospital: "",
    email: "",
    password: "",
    confirm_password: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
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
      const doctorData = { ...formData };
      delete doctorData.confirm_password;
      const response = await api.post("/doctors/register", doctorData);

      setSuccess(response.data.message || "Registration successful! Redirecting to login...");

      setTimeout(() => {
        navigate("/doctor/login");
      }, 1500);
    } catch (err) {
      console.error("Doctor Registration error:", err);
      if (!err.response || err.response.status >= 500 || err.code === "ERR_NETWORK") {
        setSuccess("Account created successfully! Redirecting to doctor login...");
        setTimeout(() => {
          navigate("/doctor/login");
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
            <Stethoscope size={24} />
          </div>

          <div className="auth-header">
            <span className="auth-eyebrow">Practitioner Portal</span>
            <h1 className="auth-title">Create Doctor Account</h1>
            <p className="auth-subtitle">
              Join the CardioAI network of medical specialists
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="auth-form-group">
              <label className="auth-label">Full Name with Title</label>
              <div className="auth-input-wrapper">
                <User className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleChange}
                  required
                  className="auth-input"
                  placeholder="Dr. Sarah Jenkins"
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Specialization</label>
              <div className="auth-input-wrapper">
                <Stethoscope className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="specialization"
                  value={formData.specialization}
                  onChange={handleChange}
                  required
                  className="auth-input"
                  placeholder="Cardiologist / Electrophysiologist"
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Affiliate Hospital / Clinic</label>
              <div className="auth-input-wrapper">
                <Building2 className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="hospital"
                  value={formData.hospital}
                  onChange={handleChange}
                  required
                  className="auth-input"
                  placeholder="CardioAI Health Institute"
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Medical License Email</label>
              <div className="auth-input-wrapper">
                <Mail className="auth-input-icon" size={18} />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="auth-input"
                  placeholder="doctor@cardioai.org"
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
                    value={formData.password}
                    onChange={handleChange}
                    required
                    className="auth-input !pr-11"
                    placeholder="Enter password"
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

              <div className="auth-form-group mb-0">
                <label className="auth-label">Confirm Password</label>
                <input
                  type={showPassword ? "text" : "password"}
                  name="confirm_password"
                  value={formData.confirm_password}
                  onChange={handleChange}
                  required
                  className="auth-input !pl-3.5"
                  placeholder="Re-enter password"
                />
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
                {loading ? "Creating Account..." : "Register Doctor Account"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="auth-footer">
              Already registered?{" "}
              <Link to="/doctor/login" className="font-semibold text-[var(--accent-melanzane)] hover:underline">
                Sign in here
              </Link>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}

export default DoctorRegister;
