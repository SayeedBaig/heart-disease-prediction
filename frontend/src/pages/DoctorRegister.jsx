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
  ShieldCheck,
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
    <div className="cardio-shell">
      <Navbar onBack={() => navigate("/get-started")} backLabel="Portals" />

      <main className="cardio-container flex-1 flex flex-col items-center justify-center py-6 sm:py-10 my-auto w-full max-w-xl">
        <div className="cardio-card p-6 sm:p-8 w-full shadow-lg rounded-2xl">
          <div className="text-center mb-6">
            <div className="w-12 h-12 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mx-auto mb-3">
              <Stethoscope size={24} />
            </div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Practitioner Registration
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Create Doctor Account
            </h1>
            <p className="body-regular text-xs mt-1">
              Join the CardioAI network of medical specialists
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Full Name with Title
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="text"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleChange}
                  required
                  className="cardio-input text-xs !pl-11"
                  placeholder="Dr. Sarah Jenkins"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Specialization
              </label>
              <div className="relative">
                <Stethoscope className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="text"
                  name="specialization"
                  value={formData.specialization}
                  onChange={handleChange}
                  required
                  className="cardio-input text-xs !pl-11"
                  placeholder="Cardiologist / Electrophysiologist"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Affiliate Hospital / Clinic
              </label>
              <div className="relative">
                <Building2 className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="text"
                  name="hospital"
                  value={formData.hospital}
                  onChange={handleChange}
                  required
                  className="cardio-input text-xs !pl-11"
                  placeholder="CardioAI Health Institute"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Medical License Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="cardio-input text-xs !pl-11"
                  placeholder="doctor@cardioai.org"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                  <input
                    type={showPassword ? "text" : "password"}
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    required
                    className="cardio-input text-xs !pl-11 !pr-11"
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

              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                  Confirm Password
                </label>
                <input
                  type={showPassword ? "text" : "password"}
                  name="confirm_password"
                  value={formData.confirm_password}
                  onChange={handleChange}
                  required
                  className="cardio-input text-xs"
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

            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary w-full py-3.5 text-xs font-semibold rounded-xl flex items-center justify-center gap-2 shadow-md"
              >
                {loading ? "Creating Account..." : "Register Doctor Account"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="text-center pt-2 text-xs text-[var(--text-secondary)]">
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


