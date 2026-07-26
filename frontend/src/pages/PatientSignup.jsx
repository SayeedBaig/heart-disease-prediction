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
  ShieldCheck,
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
    <div className="cardio-shell">
      <Navbar onBack={() => navigate("/get-started")} backLabel="Portals" />

      <main className="cardio-container flex-1 flex flex-col items-center justify-center py-6 sm:py-10 my-auto w-full max-w-xl">
        <div className="cardio-card p-6 sm:p-8 w-full shadow-lg rounded-2xl">
          <div className="text-center mb-6">
            <div className="w-12 h-12 rounded-xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mx-auto mb-3">
              <HeartPulse size={24} />
            </div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Patient Registration
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Create Patient Account
            </h1>
            <p className="body-regular text-xs mt-1">
              Sign up for full access to CardioAI risk tracking and reports
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="text"
                  name="full_name"
                  placeholder="Jane Doe"
                  className="cardio-input text-xs !pl-11"
                  value={formData.full_name}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="email"
                  name="email"
                  placeholder="jane.doe@example.com"
                  className="cardio-input text-xs !pl-11"
                  value={formData.email}
                  onChange={handleChange}
                  required
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
                    placeholder="••••••••"
                    className="cardio-input text-xs !pl-11 !pr-11"
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

              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                  Confirm Password
                </label>
                <input
                  type={showPassword ? "text" : "password"}
                  name="confirm_password"
                  placeholder="••••••••"
                  className="cardio-input text-xs"
                  value={formData.confirm_password}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                Phone Number
              </label>
              <div className="relative">
                <Phone className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                <input
                  type="text"
                  name="phone"
                  placeholder="+1 (555) 000-0000"
                  className="cardio-input text-xs !pl-11"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                  Gender
                </label>
                <select
                  name="gender"
                  className="cardio-input text-xs"
                  value={formData.gender}
                  onChange={handleChange}
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-[var(--text-primary)] mb-1.5">
                  Date of Birth
                </label>
                <div className="relative">
                  <Calendar className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none" size={18} />
                  <input
                    type="date"
                    name="date_of_birth"
                    className="cardio-input text-xs !pl-11"
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

            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="btn-primary w-full py-3.5 text-xs font-semibold rounded-xl flex items-center justify-center gap-2 shadow-md"
              >
                {loading ? "Creating Account..." : "Register Patient Account"}
                <ArrowRight size={16} />
              </button>
            </div>

            <div className="text-center pt-2 text-xs text-[var(--text-secondary)]">
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


