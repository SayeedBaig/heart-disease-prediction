import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { HeartPulse, ArrowRight } from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";

function RegisterPage() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    phone: "",
    gender: "",
    date_of_birth: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleRegister = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await api.post("/patients/register", formData);

      // Save patient information
      localStorage.setItem("patient_id", response.data.patient_id);
      localStorage.setItem(
        "patient_name",
        response.data.full_name
      );
      localStorage.setItem("patient_email", formData.email);

      navigate("/dashboard");

    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail ||
        "Registration failed."
      );

    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-wrapper">
      <Navbar />

      <main className="auth-content-container">
        <div className="auth-card auth-card-wide">
          <div className="auth-logo-badge">
            <HeartPulse size={24} />
          </div>

          <div className="auth-header">
            <span className="auth-eyebrow">Diagnostic Registration</span>
            <h1 className="auth-title">Patient Registration</h1>
            <p className="auth-subtitle">
              Register details before starting your AI diagnosis
            </p>
          </div>

          <div className="space-y-4">
            <div className="auth-form-group">
              <label className="auth-label">Full Name</label>
              <input
                type="text"
                name="full_name"
                placeholder="Full Name"
                value={formData.full_name}
                onChange={handleChange}
                className="auth-input !pl-3.5"
              />
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Email Address</label>
              <input
                type="email"
                name="email"
                placeholder="Email Address"
                value={formData.email}
                onChange={handleChange}
                className="auth-input !pl-3.5"
              />
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Phone Number</label>
              <input
                type="text"
                name="phone"
                placeholder="Phone Number"
                value={formData.phone}
                onChange={handleChange}
                className="auth-input !pl-3.5"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="auth-form-group mb-0">
                <label className="auth-label">Gender</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="auth-input !pl-3.5"
                >
                  <option value="">Select Gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              <div className="auth-form-group mb-0">
                <label className="auth-label">Date of Birth</label>
                <input
                  type="date"
                  name="date_of_birth"
                  value={formData.date_of_birth}
                  onChange={handleChange}
                  className="auth-input !pl-3.5"
                />
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-medium">
                {error}
              </div>
            )}

            <div className="pt-2">
              <button
                onClick={handleRegister}
                disabled={loading}
                className="btn-primary auth-btn-primary flex items-center justify-center gap-2"
              >
                {loading
                  ? "Registering..."
                  : "Continue to Diagnosis"}
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default RegisterPage;