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
  HeartPulse,
} from "lucide-react";
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

      setSuccess(response.data.message);

      setTimeout(() => {
        navigate("/doctor/login");
      }, 1500);
    } catch (err) {
      setError(err.response?.data?.detail || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-cyan-100 flex items-center justify-center p-6">

      <div className="w-full max-w-lg bg-white rounded-3xl shadow-2xl border border-slate-200 overflow-hidden">

        {/* Header */}

        <div className="bg-gradient-to-r from-blue-600 to-cyan-500 text-white p-8 text-center">

          <div className="flex justify-center mb-4">
            <div className="bg-white/20 p-4 rounded-2xl">
              <HeartPulse size={42} />
            </div>
          </div>

          <h1 className="text-3xl font-bold">
            Doctor Registration
          </h1>

          <p className="mt-2 text-blue-100">
            Create your CardioAI doctor account
          </p>

        </div>

        {/* Form */}

        <form
          onSubmit={handleSubmit}
          className="p-8 space-y-6"
        >

          {/* Full Name */}

          <div>

            <label className="block mb-2 font-medium text-slate-700">
              Full Name
            </label>

            <div className="relative">

              <User
                className="absolute left-4 top-3 text-slate-400"
                size={20}
              />

              <input
                type="text"
                name="full_name"
                value={formData.full_name}
                onChange={handleChange}
                required
                className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Dr. John Doe"
              />

            </div>

          </div>

          {/* Specialization */}

          <div>

            <label className="block mb-2 font-medium text-slate-700">
              Specialization
            </label>

            <div className="relative">

              <Stethoscope
                className="absolute left-4 top-3 text-slate-400"
                size={20}
              />

              <input
                type="text"
                name="specialization"
                value={formData.specialization}
                onChange={handleChange}
                required
                className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Cardiologist"
              />

            </div>

          </div>

          {/* Hospital */}

          <div>

            <label className="block mb-2 font-medium text-slate-700">
              Hospital
            </label>

            <div className="relative">

              <Building2
                className="absolute left-4 top-3 text-slate-400"
                size={20}
              />

              <input
                type="text"
                name="hospital"
                value={formData.hospital}
                onChange={handleChange}
                required
                className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Apollo Hospital"
              />

            </div>

          </div>

          {/* Email */}

          <div>

            <label className="block mb-2 font-medium text-slate-700">
              Email Address
            </label>

            <div className="relative">

              <Mail
                className="absolute left-4 top-3 text-slate-400"
                size={20}
              />

              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="doctor@example.com"
              />

            </div>

          </div>

          {/* Password */}

          <div>

            <label className="block mb-2 font-medium text-slate-700">
              Password
            </label>

            <div className="relative">

              <Lock
                className="absolute left-4 top-3 text-slate-400"
                size={20}
              />

              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                className="w-full pl-12 pr-12 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter password"
              />

              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-3 text-slate-500"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>

            </div>

          </div>

          {/* Error */}

          <div>
            <label className="block mb-2 font-medium text-slate-700">
              Confirm Password
            </label>
            <input
              type={showPassword ? "text" : "password"}
              name="confirm_password"
              value={formData.confirm_password}
              onChange={handleChange}
              required
              className="w-full px-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Confirm password"
            />
          </div>

          {error && (
            <div className="bg-red-100 border border-red-300 text-red-700 rounded-xl p-3">
              {error}
            </div>
          )}

          {/* Success */}

          {success && (
            <div className="bg-green-100 border border-green-300 text-green-700 rounded-xl p-3">
              {success}
            </div>
          )}

          {/* Button */}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-xl font-semibold text-lg hover:shadow-lg transition duration-300 disabled:opacity-60"
          >
            {loading ? "Creating Account..." : "Create Account"}
          </button>

          {/* Login */}

          <div className="text-center text-slate-600">

            Already have an account?{" "}

            <Link
              to="/doctor/login"
              className="text-blue-600 font-semibold hover:underline"
            >
              Login
            </Link>

          </div>

        </form>

      </div>

    </div>
  );
}

export default DoctorRegister;
