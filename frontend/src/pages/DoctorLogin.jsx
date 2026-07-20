import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Mail,
  Lock,
  Eye,
  EyeOff,
  HeartPulse,
} from "lucide-react";
import api from "../services/api";

function DoctorLogin() {
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

    try {
      const response = await api.post("/doctors/login", {
        email,
        password,
      });

      // Save JWT Token
      localStorage.setItem(
        "access_token",
        response.data.access_token
      );

      // Save Doctor Details
      localStorage.setItem(
        "doctor",
        JSON.stringify(response.data.doctor)
      );

      navigate("/doctor/dashboard");
    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail || "Login failed."
      );
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
            Welcome Back
          </h1>

          <p className="mt-2 text-blue-100">
            Sign in to your CardioAI Doctor Account
          </p>

        </div>

        {/* Login Form */}

        <form
          onSubmit={handleLogin}
          className="p-8 space-y-6"
        >

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
                placeholder="doctor@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full pl-12 pr-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
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
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full pl-12 pr-12 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
              />

              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-3 text-slate-500 hover:text-blue-600"
              >
                {showPassword ? (
                  <EyeOff size={20} />
                ) : (
                  <Eye size={20} />
                )}
              </button>

            </div>

          </div>

          {/* Error */}

          {error && (
            <div className="bg-red-100 border border-red-300 text-red-700 rounded-xl p-3">
              {error}
            </div>
          )}

          {/* Login Button */}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 text-white py-3 rounded-xl font-semibold text-lg hover:shadow-lg transition duration-300 disabled:opacity-60"
          >
            {loading ? "Signing In..." : "Sign In"}
          </button>

          {/* Register */}

          <div className="text-center text-slate-600">

            Don't have an account?{" "}

            <Link
              to="/doctor/register"
              className="text-blue-600 font-semibold hover:underline"
            >
              Register
            </Link>

          </div>

        </form>

      </div>

    </div>
  );
}

export default DoctorLogin;