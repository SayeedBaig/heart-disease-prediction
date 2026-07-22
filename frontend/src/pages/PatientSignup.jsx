import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function PatientSignup() {
  const navigate = useNavigate();

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

      setSuccess("Registration successful! Redirecting to login...");

      setTimeout(() => {
        navigate("/patient/login");
      }, 1500);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Registration failed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100">
      <div className="w-full max-w-lg bg-white rounded-2xl shadow-lg p-8">

        <h1 className="text-3xl font-bold text-center mb-2">
          Patient Registration
        </h1>

        <p className="text-center text-gray-500 mb-8">
          Create your CardioAI account
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">

          <input
            type="text"
            name="full_name"
            placeholder="Full Name"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.full_name}
            onChange={handleChange}
            required
          />

          <input
            type="email"
            name="email"
            placeholder="Email"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.email}
            onChange={handleChange}
            required
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.password}
            onChange={handleChange}
            required
          />

          <input
            type="password"
            name="confirm_password"
            placeholder="Confirm Password"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.confirm_password}
            onChange={handleChange}
            required
          />

          <input
            type="text"
            name="phone"
            placeholder="Phone Number"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.phone}
            onChange={handleChange}
            required
          />

          <select
            name="gender"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.gender}
            onChange={handleChange}
          >
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>

          <input
            type="date"
            name="date_of_birth"
            className="w-full border rounded-lg px-4 py-3"
            value={formData.date_of_birth}
            onChange={handleChange}
            required
          />

          {error && (
            <p className="text-red-600 text-center">
              {error}
            </p>
          )}

          {success && (
            <p className="text-green-600 text-center">
              {success}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700"
          >
            {loading ? "Creating Account..." : "Register"}
          </button>

          <div className="text-center">
            Already have an account?{" "}
            <span
              className="text-blue-600 cursor-pointer hover:underline"
              onClick={() => navigate("/patient/login")}
            >
              Login
            </span>
          </div>

        </form>

      </div>
    </div>
  );
}

export default PatientSignup;
