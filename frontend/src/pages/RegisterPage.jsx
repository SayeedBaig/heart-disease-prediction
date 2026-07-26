import { useState } from "react";
import { useNavigate } from "react-router-dom";
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
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 flex items-center justify-center py-10">

        <div className="bg-white shadow-xl rounded-xl p-10 w-full max-w-xl">

          <h1 className="text-4xl font-bold text-blue-900 mb-2">
            Patient Registration
          </h1>

          <p className="text-gray-500 mb-8">
            Register before starting the diagnosis.
          </p>

          <div className="space-y-5">

            <input
              type="text"
              name="full_name"
              placeholder="Full Name"
              value={formData.full_name}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
            />

            <input
              type="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
            />

            <input
              type="text"
              name="phone"
              placeholder="Phone Number"
              value={formData.phone}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
            />

            <select
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>

            <input
              type="date"
              name="date_of_birth"
              value={formData.date_of_birth}
              onChange={handleChange}
              className="w-full border rounded-lg p-3"
            />

            {error && (
              <p className="text-red-600">
                {error}
              </p>
            )}

            <button
              onClick={handleRegister}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-semibold"
            >
              {loading
                ? "Registering..."
                : "Continue to Diagnosis"}
            </button>

          </div>

        </div>

      </div>
    </>
  );
}

export default RegisterPage;