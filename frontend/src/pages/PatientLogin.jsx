import { useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function PatientLogin() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");

    try {
      // Step 1: Login
const loginResponse = await api.post("/patients/login", {
  email,
  password,
});

// Step 2: Store JWT
const token = loginResponse.data.access_token;

localStorage.setItem("access_token", token);

// Step 3: Fetch logged-in patient profile
const profileResponse = await api.get("/patients/me", {
  headers: {
    Authorization: `Bearer ${token}`,
  },
});

// Step 4: Store patient profile
localStorage.setItem(
  "patient",
  JSON.stringify(profileResponse.data)
);

// Step 5: Navigate
navigate("/patient/dashboard");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Invalid email or password."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2">
          Patient Login
        </h1>

        <p className="text-center text-gray-500 mb-8">
          Login to access your health records
        </p>

        <form onSubmit={handleLogin} className="space-y-5">
          <div>
            <label className="block mb-2 font-medium">
              Email
            </label>

            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full border rounded-lg px-4 py-3"
              placeholder="Enter your email"
            />
          </div>

          <div>
            <label className="block mb-2 font-medium">
              Password
            </label>

            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full border rounded-lg px-4 py-3"
              placeholder="Enter your password"
            />
          </div>

          {error && (
            <div className="text-red-600 text-sm text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? "Logging in..." : "Login"}
          </button>
          <div className="text-center mt-4">
  <p className="text-gray-600">
    Don't have an account?{" "}
    <span
      className="text-blue-600 cursor-pointer hover:underline"
      onClick={() => navigate("/patient/signup")}
    >
      Register
    </span>
  </p>
</div>
        </form>
      </div>
    </div>
  );
}

export default PatientLogin;