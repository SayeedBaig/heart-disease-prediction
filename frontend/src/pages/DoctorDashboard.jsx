import { useNavigate } from "react-router-dom";

function DoctorDashboard() {
  const doctor = JSON.parse(localStorage.getItem("doctor"));
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-100">

      {/* Header */}
      <div className="bg-blue-700 text-white py-6 shadow-md">
        <div className="max-w-6xl mx-auto px-6">
          <h1 className="text-4xl font-bold">Doctor Dashboard</h1>
          <p className="mt-2 text-lg">
            Welcome, {doctor?.full_name || "Doctor"}
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-8">

        {/* Doctor Information */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-10">

          <h2 className="text-2xl font-semibold text-blue-700 mb-6">
            Doctor Information
          </h2>

          <div className="grid md:grid-cols-2 gap-6">

            <div>
              <p className="text-gray-500">Full Name</p>
              <h3 className="text-xl font-semibold">
                {doctor?.full_name || "Doctor"}
              </h3>
            </div>

            <div>
              <p className="text-gray-500">Email</p>
              <h3 className="text-xl font-semibold">
                {doctor?.email || "-"}
              </h3>
            </div>

            <div>
              <p className="text-gray-500">Hospital</p>
              <h3 className="text-xl font-semibold">
                {doctor?.hospital || "-"}
              </h3>
            </div>

          </div>

        </div>

        {/* Quick Actions */}

        <h2 className="text-3xl font-bold mb-6 text-gray-800">
          Quick Actions
        </h2>

        <div className="grid md:grid-cols-3 gap-6">

          {/* Patients */}

          <div
            onClick={() => navigate("/doctor/patients")}
            className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition duration-300"
          >
            <div className="text-5xl mb-4">👥</div>

            <h3 className="text-2xl font-bold text-blue-700">
              Patients
            </h3>

            <p className="text-gray-600 mt-3">
              View all registered patients and their information.
            </p>
          </div>

          {/* History */}

          <div
            onClick={() => navigate("/history")}
            className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition duration-300"
          >
            <div className="text-5xl mb-4">📜</div>

            <h3 className="text-2xl font-bold text-green-700">
              History
            </h3>

            <p className="text-gray-600 mt-3">
              Review previous diagnoses and reports.
            </p>
          </div>

          {/* Appointments */}

          <div
            onClick={() => navigate("/appointments")}
            className="bg-white rounded-2xl shadow-lg p-8 cursor-pointer hover:shadow-2xl transition duration-300"
          >
            <div className="text-5xl mb-4">📅</div>

            <h3 className="text-2xl font-bold text-purple-700">
              Appointments
            </h3>

            <p className="text-gray-600 mt-3">
              Manage patient appointment requests.
            </p>
          </div>

        </div>

      </div>
    </div>
  );
}

export default DoctorDashboard;