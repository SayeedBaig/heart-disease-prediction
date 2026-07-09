import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  User,
  HeartPulse,
  FileText,
  History,
} from "lucide-react";

function Dashboard() {
  const patientId = localStorage.getItem("patient_id");
const patientName = localStorage.getItem("patient_name");
const patientEmail = localStorage.getItem("patient_email");
  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-slate-100 py-10 px-6">

        <div className="max-w-6xl mx-auto">

          {/* Welcome */}

          <div className="bg-gradient-to-r from-blue-600 to-blue-800 rounded-2xl text-white p-8 shadow-xl">

            <h1 className="text-4xl font-bold">
              Welcome 👋
            </h1>

            <p className="text-xl mt-2">
              {patientName || "Patient"}
            </p>

            <p className="mt-3 text-blue-100">
              Patient ID :
              <span className="font-semibold ml-2">
                {patientId || "Not Available"}
              </span>
            </p>
            <p className="mt-2 text-blue-100">
  Email :
  <span className="font-semibold ml-2">
    {patientEmail || "Not Available"}
  </span>
</p>

          </div>

          {/* Cards */}

          <div className="grid md:grid-cols-3 gap-8 mt-10">

            {/* Patient */}

            <div className="bg-white rounded-xl shadow-lg p-6">

              <User
                size={42}
                className="text-blue-600 mb-4"
              />

              <h2 className="text-xl font-bold">
                Patient Profile
              </h2>

              <p className="text-gray-500 mt-2">
                Registered successfully.
              </p>

            </div>

            {/* Latest */}

            <div className="bg-white rounded-xl shadow-lg p-6">

              <HeartPulse
                size={42}
                className="text-red-500 mb-4"
              />

              <h2 className="text-xl font-bold">
                Latest Prediction
              </h2>

              <p className="text-gray-500 mt-2">
                No prediction available yet.
              </p>

            </div>

            {/* History */}

            <div className="bg-white rounded-xl shadow-lg p-6">

              <History
                size={42}
                className="text-green-600 mb-4"
              />

              <h2 className="text-xl font-bold">
                Prediction History
              </h2>

              <p className="text-gray-500 mt-2">
                View all previous predictions.
              </p>

            </div>

          </div>

          {/* Quick Actions */}

          <div className="bg-white rounded-xl shadow-lg p-8 mt-10">

            <h2 className="text-2xl font-bold mb-8">
              Quick Actions
            </h2>

            <div className="flex flex-wrap gap-6">

              <Link
                to="/diagnose"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-xl font-semibold"
              >
                🫀 Start Diagnosis
              </Link>

              <Link
                to="/reports"
                className="bg-green-600 hover:bg-green-700 text-white px-8 py-4 rounded-xl font-semibold"
              >
                <FileText className="inline mr-2" size={18} />
                Reports
              </Link>

              <Link
                to="/history"
                className="bg-slate-700 hover:bg-slate-800 text-white px-8 py-4 rounded-xl font-semibold"
              >
                <History className="inline mr-2" size={18} />
                History
              </Link>

            </div>

          </div>

        </div>

      </div>
    </>
  );
}

export default Dashboard;