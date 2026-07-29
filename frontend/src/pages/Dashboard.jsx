import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import PageBackground from "../components/PageBackground";

import {
  User,
  HeartPulse,
  FileText,
  History,
  Activity,
} from "lucide-react";

function Dashboard() {
  const patientId = localStorage.getItem("patient_id");
  const patientName = localStorage.getItem("patient_name");
  const patientEmail = localStorage.getItem("patient_email");

  return (
    <div className="cardio-shell">
      <Navbar />

      <main className="cardio-container py-8 flex-1 w-full relative overflow-hidden">
        <PageBackground />

        <div className="relative z-10 w-full">

          {/* ================= Welcome Card ================= */}

          <div
            className="
            relative
            overflow-hidden
            rounded-3xl
            bg-gradient-to-r
            from-blue-600
            via-blue-700
            to-indigo-800
            shadow-2xl
            p-10
            text-white"
          >

            <Activity
              className="
              absolute
              right-8
              top-1/2
              -translate-y-1/2
              w-56
              h-56
              text-white/10"
            />

            <div className="relative z-10">

              <h1 className="text-4xl font-bold">
                Welcome 👋
              </h1>

              <p className="text-2xl mt-3 font-medium">
                {patientName || "Patient"}
              </p>

              <p className="mt-5 text-blue-100">
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

          </div>

          {/* ================= Dashboard Cards ================= */}

          <div className="grid md:grid-cols-3 gap-8 mt-10">

            {/* Patient */}

            <div
              className="
              bg-white
              rounded-2xl
              border
              border-slate-100
              shadow-lg
              p-7
              transition-all
              duration-300"
            >

              <div className="w-16 h-16 rounded-2xl bg-blue-100 flex items-center justify-center mb-5">
                <User
                  size={30}
                  className="text-blue-600"
                />
              </div>

              <h2 className="text-2xl font-bold">
                Patient Profile
              </h2>

              <p className="text-gray-500 mt-3">
                Registered successfully.
              </p>

            </div>

            {/* Latest Prediction */}

            <div
              className="
              bg-white
              rounded-2xl
              border
              border-slate-100
              shadow-lg
              p-7
              transition-all
              duration-300"
            >

              <div className="w-16 h-16 rounded-2xl bg-red-100 flex items-center justify-center mb-5">

                <HeartPulse
                  size={30}
                  className="text-red-500"
                />

              </div>

              <h2 className="text-2xl font-bold">
                Latest Prediction
              </h2>

              <p className="text-gray-500 mt-3">
                No prediction available yet.
              </p>

            </div>

            {/* History */}

            <div
              className="
              bg-white
              rounded-2xl
              border
              border-slate-100
              shadow-lg
              p-7
              transition-all
              duration-300"
            >

              <div className="w-16 h-16 rounded-2xl bg-green-100 flex items-center justify-center mb-5">

                <History
                  size={30}
                  className="text-green-600"
                />

              </div>

              <h2 className="text-2xl font-bold">
                Prediction History
              </h2>

              <p className="text-gray-500 mt-3">
                View all previous predictions.
              </p>

            </div>

          </div>

          {/* ================= Quick Actions ================= */}

          <div
            className="
            bg-white
            rounded-3xl
            border
            border-slate-100
            shadow-xl
            p-8
            mt-10"
          >

            <h2 className="text-3xl font-bold mb-8">
              Quick Actions
            </h2>

            <div className="grid md:grid-cols-3 gap-6">

              {/* Diagnosis */}

              <Link
                to="/diagnose"
                className="
                bg-gradient-to-r
                from-blue-600
                to-indigo-700
                rounded-2xl
                p-6
                text-white
                transition-all
                duration-300"
              >

                <HeartPulse size={36} />

                <h3 className="mt-4 text-xl font-bold">
                  Start Diagnosis
                </h3>

                <p className="mt-2 text-blue-100">
                  Begin a new AI heart disease assessment.
                </p>

              </Link>

              {/* Reports */}

              <Link
                to="/reports"
                className="
                bg-white
                border
                border-slate-200
                rounded-2xl
                p-6
                transition-all
                duration-300"
              >

                <FileText
                  size={36}
                  className="text-green-600"
                />

                <h3 className="mt-4 text-xl font-bold">
                  Reports
                </h3>

                <p className="mt-2 text-gray-500">
                  Download patient and doctor reports.
                </p>

              </Link>

              {/* History */}

              <Link
                to="/history"
                className="
                bg-white
                border
                border-slate-200
                rounded-2xl
                p-6
                transition-all
                duration-300"
              >

                <History
                  size={36}
                  className="text-blue-600"
                />

                <h3 className="mt-4 text-xl font-bold">
                  Prediction History
                </h3>

                <p className="mt-2 text-gray-500">
                  Review previous AI predictions.
                </p>

              </Link>

            </div>

          </div>

        </div>

      </main>
    </div>
  );
}

export default Dashboard;