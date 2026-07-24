import { useState } from "react";
import { useNavigate } from "react-router-dom";

import Navbar from "../components/Navbar";
import api from "../services/api";

function Appointments() {
  const navigate = useNavigate();

  // -----------------------------
  // Patient Information
  // -----------------------------

  const patientId = localStorage.getItem("selected_patient_id");
  const patientCode = localStorage.getItem("selected_patient_code");
  const patientName = localStorage.getItem("selected_patient_name");

  const doctor = JSON.parse(localStorage.getItem("doctor"));

  // -----------------------------
  // Form State
  // -----------------------------

  const [appointment, setAppointment] = useState({
    preferred_date: "",
    preferred_time: "",
    symptoms: "",
    reason: "",
  });

  const [loading, setLoading] = useState(false);

  // -----------------------------
  // Book Appointment
  // -----------------------------

  const handleBookAppointment = async () => {
    if (
      !appointment.preferred_date ||
      !appointment.preferred_time ||
      !appointment.reason.trim()
    ) {
      alert("Please fill all required fields.");
      return;
    }

    try {
      setLoading(true);

      const response = await api.post("/appointments/", {
        patient_id: Number(patientId),
        doctor_id: doctor.doctor_id,
        preferred_date: appointment.preferred_date,
        preferred_time: appointment.preferred_time,
        symptoms: appointment.symptoms,
        reason: appointment.reason,
      });

      localStorage.setItem(
        "appointment_id",
        response.data.appointment_id
      );

      alert("Appointment booked successfully!");

      setAppointment({
        preferred_date: "",
        preferred_time: "",
        symptoms: "",
        reason: "",
      });

      navigate("/doctor/dashboard");

    } catch (err) {
      console.error(err);

      alert(
        err.response?.data?.detail ||
        "Failed to book appointment."
      );
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // UI
  // -----------------------------

  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-cyan-100 py-12">

        <div className="max-w-6xl mx-auto px-6">

          {/* Page Header */}

          <div className="text-center mb-12">

            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-blue-100 mb-5">

              <span className="text-4xl">
                📅
              </span>

            </div>

            <h1 className="text-5xl font-extrabold text-slate-900">
              Book Appointment
            </h1>

            <p className="text-gray-600 text-lg mt-4">
              Schedule a follow-up consultation with your doctor.
            </p>

          </div>

          {/* Patient Details */}

          <div className="bg-white rounded-3xl shadow-xl p-8 mb-10">

            <h2 className="text-3xl font-bold text-blue-700 mb-8">
              👤 Patient Details
            </h2>

            <div className="grid md:grid-cols-3 gap-6">

              <div className="bg-slate-50 rounded-2xl shadow-md p-6">

                <p className="text-gray-500">
                  Patient ID
                </p>

                <h3 className="text-2xl font-bold mt-2">
                  {patientCode || "N/A"}
                </h3>

              </div>

              <div className="bg-slate-50 rounded-2xl shadow-md p-6">

                <p className="text-gray-500">
                  Patient Name
                </p>

                <h3 className="text-2xl font-bold mt-2">
                  {patientName || "N/A"}
                </h3>

              </div>

              <div className="bg-slate-50 rounded-2xl shadow-md p-6">

                <p className="text-gray-500">
                  Database ID
                </p>

                <h3 className="text-2xl font-bold mt-2">
                  {patientId || "N/A"}
                </h3>

              </div>

            </div>

          </div>

          {/* Appointment Form */}

          <div className="bg-white rounded-3xl shadow-xl p-8">

            <h2 className="text-3xl font-bold text-blue-700 mb-8">
              📋 Appointment Details
            </h2>

            <div className="grid gap-6">
                          {/* Preferred Date */}

            <div>

              <label className="block text-lg font-semibold text-slate-700 mb-3">
                Preferred Date <span className="text-red-500">*</span>
              </label>

              <input
                type="date"
                value={appointment.preferred_date}
                onChange={(e) =>
                  setAppointment({
                    ...appointment,
                    preferred_date: e.target.value,
                  })
                }
                className="w-full border border-slate-300 rounded-2xl p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              />

            </div>

            {/* Preferred Time */}

            <div>

              <label className="block text-lg font-semibold text-slate-700 mb-3">
                Preferred Time <span className="text-red-500">*</span>
              </label>

              <input
                type="time"
                value={appointment.preferred_time}
                onChange={(e) =>
                  setAppointment({
                    ...appointment,
                    preferred_time: e.target.value,
                  })
                }
                className="w-full border border-slate-300 rounded-2xl p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              />

            </div>

            {/* Symptoms */}

            <div>

              <label className="block text-lg font-semibold text-slate-700 mb-3">
                Symptoms
              </label>

              <textarea
                rows="4"
                placeholder="Example: Chest pain, dizziness, shortness of breath..."
                value={appointment.symptoms}
                onChange={(e) =>
                  setAppointment({
                    ...appointment,
                    symptoms: e.target.value,
                  })
                }
                className="w-full border border-slate-300 rounded-2xl p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none"
              />

            </div>

            {/* Reason */}

            <div>

              <label className="block text-lg font-semibold text-slate-700 mb-3">
                Reason for Appointment <span className="text-red-500">*</span>
              </label>

              <textarea
                rows="4"
                placeholder="Briefly explain why you want to consult the doctor..."
                value={appointment.reason}
                onChange={(e) =>
                  setAppointment({
                    ...appointment,
                    reason: e.target.value,
                  })
                }
                className="w-full border border-slate-300 rounded-2xl p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all resize-none"
              />

            </div>

            {/* Appointment Tips */}

            <div className="bg-blue-50 border-l-4 border-blue-500 rounded-2xl p-6">

              <h3 className="text-xl font-bold text-blue-700 mb-3">
                💡 Appointment Tips
              </h3>

              <ul className="list-disc list-inside space-y-2 text-gray-700">

                <li>Select a convenient date and time.</li>

                <li>Clearly describe your symptoms.</li>

                <li>Mention any recent health changes.</li>

                <li>Bring previous medical reports during your visit.</li>

              </ul>

            </div>
                        {/* Action Buttons */}

            <div className="flex flex-col sm:flex-row gap-4 pt-4">

              <button
                onClick={handleBookAppointment}
                disabled={loading}
                className={`flex-1 py-4 rounded-2xl text-lg font-semibold transition-all duration-300 ${
                  loading
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              >
                {loading ? "Booking Appointment..." : "📅 Book Appointment"}
              </button>

              <button
                 onClick={() => navigate("/doctor/dashboard")}
                className="flex-1 py-4 rounded-2xl text-lg font-semibold border-2 border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white transition-all duration-300"
              >
                ← Back to Dashboard
              </button>

                        </div> {/* End Action Buttons */}

          </div> {/* End Grid */}

        </div> {/* End Appointment Form */}

      </div> {/* End max-width container */}

    </div> {/* End Background */}

    </>
  );
}

export default Appointments;