import { useNavigate } from "react-router-dom";
import { CalendarDays, ArrowLeft } from "lucide-react";

function PatientAppointments() {
  const navigate = useNavigate();

  const appointments = [
    {
      id: 1,
      doctor: "Dr. Cardiologist",
      date: "25 Jul 2026",
      time: "10:30 AM",
      status: "Confirmed",
    },
    {
      id: 2,
      doctor: "Dr. Heart Specialist",
      date: "02 Aug 2026",
      time: "02:00 PM",
      status: "Pending",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-100 p-8">

      {/* Header */}

      <button
        onClick={() => navigate("/patient/dashboard")}
        className="flex items-center gap-2 text-blue-600 hover:text-blue-800 mb-5"
      >
        <ArrowLeft size={18} />
        Back to Dashboard
      </button>

      <h1 className="text-4xl font-bold text-slate-800">
        My Appointments
      </h1>

      <p className="text-slate-500 mt-2 mb-8">
        View all your scheduled appointments.
      </p>

      {/* Appointment Cards */}

      <div className="space-y-6">

        {appointments.map((appointment) => (

          <div
            key={appointment.id}
            className="bg-white rounded-2xl shadow-md p-6"
          >

            <div className="flex justify-between items-center">

              <div>

                <div className="flex items-center gap-3 mb-3">
                  <CalendarDays className="text-blue-600" />

                  <h2 className="text-xl font-semibold">
                    {appointment.doctor}
                  </h2>
                </div>

                <p>
                  <strong>Date:</strong> {appointment.date}
                </p>

                <p>
                  <strong>Time:</strong> {appointment.time}
                </p>

              </div>

              <span
                className={`px-4 py-2 rounded-full text-sm font-semibold ${
                  appointment.status === "Confirmed"
                    ? "bg-green-100 text-green-700"
                    : "bg-yellow-100 text-yellow-700"
                }`}
              >
                {appointment.status}
              </span>

            </div>

          </div>

        ))}

      </div>

      {/* Book Appointment */}

      <div className="mt-8">
        <button
          className="bg-blue-600 text-white px-6 py-3 rounded-xl hover:bg-blue-700"
        >
          + Book New Appointment
        </button>
      </div>

    </div>
  );
}

export default PatientAppointments;