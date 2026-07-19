import { useState } from "react";
import api from "../services/api";

function Appointments() {
  const patientId = localStorage.getItem("selected_patient_id");
  const patientCode = localStorage.getItem("selected_patient_code");
  const patientName = localStorage.getItem("selected_patient_name");

  const [appointment, setAppointment] = useState({
    preferred_date: "",
    preferred_time: "",
    symptoms: "",
    reason: "",
  });

  const handleBookAppointment = async () => {
    try {
      const doctor = JSON.parse(localStorage.getItem("doctor"));

      const response = await api.post("/appointments/", {
        patient_id: Number(patientId),
        doctor_id: doctor.doctor_id,
        preferred_date: appointment.preferred_date,
        preferred_time: appointment.preferred_time,
        symptoms: appointment.symptoms,
        reason: appointment.reason,
      });

      console.log(response.data);

      alert("Appointment booked successfully!");

      localStorage.setItem(
        "appointment_id",
        response.data.appointment_id
      );

      setAppointment({
        preferred_date: "",
        preferred_time: "",
        symptoms: "",
        reason: "",
      });

    } catch (err) {
      console.error("Appointment Error:", err);

      if (err.response) {
        console.log("Status:", err.response.status);
        console.log("Data:", err.response.data);
      }

      alert("Failed to book appointment.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 p-10">
      <h1 className="text-4xl font-bold mb-8">
        Book Appointment
      </h1>

      <div className="bg-white rounded-xl shadow-lg p-8 max-w-3xl">
        <h2 className="text-2xl font-semibold mb-6">
          Patient Details
        </h2>

        <p>
          <strong>Patient ID:</strong> {patientCode}
        </p>

        <p>
          <strong>Name:</strong> {patientName}
        </p>

        <p>
          <strong>Database ID:</strong> {patientId}
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-8 max-w-3xl mt-8">
        <h2 className="text-2xl font-semibold mb-6">
          Appointment Details
        </h2>

        <div className="grid gap-5">
          <div>
            <label className="block font-medium mb-2">
              Preferred Date
            </label>

            <input
              type="date"
              className="w-full border rounded-lg p-3"
              value={appointment.preferred_date}
              onChange={(e) =>
                setAppointment({
                  ...appointment,
                  preferred_date: e.target.value,
                })
              }
            />
          </div>

          <div>
            <label className="block font-medium mb-2">
              Preferred Time
            </label>

            <input
              type="time"
              className="w-full border rounded-lg p-3"
              value={appointment.preferred_time}
              onChange={(e) =>
                setAppointment({
                  ...appointment,
                  preferred_time: e.target.value,
                })
              }
            />
          </div>

          <div>
            <label className="block font-medium mb-2">
              Symptoms
            </label>

            <textarea
              rows="3"
              className="w-full border rounded-lg p-3"
              value={appointment.symptoms}
              onChange={(e) =>
                setAppointment({
                  ...appointment,
                  symptoms: e.target.value,
                })
              }
            />
          </div>

          <div>
            <label className="block font-medium mb-2">
              Reason
            </label>

            <textarea
              rows="3"
              className="w-full border rounded-lg p-3"
              value={appointment.reason}
              onChange={(e) =>
                setAppointment({
                  ...appointment,
                  reason: e.target.value,
                })
              }
            />
          </div>

          <button
            onClick={handleBookAppointment}
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-3 transition"
          >
            Book Appointment
          </button>
        </div>
      </div>
    </div>
  );
}

export default Appointments;