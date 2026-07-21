import { useEffect, useState } from "react";
import api from "../services/api";

function AppointmentManagement() {
 const [appointments, setAppointments] = useState([]);
const [loading, setLoading] = useState(true);
const [error, setError] = useState("");
const [selectedAppointment, setSelectedAppointment] = useState(null);
const [search, setSearch] = useState("");
const [statusFilter, setStatusFilter] = useState("All");
  useEffect(() => {
    const fetchAppointments = async () => {
      try {
        const response = await api.get("/appointments");
        console.log(response.data);
        setAppointments(response.data);
      } catch (err) {
        console.error(err);

        if (err.response) {
          setError(
            `Error ${err.response.status}: ${JSON.stringify(
              err.response.data
            )}`
          );
        } else {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchAppointments();
  }, []);

  
  

  if (loading) {
    return (
      <div className="min-h-screen flex justify-center items-center">
        <h2 className="text-xl font-semibold">
          Loading appointments...
        </h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex justify-center items-center">
        <h2 className="text-red-600 text-lg">{error}</h2>
      </div>
    );
  }
  const handleApprove = async (appointmentId) => {
    const confirmApprove = window.confirm(
  "Are you sure you want to approve this appointment?"
);

if (!confirmApprove) return;
  try {
    await api.put(`/appointments/${appointmentId}/approve`);

    setAppointments((prev) =>
      prev.map((appointment) =>
        appointment.appointment_id === appointmentId
          ? { ...appointment, status: "Approved" }
          : appointment
      )
    );

    alert("Appointment approved successfully!");
  } catch (err) {
    console.error(err);
    alert("Failed to approve appointment.");
  }
};
const handleReject = async (appointmentId) => {
  const confirmReject = window.confirm(
  "Are you sure you want to reject this appointment?"
);

if (!confirmReject) return;
  try {
    await api.put(`/appointments/${appointmentId}/reject`);

    setAppointments((prev) =>
      prev.map((appointment) =>
        appointment.appointment_id === appointmentId
          ? { ...appointment, status: "Rejected" }
          : appointment
      )
    );

    alert("Appointment rejected successfully!");
  } catch (err) {
    console.error(err);
    alert("Failed to reject appointment.");
  }
};

const handleComplete = async (appointmentId) => {
  const confirmComplete = window.confirm(
  "Mark this appointment as completed?"
);

if (!confirmComplete) return;
  try {
    await api.put(`/appointments/${appointmentId}/complete`);

    setAppointments((prev) =>
      prev.map((appointment) =>
        appointment.appointment_id === appointmentId
          ? { ...appointment, status: "Completed" }
          : appointment
      )
    );

    alert("Appointment completed successfully!");
  } catch (err) {
    console.error(err);
    alert("Failed to complete appointment.");
  }
};
const filteredAppointments = appointments
  .filter((appointment) => {
    const matchesSearch =
      search === "" ||
      appointment.patient_id
        .toString()
        .toLowerCase()
        .includes(search.toLowerCase());

    const matchesStatus =
      statusFilter === "All" ||
      appointment.status === statusFilter;

    return matchesSearch && matchesStatus;
  })
  .sort((a, b) => {
    const dateA = new Date(`${a.preferred_date}T${a.preferred_time}`);
    const dateB = new Date(`${b.preferred_date}T${b.preferred_time}`);

    return dateB - dateA; // Newest first
  });
  

const formatDate = (date) => {
  return new Date(date).toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
};

const formatTime = (time) => {
  const [hours, minutes] = time.split(":");

  const date = new Date();
  date.setHours(hours);
  date.setMinutes(minutes);

  return date.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
};

  return (
    <div className="min-h-screen bg-slate-100 p-8">
      <div className="max-w-7xl mx-auto bg-white rounded-2xl shadow-lg p-8">

        <h1 className="text-3xl font-bold text-slate-800 mb-8">
  Appointment Management
</h1>

<div className="flex justify-end mb-6 gap-3">
  <button
    onClick={() => window.location.reload()}
    className="bg-blue-600 text-white px-5 py-2 rounded-lg hover:bg-blue-700"
  >
    🔄 Refresh
  </button>

  <button
    className="bg-green-600 text-white px-5 py-2 rounded-lg hover:bg-green-700"
  >
    📥 Export CSV
  </button>
</div>

       <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">

  <div className="bg-blue-600 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer">
    <h3 className="text-lg font-semibold">
      Total Appointments
    </h3>

    <p className="text-3xl font-bold mt-2">
      {appointments.length}
    </p>
  </div>

  <div className="bg-blue-600 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer">
    <h3 className="text-lg font-semibold">
      Pending
    </h3>

    <p className="text-3xl font-bold mt-2">
      {appointments.filter(a => a.status === "Pending").length}
    </p>
  </div>

  <div className="bg-green-600 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer">
    <h3 className="text-lg font-semibold">
      Approved
    </h3>

    <p className="text-3xl font-bold mt-2">
      {appointments.filter(a => a.status === "Approved").length}
    </p>
  </div>

  <div className="bg-red-600 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer">
  <h3 className="text-lg font-semibold">
    Rejected
  </h3>

  <p className="text-3xl font-bold mt-2">
    {appointments.filter(a => a.status === "Rejected").length}
  </p>
</div>

  <div className="bg-indigo-600 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 cursor-pointer">
    <h3 className="text-lg font-semibold">
      Completed
    </h3>

    <p className="text-3xl font-bold mt-2">
      {appointments.filter(a => a.status === "Completed").length}
    </p>
  </div>

</div>

<div className="flex flex-col md:flex-row gap-4 mb-6">

  <input
    type="text"
    placeholder="Search by Patient ID..."
    value={search}
    onChange={(e) => setSearch(e.target.value)}
    className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
  />

  <select
    value={statusFilter}
    onChange={(e) => setStatusFilter(e.target.value)}
    className="border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
  >
    <option value="All">All Status</option>
    <option value="Pending">Pending</option>
    <option value="Approved">Approved</option>
    <option value="Rejected">Rejected</option>
    <option value="Completed">Completed</option>
  </select>

</div>

       {filteredAppointments.length === 0 ? (
          <div className="text-center py-16">
  <h2 className="text-2xl font-semibold text-gray-600">
    No appointments found
  </h2>

  <p className="text-gray-500 mt-2">
    Try changing the search text or status filter.
  </p>
</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">

              <thead>
                <tr className="bg-slate-200">
                  <th className="border p-3">S.No</th>
<th className="border p-3">Patient ID</th>
                  <th className="border p-3">Preferred Date</th>
                  <th className="border p-3">Preferred Time</th>
                  <th className="border p-3">Symptoms</th>
                  <th className="border p-3">Reason</th>
                  <th className="border p-3">Status</th>
                  <th className="border p-3">Actions</th>
                </tr>
              </thead>

              <tbody>
                {filteredAppointments.map((appointment, index) => (
                  <tr
                    key={appointment.appointment_id}
                    className="hover:bg-slate-50 transition"
                  >
                    <td className="border p-3 text-center font-medium">
  {index + 1}
</td>
                    <td className="border p-3">
                      {appointment.patient_id}
                    </td>
                    

                   <td className="border p-3">
  {formatDate(appointment.preferred_date)}
</td>

                  <td className="border p-3">
  {formatTime(appointment.preferred_time)}
</td>

                    <td className="border p-3">
                      {appointment.symptoms}
                    </td>

                    <td className="border p-3">
                      {appointment.reason}
                    </td>

                    <td className="border p-3">
  <span
    className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold ${
      appointment.status === "Pending"
        ? "bg-yellow-100 text-yellow-700"
        : appointment.status === "Approved"
        ? "bg-green-100 text-green-700"
        : appointment.status === "Rejected"
        ? "bg-red-100 text-red-700"
        : "bg-blue-100 text-blue-700"
    }`}
  >
    <span
      className={`w-2 h-2 rounded-full ${
        appointment.status === "Pending"
          ? "bg-yellow-500"
          : appointment.status === "Approved"
          ? "bg-green-500"
          : appointment.status === "Rejected"
          ? "bg-red-500"
          : "bg-blue-500"
      }`}
    ></span>

    {appointment.status}
  </span>
</td>

                   <td className="border p-3">
  <div className="flex flex-wrap gap-2">

    <button
      onClick={() => setSelectedAppointment(appointment)}
      className="bg-gray-700 hover:bg-gray-800 text-white px-3 py-1 rounded transition"
    >
      View
    </button>
    {appointment.status === "Pending" && (
      <>
        <button
          onClick={() => handleApprove(appointment.appointment_id)}
          className="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded transition"
        >
          Approve
        </button>

        <button
          onClick={() => handleReject(appointment.appointment_id)}
          className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded transition"
        >
          Reject
        </button>
      </>
    )}

    {appointment.status === "Approved" && (
      <button
        onClick={() => handleComplete(appointment.appointment_id)}
        className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded transition"
      >
        Complete
      </button>
    )}

    {(appointment.status === "Rejected" ||
      appointment.status === "Completed") && (
      <span className="text-gray-500 italic">
        No Actions
      </span>
    )}

  </div>
</td>

                  </tr>
                ))}
              </tbody>

            </table>
          </div>
        )}

           </div>

      {/* Appointment Details Modal */}
      {selectedAppointment && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl p-8">

            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold">
                Appointment Details
              </h2>

              <button
                onClick={() => setSelectedAppointment(null)}
                className="text-gray-500 hover:text-red-600 text-2xl"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">

              <div>
                <span className="font-semibold">Patient ID:</span>{" "}
                {selectedAppointment.patient_id}
              </div>

              <div>
                <span className="font-semibold">Preferred Date:</span>{" "}
                {formatDate(selectedAppointment.preferred_date)}
              </div>

              <div>
                <span className="font-semibold">Preferred Time:</span>{" "}
                {formatTime(selectedAppointment.preferred_time)}
              </div>

              <div>
                <span className="font-semibold">Symptoms:</span>
                <p className="mt-2 bg-slate-100 p-3 rounded-lg">
                  {selectedAppointment.symptoms}
                </p>
              </div>

              <div>
                <span className="font-semibold">Reason:</span>
                <p className="mt-2 bg-slate-100 p-3 rounded-lg">
                  {selectedAppointment.reason}
                </p>
              </div>

              <div>
                <span className="font-semibold">Status:</span>{" "}
                {selectedAppointment.status}
              </div>

            </div>

            <div className="mt-8 text-right">
              <button
                onClick={() => setSelectedAppointment(null)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg"
              >
                Close
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
    
  );
}

export default AppointmentManagement;