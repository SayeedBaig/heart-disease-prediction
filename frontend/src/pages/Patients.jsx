import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";

function Patients() {
  const navigate = useNavigate();

  const [patients, setPatients] = useState([]);
  const [loading, setLoading] =useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await api.get("/patients");

        setPatients(response.data);
      } catch (err) {
        console.error(err);
        setError("Unable to load patients.");
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, []);

  const handleViewHistory = (patient) => {
    // Database primary key (required for History API)
    localStorage.setItem("selected_patient_id", patient.id);

    // Generated patient ID (PT000001)
    localStorage.setItem("selected_patient_code", patient.patient_id);

    // Patient name
    localStorage.setItem("selected_patient_name", patient.full_name);

    navigate("/history");
  };
  const handleBookAppointment = (patient) => {
  // Database primary key
  localStorage.setItem("selected_patient_id", patient.id);

  // PT000007
  localStorage.setItem("selected_patient_code", patient.patient_id);

  // Patient name
  localStorage.setItem("selected_patient_name", patient.full_name);

  navigate("/appointments");
};

  if (loading) {
    return (
      <div className="min-h-screen flex justify-center items-center text-xl">
        Loading patients...
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex justify-center items-center text-red-600 text-xl">
        {error}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-100 p-10">

      <h1 className="text-4xl font-bold mb-8">
        Patients
      </h1>

      <div className="bg-white rounded-2xl shadow-lg overflow-hidden">

        <table className="w-full">

          <thead className="bg-slate-200">

            <tr>
              <th className="p-4 text-left">Patient ID</th>
              <th className="p-4 text-left">Name</th>
              <th className="p-4 text-left">Email</th>
              <th className="p-4 text-left">Phone</th>
              <th className="p-4 text-center">Actions</th>
            </tr>

          </thead>

          <tbody>

            {patients.map((patient) => (

              <tr
                key={patient.id}
                className="border-b hover:bg-slate-50 transition"
              >
                <td className="p-4 font-medium">
                  {patient.patient_id}
                </td>

                <td className="p-4">
                  {patient.full_name}
                </td>

                <td className="p-4">
                  {patient.email}
                </td>

                <td className="p-4">
                  {patient.phone}
                </td>

                <td className="p-4 text-center">

                 <div className="flex justify-center gap-3">

  <button
    onClick={() => handleViewHistory(patient)}
    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition"
  >
    View History
  </button>

  <button
    onClick={() => handleBookAppointment(patient)}
    className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition"
  >
    Book Appointment
  </button>

</div>

                </td>

              </tr>

            ))}

          </tbody>

        </table>

      </div>

    </div>
  );
}

export default Patients;