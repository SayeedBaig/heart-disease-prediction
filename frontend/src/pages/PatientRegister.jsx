import { useState } from "react";
import { useNavigate } from "react-router-dom";

import Sidebar from "../components/dashboard/Sidebar";
import Header from "../components/dashboard/Header";
import PatientForm from "../components/patient/PatientForm";
import SuccessModal from "../components/patient/SuccessModal";

import { registerPatient } from "../services/patientService";

function PatientRegister() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    full_name: "",
    email: "",
    password: "",
    confirm_password: "",
    phone: "",
    gender: "",
    date_of_birth: "",
  });

  const [loading, setLoading] = useState(false);

  const [successData, setSuccessData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);

      if (formData.password !== formData.confirm_password) {
        throw new Error("Passwords do not match.");
      }

      const patientData = { ...formData };
      delete patientData.confirm_password;
      const data = await registerPatient(patientData);

      // Save patient details
      localStorage.setItem("patient_id", data.patient_id);
      localStorage.setItem("selected_patient_id", String(data.id));
      localStorage.setItem("selected_patient_code", data.patient_id);
      localStorage.setItem("selected_patient_name", data.full_name);
      localStorage.setItem("patient_name", data.full_name);
      localStorage.setItem("patient_email", data.email);
      localStorage.setItem("patient_gender", formData.gender);
      localStorage.setItem("patient_dob", formData.date_of_birth);

      // Show Success Modal
      setSuccessData(data);

      console.log(data);
    } catch (error) {
      console.error(error);

      alert(
        error.response?.data?.detail ||
          "Unable to register patient."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleContinue = () => {
    // Clear form
    setFormData({
      full_name: "",
      email: "",
      password: "",
      confirm_password: "",
      phone: "",
      gender: "",
      date_of_birth: "",
    });

    // Close modal
    setSuccessData(null);

    navigate("/appointment-management");
  };

  return (
    <div className="cardio-shell">
      <div className="flex min-h-screen">
        <Sidebar />

        <div className="flex-1 flex flex-col min-w-0">
          <Header />

          <main className="p-6 md:p-10 flex-1 w-full max-w-6xl mx-auto">
            <div className="w-full">
              <div className="mb-8">
                <h1 className="h2-semibold text-[var(--text-primary)]">
                  Register New Patient
                </h1>

                <p className="body-regular text-xs mt-1">
                  Register the patient before starting AI diagnosis.
                </p>
              </div>

              <PatientForm
                formData={formData}
                setFormData={setFormData}
                onSubmit={handleSubmit}
                loading={loading}
              />
            </div>
          </main>
        </div>
      </div>

      <SuccessModal
        open={!!successData}
        patientId={successData?.patient_id}
        patientName={successData?.full_name}
        onContinue={handleContinue}
      />
    </>
  );
}

export default PatientRegister;
