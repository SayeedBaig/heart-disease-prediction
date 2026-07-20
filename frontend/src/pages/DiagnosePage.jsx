import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import ProgressBar from "../components/ProgressBar";
import StepUploadECG from "../components/StepUploadECG";
import StepUploadEcho from "../components/StepUploadEcho";

import StepOne from "../components/StepOne";
import StepTwo from "../components/StepTwo";
import StepThree from "../components/StepThree";
import api from "../services/api";
import ResultCard from "../components/ResultCard";

function DiagnosePage() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);

  const [formData, setFormData] = useState({
  age: "",
  height: "",
  weight: "",
  ap_hi: "",
  ap_lo: "",
  cholesterol: "",
  gluc: "",
  smoke: "",
  alco: "",
  active: "",
  ecgFile: null,
  echoFile: null,
});

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // ---------------- Handle Input ----------------

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleECGFileChange = (e) => {
  setFormData((prev) => ({
    ...prev,
    ecgFile: e.target.files[0],
  }));
};

const handleEchoFileChange = (e) => {
  setFormData((prev) => ({
    ...prev,
    echoFile: e.target.files[0],
  }));
};

  // ---------------- Step 1 Validation ----------------

  const validateStepOne = () => {
    const newErrors = {};

    if (!formData.age || formData.age < 1 || formData.age > 120) {
      newErrors.age = "Age must be between 1 and 120";
    }

    if (!formData.height || formData.height < 100 || formData.height > 250) {
      newErrors.height = "Height must be between 100 and 250 cm";
    }

    if (!formData.weight || formData.weight < 30 || formData.weight > 300) {
      newErrors.weight = "Weight must be between 30 and 300 kg";
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  // ---------------- Step 2 Validation ----------------

  const validateStepTwo = () => {
    const newErrors = {};

    if (!formData.ap_hi || formData.ap_hi < 50 || formData.ap_hi > 300) {
      newErrors.ap_hi =
        "Systolic Blood Pressure must be between 50 and 300";
    }

    if (!formData.ap_lo || formData.ap_lo < 30 || formData.ap_lo > 200) {
      newErrors.ap_lo =
        "Diastolic Blood Pressure must be between 30 and 200";
    }

    if (!formData.cholesterol) {
      newErrors.cholesterol = "Please select cholesterol level";
    }

    if (!formData.gluc) {
      newErrors.gluc = "Please select glucose level";
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  // ---------------- Step 3 Validation ----------------

  const validateStepThree = () => {
    const newErrors = {};

    if (!formData.smoke) {
      newErrors.smoke = "Please select smoking status";
    }

    if (!formData.alco) {
      newErrors.alco = "Please select alcohol consumption";
    }

    if (!formData.active) {
      newErrors.active = "Please select physical activity";
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  // ---------------- Navigation ----------------

  const nextStep = () => {

  // Patient Details
  if (step === 1) {
    if (!validateStepOne()) return;

    setStep(2);
    return;
  }

  // Health Details
  if (step === 2) {
    if (!validateStepTwo()) return;

    setStep(3);
    return;
  }

  // Lifestyle
  if (step === 3) {
    if (!validateStepThree()) return;

    setStep(4);
    return;
  }

  // ECG Upload
  if (step === 4) {
    setStep(5);
    return;
  }

};
  // ---------------- Upload ECG ----------------
const prevStep = () => {
  setStep((prev) => prev - 1);
};
const uploadECG = async () => {
  if (!formData.ecgFile) return null;

  const data = new FormData();
  data.append("file", formData.ecgFile);

  const response = await api.post("/upload/ecg", data, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data.file_path;
};

// ---------------- Upload Echo ----------------

const uploadEcho = async () => {
  if (!formData.echoFile) return null;

  const data = new FormData();
  data.append("file", formData.echoFile);

  const response = await api.post("/upload/echo", data, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data.file_path;
};

const handleSubmit = async () => {
  if (!validateStepThree()) {
    return;
  }

  setLoading(true);
  setError("");
  setResult(null);

  try {

    // Upload ECG
    const ecgPath = await uploadECG();

    // Upload Echo
    const echoPath = await uploadEcho();

    console.log("Uploaded ECG:", ecgPath);
    console.log("Uploaded Echo:", echoPath);

    const storedGender = localStorage.getItem("patient_gender");
    const genderValue = storedGender === "Male" ? 2 : 1;

    // Clinical data only
    
    const clinicalData = {
  patient_id: localStorage.getItem("patient_id"),

  age: Number(formData.age),
  gender: genderValue,
  height: Number(formData.height),
  weight: Number(formData.weight),
  ap_hi: Number(formData.ap_hi),
  ap_lo: Number(formData.ap_lo),
  cholesterol: Number(formData.cholesterol),
  gluc: Number(formData.gluc),
  smoke: Number(formData.smoke),
  alco: Number(formData.alco),
  active: Number(formData.active),

  ecg_path: ecgPath,
  echo_path: echoPath,
};
    console.log(clinicalData);
   const response = await api.post("/predict", clinicalData);

console.log(response.data);

// If backend returned validation errors
if (response.data.success === false) {
  setError(response.data.errors.join(", "));
  return;
}

// Save prediction_id
if (response.data.prediction_id) {
  localStorage.setItem(
    "prediction_id",
    response.data.prediction_id
  );
}

setResult(response.data);

  } catch (err) {

  console.error(err);

  setError(
    err.response?.data?.detail ||
    "Prediction failed. Please try again."
  );

} finally {

    setLoading(false);

  }
};
  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 py-10">
        <ProgressBar step={step} />

       {/* Step 1 - Patient Details */}
{step === 1 && (
  <StepOne
    formData={formData}
    handleChange={handleChange}
    nextStep={nextStep}
    errors={errors}
  />
)}

{/* Step 2 - Health Details */}
{step === 2 && (
  <StepTwo
    formData={formData}
    handleChange={handleChange}
    nextStep={nextStep}
    prevStep={prevStep}
    errors={errors}
  />
)}

{/* Step 3 - Lifestyle */}
{step === 3 && (
  <StepThree
  formData={formData}
  handleChange={handleChange}
  nextStep={nextStep}
  prevStep={prevStep}
  errors={errors}
/>
)}

{/* Step 4 - ECG */}
{step === 4 && (
  <StepUploadECG
    formData={formData}
    handleECGFileChange={handleECGFileChange}
    nextStep={nextStep}
    prevStep={prevStep}
  />
)}

{/* Step 5 - Echo */}
{step === 5 && (
  <StepUploadEcho
    formData={formData}
    handleEchoFileChange={handleEchoFileChange}
    prevStep={prevStep}
    handleSubmit={handleSubmit}
    loading={loading}
    error={error}
  />

)}


        {result && (
  <>
    <ResultCard result={result} />

    <div className="flex justify-center mt-8 mb-10">
      <button
        onClick={() => navigate("/reports")}
        className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-3 rounded-lg shadow-md transition"
      >
        📄 View Reports
      </button>
    </div>
  </>
)}
      </div>
    </>
  );
}

export default DiagnosePage;