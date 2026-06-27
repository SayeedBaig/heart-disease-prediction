import { useState } from "react";
import Navbar from "../components/Navbar";
import ProgressBar from "../components/ProgressBar";
import StepOne from "../components/StepOne";
import StepTwo from "../components/StepTwo";
import StepThree from "../components/StepThree";
import api from "../services/api";
import ResultCard from "../components/ResultCard";

function DiagnosePage() {
  const [step, setStep] = useState(1);

  const [formData, setFormData] = useState({
    age: "",
    gender: "",
    height: "",
    weight: "",
    ap_hi: "",
    ap_lo: "",
    cholesterol: "",
    gluc: "",
    smoke: "",
    alco: "",
    active: "",
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

  // ---------------- Step 1 Validation ----------------

  const validateStepOne = () => {
    const newErrors = {};

    if (!formData.age || formData.age < 1 || formData.age > 120) {
      newErrors.age = "Age must be between 1 and 120";
    }

    if (!formData.gender) {
      newErrors.gender = "Please select gender";
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
    if (step === 1 && !validateStepOne()) {
      return;
    }

    if (step === 2 && !validateStepTwo()) {
      return;
    }

    setStep(step + 1);
  };

  const prevStep = () => {
    setStep(step - 1);
  };

  // ---------------- Submit ----------------

const handleSubmit = async () => {

  

  if (!validateStepThree()) {
    return;
  }

  setLoading(true);
  setError("");

  try {

    console.log("Sending Data:", formData);

    const response = await api.post("/predict", formData);

    console.log("Backend Response:", response.data);

    setResult(response.data);


  } catch (err) {

  console.error("Full Error:", err);

  if (err.response) {
  console.log("Status:", err.response.status);
  console.log("Response:", err.response.data);
} else if (err.request) {
  console.log("No response received from backend.");
} else {
  console.log("Error:", err.message);
}

  setError("Prediction failed. Please try again.");


  } finally {

    setLoading(false);

  }

};
  return (
    <>
      <Navbar />

      <div className="min-h-screen bg-gray-100 py-10">
        <ProgressBar step={step} />

        {step === 1 && (
          <StepOne
            formData={formData}
            handleChange={handleChange}
            nextStep={nextStep}
            errors={errors}
          />
        )}

        {step === 2 && (
          <StepTwo
            formData={formData}
            handleChange={handleChange}
            nextStep={nextStep}
            prevStep={prevStep}
            errors={errors}
          />
        )}

        {step === 3 && (
          <StepThree
            formData={formData}
            handleChange={handleChange}
            prevStep={prevStep}
            errors={errors}
            handleSubmit={handleSubmit}
            loading={loading}
            error={error}
          />
        )}
        {result && <ResultCard result={result} />}
      </div>
    </>
  );
}

export default DiagnosePage;