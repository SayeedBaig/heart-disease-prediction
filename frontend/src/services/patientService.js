import api from "./api";

export const registerPatient = async (patientData) => {
  const response = await api.post("/patients/register", patientData);
  return response.data;
};