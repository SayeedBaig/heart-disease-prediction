import api from "./api";

const patientSessionKeys = [
  "access_token", "patient", "cardio-patient", "cardio-prediction", "prediction_id",
  "cardio-data", "cardio-digital-twin", "cardio-digital-twin-report",
  "patient_id", "patient_name", "patient_email",
];

export function clearPatientSession() {
  patientSessionKeys.forEach((key) => localStorage.removeItem(key));
}

const numberOr = (v) => v === "" || v === null || v === undefined ? null : Number(v);

function clinicalPayload(data, patient, paths) {
  return {
    patient_id: patient.patient_id,
    age: numberOr(data.age),
    gender: patient.gender === "Female" ? 1 : patient.gender === "Male" ? 2 : null,
    height: numberOr(data.height),
    weight: numberOr(data.weight),
    ap_hi: numberOr(data.systolic),
    ap_lo: numberOr(data.diastolic),
    cholesterol: numberOr(data.cholesterol),
    gluc: numberOr(data.glucose),
    smoke: numberOr(data.smoking),
    alco: numberOr(data.alcohol),
    active: numberOr(data.active),
    ecg_path: paths.ecg || null,
    echo_path: paths.echo || null,
  };
}

export async function createPatientAccount(account) {
  await api.post("/patients/register", {
    full_name: account.name,
    email: account.email,
    phone: account.phone,
    password: account.password,
    gender: account.gender,
    date_of_birth: account.dateOfBirth,
  });
  return loginPatient(account.email, account.password);
}

export async function loginPatient(email, password) {
  clearPatientSession();
  const { data } = await api.post("/patients/login", { email, password });
  localStorage.setItem("access_token", data.access_token);
  const profile = await api.get("/patients/me");
  localStorage.setItem("cardio-patient", JSON.stringify(profile.data));
  return profile.data;
}

async function upload(path, file) {
  if (!file) return null;
  const body = new FormData();
  body.append("file", file);
  const { data } = await api.post(path, body, { headers: { "Content-Type": "multipart/form-data" } });
  return data.file_path;
}

export async function analyzePatientData(data, files, patient) {
  const [ecg, echo] = await Promise.all([upload("/upload/ecg", files.ecg), upload("/upload/echo", files.echo)]);
  const response = await api.post("/predict", clinicalPayload(data, patient, { ecg, echo }));
  return response.data;
}

export async function getPatientReport(predictionId) {
  const { data } = await api.get(`/reports/${predictionId}/patient`);
  return data;
}

export async function getPatientPredictions(patientId) {
  const { data } = await api.get(`/patients/${patientId}/predictions`);
  return data;
}

export async function downloadPatientReport(predictionId) {
  const response = await api.get(`/reports/${predictionId}/patient/pdf`, { responseType: "blob" });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = `cardio-ai-report-${predictionId}.pdf`;
  link.click();
  URL.revokeObjectURL(url);
}

export async function viewPatientReportPdf(predictionId) {
  const previewWindow = window.open("", "_blank");
  try {
    const response = await api.get(`/reports/${predictionId}/patient/pdf`, { responseType: "blob" });
    const url = URL.createObjectURL(response.data);
    if (previewWindow) previewWindow.location.href = url;
    else window.open(url, "_blank");
    window.setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) {
    previewWindow?.close();
    throw error;
  }
}

export async function emailPatientReport(predictionId) {
  const { data } = await api.post(`/reports/${predictionId}/patient/email`);
  return data;
}

export async function emailDigitalTwinReport(report) {
  const { data } = await api.post("/reports/digital-twin/email", report);
  return data;
}

/* ── Appointments ───────────────────────────────────────────────── */

export async function getAvailableDoctors() {
  const { data } = await api.get("/doctors/");
  return data;
}

export async function bookAppointment({ doctorId, date, time, symptoms, reason }) {
  const { data } = await api.post("/appointments/book", {
    doctor_id: doctorId,
    preferred_date: date,         // "YYYY-MM-DD"
    preferred_time: time + ":00", // "HH:MM:SS"
    symptoms: symptoms || null,
    reason: reason || null,
  });
  return data;
}
