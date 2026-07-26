const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ── Shared fetch helper (always sends doctor_token, no interceptor) ── */
async function doctorFetch(path, { method = "GET", body, responseType } = {}) {
  const token = localStorage.getItem("doctor_token") || localStorage.getItem("doctor_access_token") || localStorage.getItem("access_token");
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (responseType === "blob") {
    if (!res.ok) throw { response: { data: { detail: "Download failed." } } };
    return res.blob();
  }

  const data = res.status === 204 ? null : await res.json().catch(() => ({}));
  if (!res.ok) throw { response: { data } };
  return data;
}

/* ── Auth (plain fetch — no patient-token interference) ─────────── */

export async function registerDoctor(form) {
  const res = await fetch(`${BASE}/doctors/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      full_name: form.fullName,
      email: form.email,
      password: form.password,
      specialization: form.specialization,
      hospital: form.hospital,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw { response: { data } };
  return data;
}

export async function loginDoctor(email, password) {
  const res = await fetch(`${BASE}/doctors/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw { response: { data } };
  localStorage.setItem("doctor_token", data.access_token);
  localStorage.setItem("cardio-doctor", JSON.stringify(data.doctor));
  return data.doctor;
}

export function logoutDoctor() {
  localStorage.removeItem("doctor_token");
  localStorage.removeItem("cardio-doctor");
}

export function getDoctorFromStorage() {
  try {
    return JSON.parse(localStorage.getItem("cardio-doctor") || "null");
  } catch {
    return null;
  }
}

/* ── Appointments (doctor-filtered with patient info) ────────────── */

export async function getPendingAppointments() {
  return doctorFetch("/appointments/my/pending");
}

export async function getApprovedAppointments() {
  return doctorFetch("/appointments/my/approved");
}

export async function approveAppointment(appointmentId) {
  return doctorFetch(`/appointments/${appointmentId}/approve`, { method: "PUT" });
}

export async function rejectAppointment(appointmentId) {
  return doctorFetch(`/appointments/${appointmentId}/reject`, { method: "PUT" });
}

/* ── Patients ────────────────────────────────────────────────────── */

export async function getAllPatients() {
  const result = await doctorFetch("/doctor/patients?page_size=100");
  return result.items;
}

export async function getDoctorPatientDetails(patientId) {
  return doctorFetch(`/doctor/patients/${patientId}`);
}

export async function getPatientPredictions(patientId) {
  return doctorFetch(`/patients/${patientId}/predictions`);
}

// The patient list is derived from approved appointments, while this endpoint
// returns the complete registration profile for the read-only detail view.
export async function getPatientRecord(patientId) {
  return doctorFetch(`/patients/${patientId}`);
}

/* ── Reports ─────────────────────────────────────────────────────── */

export async function getDoctorReport(predictionId) {
  const report = await doctorFetch(`/reports/${predictionId}/doctor`);
  return {
    ...report,
    prediction: report.prediction ?? {
      risk_level: report.final_prediction?.final_level,
      risk_percentage: report.final_prediction?.risk_percentage,
      clinical_level: report.clinical_analysis?.level,
      ecg_level: report.ecg_analysis?.level,
      echo_level: report.echo_analysis?.level,
    },
    clinical: report.clinical ?? {
      ...report.clinical_analysis,
      reason: report.ai_recommendation?.explanation || report.medical_explanation?.summary,
    },
  };
}

export async function downloadDoctorReport(predictionId) {
  const blob = await doctorFetch(`/reports/${predictionId}/doctor/pdf`, { responseType: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `cardio-doctor-report-${predictionId}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function viewDoctorReportPdf(predictionId) {
  const previewWindow = window.open("", "_blank");
  try {
    const blob = await doctorFetch(`/reports/${predictionId}/doctor/pdf`, { responseType: "blob" });
    const url = URL.createObjectURL(blob);
    if (previewWindow) previewWindow.location.href = url;
    else window.open(url, "_blank");
    window.setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) {
    previewWindow?.close();
    throw error;
  }
}

export async function emailDoctorReport(predictionId) {
  return doctorFetch(`/reports/${predictionId}/doctor/email`, { method: "POST" });
}
