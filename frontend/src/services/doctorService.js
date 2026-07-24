const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ── Shared fetch helper (always sends doctor_token, no interceptor) ── */
async function doctorFetch(path, { method = "GET", body, responseType } = {}) {
  const token = localStorage.getItem("doctor_token");
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
  return doctorFetch("/patients/");
}

export async function getPatientPredictions(patientId) {
  return doctorFetch(`/patients/${patientId}/predictions`);
}

/* ── Reports ─────────────────────────────────────────────────────── */

export async function getDoctorReport(predictionId) {
  return doctorFetch(`/reports/${predictionId}/doctor`);
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
