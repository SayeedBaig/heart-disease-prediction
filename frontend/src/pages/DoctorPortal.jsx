import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import DoctorDashboard from "./DoctorDashboard";
import AppointmentManagement from "./AppointmentManagement";
import Reports from "./Reports";
import { emailDoctorReport, getAllPatients, getDoctorReport, viewDoctorReportPdf } from "../services/doctorService";

export default function DoctorPortal() {
  const navigate = useNavigate();
  const location = useLocation();

  const [screen, setScreen] = useState(() => {
    const path = location.pathname;
    if (path.includes("/appointments")) return "appointments";
    if (path.includes("/reports")) return "reports";
    if (path.includes("/patients")) return "patients";
    if (path.includes("/profile")) return "profile";
    return "dashboard";
  });

  const [auth, setAuth] = useState({ email: "", password: "" });
  const [error, setError] = useState("");
  const [doctor, setDoctor] = useState(() => JSON.parse(localStorage.getItem("cardio-doctor") || "{}"));
  const [patients, setPatients] = useState([]);
  const [patientsError, setPatientsError] = useState("");
  const [patientsLoading, setPatientsLoading] = useState(false);
  const [recordActionId, setRecordActionId] = useState(null);
  const [recordStatus, setRecordStatus] = useState("");
  const [recordSummary, setRecordSummary] = useState("");

  useEffect(() => {
    if (screen !== "patients" || !doctor.doctor_id) return;

    let active = true;
    const loadPatients = () => {
      setPatientsLoading(true);
      getAllPatients()
        .then((items) => active && setPatients(items))
        .catch((requestError) => active && setPatientsError(requestError.response?.data?.detail || "Unable to load patient records."))
        .finally(() => active && setPatientsLoading(false));
    };

    setPatientsError("");
    loadPatients();
    const refreshId = window.setInterval(loadPatients, 15000);

    return () => {
      active = false;
      window.clearInterval(refreshId);
    };
  }, [screen, doctor.doctor_id]);

  const handleLogout = () => {
    localStorage.removeItem("doctor_access_token");
    localStorage.removeItem("cardio-doctor");
    navigate("/");
  };

  const handleAuthSubmit = (e) => {
    e.preventDefault();
    navigate("/doctor/login");
  };

  if (!doctor.doctor_id && screen !== "login") {
    return (
      <div className="cardio-shell">
        <Navbar onBack={() => navigate("/")} breadcrumb="Doctor Login" />

        <main className="cardio-container flex-1">
          <div className="cardio-card p-8">
            <div className="text-center mb-6">
              <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold">
                Clinician Portal
              </span>
              <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
                Doctor Authentication
              </h1>
            </div>

            <form onSubmit={handleAuthSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold mb-1">Medical License Email</label>
                <input
                  type="email"
                  required
                  placeholder="doctor@cardioai.org"
                  value={auth.email}
                  onChange={(e) => setAuth(prev => ({ ...prev, email: e.target.value }))}
                  className="cardio-input text-xs"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold mb-1">Password</label>
                <input
                  type="password"
                  required
                  placeholder="••••••••"
                  value={auth.password}
                  onChange={(e) => setAuth(prev => ({ ...prev, password: e.target.value }))}
                  className="cardio-input text-xs"
                />
              </div>

              {error && <p className="text-xs text-red-500 font-semibold">{error}</p>}

              <button type="submit" className="btn-primary w-full py-3 text-xs font-semibold rounded-xl">
                Log In to Clinician Portal
              </button>
            </form>
          </div>
        </main>
      </div>
    );
  }

  const renderDashboardWrap = (children, activeTab) => (
    <DoctorDashboard
      doctor={doctor}
      activeTab={activeTab}
      onNavigate={(target) => setScreen(target)}
      onLogout={handleLogout}
    >
      {children}
    </DoctorDashboard>
  );

  // Doctor Dashboard (Minimal 5 Cards View)
  if (screen === "dashboard") {
    return renderDashboardWrap(null, "dashboard");
  }

  // Appointments Management
  if (screen === "appointments") {
    return renderDashboardWrap(<AppointmentManagement />, "appointments");
  }

  // Reports Management
  if (screen === "reports") {
    return renderDashboardWrap(<Reports />, "reports");
  }

  // Patients Management
  if (screen === "patients") {
    return renderDashboardWrap(
      <div>
          <div className="mb-8 border-b border-[var(--border-color)] pb-6">
            <span className="caption-small font-bold uppercase tracking-wider text-[var(--accent-melanzane)]">Patient Management</span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">Patient Roster</h1>
            <p className="body-regular text-xs mt-2">Live patient records with each patient&apos;s latest assessment.</p>
          </div>
          <div className="cardio-card p-6">


            {patientsLoading && <p className="caption-small">Loading patient records…</p>}
            {patientsError && <p className="text-xs font-semibold text-red-500">{patientsError}</p>}
            {!patientsLoading && !patientsError && patients.length === 0 && (
              <p className="caption-small">No registered patients yet.</p>
            )}

            <div className="space-y-3 text-xs">
              {patients.map((patient) => {
                const report = patient.latest_prediction;
                return (
                  <div key={patient.patient_id} className="flex flex-col gap-3 rounded-xl bg-[var(--bg-secondary)] p-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <div className="font-bold text-[var(--text-primary)]">{patient.public_patient_id} — {patient.name}</div>
                      <div className="caption-small mt-1">
                        {patient.age ? `Age ${patient.age}` : "Age unavailable"} · {patient.gender || "Gender unavailable"}
                        {report ? ` · ${report.risk_level || "Unknown"} risk (${Math.round(report.risk_score || 0)}%)` : " · No report yet"}
                      </div>
                    </div>
                    {report && (
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => viewDoctorReportPdf(report.prediction_id).catch((requestError) => setRecordStatus(requestError.response?.data?.detail || "Unable to open this report."))}
                          className="btn-primary shrink-0 text-xs py-1.5 px-3"
                        >
                          View
                        </button>
                        <button
                          disabled={recordActionId === `mail-${report.prediction_id}`}
                          onClick={async () => {
                            setRecordStatus("");
                            setRecordActionId(`mail-${report.prediction_id}`);
                            try {
                              await emailDoctorReport(report.prediction_id);
                              setRecordStatus("Report emailed to the patient's registered email address.");
                            } catch (requestError) {
                              setRecordStatus(requestError.response?.data?.detail || "Unable to email this report.");
                            } finally {
                              setRecordActionId(null);
                            }
                          }}
                          className="btn-secondary shrink-0 text-xs py-1.5 px-3 disabled:opacity-60"
                        >
                          {recordActionId === `mail-${report.prediction_id}` ? "Sending..." : "Mail"}
                        </button>
                        <button
                          disabled={recordActionId === `summary-${report.prediction_id}`}
                          onClick={async () => {
                            setRecordStatus("");
                            setRecordSummary("");
                            setRecordActionId(`summary-${report.prediction_id}`);
                            try {
                              const reportData = await getDoctorReport(report.prediction_id);
                              setRecordSummary(reportData.medical_explanation?.summary || reportData.ai_recommendation?.explanation || "No written summary is available for this report yet.");
                            } catch (requestError) {
                              setRecordStatus(requestError.response?.data?.detail || "Unable to summarize this report.");
                            } finally {
                              setRecordActionId(null);
                            }
                          }}
                          className="btn-secondary shrink-0 text-xs py-1.5 px-3 disabled:opacity-60"
                        >
                          {recordActionId === `summary-${report.prediction_id}` ? "Loading..." : "Summarize"}
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}

              <div className="hidden">
              <div className="p-4 rounded-xl bg-[var(--bg-secondary)] flex justify-between items-center">
                <div>
                  <div className="font-bold text-[var(--text-primary)]">PT-84920 - Jane Doe</div>
                  <div className="caption-small">Age 52 · Female · Moderate Risk (24%)</div>
                </div>
                <button onClick={() => setScreen("reports")} className="btn-secondary text-xs py-1.5 px-3">
                  View Report
                </button>
              </div>

              <div className="p-4 rounded-xl bg-[var(--bg-secondary)] flex justify-between items-center">
                <div>
                  <div className="font-bold text-[var(--text-primary)]">PT-73819 - Robert Smith</div>
                  <div className="caption-small">Age 61 · Male · Low Risk (14%)</div>
                </div>
                <button onClick={() => setScreen("reports")} className="btn-secondary text-xs py-1.5 px-3">
                  View Report
                </button>
              </div>
              </div>
            </div>
            {recordStatus && <p className="mt-4 text-xs font-semibold text-[var(--accent-melanzane)]">{recordStatus}</p>}
            {recordSummary && (
              <div className="mt-4 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] p-4 text-sm leading-6 text-[var(--text-secondary)]">
                <div className="mb-1 text-xs font-bold uppercase tracking-wide text-[var(--accent-melanzane)]">Clinical Report Summary</div>
                {recordSummary}
              </div>
            )}
          </div>
      </div>,
      "patients"
    );
  }

  // Profile View
  if (screen === "profile") {
    return renderDashboardWrap(
      <div>
        <div className="mb-8 border-b border-[var(--border-color)] pb-6">
          <span className="caption-small font-bold uppercase tracking-wider text-[var(--accent-melanzane)]">Doctor Profile</span>
          <h1 className="h2-semibold text-[var(--text-primary)] mt-1">Clinician Credentials</h1>
          <p className="body-regular text-xs mt-2">Your verified medical credentials.</p>
        </div>
        <div className="cardio-card p-8 space-y-4 text-xs">
            <div>
              <span className="caption-small">Full Name</span>
              <div className="font-bold text-sm text-[var(--text-primary)]">{doctor.full_name}</div>
            </div>
            <div>
              <span className="caption-small">Specialization</span>
              <div className="font-bold text-[var(--text-primary)]">{doctor.specialization}</div>
            </div>
            <div>
              <span className="caption-small">Affiliate Hospital</span>
              <div className="font-bold text-[var(--text-primary)]">{doctor.hospital}</div>
            </div>
          </div>
      </div>,
      "profile"
    );
  }

  return renderDashboardWrap(null, "dashboard");
}
