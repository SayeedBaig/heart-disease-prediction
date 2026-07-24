import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowLeft, ArrowRight, CheckCircle2,
  ClipboardList, Download, FileText, HeartPulse,
  LogOut, Mail, Stethoscope, UserRound, Users, XCircle,
} from "lucide-react";
import SpecularButton from "../components/SpecularButton";
import {
  approveAppointment,
  getApprovedAppointments,
  getDoctorFromStorage,
  getDoctorReport,
  downloadDoctorReport,
  loginDoctor,
  logoutDoctor,
  registerDoctor,
  getPatientPredictions,
  getPendingAppointments,
  rejectAppointment,
} from "../services/doctorService";

/* ─── tiny helpers ─────────────────────────────────────────────── */
function Field({ label, value, onChange, type = "text", required }) {
  return (
    <div className="field">
      <label>{label}{required && " *"}</label>
      <input
        type={type}
        value={value}
        required={required}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function StatusBadge({ status }) {
  const map = {
    Pending:   { bg: "rgba(251,191,36,.15)",  color: "#f59e0b" },
    Approved:  { bg: "rgba(52,211,153,.15)",  color: "#10b981" },
    Rejected:  { bg: "rgba(248,113,113,.15)", color: "#ef4444" },
    Completed: { bg: "rgba(99,102,241,.15)",  color: "#6366f1" },
  };
  const s = map[status] || map.Pending;
  return (
    <span style={{
      padding: "3px 12px", borderRadius: 99, fontSize: 12,
      fontWeight: 700, background: s.bg, color: s.color,
    }}>
      {status}
    </span>
  );
}

/* ─── Auth page ────────────────────────────────────────────────── */
function DoctorAuth({ onSuccess }) {
  const [mode, setMode]   = useState("login");
  const [form, setForm]   = useState({
    fullName: "", email: "", password: "", confirm: "",
    specialization: "", hospital: "",
  });
  const [error, setError]  = useState("");
  const [busy, setBusy]    = useState(false);

  const set = (k) => (v) => setForm((f) => ({ ...f, [k]: v }));

  async function submit(e) {
    e.preventDefault();
    setError("");
    if (mode === "signup" && form.password !== form.confirm) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    try {
      if (mode === "login") {
        await loginDoctor(form.email, form.password);
      } else {
        await registerDoctor(form);
        await loginDoctor(form.email, form.password);
      }
      onSuccess();
    } catch (err) {
      setError(err?.response?.data?.detail || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page auth-layout">
      <section className="auth-copy">
        <span className="eyebrow">Doctor access</span>
        <h1>Clinician workspace.</h1>
        <p>
          Review patient reports, accept or reject submissions, and
          manage your patient list — all in one secure portal.
        </p>
      </section>

      <form className="auth-panel" onSubmit={submit}>
        <div className="toggle">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>
            Log in
          </button>
          <button type="button" className={mode === "signup" ? "active" : ""} onClick={() => setMode("signup")}>
            Sign up
          </button>
        </div>

        <div className="fields">
          {mode === "signup" && (
            <>
              <Field label="Full name" required value={form.fullName}        onChange={set("fullName")} />
              <Field label="Specialization" required value={form.specialization} onChange={set("specialization")} />
              <Field label="Hospital / Clinic" required value={form.hospital}      onChange={set("hospital")} />
            </>
          )}
          <Field label="Email" required type="email" value={form.email} onChange={set("email")} />
          <Field label="Password" required type="password" value={form.password} onChange={set("password")} />
          {mode === "signup" && (
            <Field label="Confirm password" required type="password" value={form.confirm} onChange={set("confirm")} />
          )}
        </div>

        {error && <p className="form-error">{error}</p>}

        <SpecularButton
          type="submit"
          tint="var(--accent-primary)" tintOpacity={0.2}
          textColor="var(--text-primary)"
        >
          {busy ? "Please wait…" : mode === "login" ? "Continue securely" : "Create account"}
          <ArrowRight size={17} style={{ verticalAlign: "middle", marginLeft: 8 }} />
        </SpecularButton>
        <p className="form-caption">
          Clinician portal is for authorised healthcare professionals only.
        </p>

        {/* Demo helper — one-click fill */}
        {mode === "login" && (
          <div style={{
            marginTop: -8, padding: "10px 14px", borderRadius: 12,
            background: "rgba(45,212,191,0.06)", border: "1px solid rgba(45,212,191,0.18)",
            fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6,
          }}>
            <strong style={{ color: "var(--accent-primary)" }}>Demo account</strong>
            <br />
            Email: <code style={{ fontSize: 12 }}>doctor@cardioai.demo</code>
            &nbsp;&nbsp;Password: <code style={{ fontSize: 12 }}>Demo@1234</code>
            <br />
            <button
              type="button"
              onClick={() => setForm(f => ({ ...f, email: "doctor@cardioai.demo", password: "Demo@1234" }))}
              style={{
                marginTop: 6, fontSize: 12, fontWeight: 700,
                color: "var(--accent-primary)", background: "none",
                border: "none", cursor: "pointer", padding: 0, textDecoration: "underline",
              }}
            >
              Click to auto-fill →
            </button>
          </div>
        )}
      </form>
    </main>
  );
}

/* ─── Hub ───────────────────────────────────────────────────────── */
function DoctorHub({ doctor, onNavigate }) {
  return (
    <main className="page narrow">
      <div className="section-head">
        <span className="eyebrow">Welcome, Dr. {doctor.full_name}</span>
        <h1>Doctor Dashboard</h1>
        <p>Select a workspace to continue.</p>
      </div>

      <div className="hub-grid">
        <HubCard
          icon={<ClipboardList size={38} />}
          title="View Appointments"
          copy="Review pending patient submissions awaiting your decision."
          onClick={() => onNavigate("appointments")}
        />
        <HubCard
          icon={<Users size={38} />}
          title="View Patients"
          copy="Browse all patients whose reports you have already approved."
          onClick={() => onNavigate("patients")}
        />
      </div>

      <div style={{ textAlign: "center", marginTop: 8, fontSize: 13, color: "var(--text-secondary)" }}>
        {doctor.specialization} · {doctor.hospital}
      </div>
    </main>
  );
}

function HubCard({ icon, title, copy, onClick }) {
  return (
    <button className="hub-card" onClick={onClick} type="button">
      <span className="hub-icon">{icon}</span>
      <h2>{title}</h2>
      <p>{copy}</p>
    </button>
  );
}

/* ─── Appointments list ─────────────────────────────────────────── */
function AppointmentsList({ onSelect, onBack }) {
  const [rows, setRows]     = useState([]);
  const [loading, setLoad]  = useState(true);
  const [error, setError]   = useState("");

  useEffect(() => {
    getPendingAppointments()
      .then(setRows)
      .catch(() => setError("Failed to load appointments."))
      .finally(() => setLoad(false));
  }, []);

  return (
    <main className="page">
      <div className="report-top" style={{ marginBottom: 28 }}>
        <div>
          <span className="eyebrow">Doctor Portal</span>
          <h1>Pending Appointments</h1>
        </div>
        <button className="button small secondary" onClick={onBack}>
          <ArrowLeft size={15} /> Dashboard
        </button>
      </div>

      {loading && <p style={{ color: "var(--text-secondary)" }}>Loading…</p>}
      {error   && <p className="form-error">{error}</p>}

      {!loading && !error && (
        rows.length === 0
          ? (
            <div className="report-card" style={{ textAlign: "center", padding: "48px 24px" }}>
              <ClipboardList size={40} color="var(--accent-primary)" style={{ marginBottom: 16 }} />
              <h2 style={{ marginBottom: 8 }}>No pending appointments</h2>
              <p style={{ color: "var(--text-secondary)" }}>All submissions have been reviewed.</p>
            </div>
          ) : (
            <section className="report-card" style={{ padding: 0, overflow: "hidden" }}>
              <table className="doctor-table">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Patient ID</th>
                    <th>Date</th>
                    <th>Reason</th>
                    <th>Status</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((a) => (
                    <tr key={a.appointment_id} onClick={() => onSelect(a)} className="clickable-row">
                      <td style={{ fontWeight: 700 }}>{a.patient_name || `Patient #${a.patient_id}`}</td>
                      <td style={{ fontFamily: "monospace", fontSize: 13 }}>{a.patient_pid || a.patient_id}</td>
                      <td>{a.preferred_date}</td>
                      <td style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {a.reason || "—"}
                      </td>
                      <td><StatusBadge status={a.status} /></td>
                      <td style={{ textAlign: "right" }}>
                        <span style={{ color: "var(--accent-primary)", fontSize: 13, fontWeight: 600 }}>Review →</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )
      )}
    </main>
  );
}

/* ─── Appointment detail + Accept / Reject ──────────────────────── */
function AppointmentDetail({ appointment, onBack, onDecision }) {
  const [report, setReport]   = useState(null);
  const [loading, setLoad]    = useState(true);
  const [busy, setBusy]       = useState(null); // "accept"|"reject"
  const [message, setMessage] = useState("");
  const [error, setError]     = useState("");

  /* fetch the latest prediction report for this patient */
  useEffect(() => {
    getPatientPredictions(appointment.patient_id)
      .then(async (history) => {
        if (!history.predictions?.length) { setLoad(false); return; }
        const latest = history.predictions[0];
        const r = await getDoctorReport(latest.prediction_id);
        setReport({ ...r, predictionId: latest.prediction_id });
      })
      .catch(() => setError("Could not load report."))
      .finally(() => setLoad(false));
  }, [appointment.patient_id]);

  async function decide(action) {
    setBusy(action);
    setMessage("");
    setError("");
    try {
      if (action === "accept") {
        await approveAppointment(appointment.appointment_id);
        setMessage("✓ Appointment approved. Patient notified by email.");
      } else {
        await rejectAppointment(appointment.appointment_id);
        setMessage("Appointment rejected. Patient notified by email.");
      }
      setTimeout(() => onDecision(), 1800);
    } catch (err) {
      setError(err?.response?.data?.detail || "Action failed.");
    } finally {
      setBusy(null);
    }
  }

  const pat = report?.patient;
  const pred = report?.prediction;

  return (
    <main className="page">
      <div className="report-top" style={{ marginBottom: 28 }}>
        <div>
          <span className="eyebrow">Appointment Review</span>
          <h1>{pat?.full_name ?? `Patient #${appointment.patient_id}`}</h1>
        </div>
        <button className="button small secondary" onClick={onBack}>
          <ArrowLeft size={15} /> Appointments
        </button>
      </div>

      {/* Patient meta */}
      <section className="report-card" style={{ marginBottom: 20 }}>
        <h3 style={{ marginBottom: 16 }}>Patient Information</h3>
        <div className="metric-list">
          {[
            ["Patient ID",    pat?.patient_id ?? appointment.patient_id],
            ["Name",          pat?.full_name  ?? "—"],
            ["Email",         pat?.email      ?? "—"],
            ["Gender",        pat?.gender     ?? "—"],
            ["Date of Birth", pat?.date_of_birth ?? "—"],
            ["Appointment",   appointment.preferred_date],
            ["Reason",        appointment.reason ?? "—"],
            ["Symptoms",      appointment.symptoms ?? "—"],
          ].map(([k, v]) => (
            <span className="metric" key={k}>{k}<b>{v}</b></span>
          ))}
        </div>
      </section>

      {/* Report */}
      {loading && <p style={{ color: "var(--text-secondary)" }}>Loading report…</p>}
      {error   && <p className="form-error">{error}</p>}

      {report && (
        <>
          {/* Risk summary */}
          <section className="report-card" style={{ marginBottom: 20 }}>
            <div className="report-top" style={{ marginBottom: 16 }}>
              <h3>Prediction Summary</h3>
              {report.predictionId && (
                <button className="button small secondary"
                  onClick={() => downloadDoctorReport(report.predictionId)}>
                  <Download size={14} /> Download PDF
                </button>
              )}
            </div>
            <div className="risk" style={{ marginBottom: 16 }}>
              <Activity size={28} />
              <div>
                <strong>
                  {pred?.risk_level ?? "—"} risk ·{" "}
                  {pred?.risk_percentage != null
                    ? `${(pred.risk_percentage * 100).toFixed(1)}%`
                    : "—"}
                </strong>
                <p>AI-assisted cardiac risk assessment.</p>
              </div>
            </div>
            <div className="metric-list">
              {[
                ["Clinical Level",  pred?.clinical_level],
                ["ECG Level",       pred?.ecg_level],
                ["Echo Level",      pred?.echo_level],
              ].map(([k, v]) => v != null && (
                <span className="metric" key={k}>{k}<b>{v}</b></span>
              ))}
            </div>
          </section>

          {/* Clinical inputs */}
          {report.clinical && (
            <section className="report-card" style={{ marginBottom: 20 }}>
              <h3 style={{ marginBottom: 16 }}>Clinical Inputs</h3>
              <div className="metric-list">
                {Object.entries(report.clinical)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <span className="metric" key={k}>
                      {k.replace(/_/g, " ")}<b>{String(v)}</b>
                    </span>
                  ))}
              </div>
            </section>
          )}

          {/* AI explanation */}
          {report.clinical?.reason && (
            <section className="report-card" style={{ marginBottom: 20 }}>
              <h3 style={{ marginBottom: 12 }}>AI Explanation</h3>
              <p style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
                {report.clinical.reason}
              </p>
            </section>
          )}
        </>
      )}

      {/* Decision */}
      {!message && (
        <div className="report-card" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <h3 style={{ flex: 1 }}>Decision</h3>
          <SpecularButton
            onClick={() => decide("accept")}
            tint="rgba(16,185,129,1)" tintOpacity={0.18}
            textColor="var(--text-primary)"
          >
            <CheckCircle2 size={16} style={{ verticalAlign: "middle", marginRight: 6 }} />
            {busy === "accept" ? "Approving…" : "Accept"}
          </SpecularButton>
          <SpecularButton
            onClick={() => decide("reject")}
            tint="rgba(239,68,68,1)" tintOpacity={0.18}
            textColor="var(--text-primary)"
          >
            <XCircle size={16} style={{ verticalAlign: "middle", marginRight: 6 }} />
            {busy === "reject" ? "Rejecting…" : "Reject"}
          </SpecularButton>
        </div>
      )}
      {message && (
        <div className="report-card" style={{ textAlign: "center", color: "var(--color-success)" }}>
          {message}
        </div>
      )}
    </main>
  );
}

/* ─── Patients list ─────────────────────────────────────────────── */
function PatientsList({ onSelect, onBack }) {
  const [rows, setRows]    = useState([]);
  const [loading, setLoad] = useState(true);
  const [error, setError]  = useState("");

  useEffect(() => {
    getApprovedAppointments()
      .then(setRows)
      .catch(() => setError("Failed to load patients."))
      .finally(() => setLoad(false));
  }, []);

  return (
    <main className="page">
      <div className="report-top" style={{ marginBottom: 28 }}>
        <div>
          <span className="eyebrow">Doctor Portal</span>
          <h1>Approved Patients</h1>
        </div>
        <button className="button small secondary" onClick={onBack}>
          <ArrowLeft size={15} /> Dashboard
        </button>
      </div>

      {loading && <p style={{ color: "var(--text-secondary)" }}>Loading…</p>}
      {error   && <p className="form-error">{error}</p>}

      {!loading && !error && (
        rows.length === 0
          ? (
            <div className="report-card" style={{ textAlign: "center", padding: "48px 24px" }}>
              <Users size={40} color="var(--accent-primary)" style={{ marginBottom: 16 }} />
              <h2 style={{ marginBottom: 8 }}>No approved patients yet</h2>
              <p style={{ color: "var(--text-secondary)" }}>Patients appear here after you approve their appointments.</p>
            </div>
          ) : (
            <section className="report-card" style={{ padding: 0, overflow: "hidden" }}>
              <table className="doctor-table">
                <thead>
                  <tr>
                    <th>Patient</th>
                    <th>Patient ID</th>
                    <th>Email</th>
                    <th>Appointment Date</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((a) => (
                    <tr key={a.appointment_id} onClick={() => onSelect(a)} className="clickable-row">
                      <td style={{ fontWeight: 700 }}>{a.patient_name || `Patient #${a.patient_id}`}</td>
                      <td style={{ fontFamily: "monospace", fontSize: 13 }}>{a.patient_pid || a.patient_id}</td>
                      <td>{a.patient_email || "—"}</td>
                      <td>{a.preferred_date}</td>
                      <td style={{ textAlign: "right" }}>
                        <span style={{ color: "var(--accent-primary)", fontSize: 13, fontWeight: 600 }}>View →</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )
      )}
    </main>
  );
}

/* ─── Patient detail (read-only) ────────────────────────────────── */
function PatientDetail({ patient, onBack }) {
  const [history, setHistory] = useState(null);
  const [report, setReport]   = useState(null);
  const [loading, setLoad]    = useState(true);
  const [error, setError]     = useState("");

  useEffect(() => {
    getPatientPredictions(patient.id)
      .then(async (h) => {
        setHistory(h);
        if (h.predictions?.length) {
          const r = await getDoctorReport(h.predictions[0].prediction_id);
          setReport({ ...r, predictionId: h.predictions[0].prediction_id });
        }
      })
      .catch(() => setError("Could not load patient data."))
      .finally(() => setLoad(false));
  }, [patient.id]);

  const pred = report?.prediction;

  return (
    <main className="page">
      <div className="report-top" style={{ marginBottom: 28 }}>
        <div>
          <span className="eyebrow">Patient Record</span>
          <h1>{patient.full_name}</h1>
        </div>
        <button className="button small secondary" onClick={onBack}>
          <ArrowLeft size={15} /> Patients
        </button>
      </div>

      {/* Profile */}
      <section className="report-card" style={{ marginBottom: 20 }}>
        <h3 style={{ marginBottom: 16 }}>Patient Profile</h3>
        <div className="metric-list">
          {[
            ["Patient ID",    patient.patient_id],
            ["Full Name",     patient.full_name],
            ["Email",         patient.email],
            ["Gender",        patient.gender],
            ["Date of Birth", patient.date_of_birth],
            ["Phone",         patient.phone],
          ].map(([k, v]) => (
            <span className="metric" key={k}>{k}<b>{v ?? "—"}</b></span>
          ))}
        </div>
      </section>

      {loading && <p style={{ color: "var(--text-secondary)" }}>Loading records…</p>}
      {error   && <p className="form-error">{error}</p>}

      {/* Prediction history */}
      {history?.predictions?.length > 0 && (
        <section className="report-card" style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 16 }}>Prediction History ({history.total_predictions})</h3>
          <div className="metric-list">
            {history.predictions.map((p) => (
              <span className="metric" key={p.prediction_id}>
                #{p.prediction_id} · {p.risk_level}
                <b>{p.risk_percentage != null
                  ? `${(p.risk_percentage * 100).toFixed(1)}%`
                  : "—"}</b>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Latest report */}
      {report && (
        <>
          <section className="report-card" style={{ marginBottom: 20 }}>
            <div className="report-top" style={{ marginBottom: 16 }}>
              <h3>Latest Report</h3>
              <button className="button small secondary"
                onClick={() => downloadDoctorReport(report.predictionId)}>
                <Download size={14} /> Download PDF
              </button>
            </div>
            <div className="risk" style={{ marginBottom: 16 }}>
              <Activity size={28} />
              <div>
                <strong>
                  {pred?.risk_level ?? "—"} risk ·{" "}
                  {pred?.risk_percentage != null
                    ? `${(pred.risk_percentage * 100).toFixed(1)}%`
                    : "—"}
                </strong>
                <p>AI-assisted cardiac risk assessment (read-only).</p>
              </div>
            </div>
          </section>

          {report.clinical && (
            <section className="report-card" style={{ marginBottom: 20 }}>
              <h3 style={{ marginBottom: 16 }}>Clinical Information</h3>
              <div className="metric-list">
                {Object.entries(report.clinical)
                  .filter(([k, v]) => k !== "reason" && v != null)
                  .map(([k, v]) => (
                    <span className="metric" key={k}>
                      {k.replace(/_/g, " ")}<b>{String(v)}</b>
                    </span>
                  ))}
              </div>
            </section>
          )}

          {report.clinical?.reason && (
            <section className="report-card">
              <h3 style={{ marginBottom: 12 }}>AI Explanation</h3>
              <p style={{ color: "var(--text-secondary)", lineHeight: 1.7 }}>
                {report.clinical.reason}
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}

/* ─── ROOT COMPONENT ────────────────────────────────────────────── */
export default function DoctorPortal() {
  const [doctor, setDoctor]         = useState(() => getDoctorFromStorage());
  const [screen, setScreen]         = useState("hub");
  const [selected, setSelected]     = useState(null); // appointment or patient object

  const go = (s, payload = null) => {
    setSelected(payload);
    setScreen(s);
  };

  function handleLogout() {
    logoutDoctor();
    setDoctor(null);
    setScreen("hub");
  }

  function handleAuthSuccess() {
    setDoctor(getDoctorFromStorage());
    setScreen("hub");
  }

  /* Not logged in → show auth */
  if (!doctor) {
    return (
      <div className="cardio-shell">
        <header className="topbar">
          <button className="brand">
            <span className="brand-mark"><HeartPulse size={20} /></span>Cardio AI
          </button>
          <span className="topbar-tag">Doctor Portal</span>
        </header>
        <DoctorAuth onSuccess={handleAuthSuccess} />
      </div>
    );
  }

  return (
    <div className="cardio-shell">
      {/* Top bar */}
      <header className="topbar">
        <button className="brand" onClick={() => go("hub")}>
          <span className="brand-mark"><HeartPulse size={20} /></span>Cardio AI
        </button>
        <span className="topbar-tag">Doctor Portal</span>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ fontSize: 14, color: "var(--text-secondary)" }}>
            Dr. {doctor.full_name}
          </span>
          <button
            className="button small secondary"
            onClick={handleLogout}
            style={{ display: "flex", alignItems: "center", gap: 6 }}
          >
            <LogOut size={14} /> Sign out
          </button>
        </div>
      </header>

      {/* Screens */}
      {screen === "hub" && (
        <DoctorHub doctor={doctor} onNavigate={(s) => go(s)} />
      )}

      {screen === "appointments" && (
        <AppointmentsList
          onSelect={(a) => go("appointment-detail", a)}
          onBack={() => go("hub")}
        />
      )}

      {screen === "appointment-detail" && selected && (
        <AppointmentDetail
          appointment={selected}
          onBack={() => go("appointments")}
          onDecision={() => go("appointments")}
        />
      )}

      {screen === "patients" && (
        <PatientsList
          onSelect={(p) => go("patient-detail", p)}
          onBack={() => go("hub")}
        />
      )}

      {screen === "patient-detail" && selected && (
        <PatientDetail
          patient={selected}
          onBack={() => go("patients")}
        />
      )}
    </div>
  );
}
