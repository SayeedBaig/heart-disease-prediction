import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, ArrowLeft, ArrowRight, BrainCircuit, Download, FileHeart, HeartPulse, Mail, ScanHeart, Stethoscope, UserRound, UploadCloud } from "lucide-react";
import heart from "../assets/heart.png";
import { analyzePatientData, bookAppointment, createPatientAccount, downloadPatientReport, emailPatientReport, getAvailableDoctors, loginPatient } from "../services/portalService";
import SpecularButton from "../components/SpecularButton";
import LineSidebar from "../components/LineSidebar";
import DigitalTwin from "./DigitalTwin";

const blank = { age: "", height: "", weight: "", systolic: "", diastolic: "", cholesterol: "", glucose: "", smoking: "", alcohol: "", active: "", ecg: "", echo: "" };
const labels = { age: "Age", height: "Height (cm)", weight: "Weight (kg)", systolic: "Systolic Blood Pressure", diastolic: "Diastolic Blood Pressure", cholesterol: "Cholesterol", glucose: "Glucose" };
const steps = [["Your basics", "A few simple measurements give the model useful context.", ["age", "height", "weight"]], ["Clinical markers", "Add recent readings if you have them. You can leave anything blank.", ["systolic", "diastolic", "cholesterol", "glucose"]], ["Lifestyle", "These answers are private and optional.", ["smoking", "alcohol", "active"]], ["ECG", "Upload an ECG image if one is available.", ["ecg"]], ["Echocardiogram", "Upload an echo image or video if you have one.", ["echo"]]];

export default function PatientPortal() {
  const [screen, setScreen] = useState("landing");
  const [role, setRole] = useState("patient");
  const [mode, setMode] = useState("login");
  const [auth, setAuth] = useState({ name: "", email: "", phone: "", password: "", confirm: "", gender: "", dateOfBirth: "", specialization: "", hospital: "" });
  const [error, setError] = useState("");
  const [data, setData] = useState(() => ({ ...blank, ...JSON.parse(localStorage.getItem("cardio-data") || "{}") }));
  const [files, setFiles] = useState({ ecg: null, echo: null });
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [prediction, setPrediction] = useState(() => JSON.parse(localStorage.getItem("cardio-prediction") || "null"));
  const [processingError, setProcessingError] = useState("");
  const [message, setMessage] = useState("");
  const [theme, setTheme] = useState(() => localStorage.getItem("cardio-theme") || "light");
  const [appointmentDoctors, setAppointmentDoctors] = useState([]);
  const [apptForm, setApptForm] = useState({ doctorId: "", date: "", time: "09:00", symptoms: "", reason: "" });
  const [apptMsg, setApptMsg] = useState("");
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");

  useEffect(() => localStorage.setItem("cardio-data", JSON.stringify(data)), [data]);
  useEffect(() => { localStorage.setItem("cardio-theme", theme); document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => {
    if (screen === "appointment" && appointmentDoctors.length === 0) {
      getAvailableDoctors().then(setAppointmentDoctors).catch(() => {});
    }
  }, [screen]);
  useEffect(() => {
    if (screen !== "processing") return undefined;
    const timer = setInterval(() => setProgress(value => Math.min(90, value + 5)), 260);
    return () => clearInterval(timer);
  }, [screen]);

  const fallbackRisk = useMemo(() => {
    let score = 12;
    if (+data.age >= 55) score += 18;
    if (+data.systolic >= 140) score += 21;
    if (+data.diastolic >= 90) score += 11;
    if (+data.cholesterol === 3) score += 14;
    if (data.smoking === "1") score += 11;
    if (data.active === "0") score += 6;
    return Math.min(score, 88);
  }, [data]);
  const model = prediction?.prediction;
  const risk = Number.isFinite(Number(model?.fusion?.risk_percentage)) ? Math.round(Number(model.fusion.risk_percentage)) : fallbackRisk;
  const level = model?.fusion?.final_level || (risk < 25 ? "Low" : risk < 50 ? "Moderate" : "Elevated");
  const navigate = useNavigate();
  const go = next => { if (next === "processing") setProgress(8); setScreen(next); window.scrollTo(0, 0); };
  const update = (key, value) => setData(current => ({ ...current, [key]: value }));
  const changeAuth = (key, value) => setAuth(current => ({ ...current, [key]: value }));
  const setFile = (key, file) => { update(key, file?.name || ""); setFiles(current => ({ ...current, [key]: file || null })); };

  async function submitAuth(event) {
    event.preventDefault(); setError("");
    if (role !== "patient") { navigate("/doctor"); return; }
    if (mode === "signup" && auth.password !== auth.confirm) { setError("The passwords do not match."); return; }
    try {
      if (mode === "signup") await createPatientAccount(auth);
      else await loginPatient(auth.email, auth.password);
      go("intake");
    } catch (requestError) { setError(requestError.response?.data?.detail || "Unable to continue. Check that the Cardio AI backend is running."); }
  }

  async function startAnalysis() {
    const required = ["age", "height", "weight", "systolic", "diastolic", "cholesterol", "glucose", "smoking", "alcohol", "active"];
    if (required.some(key => !data[key]) || !patient.gender || patient.gender === "Other") {
      window.alert("To preserve ML accuracy, add Gender and all clinical values before diagnosing. ECG and echocardiogram uploads are optional.");
      setStep(!data.age || !data.height || !data.weight ? 0 : !data.systolic || !data.diastolic || !data.cholesterol || !data.glucose ? 1 : 2);
      go("intake"); return;
    }
    setError(""); setProcessingError(""); setMessage(""); go("processing");
    try {
      const result = await analyzePatientData(data, files, JSON.parse(localStorage.getItem("cardio-patient") || "{}"));
      setPrediction(result); localStorage.setItem("cardio-prediction", JSON.stringify(result)); setProgress(100);
      setTimeout(() => go("hub"), 350);
    } catch (requestError) { setProcessingError(requestError.response?.data?.detail || "Analysis could not be completed. Please check your connection and try again."); }
  }

  async function sendReport() {
    try { await emailPatientReport(prediction.prediction_id); setMessage("Your report has been sent to your registered email address."); }
    catch (requestError) { setMessage(requestError.response?.data?.detail || "Unable to send the report. Configure SMTP on the backend and try again."); }
  }

  // ── Full-page Digital Twin (rendered outside cardio-shell) ──
  if (screen === "twin") {
    return (
      <DigitalTwin
        initialData={data}
        onBack={() => go("hub")}
      />
    );
  }

  return <div className="cardio-shell">
    <header className="topbar"><button className="brand" onClick={() => go("landing")}><span className="brand-mark"><HeartPulse size={20} /></span>Cardio AI</button><span className="topbar-tag">Intelligent cardiac care</span><button className="theme-toggle" onClick={() => setTheme(value => value === "light" ? "dark" : "light")}>{theme === "light" ? "Dark" : "Light"}</button></header>
    {screen === "landing" && <main className="page landing"><section><span className="eyebrow">Precision cardiac intelligence</span><h1>Know your heart. <em>Shape your future.</em></h1><p className="lead">Cardio AI turns your health information, ECG, and echocardiogram data into a clear, personal cardiac-health view—so you can take the next step with confidence.</p><SpecularButton onClick={() => go("roles")} tint="var(--accent-primary)" tintOpacity={0.15} textColor="var(--text-primary)">Get Started <ArrowRight size={17} style={{ verticalAlign: "middle", marginLeft: 8 }} /></SpecularButton><div className="trust-line"><span>✦ Private by design</span><span>✦ Patient-first insights</span><span>✦ AI-assisted analysis</span></div></section><section className="heart-scene"><div className="heart-glow" /><div className="heart-orbit" /><img className="heart-image" src={heart} alt="Animated anatomical heart" /><span className="scene-label">Live cardiac visualisation</span></section></main>}
    {screen === "roles" && <main className="page narrow"><div className="section-head"><span className="eyebrow">Start your journey</span><h1>Choose your portal</h1><p>Select the experience that is right for you. Patient tools are available now.</p></div><div className="role-grid"><Role icon={<UserRound size={38} />} title="Patient" copy="Understand your heart health, explore your digital twin, and share your report." onClick={() => { setRole("patient"); go("auth"); }} /><Role icon={<Stethoscope size={38} />} title="Doctor" copy="Access the secure clinician workspace and patient-care tools." onClick={() => { navigate("/doctor"); }} /></div><div className="doctor-note">Doctor Portal: secure clinician workspace for reviewing patient reports.</div></main>}
    {screen === "auth" && <main className="page auth-layout"><section className="auth-copy"><span className="eyebrow">{role} access</span><h1>{role === "patient" ? "Your heart health, in one place." : "Clinician access."}</h1><p>{role === "patient" ? "Create a secure profile to start your personalised cardiac assessment." : "The Doctor Portal is being prepared. Patient tools are the active experience in this release."}</p></section><form className="auth-panel" onSubmit={submitAuth}><div className="toggle"><button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>Log in</button><button type="button" className={mode === "signup" ? "active" : ""} onClick={() => setMode("signup")}>Sign up</button></div><div className="fields">{mode === "signup" && <><Field label="Full name" required value={auth.name} onChange={value => changeAuth("name", value)} /><SelectField label="Gender" required value={auth.gender} options={["Female", "Male"]} onChange={value => changeAuth("gender", value)} /><Field label="Date of birth" required type="date" value={auth.dateOfBirth} onChange={value => changeAuth("dateOfBirth", value)} /></>}<Field label="Email" required type="email" value={auth.email} onChange={value => changeAuth("email", value)} />{mode === "signup" && <Field label="Phone number" required type="tel" value={auth.phone} onChange={value => changeAuth("phone", value)} />}<Field label="Password" required type="password" value={auth.password} onChange={value => changeAuth("password", value)} />{mode === "signup" && <Field label="Confirm password" required type="password" value={auth.confirm} onChange={value => changeAuth("confirm", value)} />}</div>{error && <p className="form-error">{error}</p>}<SpecularButton type="submit" tint="var(--accent-primary)" tintOpacity={0.2} textColor="var(--text-primary)">{mode === "login" ? "Continue securely" : "Create account"}</SpecularButton><p className="form-caption">By continuing, you agree to use Cardio AI for informational support.</p></form></main>}
    {screen === "intake" && <main className="page wizard wizard-layout"><aside className="wizard-sidebar"><LineSidebar items={["Basic info", "Clinical", "Lifestyle", "ECG upload", "Echo upload"]} accentColor="var(--accent-primary)" textColor="var(--text-secondary)" markerColor="var(--accent-primary)" defaultActive={step} onItemClick={index => setStep(index)} /></aside><div><span className="eyebrow">Patient intake</span><span className="step-count">Step {step + 1} of 5</span><section className="wizard-card"><h2>{steps[step][0]}</h2><p>{steps[step][1]}</p>{step === 0 && <div className="data-grid">{steps[step][2].map(key => <Field key={key} label={labels[key]} type="number" value={data[key]} onChange={value => update(key, value)} />)}</div>}{step === 1 && <div className="data-grid"><Field label={labels.systolic} type="number" value={data.systolic} onChange={value => update("systolic", value)} /><Field label={labels.diastolic} type="number" value={data.diastolic} onChange={value => update("diastolic", value)} /><Category label="Cholesterol" value={data.cholesterol} onChange={value => update("cholesterol", value)} /><Category label="Glucose" value={data.glucose} onChange={value => update("glucose", value)} /></div>}{step === 2 && <div className="data-grid"><Binary label="Smoking" value={data.smoking} onChange={value => update("smoking", value)} /><Binary label="Alcohol consumption" value={data.alcohol} onChange={value => update("alcohol", value)} /><Binary label="Physically active" value={data.active} onChange={value => update("active", value)} /></div>}{step === 3 && <Upload name="ECG image" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.csv" file={data.ecg} onChange={file => setFile("ecg", file)} />} {step === 4 && <Upload name="Echocardiogram" accept=".mp4,.avi,.mov,.mkv" file={data.echo} onChange={file => setFile("echo", file)} />}<div className="wizard-actions"><SpecularButton size="sm" onClick={() => step ? setStep(step - 1) : go("auth")} tint="var(--bg-card)" tintOpacity={0.5} textColor="var(--text-primary)">{step ? "Previous" : "Exit"}</SpecularButton><div style={{ textAlign: "right" }}><div className="skip-note">All fields are optional</div><SpecularButton size="sm" onClick={() => step === 4 ? startAnalysis() : setStep(step + 1)} tint="var(--accent-primary)" tintOpacity={0.2} textColor="var(--text-primary)">{step === 4 ? "Diagnose" : "Next"}</SpecularButton></div></div></section></div></main>}
    {screen === "processing" && <main className="page processing"><section className="processing-box"><div className="scanner"><ScanHeart size={45} /></div><h1>Building your cardiac picture</h1><p>Our AI is securely reviewing your submitted health information.</p><div className="loading-track"><i style={{ width: `${progress}%` }} /></div>{processingError ? <><p className="form-error">{processingError}</p><button className="button" onClick={startAnalysis}>Try again</button></> : <small>{progress < 45 ? "Organising clinical signals…" : progress < 82 ? "Comparing cardiovascular patterns…" : "Preparing your results…"}</small>}</section></main>}
    {screen === "hub" && <main className="page narrow"><div className="section-head"><span className="eyebrow">Welcome back, {patient.full_name || "Patient"}</span><h1>Your results hub</h1><p>Choose how you would like to explore your cardiac-health assessment.</p></div><div className="hub-grid"><Hub icon={<FileHeart size={38} />} title="View Report" copy="Read your diagnostic summary and recommended next steps." onClick={() => go("report")} /><Hub icon={<BrainCircuit size={38} />} title="Digital Twin" copy="Experiment with your values and see how the assessment responds." onClick={() => go("twin")} /><Hub icon={<Stethoscope size={38} />} title="Book Appointment" copy="Arrange a follow-up discussion with a cardiac-care provider." onClick={() => go("appointment")} /></div></main>}
    {screen === "report" && <main className="page"><section className="report-card"><div className="report-top"><div><span className="eyebrow">Cardio AI report</span><h1>Your cardiac-health summary</h1></div><div className="report-actions"><button className="button small secondary" disabled={!prediction?.prediction_id} onClick={() => downloadPatientReport(prediction.prediction_id)}><Download size={15} /> Download report</button><button className="button small" disabled={!prediction?.prediction_id} onClick={sendReport}><Mail size={15} /> Send to email</button></div></div>{message && <p className="form-caption">{message}</p>}<div className="risk"><Activity size={31} /><div><strong>{level} risk profile · {risk}%</strong><p>This is an AI-assisted screening summary, not a medical diagnosis.</p></div></div><div className="report-grid"><section className="report-section"><h3>Assessment overview</h3><p>{model?.clinical?.reason || "Your profile was assessed using the clinical details and lifestyle data you chose to provide."}</p></section><section className="report-section"><h3>Inputs used</h3><div className="metric-list">{Object.entries(labels).map(([key, label]) => <span className="metric" key={key}>{label}<b>{data[key] || "Not provided"}</b></span>)}</div></section><section className="report-section"><h3>Suggested next steps</h3><ul><li>Discuss this screening summary with a qualified healthcare professional.</li><li>Keep track of blood pressure, activity, and lifestyle changes.</li><li>Use the Digital Twin to understand how changes may affect your profile.</li></ul></section><section className="report-section"><h3>Media review</h3><p>ECG: {data.ecg || "Not provided"}<br />Echocardiogram: {data.echo || "Not provided"}</p></section></div></section></main>}
    {screen === "appointment" && <main className="page">
      <div className="report-top" style={{ marginBottom: 28 }}>
        <div><span className="eyebrow">Patient Portal</span><h1>Book an Appointment</h1></div>
        <button className="button small secondary" onClick={() => go("hub")}>← Back</button>
      </div>

      {apptMsg
        ? <section className="report-card" style={{ textAlign: "center", padding: "48px 24px" }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>✓</div>
            <h2 style={{ marginBottom: 8 }}>Appointment Requested</h2>
            <p style={{ color: "var(--text-secondary)", lineHeight: 1.6 }}>{apptMsg}</p>
            <button className="button" style={{ marginTop: 20 }} onClick={() => { setApptMsg(""); go("hub"); }}>Back to Dashboard</button>
          </section>
        : <section className="report-card">
            <h3 style={{ marginBottom: 20 }}>Select a Doctor</h3>
            {appointmentDoctors.length === 0
              ? <p style={{ color: "var(--text-secondary)" }}>Loading doctors…</p>
              : <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 24 }}>
                  {appointmentDoctors.map(d => (
                    <button key={d.doctor_id}
                      onClick={() => setApptForm(f => ({ ...f, doctorId: d.doctor_id }))}
                      style={{
                        padding: "14px 16px", borderRadius: 14, textAlign: "left", cursor: "pointer",
                        border: apptForm.doctorId === d.doctor_id ? "2px solid var(--accent-primary)" : "1px solid var(--border-color)",
                        background: apptForm.doctorId === d.doctor_id ? "var(--accent-glow)" : "var(--bg-card)",
                        color: "var(--text-primary)", transition: "all .15s",
                      }}>
                      <div style={{ fontWeight: 700, marginBottom: 3 }}>Dr. {d.full_name}</div>
                      <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{d.specialization}</div>
                      <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{d.hospital}</div>
                    </button>
                  ))}
                </div>
            }
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              <label className="field"><span>Preferred Date</span>
                <input type="date" value={apptForm.date} onChange={e => setApptForm(f => ({ ...f, date: e.target.value }))} />
              </label>
              <label className="field"><span>Preferred Time</span>
                <input type="time" value={apptForm.time} onChange={e => setApptForm(f => ({ ...f, time: e.target.value }))} />
              </label>
            </div>
            <label className="field" style={{ marginBottom: 12 }}><span>Reason for visit</span>
              <input type="text" value={apptForm.reason} placeholder="e.g. Chest pain, follow-up…" onChange={e => setApptForm(f => ({ ...f, reason: e.target.value }))} />
            </label>
            <label className="field" style={{ marginBottom: 20 }}><span>Symptoms (optional)</span>
              <input type="text" value={apptForm.symptoms} placeholder="e.g. Shortness of breath, fatigue…" onChange={e => setApptForm(f => ({ ...f, symptoms: e.target.value }))} />
            </label>
            {error && <p className="form-error" style={{ marginBottom: 12 }}>{error}</p>}
            <SpecularButton
              onClick={async () => {
                if (!apptForm.doctorId) { setError("Please select a doctor."); return; }
                if (!apptForm.date)     { setError("Please choose a date.");   return; }
                setError("");
                try {
                  await bookAppointment(apptForm);
                  setApptMsg("Your appointment request has been sent. The doctor will review and you will receive an email confirmation.");
                } catch (err) {
                  setError(err?.response?.data?.detail || "Failed to book appointment. Try again.");
                }
              }}
              tint="var(--accent-primary)" tintOpacity={0.2} textColor="var(--text-primary)">
              Request Appointment →
            </SpecularButton>
          </section>
      }
    </main>}
    {screen === "doctor-soon" && <main className="page narrow"><section className="report-card" style={{ textAlign: "center" }}><Stethoscope size={42} color="#7de3df" /><h1>Doctor Portal is coming soon</h1><p style={{ color: "#a7c4cf" }}>The Patient Portal is the active experience in this release.</p><button className="button" onClick={() => go("roles")}>Back to role selection</button></section></main>}
  </div>;
}

function Field({ label, value, onChange, type = "text", required = false }) { return <label className="field"><span>{label}</span><input required={required} type={type} value={value} onChange={event => onChange(event.target.value)} /></label>; }
function SelectField({ label, value, onChange, options, required = false }) { return <label className="field"><span>{label}</span><select required={required} value={value} onChange={event => onChange(event.target.value)}><option value="">Select</option>{options.map(option => <option key={option} value={option}>{option}</option>)}</select></label>; }
function Category({ label, value, onChange }) { return <label className="field"><span>{label}</span><select value={value} onChange={event => onChange(event.target.value)}><option value="">Select</option><option value="1">Normal</option><option value="2">Above Normal</option><option value="3">Well Above Normal</option></select></label>; }
function Binary({ label, value, onChange }) { return <label className="field"><span>{label}</span><select value={value} onChange={event => onChange(event.target.value)}><option value="">Select</option><option value="0">No</option><option value="1">Yes</option></select></label>; }
function Upload({ name, accept, file, onChange }) { const id = `upload-${name.replaceAll(" ", "-")}`; return <div className="upload-zone"><UploadCloud size={30} /><strong>{file || `No ${name.toLowerCase()} selected`}</strong><label htmlFor={id}>Choose {name}</label><input id={id} type="file" accept={accept} onChange={event => onChange(event.target.files[0])} /><small>Optional · {accept.includes("video") ? "Image or video" : "Image"}</small></div>; }
function Role({ icon, title, copy, onClick }) { return <button className="role-card" onClick={onClick}>{icon}<h2>{title}</h2><p>{copy}</p></button>; }
function Hub({ icon, title, copy, onClick }) { return <button className="hub-card" onClick={onClick}>{icon}<h2>{title}</h2><p>{copy}</p><ArrowRight className="arrow" size={20} /></button>; }
function Twin({ title, keys, data, update }) { return <section className="twin-section"><h2>{title}</h2><div className="data-grid">{keys.map(key => <Field key={key} label={labels[key]} type="number" value={data[key]} onChange={value => update(key, value)} />)}</div></section>; }
