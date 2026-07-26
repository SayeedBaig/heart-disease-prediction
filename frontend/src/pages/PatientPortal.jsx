import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { ScanHeart, UploadCloud } from "lucide-react";
import { analyzePatientData, createPatientAccount, loginPatient } from "../services/portalService";
import { clearPatientSession } from "../services/portalService";

import Navbar from "../components/Navbar";
import LandingPage from "./LandingPage";
import PatientDashboard from "./PatientDashboard";
import DigitalTwin from "./DigitalTwin";
import Reports from "./Reports";
import PatientReports from "./PatientReports";
import Appointments from "./Appointments";
import ResultsPage from "./ResultsPage";
import DigitalTwinReport from "./DigitalTwinReport";

const blank = { age: "", height: "", weight: "", systolic: "", diastolic: "", cholesterol: "", glucose: "", smoking: "", alcohol: "", active: "", ecg: "", echo: "" };
const labels = { age: "Age (yrs)", height: "Height (cm)", weight: "Weight (kg)", systolic: "Systolic Blood Pressure (mmHg)", diastolic: "Diastolic Blood Pressure (mmHg)", cholesterol: "Total Cholesterol Level", glucose: "Fasting Blood Glucose" };
const steps = [
  ["Your basics", "Enter core physical measurements.", ["age", "height", "weight"]],
  ["Clinical markers", "Add recent diagnostic readings if available.", ["systolic", "diastolic", "cholesterol", "glucose"]],
  ["Lifestyle", "Select your lifestyle habits.", ["smoking", "alcohol", "active"]],
  ["ECG Upload", "Upload an ECG waveform image if available.", ["ecg"]],
  ["Echocardiogram Upload", "Upload an echo scan or video clip if available.", ["echo"]]
];

export default function PatientPortal() {
  const location = useLocation();

  // Determine initial view screen based on location path
  const [screen, setScreen] = useState(() => {
    const path = location.pathname;
    if (path.includes("/intake") || path.includes("/diagnose")) return "intake";
    if (path.includes("/dashboard") || (localStorage.getItem("access_token") && localStorage.getItem("cardio-patient"))) return "hub";
    if (path.includes("/digital-twin/report")) return "twin-report";
    if (path.includes("/digital-twin")) return "twin";
    if (path.includes("/reports")) return "report";
    if (path.includes("/appointments")) return "appointment";
    if (path.includes("/profile")) return "profile";
    return "landing";
  });

  useEffect(() => {
    const path = location.pathname;
    if (path.includes("/digital-twin/report")) setScreen("twin-report");
    else if (path.includes("/digital-twin")) setScreen("twin");
    else if (path.includes("/reports")) setScreen("report");
    else if (path.includes("/appointments")) setScreen("appointment");
    else if (path.includes("/profile")) setScreen("profile");
    else if (path.includes("/intake") || path.includes("/diagnose")) setScreen("intake");
    else if (path.includes("/dashboard")) setScreen("hub");
  }, [location.pathname]);

  const [mode, setMode] = useState("login");
  const [auth, setAuth] = useState({ name: "", email: "", phone: "", password: "", confirm: "", gender: "Female", dateOfBirth: "" });
  const [error, setError] = useState("");
  const [data, setData] = useState(() => ({ ...blank, ...JSON.parse(localStorage.getItem("cardio-data") || "{}") }));
  const [files, setFiles] = useState({ ecg: null, echo: null });
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [processingError, setProcessingError] = useState("");
  const [theme] = useState(() => localStorage.getItem("cardio-theme") || "light");

  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");

  useEffect(() => localStorage.setItem("cardio-data", JSON.stringify(data)), [data]);
  useEffect(() => {
    localStorage.setItem("cardio-theme", theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (screen !== "processing") return undefined;
    const timer = setInterval(() => setProgress(v => Math.min(90, v + 6)), 200);
    return () => clearInterval(timer);
  }, [screen]);

  const go = (next) => {
    if (next === "processing") setProgress(10);
    setScreen(next);
    window.scrollTo(0, 0);
  };

  const update = (key, value) => setData(c => ({ ...c, [key]: value }));
  const changeAuth = (key, value) => setAuth(c => ({ ...c, [key]: value }));
  const setFile = (key, file) => {
    update(key, file?.name || "");
    setFiles(c => ({ ...c, [key]: file || null }));
  };

  async function submitAuth(event) {
    event.preventDefault();
    setError("");
    if (mode === "signup" && auth.password !== auth.confirm) {
      setError("Passwords do not match.");
      return;
    }
    try {
      if (mode === "signup") {
        await createPatientAccount(auth);
        go("intake");
      } else {
        await loginPatient(auth.email, auth.password);
        go("hub");
      }
    } catch (requestError) {
      setError(requestError.response?.data?.detail || "Invalid login credentials. Please check back-end connection.");
    }
  }

  async function startAnalysis() {
    setError("");
    setProcessingError("");
    go("processing");
    try {
      const result = await analyzePatientData(data, files, patient);
      localStorage.setItem("cardio-prediction", JSON.stringify(result));
      setProgress(100);
      setTimeout(() => go("results"), 400);
    } catch (requestError) {
      setProcessingError(requestError.response?.data?.detail || "AI analysis failed. Please try again.");
    }
  }

  const handleLogout = () => {
    clearPatientSession();
    go("landing");
  };

  /* ──────────────── SCREEN RENDERING ──────────────── */

  // 1. Landing Screen
  if (screen === "landing") {
    return <LandingPage />;
  }

  // 2. Patient Minimal Dashboard (Hub)
  if (screen === "hub") {
    return (
      <PatientDashboard
        onNavigate={(target) => go(target)}
        onLogout={handleLogout}
      />
    );
  }

  // 3. Digital Twin Simulation Suite Showcase
  if (screen === "twin") {
    return (
      <DigitalTwin
        initialData={data}
        onBack={() => go("hub")}
      />
    );
  }

  if (screen === "twin-report") {
    return <DigitalTwinReport onBack={() => go("twin")} />;
  }

  // 4. Diagnostic Results Page
  if (screen === "results") {
    return <ResultsPage />;
  }

  // 5. Official Medical Report Page
  if (screen === "report" || screen === "reports") {
    return <PatientReports />;
  }

  // 6. Appointment Booking Page
  if (screen === "appointment" || screen === "appointments") {
    return <Appointments />;
  }

  if (screen === "profile") {
    const birthDate = patient.date_of_birth || patient.dateOfBirth;
    const age = birthDate
      ? new Date().getFullYear() - new Date(birthDate).getFullYear() -
        (new Date() < new Date(new Date(birthDate).setFullYear(new Date().getFullYear())) ? 1 : 0)
      : "Not available";
    const details = [
      ["Patient ID", patient.patient_id || "Not available"],
      ["Full Name", patient.full_name || patient.name || "Not available"],
      ["Age", age],
      ["Gender", patient.gender || "Not available"],
      ["Email", patient.email || "Not available"],
      ["Phone", patient.phone || "Not available"],
      ["Date of Birth", birthDate || "Not available"],
    ];

    return (
      <div className="cardio-shell">
        <Navbar onBack={() => go("hub")} backLabel="Dashboard" breadcrumb="My Profile" />
        <main className="cardio-container w-full max-w-4xl flex-1 py-8">
          <div className="mb-8 border-b border-[var(--border-color)] pb-6">
            <span className="caption-small font-bold uppercase tracking-wider text-[var(--accent-melanzane)]">Patient Workspace</span>
            <h1 className="h2-semibold mt-1">My Profile</h1>
            <p className="body-regular mt-2 text-sm">Your registered personal and contact details.</p>
          </div>
          <div className="cardio-card p-0 overflow-hidden">
            <div className="bg-[#39062B] p-6 text-white">
              <p className="text-sm text-white/75">Patient</p>
              <h2 className="mt-1 text-2xl font-bold">{patient.full_name || patient.name || "Patient"}</h2>
            </div>
            <dl className="grid divide-y divide-[var(--border-color)] sm:grid-cols-2 sm:divide-x sm:divide-y-0">
              {details.map(([label, value]) => (
                <div key={label} className="p-5">
                  <dt className="caption-small font-semibold uppercase tracking-wide">{label}</dt>
                  <dd className="mt-1 break-words text-sm font-semibold text-[var(--text-primary)]">{String(value)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </main>
      </div>
    );
  }

  // 7. Patient Auth Login/Signup Screen
  if (screen === "auth") {
    return (
      <div className="cardio-shell">
        <Navbar onBack={() => go("landing")} breadcrumb="Authentication" />

        <main className="max-w-md mx-auto w-full px-6 py-14">
          <div className="cardio-card p-8">
            <div className="text-center mb-6">
              <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold">
                Patient Access
              </span>
              <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
                {mode === "login" ? "Welcome Back" : "Create Account"}
              </h1>
            </div>

            <div className="flex bg-[var(--bg-secondary)] p-1 rounded-xl mb-6">
              <button
                type="button"
                className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                  mode === "login" ? "bg-[var(--card-bg)] text-[var(--text-primary)] shadow-sm" : "text-[var(--text-muted)]"
                }`}
                onClick={() => setMode("login")}
              >
                Log In
              </button>
              <button
                type="button"
                className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                  mode === "signup" ? "bg-[var(--card-bg)] text-[var(--text-primary)] shadow-sm" : "text-[var(--text-muted)]"
                }`}
                onClick={() => setMode("signup")}
              >
                Sign Up
              </button>
            </div>

            <form onSubmit={submitAuth} className="space-y-4">
              {mode === "signup" && (
                <div>
                  <label className="block text-xs font-semibold mb-1">Full Name</label>
                  <input
                    type="text"
                    required
                    value={auth.name}
                    onChange={e => changeAuth("name", e.target.value)}
                    className="cardio-input text-xs"
                  />
                </div>
              )}

              <div>
                <label className="block text-xs font-semibold mb-1">Email Address</label>
                <input
                  type="email"
                  required
                  value={auth.email}
                  onChange={e => changeAuth("email", e.target.value)}
                  className="cardio-input text-xs"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold mb-1">Password</label>
                <input
                  type="password"
                  required
                  value={auth.password}
                  onChange={e => changeAuth("password", e.target.value)}
                  className="cardio-input text-xs"
                />
              </div>

              {mode === "signup" && (
                <div>
                  <label className="block text-xs font-semibold mb-1">Confirm Password</label>
                  <input
                    type="password"
                    required
                    value={auth.confirm}
                    onChange={e => changeAuth("confirm", e.target.value)}
                    className="cardio-input text-xs"
                  />
                </div>
              )}

              {error && <p className="text-xs text-red-500 font-semibold">{error}</p>}

              <button type="submit" className="btn-primary w-full py-3 text-xs font-semibold rounded-xl">
                {mode === "login" ? "Log In" : "Create Patient Account"}
              </button>
            </form>
          </div>
        </main>
      </div>
    );
  }

  // 8. Intake / Predict Flow
  if (screen === "intake") {
    return (
      <div className="cardio-shell">
        <Navbar onBack={() => go("hub")} backLabel="Dashboard" breadcrumb="Patient Intake" />

        <main className="cardio-container py-8 flex-1 w-full max-w-4xl">
          <div className="cardio-card p-8">
            <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-4 mb-6">
              <div>
                <span className="caption-small text-[var(--accent-melanzane)] font-bold uppercase">
                  Step {step + 1} of 5
                </span>
                <h2 className="h2-semibold text-lg text-[var(--text-primary)] mt-0.5">
                  {steps[step][0]}
                </h2>
                <p className="caption-small">{steps[step][1]}</p>
              </div>

              {/* Progress pill */}
              <div className="w-10 h-10 rounded-full bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center font-bold text-xs">
                {Math.round(((step + 1) / 5) * 100)}%
              </div>
            </div>

            {/* Step Inputs */}
            {step === 0 && (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.age}</label>
                  <input type="number" value={data.age} onChange={e => update("age", e.target.value)} className="cardio-input text-xs" placeholder="e.g. 48" />
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.height}</label>
                  <input type="number" value={data.height} onChange={e => update("height", e.target.value)} className="cardio-input text-xs" placeholder="e.g. 175" />
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.weight}</label>
                  <input type="number" value={data.weight} onChange={e => update("weight", e.target.value)} className="cardio-input text-xs" placeholder="e.g. 74" />
                </div>
              </div>
            )}

            {step === 1 && (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.systolic}</label>
                  <input type="number" value={data.systolic} onChange={e => update("systolic", e.target.value)} className="cardio-input text-xs" placeholder="e.g. 125" />
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.diastolic}</label>
                  <input type="number" value={data.diastolic} onChange={e => update("diastolic", e.target.value)} className="cardio-input text-xs" placeholder="e.g. 82" />
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.cholesterol}</label>
                  <select value={data.cholesterol} onChange={e => update("cholesterol", e.target.value)} className="cardio-input text-xs">
                    <option value="">Select Level</option>
                    <option value="1">Normal (&lt; 200 mg/dL)</option>
                    <option value="2">Above Normal (200 - 239 mg/dL)</option>
                    <option value="3">High (&ge; 240 mg/dL)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">{labels.glucose}</label>
                  <select value={data.glucose} onChange={e => update("glucose", e.target.value)} className="cardio-input text-xs">
                    <option value="">Select Level</option>
                    <option value="1">Normal (&lt; 100 mg/dL)</option>
                    <option value="2">Above Normal (100 - 125 mg/dL)</option>
                    <option value="3">High (&ge; 126 mg/dL)</option>
                  </select>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold mb-1">Tobacco / Smoking Status</label>
                  <select value={data.smoking} onChange={e => update("smoking", e.target.value)} className="cardio-input text-xs">
                    <option value="">Select</option>
                    <option value="0">Non-Smoker</option>
                    <option value="1">Active Smoker</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">Alcohol Intake</label>
                  <select value={data.alcohol} onChange={e => update("alcohol", e.target.value)} className="cardio-input text-xs">
                    <option value="">Select</option>
                    <option value="0">None / Minimal</option>
                    <option value="1">Regular Consumption</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold mb-1">Physically Active (&gt; 150 mins/wk)</label>
                  <select value={data.active} onChange={e => update("active", e.target.value)} className="cardio-input text-xs">
                    <option value="">Select</option>
                    <option value="1">Yes (Active)</option>
                    <option value="0">No (Sedentary)</option>
                  </select>
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="p-6 border-2 dashed border-[var(--border-color)] rounded-xl text-center">
                <UploadCloud size={32} className="mx-auto text-[var(--accent-melanzane)] mb-2" />
                <p className="text-xs font-bold text-[var(--text-primary)]">Upload ECG File (Optional)</p>
                <p className="caption-small mt-1">Accepts PNG, JPG, CSV waveforms</p>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.csv"
                  onChange={e => setFile("ecg", e.target.files[0])}
                  className="mt-4 text-xs mx-auto"
                />
                {files.ecg && <p className="text-xs font-semibold text-emerald-500 mt-2">Selected: {files.ecg.name}</p>}
              </div>
            )}

            {step === 4 && (
              <div className="p-6 border-2 dashed border-[var(--border-color)] rounded-xl text-center">
                <UploadCloud size={32} className="mx-auto text-[var(--accent-melanzane)] mb-2" />
                <p className="text-xs font-bold text-[var(--text-primary)]">Upload Echocardiogram File (Optional)</p>
                <p className="caption-small mt-1">Accepts MP4, AVI, MOV ultrasound video or images</p>
                <input
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.png,.jpg"
                  onChange={e => setFile("echo", e.target.files[0])}
                  className="mt-4 text-xs mx-auto"
                />
                {files.echo && <p className="text-xs font-semibold text-emerald-500 mt-2">Selected: {files.echo.name}</p>}
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="flex items-center justify-between pt-6 mt-6 border-t border-[var(--border-color)]">
              <button
                type="button"
                onClick={() => (step > 0 ? setStep(step - 1) : go("hub"))}
                className="btn-secondary text-xs py-2 px-4"
              >
                {step > 0 ? "Previous" : "Dashboard"}
              </button>

              <button
                type="button"
                onClick={() => (step < 4 ? setStep(step + 1) : startAnalysis())}
                className="btn-primary text-xs py-2 px-6"
              >
                {step === 4 ? "Run AI Assessment" : "Next Step"}
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // 9. Processing Loader Screen
  if (screen === "processing") {
    return (
      <div className="cardio-shell items-center justify-center min-h-screen p-6 text-center">
        <div className="cardio-card max-w-sm w-full p-8 space-y-4">
          <div className="w-16 h-16 rounded-full bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] flex items-center justify-center mx-auto animate-pulse">
            <ScanHeart size={36} />
          </div>

          <h2 className="h2-semibold text-base text-[var(--text-primary)]">
            Building Cardiac Intelligence Picture
          </h2>
          <p className="caption-small">
            Multimodal Neural Network is analyzing your metrics...
          </p>

          <div className="w-full h-2 rounded-full bg-[var(--bg-secondary)] overflow-hidden">
            <div
              className="h-full bg-[#39062B] transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>

          {processingError && (
            <p className="text-xs text-red-500 font-semibold">{processingError}</p>
          )}
        </div>
      </div>
    );
  }

  // Default Fallback
  return <LandingPage />;
}
