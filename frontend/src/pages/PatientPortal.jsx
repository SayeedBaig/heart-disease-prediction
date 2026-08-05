import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { analyzePatientData, createPatientAccount, loginPatient } from "../services/portalService";
import { clearPatientSession } from "../services/portalService";

import Navbar from "../components/Navbar";
import LandingPage from "./LandingPage";
import PatientDashboard from "./PatientDashboard";
import PatientIntake from "./PatientIntake";
import DigitalTwin from "./DigitalTwin";
import Reports from "./Reports";
import PatientReports from "./PatientReports";
import Appointments from "./Appointments";
import ResultsPage from "./ResultsPage";
import DigitalTwinReport from "./DigitalTwinReport";

const blank = { age: "", height: "", weight: "", systolic: "", diastolic: "", cholesterol: "", glucose: "", smoking: "", alcohol: "", active: "", ecg: "", echo: "" };

export default function PatientPortal() {
  const location = useLocation();

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
      go("intake"); // Return to intake if failed
    }
  }

  const handleLogout = () => {
    clearPatientSession();
    go("landing");
  };

  /* ──────────────── SCREEN RENDERING ──────────────── */

  const renderDashboardWrap = (children, activeTab) => (
    <PatientDashboard
      activeTab={activeTab}
      onNavigate={(target) => go(target)}
      onLogout={handleLogout}
    >
      {children}
    </PatientDashboard>
  );

  if (screen === "landing") {
    return <LandingPage />;
  }

  if (screen === "hub") {
    return renderDashboardWrap(null, "hub");
  }

  if (screen === "twin") {
    return renderDashboardWrap(<DigitalTwin initialData={data} onBack={() => go("hub")} />, "twin");
  }

  if (screen === "twin-report") {
    return renderDashboardWrap(<DigitalTwinReport onBack={() => go("twin")} />, "twin");
  }

  if (screen === "results") {
    return renderDashboardWrap(<ResultsPage />, "results");
  }

  if (screen === "report" || screen === "reports") {
    return renderDashboardWrap(<PatientReports />, "report");
  }

  if (screen === "appointment" || screen === "appointments") {
    return renderDashboardWrap(<Appointments />, "appointment");
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

    return renderDashboardWrap(
      <div>
        <div className="mb-8 border-b border-[var(--border-color)] pb-6">
          <span className="caption-small font-bold uppercase tracking-wider text-[var(--accent-melanzane)]">Patient Workspace</span>
          <h1 className="h2-semibold mt-1">My Profile</h1>
          <p className="body-regular mt-2 text-sm">Your registered personal and contact details.</p>
        </div>
        <div className="cardio-card p-0 overflow-hidden">
          <div className="bg-[var(--accent-melanzane)] p-6 text-white">
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
      </div>,
      "profile"
    );
  }

  if (screen === "auth") {
    return (
      <div className="cardio-shell">
        <Navbar onBack={() => go("landing")} breadcrumb="Authentication" />
        <main className="cardio-container flex-1">
          <div className="cardio-card p-8">
            <div className="text-center mb-6">
              <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold">Patient Access</span>
              <h1 className="h2-semibold text-[var(--text-primary)] mt-1">{mode === "login" ? "Welcome Back" : "Create Account"}</h1>
            </div>
            {/* ... other auth logic ... */}
          </div>
        </main>
      </div>
    );
  }

  // 8 & 9. Intake / Predict Flow and Processing Loader Screen
  if (screen === "intake" || screen === "processing") {
    return renderDashboardWrap(
      <PatientIntake
        data={data}
        files={files}
        update={update}
        setFile={setFile}
        startAnalysis={startAnalysis}
        isProcessing={screen === "processing"}
        progress={progress}
        onNavigate={(target) => go(target)}
        onLogout={handleLogout}
        patient={patient}
      />,
      "intake"
    );
  }

  return <LandingPage />;
}
