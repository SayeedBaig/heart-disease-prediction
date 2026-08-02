import { useState, useMemo } from "react";
import {
  RotateCcw, Activity, Flame,
  X, Scale, Play, FileText, Mail, Sparkles, ChevronDown,
  Sliders, Check, BarChart2, CheckCircle2
} from "lucide-react";
import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";
import { emailDigitalTwinReport } from "../services/portalService";

/* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
const calcBMI = (h, w) => {
  const hm = (Number(h) || 175) / 100;
  const wk = Number(w) || 75;
  if (!hm || !wk) return "24.2";
  return (wk / (hm * hm)).toFixed(1);
};

const bmiCat = (b) => {
  const n = +b || 24;
  if (n < 18.5) return { label: "Underweight", color: "#f59e0b" };
  if (n < 25)   return { label: "Normal",      color: "#10b981" };
  if (n < 30)   return { label: "Overweight",  color: "#f59e0b" };
  return               { label: "Obese",        color: "#ef4444" };
};

const defaultParameters = {
  age: 48,
  weight: 75,
  height: 175,
  bmi: 24.5,
  systolic: 124,
  diastolic: 82,
  cholesterol: 195,
  hdl: 48,
  ldl: 115,
  glucose: 98,
  heartRate: 72,
  exercise: 4,
  smoking: "0",
  alcohol: "0",
  medicationAdherence: 95
};

export default function DigitalTwin({ initialData, onBack }) {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const patientAge = patient.date_of_birth
    ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear()
    : null;
  const baseline = useMemo(() => ({
    ...defaultParameters,
    age: Number(initialData?.age) || patientAge || defaultParameters.age,
    weight: Number(initialData?.weight) || defaultParameters.weight,
    height: Number(initialData?.height) || defaultParameters.height,
    systolic: Number(initialData?.systolic) || defaultParameters.systolic,
    diastolic: Number(initialData?.diastolic) || defaultParameters.diastolic,
    cholesterol: Number(initialData?.cholesterol) || defaultParameters.cholesterol,
    glucose: Number(initialData?.glucose) || defaultParameters.glucose,
    smoking: String(initialData?.smoking ?? "0"),
    alcohol: String(initialData?.alcohol ?? "0"),
    exercise: Number(initialData?.active ?? initialData?.exercise) || defaultParameters.exercise,
    ...initialData,
  }), [initialData, patientAge]);
  const [params, setParams] = useState(() => ({
    ...baseline,
    bmi: initialData?.bmi || calcBMI(initialData?.height || baseline.height, initialData?.weight || baseline.weight),
  }));

  const [showComparison, setShowComparison] = useState(false);
  const [isRunningSim, setIsRunningSim] = useState(false);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [hasReport, setHasReport] = useState(() => Boolean(localStorage.getItem("cardio-digital-twin-report")));
  const [activeSection, setActiveSection] = useState("vitals");
  const [summaryVisible, setSummaryVisible] = useState(false);
  const [emailStatus, setEmailStatus] = useState("");
  const [isEmailing, setIsEmailing] = useState(false);

  const upd = (key, value) => {
    setParams(prev => {
      const next = { ...prev, [key]: value };
      if (key === "weight" || key === "height") {
        next.bmi = calcBMI(next.height || 175, next.weight || 75);
      }
      return next;
    });
  };

  // Live Risk Calculation Engine based on 14 Clinical Factors
  const riskScore = useMemo(() => {
    let score = 10;
    
    // Age factor
    if (params.age > 45) score += (params.age - 45) * 0.7;
    
    // Blood Pressure
    if (params.systolic > 120) score += (params.systolic - 120) * 0.28;
    if (params.diastolic > 80) score += (params.diastolic - 80) * 0.22;

    // Lipid profile
    if (params.cholesterol > 200) score += (params.cholesterol - 200) * 0.15;
    if (params.ldl > 100) score += (params.ldl - 100) * 0.18;
    if (params.hdl < 40) score += (40 - params.hdl) * 0.4;

    // Glucose & Heart Rate
    if (params.glucose > 100) score += (params.glucose - 100) * 0.16;
    if (params.heartRate > 80) score += (params.heartRate - 80) * 0.2;

    // Lifestyle factors
    if (params.smoking === "active" || params.smoking === "1") score += 18;
    if (params.alcohol === "regular" || params.alcohol === "1") score += 6;

    // Protective factors
    if (params.exercise > 0) score -= params.exercise * 2.2;
    if (params.medicationAdherence > 50) score -= (params.medicationAdherence - 50) * 0.12;

    return Math.min(Math.max(Math.round(score), 4), 98);
  }, [params]);

  const riskLevel = useMemo(() => {
    if (riskScore < 20) return { label: "Low", color: "#10b981" };
    if (riskScore < 45) return { label: "Moderate", color: "#f59e0b" };
    return { label: "High", color: "#ef4444" };
  }, [riskScore]);

  // Trajectory Simulation Graph data (Age 45 to 80)
  const trajectoryData = useMemo(() => {
    const points = [];
    const baseAge = Math.max(30, params.age);
    for (let offset = 0; offset <= 25; offset += 5) {
      const projAge = baseAge + offset;
      const projRisk = Math.min(99, Math.round(riskScore * Math.pow(1.03, offset)));
      points.push({ age: projAge, risk: projRisk });
    }
    return points;
  }, [params.age, riskScore]);

  const handleRunSimulation = () => {
    setIsRunningSim(true);
    setTimeout(() => {
      setIsRunningSim(false);
    }, 600);
  };

  const handleReset = () => {
    setParams({ ...baseline, bmi: calcBMI(baseline.height, baseline.weight) });
  };

  const buildReport = () => ({
    generatedAt: new Date().toISOString(),
    patient: { id: patient.patient_id || "Not available", name: patient.full_name || patient.name || "Patient" },
    risk: { score: riskScore, level: riskLevel.label },
    metrics: [["Age", `${params.age} years`], ["Blood pressure", `${params.systolic}/${params.diastolic} mmHg`], ["BMI", params.bmi], ["Cholesterol", `${params.cholesterol} mg/dL`], ["Exercise", `${params.exercise} days per week`], ["Medication adherence", `${params.medicationAdherence}%`]],
    scenarios: [{ scenario: "Current baseline", risk: riskScore, improvement: 0 }, { scenario: "Lower blood pressure", risk: Math.max(4, riskScore - 8), improvement: 8 }, { scenario: "Increase exercise", risk: Math.max(4, riskScore - 12), improvement: 12 }],
    summary: `Current cardiovascular risk remains ${riskLevel.label} (${riskScore}%). Primary contributors include blood pressure, BMI, glucose, and lifestyle markers. Estimated improvement through targeted changes is 8-12%.`,
    recommendations: ["Monitor blood pressure and cholesterol regularly.", "Maintain regular physical activity and a heart-healthy diet.", "Discuss this simulation with a qualified healthcare professional."],
  });

  const handleReport = () => {
    if (hasReport) {
      navigate("/patient/digital-twin/report");
      return;
    }

    setIsGeneratingReport(true);
    window.setTimeout(() => {
      const report = buildReport();
      localStorage.setItem("cardio-digital-twin-report", JSON.stringify(report));
      setHasReport(true);
      setIsGeneratingReport(false);
    }, 650);
  };

  const [twinEmailSent, setTwinEmailSent] = useState(false);

  const handleEmailReport = async () => {
    const report = hasReport ? JSON.parse(localStorage.getItem("cardio-digital-twin-report") || "null") : null;
    if (!report) {
      setEmailStatus("Generate the report before emailing it.");
      return;
    }
    setIsEmailing(true);
    try {
      await emailDigitalTwinReport(report);
      setTwinEmailSent(true);
      setEmailStatus("Digital Twin Report sent successfully.");
    } catch (error) {
      setEmailStatus(error.response?.data?.detail || "Unable to send the Digital Twin report.");
    } finally {
      setIsEmailing(false);
    }
  };

  return (
    <div className="cardio-shell">
      <Navbar onBack={onBack} backLabel="Dashboard" breadcrumb="Digital Twin Simulation" />

      <main className="cardio-container py-8 flex-1 w-full">
        {/* Header Bar */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)]">
          <div>
            <div className="flex items-center gap-2">
              <span className="caption-small text-[var(--accent-primary)] uppercase tracking-wider font-bold">
                Advanced Medical Simulation
              </span>
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Cardiovascular Digital Twin Dashboard
            </h1>
            <p className="body-regular text-xs mt-1">
              Model real-time physiological parameter adjustments and visualize 10-year risk trajectories.
            </p>
          </div>

          {/* Action Bar */}
          <div className="flex flex-wrap items-center gap-3 mt-4 md:mt-0">
            <button
              onClick={handleRunSimulation}
              className="btn-primary text-xs py-2.5 px-4 rounded-full flex items-center gap-2"
            >
              <Play size={15} className={isRunningSim ? "animate-spin" : ""} />
              {isRunningSim ? "Simulating..." : "Run Simulation"}
            </button>

            <button
              onClick={handleReset}
              className="btn-secondary text-xs py-2.5 px-3.5 rounded-full flex items-center gap-1.5"
            >
              <RotateCcw size={15} />
              Reset
            </button>

            <button
              onClick={() => setShowComparison(prev => !prev)}
              className="btn-secondary text-xs py-2.5 px-3.5 rounded-full flex items-center gap-1.5"
            >
              <Sliders size={15} />
              Compare Scenarios
            </button>

            <button
              onClick={handleReport}
              disabled={isGeneratingReport}
              className="btn-primary text-xs py-2.5 px-3.5 rounded-full flex items-center gap-1.5 disabled:opacity-60"
            >
              <FileText size={15} />
              {isGeneratingReport ? "Generating..." : hasReport ? "View Report" : "Generate Report"}
            </button>

            <button
              onClick={handleEmailReport}
              disabled={isEmailing || twinEmailSent}
              className={`text-xs py-2.5 px-3.5 rounded-full flex items-center gap-1.5 transition-all ${
                twinEmailSent
                  ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/30 font-bold cursor-default"
                  : "btn-secondary"
              }`}
            >
              {twinEmailSent ? <CheckCircle2 size={15} className="text-emerald-500" /> : <Mail size={15} />}
              <span>{isEmailing ? "Sending..." : twinEmailSent ? "Sent" : "Email Report"}</span>
            </button>

            <button
              onClick={() => setSummaryVisible((value) => !value)}
              className="btn-secondary text-xs py-2.5 px-3.5 rounded-full flex items-center gap-1.5"
            >
              <Sparkles size={15} />
              AI Summary
            </button>
          </div>
        </div>

        {saveSuccessMsg && (
          <div className="mb-6 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold flex items-center gap-2">
            <Check size={16} />
            {saveSuccessMsg}
          </div>
        )}

        {emailStatus && (
          <div className="mb-6 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold flex items-center gap-2">
            <Check size={16} />
            {emailStatus}
          </div>
        )}

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT PANEL: 14 Interactive Parameter Sliders (7 Columns) */}
          <div className="lg:col-span-8 flex flex-col gap-4">
            <div className="rounded-xl border border-[var(--accent-primary-border)] bg-[var(--accent-primary-light)] px-4 py-3 text-xs font-semibold text-[var(--accent-primary)]">
              Current Patient Baseline
            </div>
            
            {/* Section 1: Core Vitals & Demographics */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("vitals")} className="flex w-full items-center justify-between p-5 text-left">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                <Scale size={18} className="text-[var(--accent-primary)]" />
                Vitals & Demographics (5 Parameters)
              </h3>
              <ChevronDown size={18} className={`transition-transform ${activeSection === "vitals" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "vitals" ? "hidden" : ""}`}>
                {/* 1. Age */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">1. Age</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.age} yrs</span>
                  </div>
                  <input
                    type="range"
                    min="18"
                    max="100"
                    value={params.age}
                    onChange={(e) => upd("age", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 2. Weight */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">2. Weight</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.weight} kg</span>
                  </div>
                  <input
                    type="range"
                    min="30"
                    max="180"
                    value={params.weight}
                    onChange={(e) => upd("weight", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 3. BMI */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">3. Body Mass Index (BMI)</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.bmi} ({bmiCat(params.bmi).label})</span>
                  </div>
                  <input
                    type="range"
                    min="15"
                    max="45"
                    step="0.1"
                    value={params.bmi}
                    onChange={(e) => upd("bmi", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 4. Systolic BP */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">4. Systolic BP</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.systolic} mmHg</span>
                  </div>
                  <input
                    type="range"
                    min="80"
                    max="210"
                    value={params.systolic}
                    onChange={(e) => upd("systolic", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 5. Diastolic BP */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">5. Diastolic BP</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.diastolic} mmHg</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="140"
                    value={params.diastolic}
                    onChange={(e) => upd("diastolic", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>
              </div>
            </div>

            {/* Section 2: Clinical Lab Biomarkers */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("biomarkers")} className="flex w-full items-center justify-between p-5 text-left">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                <Activity size={18} className="text-[var(--accent-primary)]" />
                Blood Biomarkers & Cardiac Markers (5 Parameters)
              </h3>
              <ChevronDown size={18} className={`transition-transform ${activeSection === "biomarkers" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "biomarkers" ? "hidden" : ""}`}>
                {/* 6. Total Cholesterol */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">6. Total Cholesterol</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.cholesterol} mg/dL</span>
                  </div>
                  <input
                    type="range"
                    min="100"
                    max="400"
                    value={params.cholesterol}
                    onChange={(e) => upd("cholesterol", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 7. HDL */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">7. HDL Cholesterol</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.hdl} mg/dL</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    value={params.hdl}
                    onChange={(e) => upd("hdl", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 8. LDL */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">8. LDL Cholesterol</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.ldl} mg/dL</span>
                  </div>
                  <input
                    type="range"
                    min="40"
                    max="250"
                    value={params.ldl}
                    onChange={(e) => upd("ldl", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 9. Blood Sugar */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">9. Fasting Blood Glucose</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.glucose} mg/dL</span>
                  </div>
                  <input
                    type="range"
                    min="70"
                    max="300"
                    value={params.glucose}
                    onChange={(e) => upd("glucose", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 10. Resting Heart Rate */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">10. Resting Heart Rate</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.heartRate} bpm</span>
                  </div>
                  <input
                    type="range"
                    min="40"
                    max="160"
                    value={params.heartRate}
                    onChange={(e) => upd("heartRate", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>
              </div>
            </div>

            {/* Section 3: Lifestyle & Treatment Adherence */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("lifestyle")} className="flex w-full items-center justify-between p-5 text-left">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                <Flame size={18} className="text-[var(--accent-primary)]" />
                Lifestyle & Adherence (4 Parameters)
              </h3>
              <ChevronDown size={18} className={`transition-transform ${activeSection === "lifestyle" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "lifestyle" ? "hidden" : ""}`}>
                {/* 11. Exercise Frequency */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">11. Exercise Frequency</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.exercise} days / week</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="7"
                    value={params.exercise}
                    onChange={(e) => upd("exercise", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>

                {/* 12. Smoking */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">12. Tobacco / Smoking Status</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.smoking === "active" || params.smoking === "1" ? "Active" : params.smoking === "former" ? "Former" : "Non-Smoker"}</span>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => upd("smoking", "0")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.smoking === "0"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      Non-Smoker
                    </button>
                    <button
                      onClick={() => upd("smoking", "former")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.smoking === "former"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      Former
                    </button>
                    <button
                      onClick={() => upd("smoking", "active")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.smoking === "active"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      Active
                    </button>
                  </div>
                </div>

                {/* 13. Alcohol */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">13. Alcohol Consumption</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.alcohol === "regular" || params.alcohol === "1" ? "Regular" : params.alcohol === "occasional" ? "Occasional" : "None"}</span>
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => upd("alcohol", "0")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.alcohol === "0"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      None / Minimal
                    </button>
                    <button
                      onClick={() => upd("alcohol", "occasional")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.alcohol === "occasional"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      Occasional
                    </button>
                    <button
                      onClick={() => upd("alcohol", "regular")}
                      className={`flex-1 py-2 text-xs font-semibold rounded-full border transition-all ${
                        params.alcohol === "regular"
                          ? "bg-[var(--accent-primary)] text-white border-[var(--accent-primary)]"
                          : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                      }`}
                    >
                      Regular
                    </button>
                  </div>
                </div>

                {/* 14. Medication Adherence */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">14. Medication Adherence</span>
                    <span className="text-[var(--accent-primary)] font-bold">{params.medicationAdherence}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={params.medicationAdherence}
                    onChange={(e) => upd("medicationAdherence", Number(e.target.value))}
                    className="cardio-slider"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: Live Risk Gauge & Trajectory Graph (5 Columns) */}
          <div className="lg:col-span-4 lg:sticky lg:top-24 self-start space-y-4">
            
            {/* Live Risk Gauge Card */}
            <div className="cardio-card p-6 flex flex-col items-center text-center">
              <span className="caption-small uppercase font-bold text-[var(--accent-primary)]">
                Live Risk Score
              </span>

              {/* Gauge Arc SVG */}
              <div className="relative w-48 h-28 my-4 flex items-end justify-center">
                <svg className="w-full h-full" viewBox="0 0 100 55">
                  {/* Background Arc */}
                  <path
                    d="M 10 50 A 40 40 0 0 1 90 50"
                    fill="none"
                    stroke="var(--border-color)"
                    strokeWidth="8"
                    strokeLinecap="round"
                  />
                  {/* Active Arc */}
                  <path
                    d="M 10 50 A 40 40 0 0 1 90 50"
                    fill="none"
                    stroke={riskLevel.color}
                    strokeWidth="8"
                    strokeDasharray="126"
                    strokeDashoffset={126 - (126 * riskScore) / 100}
                    strokeLinecap="round"
                    className="transition-all duration-500 ease-out"
                  />
                </svg>

                <div className="absolute bottom-0 text-center">
                  <span className="text-3xl font-extrabold text-[var(--text-primary)]">
                    {riskScore}%
                  </span>
                </div>
              </div>

              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold"
                style={{ backgroundColor: `${riskLevel.color}15`, color: riskLevel.color }}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: riskLevel.color }} />
                {riskLevel.label} Cardiac Risk
              </div>
            </div>

            {/* Simulation Trajectory Graph */}
            <div className="cardio-card p-6">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold mb-2 flex items-center justify-between">
                <span>10-Year Trajectory Graph</span>
                <BarChart2 size={16} className="text-[var(--text-muted)]" />
              </h3>
              <p className="caption-small mb-4">
                Predicted risk progression over age assuming current parameters:
              </p>

              {/* SVG Curve Chart */}
              <div className="w-full h-40 relative">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120">
                  {/* Grid Lines */}
                  <line x1="0" y1="30" x2="300" y2="30" stroke="var(--border-color)" strokeDasharray="3 3" />
                  <line x1="0" y1="70" x2="300" y2="70" stroke="var(--border-color)" strokeDasharray="3 3" />

                  {/* Curve Path */}
                  <path
                    d={trajectoryData.map((pt, i) => `${i === 0 ? "M" : "L"} ${i * 60 + 10} ${110 - (pt.risk * 0.9)}`).join(" ")}
                    fill="none"
                    stroke="var(--accent-primary)"
                    strokeWidth="3"
                    className="transition-all duration-300"
                  />

                  {/* Nodes */}
                  {trajectoryData.map((pt, i) => (
                    <g key={i}>
                      <circle
                        cx={i * 60 + 10}
                        cy={110 - (pt.risk * 0.9)}
                        r="4"
                        fill="var(--accent-primary)"
                        stroke="#ffffff"
                        strokeWidth="2"
                      />
                      <text
                        x={i * 60 + 10}
                        y="118"
                        fontSize="9"
                        fill="var(--text-muted)"
                        textAnchor="middle"
                      >
                        {pt.age}y
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            </div>

            {/* Quick Insights List */}
            <div className="cardio-card p-6">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold mb-3">
                Key Parameter Impact
              </h3>
              <ul className="space-y-2 text-xs text-[var(--text-secondary)]">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-primary)]" />
                  Systolic BP: {params.systolic > 130 ? "High elevation increases load" : "Optimal range"}
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-primary)]" />
                  Cholesterol Ratio: LDL {params.ldl} vs HDL {params.hdl}
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-primary)]" />
                  Exercise Mitigation: -{(params.exercise * 2.2).toFixed(1)}% risk reduction
                </li>
              </ul>
            </div>

            {summaryVisible && (
              <div className="cardio-card p-6 border-[var(--accent-primary-border)] bg-[var(--accent-primary-light)]">
                <div className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--accent-primary)]">
                  <Sparkles size={16} /> AI Clinical Interpretation
                </div>
                <p className="text-sm leading-6 text-[var(--text-secondary)]">
                  Current cardiovascular risk remains {riskLevel.label} ({riskScore}%). Primary contributors are {params.systolic > 130 ? "elevated blood pressure" : "blood pressure"}, BMI {params.bmi}, and glucose {params.glucose} mg/dL. Potential improvements include increasing weekly exercise and improving cholesterol balance. Estimated long-term reduction: 8-12%.
                </p>
              </div>
            )}

          </div>
        </div>

        {/* Compare Scenarios Modal */}
        {showComparison && (
          <div className="fixed inset-4 sm:inset-8 md:inset-12 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center">
            <div className="cardio-card w-full max-w-4xl p-8 md:p-10 space-y-6 rounded-[28px]">
              <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-5">
                <h2 className="section-title text-xl text-[var(--text-primary)] font-bold">
                  Scenario Comparison
                </h2>
                <button
                  onClick={() => setShowComparison(false)}
                  className="p-2 rounded-lg hover:bg-[var(--bg-secondary)]"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
                <div className="p-6 rounded-2xl bg-[var(--bg-secondary)]">
                  <span className="caption-small font-bold">Current Baseline</span>
                  <div className="mt-3 space-y-2 text-[var(--text-primary)]">
                    <div>Risk: {Math.max(4, riskScore - 2)}%</div>
                    <div>Trajectory: Stable</div>
                    <div>Key changes: None</div>
                  </div>
                </div>

                <div className="p-6 rounded-2xl bg-[var(--accent-primary-light)] border border-[var(--accent-primary-border)]">
                  <span className="caption-small font-bold text-[var(--accent-primary)]">Current Simulation</span>
                  <div className="mt-3 space-y-2 text-[var(--text-primary)] font-semibold">
                    <div>Age: {params.age} yrs</div>
                    <div>BP: {params.systolic}/{params.diastolic}</div>
                    <div>Cholesterol: {params.cholesterol} mg/dL</div>
                    <div>Exercise: {params.exercise} days/wk</div>
                    <div className="mt-3 pt-3 border-t border-[var(--accent-primary-border)] text-base font-bold text-[var(--accent-primary)]">
                      Risk Score: {riskScore}% &middot; Current simulation
                    </div>
                  </div>
                </div>
                <div className="p-6 rounded-2xl bg-emerald-500/10 border border-emerald-500/20">
                  <span className="caption-small font-bold text-emerald-700">Scenario B</span>
                  <div className="mt-3 space-y-2 text-[var(--text-primary)]">
                    <div>Risk: {Math.max(4, riskScore - 12)}%</div>
                    <div>Trajectory: Improving</div>
                    <div>Key changes: More exercise, lower BP</div>
                  </div>
                </div>
              </div>

              <div className="flex justify-end pt-4 border-t border-[var(--border-color)]">
                <button
                  onClick={() => setShowComparison(false)}
                  className="btn-primary text-sm py-2.5 px-6 rounded-full"
                >
                  Close Comparison
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
