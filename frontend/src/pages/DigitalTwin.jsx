import { useState, useMemo, useCallback } from "react";
import {
  RotateCcw, Activity, Flame, X, Scale, Play, FileText,
  Mail, Sparkles, ChevronDown, Sliders, Check, BarChart2,
  CheckCircle2, Plus, Copy, Trash2, AlertTriangle, Info,
} from "lucide-react";
import Navbar from "../components/Navbar";
import { useNavigate } from "react-router-dom";
import {
  emailDigitalTwinReport,
  runDigitalTwinScenarios,
} from "../services/portalService";

/* ─────────────────────────── HELPERS ──────────────────────────── */
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

const DEFAULT_PARAMS = {
  age: 48, weight: 75, height: 175, bmi: 24.5,
  systolic: 124, diastolic: 82,
  cholesterol: 195, hdl: 48, ldl: 115,
  glucose: 98, heartRate: 72,
  exercise: 4, smoking: "0", alcohol: "0",
  medicationAdherence: 95,
};

// ── Clinical alert thresholds (must match backend clinical_alert_engine.py) ──
const BP_THRESHOLDS = {
  systolicLow:   90,
  diastolicLow:  60,
  systolicHigh:  140,
  diastolicHigh: 90,
};

/**
 * Evaluate blood pressure clinical alerts.
 * Mirrors the shared backend clinical_alert_engine.py rules.
 *
 * @param {number} systolic   - Systolic BP in mmHg
 * @param {number} diastolic  - Diastolic BP in mmHg
 * @param {boolean} isHypothetical - True for Digital Twin scenario values
 * @param {string} contextId  - Scenario name or "baseline"
 * @returns {Array<Object>}   - Structured alert objects
 */
function computeBpAlerts(systolic, diastolic, isHypothetical = false, contextId = "baseline") {
  const alerts = [];
  const sys = Number(systolic);
  const dia = Number(diastolic);
  if (!sys || !dia) return alerts;

  const hypoLabel = isHypothetical ? " [HYPOTHETICAL SCENARIO]" : " [OBSERVED MEASUREMENT]";

  if (sys < BP_THRESHOLDS.systolicLow) {
    alerts.push({
      alert_id: "LOW_SYSTOLIC_BP",
      parameter: "Systolic Blood Pressure",
      measured_value: sys,
      unit: "mmHg",
      category: "Blood Pressure",
      severity: "warning",
      threshold_rule: `Systolic BP < ${BP_THRESHOLDS.systolicLow} mmHg (adult screening reference)`,
      explanation: isHypothetical
        ? `This hypothetical scenario contains a systolic blood pressure of ${sys} mmHg, which is below the configured adult screening threshold of ${BP_THRESHOLDS.systolicLow} mmHg. This is a simulated value and does not establish the patient's actual blood pressure.`
        : `Your recorded blood pressure is ${sys}/${dia} mmHg. The systolic component (${sys} mmHg) is below the configured adult screening threshold of ${BP_THRESHOLDS.systolicLow} mmHg, which may indicate low blood pressure. This is a screening reference and does not constitute a diagnosis.`,
      recommendation: isHypothetical
        ? "Do not interpret this hypothetical reading as an actual clinical finding. If a real measurement returns a similar value, consult a qualified healthcare professional."
        : "Consider arranging a clinical review, particularly if you experience dizziness, weakness, blurred vision, or fainting. Do not change prescribed medication without medical advice.",
      is_hypothetical: isHypothetical,
      context_id: contextId,
    });
  }
  if (dia < BP_THRESHOLDS.diastolicLow) {
    alerts.push({
      alert_id: "LOW_DIASTOLIC_BP",
      parameter: "Diastolic Blood Pressure",
      measured_value: dia,
      unit: "mmHg",
      category: "Blood Pressure",
      severity: "warning",
      threshold_rule: `Diastolic BP < ${BP_THRESHOLDS.diastolicLow} mmHg (adult screening reference)`,
      explanation: isHypothetical
        ? `This hypothetical scenario contains a diastolic blood pressure of ${dia} mmHg, which is below the configured adult screening threshold of ${BP_THRESHOLDS.diastolicLow} mmHg. This is a simulated value.`
        : `Your recorded blood pressure is ${sys}/${dia} mmHg. The diastolic component (${dia} mmHg) is below the configured adult screening threshold of ${BP_THRESHOLDS.diastolicLow} mmHg, which may indicate low diastolic blood pressure.`,
      recommendation: isHypothetical
        ? "Do not interpret this hypothetical reading as an actual clinical finding."
        : "Consider arranging a clinical review with a qualified healthcare professional. Do not change prescribed medication without medical advice.",
      is_hypothetical: isHypothetical,
      context_id: contextId,
    });
  }
  if (sys >= BP_THRESHOLDS.systolicHigh) {
    alerts.push({
      alert_id: "HIGH_SYSTOLIC_BP",
      parameter: "Systolic Blood Pressure",
      measured_value: sys,
      unit: "mmHg",
      category: "Blood Pressure",
      severity: "warning",
      threshold_rule: `Systolic BP ≥ ${BP_THRESHOLDS.systolicHigh} mmHg (adult screening reference)`,
      explanation: isHypothetical
        ? `This hypothetical scenario contains a systolic blood pressure of ${sys} mmHg, which meets or exceeds the adult screening threshold of ${BP_THRESHOLDS.systolicHigh} mmHg. This is a simulated value.`
        : `Your recorded blood pressure is ${sys}/${dia} mmHg. The systolic component (${sys} mmHg) meets or exceeds the configured adult screening threshold of ${BP_THRESHOLDS.systolicHigh} mmHg. A single reading is not sufficient to diagnose hypertension.`,
      recommendation: "Discuss this reading with a healthcare professional. A single elevated reading does not diagnose hypertension.",
      is_hypothetical: isHypothetical,
      context_id: contextId,
    });
  }
  if (dia >= BP_THRESHOLDS.diastolicHigh) {
    alerts.push({
      alert_id: "HIGH_DIASTOLIC_BP",
      parameter: "Diastolic Blood Pressure",
      measured_value: dia,
      unit: "mmHg",
      category: "Blood Pressure",
      severity: "warning",
      threshold_rule: `Diastolic BP ≥ ${BP_THRESHOLDS.diastolicHigh} mmHg (adult screening reference)`,
      explanation: isHypothetical
        ? `This hypothetical scenario contains a diastolic blood pressure of ${dia} mmHg, which meets or exceeds the adult screening threshold of ${BP_THRESHOLDS.diastolicHigh} mmHg. This is a simulated value.`
        : `Your recorded blood pressure is ${sys}/${dia} mmHg. The diastolic component (${dia} mmHg) meets or exceeds the configured adult screening threshold of ${BP_THRESHOLDS.diastolicHigh} mmHg. A single reading is not sufficient to diagnose hypertension.`,
      recommendation: "Discuss this reading with a healthcare professional. A single elevated reading does not diagnose hypertension.",
      is_hypothetical: isHypothetical,
      context_id: contextId,
    });
  }
  return alerts;
}

/** Compute the local risk score for display (rule-based composite) */
function computeRisk(p) {
  let score = 10;
  if (p.age > 45) score += (p.age - 45) * 0.7;
  if (p.systolic > 120) score += (p.systolic - 120) * 0.28;
  if (p.diastolic > 80) score += (p.diastolic - 80) * 0.22;
  if (p.cholesterol > 200) score += (p.cholesterol - 200) * 0.15;
  if (p.ldl > 100) score += (p.ldl - 100) * 0.18;
  if (p.hdl < 40) score += (40 - p.hdl) * 0.4;
  if (p.glucose > 100) score += (p.glucose - 100) * 0.16;
  if (p.heartRate > 80) score += (p.heartRate - 80) * 0.2;
  if (p.smoking === "active" || p.smoking === "1") score += 18;
  if (p.alcohol === "regular" || p.alcohol === "1") score += 6;
  if (p.exercise > 0) score -= p.exercise * 2.2;
  if (p.medicationAdherence > 50) score -= (p.medicationAdherence - 50) * 0.12;
  return Math.min(Math.max(Math.round(score), 4), 98);
}

const riskLevel = (s) => {
  if (s < 20) return { label: "Low",      color: "#10b981" };
  if (s < 45) return { label: "Moderate", color: "#f59e0b" };
  return             { label: "High",     color: "#ef4444" };
};

/** Return names of params that differ between two param objects */
function diffParams(base, scenario) {
  const changed = [];
  const LABELS = {
    age: "Age", weight: "Weight (kg)", height: "Height (cm)", bmi: "BMI",
    systolic: "Systolic BP", diastolic: "Diastolic BP",
    cholesterol: "Total Cholesterol", hdl: "HDL", ldl: "LDL",
    glucose: "Blood Glucose", heartRate: "Heart Rate",
    exercise: "Exercise (days/wk)", smoking: "Smoking", alcohol: "Alcohol",
    medicationAdherence: "Medication Adherence (%)",
  };
  for (const k of Object.keys(base)) {
    if (String(base[k]) !== String(scenario[k])) {
      changed.push({ key: k, label: LABELS[k] || k, from: base[k], to: scenario[k] });
    }
  }
  return changed;
}

let _scenarioCounter = 1;
const newScenarioId = () => `scenario-${_scenarioCounter++}`;

/* ─────────────────────────────────────────────────────────────── */

export default function DigitalTwin({ initialData, onBack }) {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const patientAge = patient.date_of_birth
    ? new Date().getFullYear() - new Date(patient.date_of_birth).getFullYear()
    : null;

  // ── Immutable Baseline (derived once from props / patient data) ──────────
  const baseline = useMemo(() => {
    const b = {
      ...DEFAULT_PARAMS,
      age:         Number(initialData?.age)       || patientAge || DEFAULT_PARAMS.age,
      weight:      Number(initialData?.weight)    || DEFAULT_PARAMS.weight,
      height:      Number(initialData?.height)    || DEFAULT_PARAMS.height,
      systolic:    Number(initialData?.systolic)  || DEFAULT_PARAMS.systolic,
      diastolic:   Number(initialData?.diastolic) || DEFAULT_PARAMS.diastolic,
      cholesterol: Number(initialData?.cholesterol) || DEFAULT_PARAMS.cholesterol,
      glucose:     Number(initialData?.glucose)   || DEFAULT_PARAMS.glucose,
      smoking:     String(initialData?.smoking ?? "0"),
      alcohol:     String(initialData?.alcohol  ?? "0"),
      exercise:    Number(initialData?.active ?? initialData?.exercise) || DEFAULT_PARAMS.exercise,
      ...initialData,
    };
    b.bmi = initialData?.bmi || calcBMI(b.height, b.weight);
    return b;
  }, [initialData, patientAge]);

  const baselineScore = useMemo(() => computeRisk(baseline), [baseline]);

  // ── Active editing state (current scenario being built) ─────────────────
  const [params, setParams] = useState(() => ({ ...baseline }));

  // ── Named saved scenarios list ───────────────────────────────────────────
  // Each entry: { id, name, params: {...}, score }
  const [savedScenarios, setSavedScenarios] = useState([]);
  const [activeScenarioId, setActiveScenarioId] = useState(null); // which saved scenario is loaded

  // ── UI State ─────────────────────────────────────────────────────────────
  const [showComparison, setShowComparison]   = useState(false);
  const [isRunningSim, setIsRunningSim]       = useState(false);
  const [activeSection, setActiveSection]     = useState("vitals");
  const [summaryVisible, setSummaryVisible]   = useState(false);
  const [saveSuccessMsg, setSaveSuccessMsg]   = useState("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [hasReport, setHasReport]             = useState(() => Boolean(localStorage.getItem("cardio-digital-twin-report")));
  const [emailStatus, setEmailStatus]         = useState("");
  const [isEmailing, setIsEmailing]           = useState(false);
  const [twinEmailSent, setTwinEmailSent]     = useState(false);
  const [scenarioNameInput, setScenarioNameInput] = useState("");
  const [showSaveDialog, setShowSaveDialog]   = useState(false);
  const [backendError, setBackendError]       = useState("");
  const [backendResults, setBackendResults]   = useState(null); // ScenarioResponse from backend

  // ── Live risk for the current editing state ──────────────────────────────
  const currentScore = useMemo(() => computeRisk(params), [params]);
  const currentLevel = useMemo(() => riskLevel(currentScore), [currentScore]);

  const scoreDelta = currentScore - baselineScore;

  // ── Trajectory data ──────────────────────────────────────────────────────
  const trajectoryData = useMemo(() => {
    const pts = [];
    const baseAge = Math.max(30, params.age);
    for (let off = 0; off <= 25; off += 5) {
      pts.push({ age: baseAge + off, risk: Math.min(99, Math.round(currentScore * Math.pow(1.03, off))) });
    }
    return pts;
  }, [params.age, currentScore]);

  // ── Parameter update helper ──────────────────────────────────────────────
  const upd = useCallback((key, value) => {
    setParams(prev => {
      const next = { ...prev, [key]: value };
      if (key === "weight" || key === "height") {
        next.bmi = calcBMI(next.height || 175, next.weight || 75);
      }
      return next;
    });
    setActiveScenarioId(null); // editing = no longer just that saved scenario
  }, []);

  // ── Reset editing state to baseline ─────────────────────────────────────
  const handleReset = () => {
    setParams({ ...baseline });
    setActiveScenarioId(null);
    setBackendResults(null);
    setBackendError("");
  };

  // ── Load a saved scenario into the editor ────────────────────────────────
  const loadScenario = (sc) => {
    setParams({ ...sc.params });
    setActiveScenarioId(sc.id);
    setBackendResults(null);
  };

  // ── Save current params as a new named scenario ──────────────────────────
  const saveCurrentScenario = () => {
    const name = scenarioNameInput.trim() || `Scenario ${savedScenarios.length + 1}`;
    const id = newScenarioId();
    const score = computeRisk(params);
    setSavedScenarios(prev => [...prev, { id, name, params: { ...params }, score }]);
    setActiveScenarioId(id);
    setScenarioNameInput("");
    setShowSaveDialog(false);
    setSaveSuccessMsg(`"${name}" saved.`);
    setTimeout(() => setSaveSuccessMsg(""), 3000);
  };

  // ── Duplicate a saved scenario ───────────────────────────────────────────
  const duplicateScenario = (sc) => {
    const id = newScenarioId();
    const name = `${sc.name} (copy)`;
    setSavedScenarios(prev => [...prev, { id, name, params: { ...sc.params }, score: sc.score }]);
  };

  // ── Delete a saved scenario ──────────────────────────────────────────────
  const deleteScenario = (id) => {
    setSavedScenarios(prev => prev.filter(s => s.id !== id));
    if (activeScenarioId === id) {
      setParams({ ...baseline });
      setActiveScenarioId(null);
    }
  };

  // ── Run all saved scenarios on the backend ───────────────────────────────
  const handleRunSimulation = async () => {
    setIsRunningSim(true);
    setBackendError("");
    setBackendResults(null);
    try {
      if (!patient.patient_id || savedScenarios.length === 0) {
        // No patient or no saved scenarios — just animate the local score
        await new Promise(r => setTimeout(r, 600));
      } else {
        const backendBaseline = {
          age: baseline.age,
          systolic_bp: baseline.systolic,
          diastolic_bp: baseline.diastolic,
          cholesterol: baseline.cholesterol,
          glucose: baseline.glucose,
          bmi: Number(baseline.bmi),
          weight: baseline.weight,
          smoking_status: baseline.smoking === "active" || baseline.smoking === "1",
          alcohol_consumption: baseline.alcohol === "regular" || baseline.alcohol === "1",
          exercise_level:
            baseline.exercise >= 5 ? "High"
            : baseline.exercise >= 2 ? "Moderate"
            : "Low",
        };
        const scenariosPayload = {};
        for (const sc of savedScenarios) {
          scenariosPayload[sc.name] = {
            age: sc.params.age,
            systolic_bp: sc.params.systolic,
            diastolic_bp: sc.params.diastolic,
            cholesterol: sc.params.cholesterol,
            glucose: sc.params.glucose,
            bmi: Number(sc.params.bmi),
            weight: sc.params.weight,
            smoking_status: sc.params.smoking === "active" || sc.params.smoking === "1",
            alcohol_consumption: sc.params.alcohol === "regular" || sc.params.alcohol === "1",
            exercise_level:
              sc.params.exercise >= 5 ? "High"
              : sc.params.exercise >= 2 ? "Moderate"
              : "Low",
          };
        }
        const result = await runDigitalTwinScenarios(
          patient.patient_id, backendBaseline, scenariosPayload
        );
        setBackendResults(result);
      }
    } catch (err) {
      setBackendError(err?.response?.data?.detail || "Simulation failed. Showing local scores.");
    } finally {
      setIsRunningSim(false);
    }
  };

  // ─── diffs for the currently edited params vs baseline ──────────────────
  const currentDiffs = useMemo(() => diffParams(baseline, params), [baseline, params]);

  // ── Clinical Alerts memo ──────────────────────────────────────────────
  const activeAlerts = useMemo(() => {
    return computeBpAlerts(
      params.systolic,
      params.diastolic,
      currentDiffs.length > 0,
      activeScenarioId ? `Scenario: ${activeScenarioId}` : "Current Simulation"
    );
  }, [params.systolic, params.diastolic, currentDiffs.length, activeScenarioId]);

  // ── Report helpers ───────────────────────────────────────────────────────
  const buildReport = () => ({
    generatedAt: new Date().toISOString(),
    patient: { id: patient.patient_id || "N/A", name: patient.full_name || "Patient" },
    risk: { score: currentScore, level: currentLevel.label },
    metrics: [
      ["Age", `${params.age} years`],
      ["Blood Pressure", `${params.systolic}/${params.diastolic} mmHg`],
      ["BMI", params.bmi],
      ["Total Cholesterol", `${params.cholesterol} mg/dL`],
      ["Exercise", `${params.exercise} days/wk`],
      ["Medication Adherence", `${params.medicationAdherence}%`],
    ],
    scenarios: savedScenarios.map(sc => ({
      scenario: sc.name,
      risk: sc.score,
      improvement: baselineScore - sc.score,
    })),
    summary: `Hypothetical cardiovascular risk is ${currentLevel.label} (composite score ${currentScore}). ` +
      `Baseline composite score: ${baselineScore}. ` +
      `These are rule-based composite scores, not calibrated clinical probabilities.`,
    recommendations: [
      "Discuss these simulations with a qualified healthcare professional.",
      "Monitor blood pressure and cholesterol regularly.",
      "Maintain regular physical activity and a heart-healthy diet.",
    ],
    clinical_alerts: activeAlerts,
  });

  const handleReport = () => {
    if (hasReport) { navigate("/patient/digital-twin/report"); return; }
    setIsGeneratingReport(true);
    window.setTimeout(() => {
      localStorage.setItem("cardio-digital-twin-report", JSON.stringify(buildReport()));
      setHasReport(true);
      setIsGeneratingReport(false);
    }, 650);
  };

  const handleEmailReport = async () => {
    const report = hasReport
      ? JSON.parse(localStorage.getItem("cardio-digital-twin-report") || "null")
      : null;
    if (!report) { setEmailStatus("Generate the report before emailing it."); return; }
    setIsEmailing(true);
    try {
      await emailDigitalTwinReport(report);
      setTwinEmailSent(true);
      setEmailStatus("Digital Twin Report sent successfully.");
    } catch (error) {
      setEmailStatus(error.response?.data?.detail || "Unable to send the report.");
    } finally {
      setIsEmailing(false);
    }
  };

  return (
    <>
      {!onBack && <Navbar breadcrumb="Digital Twin" onBack={() => navigate(-1)} />}
      <div className={!onBack ? "max-w-[1280px] w-full mx-auto px-4 md:px-8 py-8" : ""}>
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)]">
          <div>
            <div className="flex items-center gap-2">
              <span className="caption-small text-[var(--accent-melanzane)] uppercase tracking-wider font-bold">
                Advanced Medical Simulation
              </span>
            </div>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Cardiovascular Digital Twin Dashboard
            </h1>
            <p className="body-regular text-xs mt-1">
              Adjust hypothetical clinical parameters and compare scenarios against your baseline.
            </p>
          </div>

          {/* Action Bar */}
          <div className="flex flex-col items-start md:items-end gap-2.5 mt-4 md:mt-0">
            {/* Line 1: Simulation Controls */}
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                onClick={handleRunSimulation}
                className="btn-primary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2 cursor-pointer shadow-sm"
              >
                <Play size={15} className={isRunningSim ? "animate-spin" : ""} />
                {isRunningSim ? "Simulating…" : "Run Simulation"}
              </button>

              <button onClick={handleReset} className="btn-secondary text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 cursor-pointer shadow-sm">
                <RotateCcw size={15} /> Reset to Baseline
              </button>

              <button
                onClick={() => setShowSaveDialog(true)}
                className="btn-secondary text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 cursor-pointer shadow-sm"
              >
                <Plus size={15} /> Save Scenario
              </button>

              {savedScenarios.length > 0 && (
                <button
                  onClick={() => setShowComparison(true)}
                  className="btn-secondary text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 cursor-pointer shadow-sm"
                >
                  <Sliders size={15} /> Compare Scenarios
                </button>
              )}
            </div>

            {/* Line 2: Report & Insights Controls */}
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                onClick={handleReport}
                disabled={isGeneratingReport}
                className="btn-primary text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 disabled:opacity-60 cursor-pointer shadow-sm"
              >
                <FileText size={15} />
                {isGeneratingReport ? "Generating…" : hasReport ? "View Report" : "Generate Report"}
              </button>

              <button
                onClick={handleEmailReport}
                disabled={isEmailing || twinEmailSent}
                className={`text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 transition-all cursor-pointer shadow-sm ${
                  twinEmailSent
                    ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/30 font-bold cursor-default"
                    : "btn-secondary"
                }`}
              >
                {twinEmailSent ? <CheckCircle2 size={15} className="text-emerald-500" /> : <Mail size={15} />}
                <span>{isEmailing ? "Sending…" : twinEmailSent ? "Sent" : "Email Report"}</span>
              </button>

              <button
                onClick={() => setSummaryVisible(v => !v)}
                className="btn-secondary text-xs py-2.5 px-3.5 rounded-xl flex items-center gap-1.5 cursor-pointer shadow-sm"
              >
                <Sparkles size={15} /> AI Summary
              </button>
            </div>
          </div>
        </div>

        {/* ── Status / notice banners ─────────────────────────────────── */}
        {saveSuccessMsg && (
          <div className="mb-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold flex items-center gap-2">
            <Check size={16} /> {saveSuccessMsg}
          </div>
        )}
        {emailStatus && (
          <div className="mb-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold flex items-center gap-2">
            <Check size={16} /> {emailStatus}
          </div>
        )}
        {backendError && (
          <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-700 text-xs font-semibold flex items-center gap-2">
            <AlertTriangle size={16} /> {backendError}
          </div>
        )}

        {/* ── Disclaimer banner ──────────────────────────────────────── */}
        <div className="mb-6 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-700 text-xs flex items-start gap-2">
          <Info size={15} className="mt-0.5 shrink-0" />
          <span>
            <strong>Model limitation:</strong> Scores shown are rule-based composite estimates (TwinEngine), 
            not calibrated ML probabilities from the primary prediction model. 
            They reflect hypothetical parameter changes and should not replace clinical judgment.
          </span>
        </div>

        {/* ── Baseline Identification Banner ─────────────────────────── */}
        <div className="mb-6 p-4 rounded-xl border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)]">
          <div className="flex items-center gap-2 text-[var(--accent-melanzane)] font-bold text-xs mb-2">
            <Activity size={14} /> Patient Baseline (Immutable Reference)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-[var(--text-secondary)]">
            <span>Age: <strong className="text-[var(--text-primary)]">{baseline.age} yrs</strong></span>
            <span>BP: <strong className="text-[var(--text-primary)]">{baseline.systolic}/{baseline.diastolic} mmHg</strong></span>
            <span>Cholesterol: <strong className="text-[var(--text-primary)]">{baseline.cholesterol} mg/dL</strong></span>
            <span>BMI: <strong className="text-[var(--text-primary)]">{baseline.bmi}</strong></span>
            <span>Glucose: <strong className="text-[var(--text-primary)]">{baseline.glucose} mg/dL</strong></span>
            <span>Exercise: <strong className="text-[var(--text-primary)]">{baseline.exercise} days/wk</strong></span>
            <span>Smoking: <strong className="text-[var(--text-primary)]">{baseline.smoking === "active" || baseline.smoking === "1" ? "Active" : baseline.smoking === "former" ? "Former" : "None"}</strong></span>
            <span>Baseline Score: <strong className="text-[var(--accent-melanzane)]">{baselineScore}</strong></span>
          </div>
        </div>

        {/* ── Saved Scenarios List ────────────────────────────────────── */}
        {savedScenarios.length > 0 && (
          <div className="mb-6">
            <h3 className="text-xs font-bold text-[var(--text-primary)] mb-3">Saved Scenarios</h3>
            <div className="flex flex-wrap gap-2">
              {savedScenarios.map(sc => {
                const delta = sc.score - baselineScore;
                const lv = riskLevel(sc.score);
                return (
                  <div
                    key={sc.id}
                    onClick={() => loadScenario(sc)}
                    className={`cursor-pointer flex items-center gap-2 px-3 py-2 rounded-xl border text-xs transition-all ${
                      activeScenarioId === sc.id
                        ? "bg-[var(--accent-melanzane)] text-white border-[var(--accent-melanzane)]"
                        : "bg-[var(--bg-secondary)] text-[var(--text-secondary)] border-[var(--border-color)] hover:border-[var(--accent-melanzane)]"
                    }`}
                  >
                    <span className="font-semibold">{sc.name}</span>
                    <span style={{ color: activeScenarioId === sc.id ? "white" : lv.color }} className="font-bold">
                      {sc.score} ({delta >= 0 ? "+" : ""}{delta})
                    </span>
                    <button
                      onClick={e => { e.stopPropagation(); duplicateScenario(sc); }}
                      className="opacity-60 hover:opacity-100 transition"
                      title="Duplicate"
                    >
                      <Copy size={12} />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); deleteScenario(sc.id); }}
                      className="opacity-60 hover:opacity-100 transition text-red-400"
                      title="Delete"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Dashboard Grid ──────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* LEFT: Sliders */}
          <div className="lg:col-span-8 flex flex-col gap-4">

            {/* Hypothetical scenario indicator */}
            <div className={`rounded-xl border px-4 py-3 text-xs font-semibold ${
              currentDiffs.length > 0
                ? "border-amber-400/40 bg-amber-500/10 text-amber-700"
                : "border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)]"
            }`}>
              {currentDiffs.length > 0
                ? `⚡ Hypothetical Scenario — ${currentDiffs.length} parameter${currentDiffs.length > 1 ? "s" : ""} changed from baseline`
                : "Current Patient Baseline — no changes yet"
              }
            </div>

            {/* Changed params summary */}
            {currentDiffs.length > 0 && (
              <div className="cardio-card p-4">
                <div className="text-xs font-bold text-[var(--text-primary)] mb-2">Changed Parameters (vs Baseline)</div>
                <div className="space-y-1">
                  {currentDiffs.map(d => (
                    <div key={d.key} className="flex items-center gap-2 text-xs">
                      <span className="text-[var(--text-secondary)] w-40 shrink-0">{d.label}</span>
                      <span className="line-through text-red-400">{String(d.from)}</span>
                      <span className="text-emerald-600 font-semibold">→ {String(d.to)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Active Clinical Safety Alerts ─────────────────────── */}
            {activeAlerts.length > 0 && (
              <div className="cardio-card p-5 border-2 border-amber-500/40 bg-amber-500/10 space-y-3">
                <div className="flex items-center gap-2 text-amber-800 font-bold text-sm">
                  <AlertTriangle size={18} className="text-amber-600 shrink-0" />
                  <span>Clinical Safety Alert — Screening Threshold Triggered</span>
                  {currentDiffs.length > 0 && (
                    <span className="ml-auto text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-800 border border-amber-500/30">
                      Hypothetical Scenario
                    </span>
                  )}
                </div>
                <div className="space-y-3">
                  {activeAlerts.map((alt, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded-xl bg-white/80 border border-amber-300 text-xs space-y-1.5 text-[var(--text-primary)]"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-amber-950 flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
                          {alt.parameter}
                        </span>
                        <span className="font-mono text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800 font-bold">
                          {alt.measured_value} {alt.unit} ({alt.threshold_rule})
                        </span>
                      </div>
                      <p className="text-[var(--text-secondary)] leading-relaxed">{alt.explanation}</p>
                      <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 font-medium text-[11px] leading-relaxed">
                        <strong>Clinical Guidance:</strong> {alt.recommendation}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-[11px] text-amber-800/80 italic flex items-center gap-1.5 pt-1">
                  <Info size={13} className="shrink-0" />
                  <span>
                    Screening alert evaluated independently of composite risk score. Low risk scores never suppress alerts. Thresholds reflect adult population screening references.
                  </span>
                </div>
              </div>
            )}

            {/* ── Section 1: Vitals & Demographics ──────────────────── */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("vitals")} className="flex w-full items-center justify-between p-5 text-left">
                <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                  <Scale size={18} className="text-[var(--accent-melanzane)]" />
                  Vitals &amp; Demographics (5 Parameters)
                </h3>
                <ChevronDown size={18} className={`transition-transform ${activeSection === "vitals" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "vitals" ? "hidden" : ""}`}>
                {/* 1. Age */}
                <SliderField
                  label="1. Age" value={params.age} baseline={baseline.age}
                  unit="yrs" min={18} max={100}
                  onChange={v => upd("age", v)} />
                {/* 2. Weight */}
                <SliderField
                  label="2. Weight" value={params.weight} baseline={baseline.weight}
                  unit="kg" min={30} max={180}
                  onChange={v => upd("weight", v)} />
                {/* 3. BMI */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">3. Body Mass Index (BMI)</span>
                    <span className="text-[var(--accent-melanzane)] font-bold">{params.bmi} ({bmiCat(params.bmi).label})</span>
                  </div>
                  <input type="range" min="15" max="45" step="0.1" value={params.bmi}
                    onChange={e => upd("bmi", Number(e.target.value))} className="cardio-slider" />
                  <BaselineHint baseline={baseline.bmi} current={params.bmi} />
                </div>
                {/* 4. Systolic BP */}
                <SliderField
                  label="4. Systolic BP" value={params.systolic} baseline={baseline.systolic}
                  unit="mmHg" min={80} max={210}
                  onChange={v => upd("systolic", v)} />
                {/* 5. Diastolic BP */}
                <SliderField
                  label="5. Diastolic BP" value={params.diastolic} baseline={baseline.diastolic}
                  unit="mmHg" min={50} max={140}
                  onChange={v => upd("diastolic", v)} />
              </div>
            </div>

            {/* ── Section 2: Blood Biomarkers ────────────────────────── */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("biomarkers")} className="flex w-full items-center justify-between p-5 text-left">
                <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                  <Activity size={18} className="text-[var(--accent-melanzane)]" />
                  Blood Biomarkers &amp; Cardiac Markers (5 Parameters)
                </h3>
                <ChevronDown size={18} className={`transition-transform ${activeSection === "biomarkers" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "biomarkers" ? "hidden" : ""}`}>
                <SliderField label="6. Total Cholesterol" value={params.cholesterol} baseline={baseline.cholesterol}
                  unit="mg/dL" min={100} max={400} onChange={v => upd("cholesterol", v)} />
                <SliderField label="7. HDL Cholesterol" value={params.hdl} baseline={baseline.hdl}
                  unit="mg/dL" min={20} max={100} onChange={v => upd("hdl", v)} />
                <SliderField label="8. LDL Cholesterol" value={params.ldl} baseline={baseline.ldl}
                  unit="mg/dL" min={40} max={250} onChange={v => upd("ldl", v)} />
                <SliderField label="9. Fasting Blood Glucose" value={params.glucose} baseline={baseline.glucose}
                  unit="mg/dL" min={70} max={300} onChange={v => upd("glucose", v)} />
                <SliderField label="10. Resting Heart Rate" value={params.heartRate} baseline={baseline.heartRate}
                  unit="bpm" min={40} max={160} onChange={v => upd("heartRate", v)} />
              </div>
            </div>

            {/* ── Section 3: Lifestyle & Adherence ─────────────────────── */}
            <div className="cardio-card p-0 overflow-hidden">
              <button onClick={() => setActiveSection("lifestyle")} className="flex w-full items-center justify-between p-5 text-left">
                <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold flex items-center gap-2">
                  <Flame size={18} className="text-[var(--accent-melanzane)]" />
                  Lifestyle &amp; Adherence (4 Parameters)
                </h3>
                <ChevronDown size={18} className={`transition-transform ${activeSection === "lifestyle" ? "rotate-180" : ""}`} />
              </button>

              <div className={`space-y-5 px-5 pb-5 ${activeSection !== "lifestyle" ? "hidden" : ""}`}>
                <SliderField label="11. Exercise Frequency" value={params.exercise} baseline={baseline.exercise}
                  unit="days/wk" min={0} max={7} onChange={v => upd("exercise", v)} />

                {/* 12. Smoking */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">12. Tobacco / Smoking Status</span>
                    <span className="text-[var(--accent-melanzane)] font-bold">
                      {params.smoking === "active" || params.smoking === "1" ? "Active" : params.smoking === "former" ? "Former" : "Non-Smoker"}
                    </span>
                  </div>
                  <div className="flex gap-3">
                    {[["0", "Non-Smoker"], ["former", "Former"], ["active", "Active"]].map(([val, lbl]) => (
                      <button key={val} onClick={() => upd("smoking", val)}
                        className={`flex-1 py-2 text-xs font-semibold rounded-xl border transition-all ${
                          params.smoking === val
                            ? "bg-[var(--accent-melanzane)] text-white border-[var(--accent-melanzane)]"
                            : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                        }`}>
                        {lbl}
                      </button>
                    ))}
                  </div>
                  <BaselineHint baseline={baseline.smoking} current={params.smoking} textual />
                </div>

                {/* 13. Alcohol */}
                <div>
                  <div className="flex justify-between text-xs font-medium mb-1.5">
                    <span className="text-[var(--text-primary)] font-semibold">13. Alcohol Consumption</span>
                    <span className="text-[var(--accent-melanzane)] font-bold">
                      {params.alcohol === "regular" || params.alcohol === "1" ? "Regular" : params.alcohol === "occasional" ? "Occasional" : "None"}
                    </span>
                  </div>
                  <div className="flex gap-3">
                    {[["0", "None / Minimal"], ["occasional", "Occasional"], ["regular", "Regular"]].map(([val, lbl]) => (
                      <button key={val} onClick={() => upd("alcohol", val)}
                        className={`flex-1 py-2 text-xs font-semibold rounded-xl border transition-all ${
                          params.alcohol === val
                            ? "bg-[var(--accent-melanzane)] text-white border-[var(--accent-melanzane)]"
                            : "bg-[var(--input-bg)] text-[var(--text-secondary)] border-[var(--input-border)]"
                        }`}>
                        {lbl}
                      </button>
                    ))}
                  </div>
                  <BaselineHint baseline={baseline.alcohol} current={params.alcohol} textual />
                </div>

                {/* 14. Medication Adherence */}
                <SliderField label="14. Medication Adherence" value={params.medicationAdherence}
                  baseline={baseline.medicationAdherence} unit="%" min={0} max={100}
                  onChange={v => upd("medicationAdherence", v)} />
              </div>
            </div>
          </div>

          {/* RIGHT: Risk Gauge + Chart + Insights */}
          <div className="lg:col-span-4 lg:sticky lg:top-24 self-start space-y-4">

            {/* Live Risk Score Card */}
            <div className="cardio-card p-6 flex flex-col items-center text-center">
              <span className="caption-small uppercase font-bold text-[var(--accent-melanzane)]">
                Hypothetical Composite Score
              </span>
              <p className="text-[9px] text-[var(--text-muted)] mb-2">
                Rule-based estimate — not a calibrated probability
              </p>

              {/* Gauge Arc */}
              <div className="relative w-48 h-28 my-4 flex items-end justify-center">
                <svg className="w-full h-full" viewBox="0 0 100 55">
                  <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke="var(--border-color)" strokeWidth="8" strokeLinecap="round" />
                  <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={currentLevel.color}
                    strokeWidth="8" strokeDasharray="126"
                    strokeDashoffset={126 - (126 * currentScore) / 100}
                    strokeLinecap="round" className="transition-all duration-500 ease-out" />
                </svg>
                <div className="absolute bottom-0 text-center">
                  <span className="text-3xl font-extrabold text-[var(--text-primary)]">{currentScore}</span>
                </div>
              </div>

              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold"
                style={{ backgroundColor: `${currentLevel.color}15`, color: currentLevel.color }}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: currentLevel.color }} />
                {currentLevel.label} Composite Risk
              </div>

              {/* Delta vs baseline */}
              <div className={`mt-3 text-xs font-semibold ${scoreDelta > 0 ? "text-red-500" : scoreDelta < 0 ? "text-emerald-600" : "text-[var(--text-muted)]"}`}>
                {scoreDelta === 0
                  ? "Same as baseline"
                  : `${scoreDelta > 0 ? "+" : ""}${scoreDelta} vs baseline (${baselineScore})`}
              </div>

              {/* Baseline vs Scenario comparison row */}
              <div className="mt-4 w-full grid grid-cols-2 gap-2 text-xs">
                <div className="p-2 rounded-lg bg-[var(--bg-secondary)] text-center">
                  <div className="text-[var(--text-muted)]">Baseline</div>
                  <div className="font-bold text-[var(--text-primary)]">{baselineScore}</div>
                </div>
                <div className="p-2 rounded-lg bg-[var(--accent-melanzane-light)] border border-[var(--accent-melanzane-border)] text-center">
                  <div className="text-[var(--accent-melanzane)]">Scenario</div>
                  <div className="font-bold text-[var(--text-primary)]">{currentScore}</div>
                </div>
              </div>
            </div>

            {/* Trajectory Graph */}
            <div className="cardio-card p-6">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold mb-2 flex items-center justify-between">
                <span>Projected Risk Trajectory</span>
                <BarChart2 size={16} className="text-[var(--text-muted)]" />
              </h3>
              <p className="caption-small mb-1 text-[9px] text-amber-600 font-semibold">
                ⚠ Illustrative only — assumes constant current parameters. Not a validated clinical forecast.
              </p>
              <p className="caption-small mb-4">Composite score over age assuming unchanged hypothetical parameters:</p>

              <div className="w-full h-40 relative">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120">
                  <line x1="0" y1="30" x2="300" y2="30" stroke="var(--border-color)" strokeDasharray="3 3" />
                  <line x1="0" y1="70" x2="300" y2="70" stroke="var(--border-color)" strokeDasharray="3 3" />
                  <path
                    d={trajectoryData.map((pt, i) => `${i === 0 ? "M" : "L"} ${i * 60 + 10} ${110 - (pt.risk * 0.9)}`).join(" ")}
                    fill="none" stroke="var(--accent-melanzane)" strokeWidth="3"
                    className="transition-all duration-300" />
                  {trajectoryData.map((pt, i) => (
                    <g key={i}>
                      <circle cx={i * 60 + 10} cy={110 - (pt.risk * 0.9)} r="4"
                        fill="var(--accent-melanzane)" stroke="#ffffff" strokeWidth="2" />
                      <text x={i * 60 + 10} y="118" fontSize="9" fill="var(--text-muted)" textAnchor="middle">
                        {pt.age}y
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            </div>

            {/* Key Parameter Impact */}
            <div className="cardio-card p-6">
              <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold mb-3">Key Parameter Impact</h3>
              <ul className="space-y-2 text-xs text-[var(--text-secondary)]">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                  Systolic BP: {params.systolic > 130 ? "High elevation increases load" : "Optimal range"}
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                  Cholesterol Ratio: LDL {params.ldl} vs HDL {params.hdl}
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                  Exercise Mitigation: −{(params.exercise * 2.2).toFixed(1)} composite points
                </li>
              </ul>
            </div>

            {/* AI Summary */}
            {summaryVisible && (
              <div className="cardio-card p-6 border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)]">
                <div className="mb-2 flex items-center gap-2 text-sm font-bold text-[var(--accent-melanzane)]">
                  <Sparkles size={16} /> Hypothetical Scenario Interpretation
                </div>
                <p className="text-sm leading-6 text-[var(--text-secondary)]">
                  Hypothetical composite risk is {currentLevel.label} ({currentScore}).
                  Baseline composite score: {baselineScore} ({scoreDelta >= 0 ? "+" : ""}{scoreDelta} change).
                  {currentDiffs.length > 0
                    ? ` ${currentDiffs.length} parameter(s) changed from baseline: ${currentDiffs.map(d => d.label).join(", ")}.`
                    : " No parameters changed from baseline."}
                  {" "}These are rule-based estimates. Discuss with a qualified clinician before making health decisions.
                </p>
              </div>
            )}

            {/* Backend Simulation Results */}
            {backendResults && (
              <div className="cardio-card p-6">
                <h3 className="section-title text-[var(--text-primary)] text-sm font-semibold mb-3">
                  Backend Simulation Results (TwinEngine)
                </h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between font-semibold border-b border-[var(--border-color)] pb-1 mb-1">
                    <span>Baseline Score</span>
                    <span className="text-[var(--accent-melanzane)]">{backendResults.baseline_twin_risk_score}</span>
                  </div>
                  {backendResults.scenarios.map(sc => {
                    const delta = sc.twin_risk_score - backendResults.baseline_twin_risk_score;
                    return (
                      <div key={sc.scenario_name} className="flex justify-between text-[var(--text-secondary)]">
                        <span>{sc.scenario_name}</span>
                        <span className={delta >= 0 ? "text-red-500 font-semibold" : "text-emerald-600 font-semibold"}>
                          {sc.twin_risk_score} ({delta >= 0 ? "+" : ""}{delta.toFixed(2)})
                        </span>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[9px] text-[var(--text-muted)] mt-3">{backendResults.disclaimer}</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Save Scenario Dialog ────────────────────────────────────── */}
        {showSaveDialog && (
          <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6">
            <div className="cardio-card max-w-sm w-full p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="section-title text-[var(--text-primary)] font-bold">Save Scenario</h2>
                <button onClick={() => setShowSaveDialog(false)} className="p-1 rounded-lg hover:bg-[var(--bg-secondary)]">
                  <X size={18} />
                </button>
              </div>
              <div className="text-xs text-[var(--text-secondary)]">
                Saves a snapshot of the current parameter values as a named scenario for comparison.
              </div>
              <input
                type="text"
                value={scenarioNameInput}
                onChange={e => setScenarioNameInput(e.target.value)}
                placeholder={`Scenario ${savedScenarios.length + 1}`}
                className="w-full px-3 py-2 text-sm rounded-xl border border-[var(--input-border)] bg-[var(--input-bg)] text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-melanzane)]"
                onKeyDown={e => e.key === "Enter" && saveCurrentScenario()}
                autoFocus
              />
              <div className="flex gap-3 justify-end">
                <button onClick={() => setShowSaveDialog(false)} className="btn-secondary text-xs py-2 px-4">Cancel</button>
                <button onClick={saveCurrentScenario} className="btn-primary text-xs py-2 px-4">Save</button>
              </div>
            </div>
          </div>
        )}

        {/* ── Compare Scenarios Modal ─────────────────────────────────── */}
        {showComparison && (
          <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6 overflow-y-auto">
            <div className="cardio-card max-w-5xl w-full p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-3">
                <h2 className="section-title text-[var(--text-primary)] font-bold">Scenario Comparison</h2>
                <button onClick={() => setShowComparison(false)} className="p-1 rounded-lg hover:bg-[var(--bg-secondary)]">
                  <X size={18} />
                </button>
              </div>

              <div className="text-xs text-amber-700 bg-amber-500/10 border border-amber-400/30 rounded-xl p-3 flex gap-2">
                <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                Scores below are TwinEngine composite estimates (rule-based). They are NOT calibrated ML probabilities.
                Absolute differences represent changes in the composite score only.
              </div>

              {/* Comparison Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-separate border-spacing-0">
                  <thead>
                    <tr>
                      <th className="text-left py-2 px-3 bg-[var(--bg-secondary)] rounded-tl-xl text-[var(--text-secondary)] font-semibold">Scenario</th>
                      <th className="text-center py-2 px-3 bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-semibold">Score</th>
                      <th className="text-center py-2 px-3 bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-semibold">vs Baseline</th>
                      <th className="text-center py-2 px-3 bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-semibold">Level</th>
                      <th className="text-left py-2 px-3 bg-[var(--bg-secondary)] rounded-tr-xl text-[var(--text-secondary)] font-semibold">Changed Parameters</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Baseline row */}
                    <tr className="border-b border-[var(--border-color)]">
                      <td className="py-2 px-3 font-bold text-[var(--accent-melanzane)]">📍 Baseline</td>
                      <td className="py-2 px-3 text-center font-bold">{baselineScore}</td>
                      <td className="py-2 px-3 text-center text-[var(--text-muted)]">—</td>
                      <td className="py-2 px-3 text-center">
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold"
                          style={{ backgroundColor: `${riskLevel(baselineScore).color}15`, color: riskLevel(baselineScore).color }}>
                          {riskLevel(baselineScore).label}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-[var(--text-muted)]">Original measurements</td>
                    </tr>
                    {/* Saved scenarios */}
                    {savedScenarios.map(sc => {
                      const delta = sc.score - baselineScore;
                      const lv = riskLevel(sc.score);
                      const diffs = diffParams(baseline, sc.params);
                      return (
                        <tr key={sc.id} className="border-b border-[var(--border-color)]">
                          <td className="py-2 px-3 font-semibold text-[var(--text-primary)]">{sc.name}</td>
                          <td className="py-2 px-3 text-center font-bold">{sc.score}</td>
                          <td className={`py-2 px-3 text-center font-bold ${delta > 0 ? "text-red-500" : delta < 0 ? "text-emerald-600" : "text-[var(--text-muted)]"}`}>
                            {delta >= 0 ? "+" : ""}{delta}
                          </td>
                          <td className="py-2 px-3 text-center">
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold"
                              style={{ backgroundColor: `${lv.color}15`, color: lv.color }}>
                              {lv.label}
                            </span>
                          </td>
                          <td className="py-2 px-3 text-[var(--text-secondary)]">
                            {diffs.length === 0
                              ? "None"
                              : diffs.map(d => `${d.label}: ${d.from}→${d.to}`).join(", ")}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Mini chart comparing scenarios */}
              <div className="pt-2">
                <h4 className="text-xs font-bold text-[var(--text-primary)] mb-3">Score Comparison Chart</h4>
                <div className="flex items-end gap-3 h-28 overflow-x-auto pb-1">
                  {[{ name: "Baseline", score: baselineScore }, ...savedScenarios].map((item, i) => {
                    const maxScore = Math.max(baselineScore, ...savedScenarios.map(s => s.score));
                    const lv = riskLevel(item.score);
                    return (
                      <div key={i} className="flex flex-col items-center gap-1 min-w-[60px]">
                        <span className="text-[10px] font-bold" style={{ color: lv.color }}>{item.score}</span>
                        <div
                          className="w-10 rounded-t-lg transition-all duration-500"
                          style={{
                            height: `${Math.max(8, (item.score / maxScore) * 80)}px`,
                            backgroundColor: i === 0 ? "var(--accent-melanzane)" : lv.color,
                            opacity: i === 0 ? 1 : 0.8,
                          }}
                        />
                        <span className="text-[9px] text-[var(--text-muted)] text-center leading-tight">{item.name}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button onClick={() => setShowComparison(false)} className="btn-primary text-xs py-2 px-4">Close</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

/* ─── Sub-components ─────────────────────────────────────────── */

function BaselineHint({ baseline, current, unit = "", textual = false }) {
  const changed = String(baseline) !== String(current);
  if (!changed) return null;
  return (
    <div className="mt-1 text-[10px] text-amber-600 font-semibold">
      Baseline: {textual ? String(baseline) : `${baseline}${unit ? " " + unit : ""}`}
    </div>
  );
}

function SliderField({ label, value, baseline, unit, min, max, step = 1, onChange }) {
  const changed = value !== baseline;
  return (
    <div>
      <div className="flex justify-between text-xs font-medium mb-1.5">
        <span className={`font-semibold ${changed ? "text-amber-700" : "text-[var(--text-primary)]"}`}>
          {label}{changed ? " ✎" : ""}
        </span>
        <span className="text-[var(--accent-melanzane)] font-bold">{value} {unit}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="cardio-slider"
      />
      <BaselineHint baseline={baseline} current={value} unit={unit} />
    </div>
  );
}
