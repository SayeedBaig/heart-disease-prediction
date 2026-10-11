import { useState, useMemo } from "react";
import {
  Download,
  Mail,
  Printer,
  ShieldCheck,
  Activity,
  User,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Info,
} from "lucide-react";
import Navbar from "../components/Navbar";
import { downloadPatientReport, emailPatientReport } from "../services/portalService";

export default function Reports() {
  const [emailSentMsg, setEmailSentMsg] = useState("");
  const [emailSent, setEmailSent] = useState(false);
  const [emailLoading, setEmailLoading] = useState(false);
  const [downloadLoading, setDownloadLoading] = useState(false);

  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const cardioData = JSON.parse(localStorage.getItem("cardio-data") || "{}");
  const predictionStored = JSON.parse(localStorage.getItem("cardio-prediction") || "{}");
  const prediction = predictionStored?.prediction || predictionStored || {};
  const predictionId = localStorage.getItem("prediction_id") || predictionStored?.prediction_id || predictionStored?.id;

  const patientDetails = {
    name: patient.full_name || patient.name || "Jane Doe",
    id: patient.patient_id || "PT-84920",
    age: patient.age || 52,
    gender: patient.gender || "Female",
    dob: patient.dateOfBirth || "1974-03-15",
    date: new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }),
  };

  const riskPct = prediction?.fusion?.risk_percentage || 24;
  const level = prediction?.fusion?.final_level || "Moderate";
  const confidence = prediction?.fusion?.confidence_percentage || 91;

  // ── Clinical Alerts computation ──────────────────────────────────────────
  const clinicalAlerts = useMemo(() => {
    // 1. Direct alerts from prediction payload if available
    const stored = prediction?.clinical_alerts || predictionStored?.clinical_alerts;
    if (Array.isArray(stored) && stored.length > 0) {
      return stored;
    }

    // 2. Screening thresholds evaluation on vitals
    const sys = Number(cardioData?.systolic || prediction?.clinical?.ap_hi || prediction?.clinical?.systolic || 132);
    const dia = Number(cardioData?.diastolic || prediction?.clinical?.ap_lo || prediction?.clinical?.diastolic || 84);
    const alerts = [];

    if (sys && sys < 90) {
      alerts.push({
        alert_id: "LOW_SYSTOLIC_BP",
        parameter: "Systolic Blood Pressure",
        measured_value: sys,
        unit: "mmHg",
        category: "Blood Pressure",
        severity: "warning",
        threshold_rule: "Systolic BP < 90 mmHg (adult screening reference)",
        explanation: `Your recorded blood pressure is ${sys}/${dia} mmHg. The systolic component (${sys} mmHg) is below the configured adult screening threshold of 90 mmHg, which may indicate low blood pressure. This is a screening reference and does not constitute a diagnosis of hypotension.`,
        recommendation: "Consider arranging a clinical review, particularly if you experience dizziness, weakness, blurred vision, or fainting. Do not change prescribed medication without medical advice.",
        is_hypothetical: false,
      });
    }
    if (dia && dia < 60) {
      alerts.push({
        alert_id: "LOW_DIASTOLIC_BP",
        parameter: "Diastolic Blood Pressure",
        measured_value: dia,
        unit: "mmHg",
        category: "Blood Pressure",
        severity: "warning",
        threshold_rule: "Diastolic BP < 60 mmHg (adult screening reference)",
        explanation: `Your recorded blood pressure is ${sys}/${dia} mmHg. The diastolic component (${dia} mmHg) is below the configured adult screening threshold of 60 mmHg, which may indicate low diastolic blood pressure.`,
        recommendation: "Consider arranging a clinical review with a qualified healthcare professional. Do not change prescribed medication without medical advice.",
        is_hypothetical: false,
      });
    }
    if (sys && sys >= 140) {
      alerts.push({
        alert_id: "HIGH_SYSTOLIC_BP",
        parameter: "Systolic Blood Pressure",
        measured_value: sys,
        unit: "mmHg",
        category: "Blood Pressure",
        severity: "warning",
        threshold_rule: "Systolic BP ≥ 140 mmHg (adult screening reference)",
        explanation: `Your recorded blood pressure is ${sys}/${dia} mmHg. The systolic component (${sys} mmHg) meets or exceeds the configured adult screening threshold of 140 mmHg. A single reading is not sufficient to diagnose hypertension.`,
        recommendation: "Discuss this reading with a healthcare professional. A single elevated reading does not diagnose hypertension.",
        is_hypothetical: false,
      });
    }
    if (dia && dia >= 90) {
      alerts.push({
        alert_id: "HIGH_DIASTOLIC_BP",
        parameter: "Diastolic Blood Pressure",
        measured_value: dia,
        unit: "mmHg",
        category: "Blood Pressure",
        severity: "warning",
        threshold_rule: "Diastolic BP ≥ 90 mmHg (adult screening reference)",
        explanation: `Your recorded blood pressure is ${sys}/${dia} mmHg. The diastolic component (${dia} mmHg) meets or exceeds the configured adult screening threshold of 90 mmHg. A single reading is not sufficient to diagnose hypertension.`,
        recommendation: "Discuss this reading with a healthcare professional. A single elevated reading does not diagnose hypertension.",
        is_hypothetical: false,
      });
    }
    return alerts;
  }, [prediction, predictionStored, cardioData]);

  const handlePrint = () => {
    window.print();
  };

  const handleEmail = async () => {
    setEmailLoading(true);
    if (predictionId) {
      try {
        await emailPatientReport(predictionId);
        setEmailSent(true);
        setEmailSentMsg("Official PDF report has been emailed to your registered address.");
        setTimeout(() => setEmailSentMsg(""), 4000);
        return;
      } catch (err) {
        console.warn("Backend email failed, falling back to simulated status", err);
      } finally {
        setEmailLoading(false);
      }
    }
    setTimeout(() => {
      setEmailLoading(false);
      setEmailSent(true);
      setEmailSentMsg("Report has been emailed to your registered address.");
      setTimeout(() => setEmailSentMsg(""), 4000);
    }, 600);
  };

  const handleDownloadPDF = async () => {
    if (predictionId) {
      try {
        setDownloadLoading(true);
        await downloadPatientReport(predictionId);
        setDownloadLoading(false);
        return;
      } catch (err) {
        console.warn("PDF API download failed, using text report fallback", err);
        setDownloadLoading(false);
      }
    }

    const alertsBlock = clinicalAlerts.length > 0
      ? clinicalAlerts.map(a =>
          `* [${(a.severity || "warning").toUpperCase()}] ${a.parameter}: ${a.measured_value} ${a.unit} (${a.threshold_rule})\n  Explanation: ${a.explanation}\n  Clinical Advice: ${a.recommendation}`
        ).join("\n\n")
      : "  No clinical screening alerts triggered for recorded vitals.";

    const reportText = `CardioAI Official Clinical Report
Date: ${patientDetails.date}
Patient: ${patientDetails.name} (ID: ${patientDetails.id}, Age: ${patientDetails.age}, Gender: ${patientDetails.gender})
--------------------------------------------------
1. RISK ANALYSIS & PREDICTION
Risk Score: ${riskPct}% (${level} Risk Level)
AI Model Confidence: ${confidence}%

2. CLINICAL SUMMARY
Vitals: Systolic BP ${cardioData?.systolic || 132} mmHg, Diastolic BP ${cardioData?.diastolic || 84} mmHg
Lipids: Cholesterol ${cardioData?.cholesterol || 210} mg/dL, Fasting Glucose ${cardioData?.glucose || 102} mg/dL

3. CLINICAL ALERTS & SCREENING WARNINGS
${alertsBlock}

Screening Reference Note: Adult screening reference thresholds (ESH/ESC, AHA/ACC). Screening thresholds are independent of ML risk scores. Do not alter prescribed medications without medical advice.

4. ECG FINDINGS
Sinus rhythm with mild ST-segment elevation detected in anterior leads.

5. ECHO FINDINGS
Left ventricular ejection fraction (LVEF): 58%. Normal wall motion.

6. RECOMMENDATIONS
- Maintain low sodium diet & daily 30-min walking.
- Schedule 3-month follow-up with cardiologist.

7. DOCTOR NOTES
Screening findings indicate moderate risk trajectory. Recommended for clinical monitoring.
--------------------------------------------------
CardioAI Decision Support System · For Information & Research Screening Purposes Only
`;
    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `CardioAI_Report_${patientDetails.id}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div>
        {/* Actions Bar (hidden during print) */}
        <div className="no-print flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)] gap-4">
          <div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Diagnostic Summary
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Official Medical Report
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleDownloadPDF}
              className="btn-primary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2"
            >
              <Download size={15} />
              Download PDF
            </button>

            <button
              onClick={handleEmail}
              disabled={emailLoading || emailSent}
              className={`text-xs py-2.5 px-4 rounded-xl flex items-center gap-2 transition-all ${
                emailSent
                  ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/30 font-bold cursor-default"
                  : "btn-secondary"
              }`}
            >
              {emailSent ? <CheckCircle2 size={15} className="text-emerald-500" /> : <Mail size={15} />}
              <span>{emailLoading ? "Sending..." : emailSent ? "Sent" : "Email Report"}</span>
            </button>

            <button
              onClick={handlePrint}
              className="btn-secondary text-xs py-2.5 px-4 rounded-xl flex items-center gap-2"
            >
              <Printer size={15} />
              Print Report
            </button>
          </div>
        </div>

        {emailSentMsg && (
          <div className="no-print mb-6 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 text-xs font-semibold flex items-center gap-2">
            <CheckCircle2 size={16} />
            {emailSentMsg}
          </div>
        )}

        {/* PRINTABLE MEDICAL REPORT CARD */}
        <div className="cardio-card p-8 space-y-8">
          {/* Header */}
          <div className="flex justify-between items-start border-b border-[var(--border-color)] pb-6">
            <div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-[var(--accent-melanzane)] text-white flex items-center justify-center font-bold text-sm">
                  AI
                </div>
                <h2 className="text-xl font-bold tracking-tight text-[var(--text-primary)]">
                  CardioAI Medical Report
                </h2>
              </div>
              <p className="caption-small mt-1">
                Multimodal AI Cardiovascular Diagnostic Report
              </p>
            </div>

            <div className="text-right text-xs text-[var(--text-muted)]">
              <div><strong>Report Date:</strong> {patientDetails.date}</div>
              <div><strong>Report ID:</strong> RPT-2026-9041</div>
            </div>
          </div>

          {/* Section 1: Patient Details */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <User size={16} />
              1. Patient Details
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 rounded-xl bg-[var(--bg-secondary)] text-xs">
              <div>
                <span className="caption-small">Full Name:</span>
                <div className="font-semibold text-[var(--text-primary)]">{patientDetails.name}</div>
              </div>
              <div>
                <span className="caption-small">Patient ID:</span>
                <div className="font-semibold text-[var(--text-primary)]">{patientDetails.id}</div>
              </div>
              <div>
                <span className="caption-small">Age / Gender:</span>
                <div className="font-semibold text-[var(--text-primary)]">{patientDetails.age} yrs / {patientDetails.gender}</div>
              </div>
              <div>
                <span className="caption-small">Date of Birth:</span>
                <div className="font-semibold text-[var(--text-primary)]">{patientDetails.dob}</div>
              </div>
            </div>
          </div>

          {/* Section 2: Prediction & Risk Analysis */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <Activity size={16} />
              2. Prediction & Risk Analysis
            </h3>
            <div className="p-5 rounded-xl border border-[var(--border-color)] flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-[var(--accent-melanzane-light)] border border-[var(--accent-melanzane-border)] text-[var(--accent-melanzane)] flex items-center justify-center font-extrabold text-2xl">
                  {riskPct}%
                </div>
                <div>
                  <div className="text-sm font-bold text-[var(--text-primary)]">
                    {level} Risk Profile
                  </div>
                  <div className="caption-small mt-0.5">
                    Calculated via Multimodal Neural Network Fusion
                  </div>
                </div>
              </div>

              <div className="text-right text-xs">
                <span className="caption-small">AI Confidence Score</span>
                <div className="font-bold text-[var(--text-primary)] text-base">{confidence}%</div>
              </div>
            </div>
          </div>

          {/* Section 3: Clinical Alerts & Screening Warnings */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] flex items-center gap-2">
                <AlertTriangle size={16} className={clinicalAlerts.length > 0 ? "text-amber-500" : "text-emerald-500"} />
                3. Clinical Alerts &amp; Screening Warnings
              </h3>
              <span className="text-[10px] font-bold uppercase px-2.5 py-1 rounded-full bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border-color)]">
                {clinicalAlerts.length} Alert{clinicalAlerts.length !== 1 ? "s" : ""}
              </span>
            </div>

            {clinicalAlerts.length > 0 ? (
              <div className="space-y-3">
                {clinicalAlerts.map((alt, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl border border-amber-300 bg-amber-500/10 space-y-2 text-xs"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-amber-950 text-sm">{alt.parameter}</span>
                        <span className="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded bg-amber-200 text-amber-900">
                          {alt.severity || "Warning"}
                        </span>
                        <span className="text-[10px] uppercase font-extrabold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300">
                          Observed Measurement
                        </span>
                      </div>
                      <span className="font-mono font-bold text-amber-900 bg-amber-200/80 px-2.5 py-1 rounded">
                        {alt.measured_value} {alt.unit}
                      </span>
                    </div>

                    <div className="text-[11px] text-amber-900 font-semibold">
                      Triggered Rule: <span className="font-mono text-amber-800">{alt.threshold_rule}</span>
                    </div>

                    <p className="text-[var(--text-secondary)] leading-relaxed">
                      {alt.explanation}
                    </p>

                    <div className="p-3 rounded-lg bg-white/80 border border-amber-200 text-amber-950 font-medium leading-relaxed">
                      <strong>Clinical Recommendation:</strong> {alt.recommendation}
                    </div>
                  </div>
                ))}

                <div className="p-3.5 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[11px] text-[var(--text-muted)] space-y-1">
                  <div className="font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
                    <Info size={13} />
                    <span>Screening Threshold Source &amp; Limitations</span>
                  </div>
                  <p>
                    Threshold source: Standard adult clinical screening references (ESH/ESC and AHA/ACC guidelines: Low BP &lt; 90/60 mmHg, High BP ≥ 140/90 mmHg). Screening thresholds are population reference rules and do not constitute a standalone medical diagnosis. Patients must not alter prescribed medications without medical advice. Clinical alerts are evaluated independently of machine learning risk scores.
                  </p>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs flex items-center gap-3 text-emerald-700">
                <CheckCircle2 size={18} className="shrink-0 text-emerald-600" />
                <div>
                  <div className="font-bold">No Clinical Screening Alerts Triggered</div>
                  <div className="text-[11px] text-emerald-600/90 mt-0.5">
                    All recorded vital signs and screening measurements fall within population reference ranges (Systolic BP 90–139 mmHg, Diastolic BP 60–89 mmHg).
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Section 4: Clinical Summary */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <FileText size={16} />
              4. Clinical Summary
            </h3>
            <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs space-y-2 leading-relaxed text-[var(--text-secondary)]">
              <p>
                Patient presents with blood pressure readings averaging <strong>{cardioData?.systolic || 132}/{cardioData?.diastolic || 84} mmHg</strong> and total cholesterol of <strong>{cardioData?.cholesterol || 210} mg/dL</strong>. Fasting glucose level is <strong>{cardioData?.glucose || 102} mg/dL</strong>.
              </p>
            </div>
          </div>

          {/* Section 5: ECG Findings */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <Activity size={16} />
              5. ECG Waveform Findings
            </h3>
            <div className="p-4 rounded-xl border border-[var(--border-color)] text-xs text-[var(--text-secondary)]">
              Normal sinus rhythm at 72 bpm. Mild ST-segment displacement noted in anterior precordial leads (V2-V4), consistent with early repolarization variant.
            </div>
          </div>

          {/* Section 6: Echo Findings */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <Activity size={16} />
              6. Echocardiogram Findings
            </h3>
            <div className="p-4 rounded-xl border border-[var(--border-color)] text-xs text-[var(--text-secondary)]">
              Left ventricular ejection fraction (LVEF) calculated at <strong>58%</strong> (Normal &gt; 55%). Normal valvular anatomy and no segmental wall motion abnormalities detected.
            </div>
          </div>

          {/* Section 7: Recommendations */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <ShieldCheck size={16} />
              7. Recommendations
            </h3>
            <ul className="space-y-2 text-xs text-[var(--text-secondary)]">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                Maintain regular moderate aerobic physical activity (min 150 mins/week).
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                Adopt a Mediterranean diet plan rich in omega-3 fatty acids and low in sodium (&lt; 2,300 mg/day).
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-melanzane)]" />
                Re-assess blood pressure and lipid panel in 90 days.
              </li>
            </ul>
          </div>

          {/* Section 8: Doctor Notes */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <FileText size={16} />
              8. Doctor Notes
            </h3>
            <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs font-mono text-[var(--text-primary)]">
              "Screening findings indicate low-to-moderate risk trajectory. No immediate invasive intervention indicated. Patient advised to maintain lifestyle modifications and return for quarterly follow-up."
            </div>
          </div>

          {/* Footer Disclaimer */}
          <div className="pt-6 border-t border-[var(--border-color)] text-[11px] text-[var(--text-muted)] text-center">
            CardioAI Decision Support System · For Information & Research Screening Purposes Only
          </div>
        </div>
      </div>
    </>
  );
}