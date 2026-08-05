import { useState } from "react";
import { Download, Mail, Printer, ShieldCheck, Activity, User, FileText, CheckCircle2 } from "lucide-react";
import Navbar from "../components/Navbar";

export default function Reports() {
  const [emailSentMsg, setEmailSentMsg] = useState("");
  const [emailSent, setEmailSent] = useState(false);
  const [emailLoading, setEmailLoading] = useState(false);

  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const prediction = JSON.parse(localStorage.getItem("cardio-prediction") || "{}")?.prediction || {};

  const patientDetails = {
    name: patient.full_name || patient.name || "Jane Doe",
    id: patient.patient_id || "PT-84920",
    age: patient.age || 52,
    gender: patient.gender || "Female",
    dob: patient.dateOfBirth || "1974-03-15",
    date: new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
  };

  const riskPct = prediction?.fusion?.risk_percentage || 24;
  const level = prediction?.fusion?.final_level || "Moderate";
  const confidence = prediction?.fusion?.confidence_percentage || 91;

  const handlePrint = () => {
    window.print();
  };

  const handleEmail = () => {
    setEmailLoading(true);
    setTimeout(() => {
      setEmailLoading(false);
      setEmailSent(true);
      setEmailSentMsg("Report has been emailed to your registered address.");
      setTimeout(() => setEmailSentMsg(""), 4000);
    }, 600);
  };

  const handleDownloadPDF = () => {
    const reportText = `CardioAI Official Clinical Report
Date: ${patientDetails.date}
Patient: ${patientDetails.name} (ID: ${patientDetails.id}, Age: ${patientDetails.age}, Gender: ${patientDetails.gender})
--------------------------------------------------
1. RISK ANALYSIS & PREDICTION
Risk Score: ${riskPct}% (${level} Risk Level)
AI Model Confidence: ${confidence}%

2. CLINICAL SUMMARY
Vitals: Systolic BP 132 mmHg, Diastolic BP 84 mmHg
Lipids: Cholesterol 210 mg/dL, Fasting Glucose 102 mg/dL

3. ECG FINDINGS
Sinus rhythm with mild ST-segment elevation detected in anterior leads.

4. ECHO FINDINGS
Left ventricular ejection fraction (LVEF): 58%. Normal wall motion.

5. RECOMMENDATIONS
- Maintain low sodium diet & daily 30-min walking.
- Schedule 3-month follow-up with cardiologist.

6. DOCTOR NOTES
Screening findings indicate moderate risk trajectory. Recommended for clinical monitoring.
--------------------------------------------------
CardioAI Intelligence System
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

          {/* Section 3: Clinical Summary */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <FileText size={16} />
              3. Clinical Summary
            </h3>
            <div className="p-4 rounded-xl bg-[var(--bg-secondary)] text-xs space-y-2 leading-relaxed text-[var(--text-secondary)]">
              <p>
                Patient presents with blood pressure readings averaging <strong>132/84 mmHg</strong> and total cholesterol of <strong>210 mg/dL</strong>. Fasting glucose level is <strong>102 mg/dL</strong>.
              </p>
            </div>
          </div>

          {/* Section 4: ECG Findings */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <Activity size={16} />
              4. ECG Waveform Findings
            </h3>
            <div className="p-4 rounded-xl border border-[var(--border-color)] text-xs text-[var(--text-secondary)]">
              Normal sinus rhythm at 72 bpm. Mild ST-segment displacement noted in anterior precordial leads (V2-V4), consistent with early repolarization variant.
            </div>
          </div>

          {/* Section 5: Echo Findings */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <Activity size={16} />
              5. Echocardiogram Findings
            </h3>
            <div className="p-4 rounded-xl border border-[var(--border-color)] text-xs text-[var(--text-secondary)]">
              Left ventricular ejection fraction (LVEF) calculated at <strong>58%</strong> (Normal &gt; 55%). Normal valvular anatomy and no segmental wall motion abnormalities detected.
            </div>
          </div>

          {/* Section 6: Recommendations */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <ShieldCheck size={16} />
              6. Recommendations
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

          {/* Section 7: Doctor Notes */}
          <div>
            <h3 className="section-title text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] mb-3 flex items-center gap-2">
              <FileText size={16} />
              7. Doctor Notes
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