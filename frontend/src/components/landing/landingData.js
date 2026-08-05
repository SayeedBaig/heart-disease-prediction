/**
 * Landing page content data.
 * Separated from presentation for maintainability.
 */
import {
  Activity,
  Brain,
  Calendar,
  FileText,
  HeartPulse,
  Layers,
  ShieldCheck,
  Stethoscope,
  UserCheck,
} from "lucide-react";

export const stats = [
  {
    value: "96%",
    label: "Prediction Confidence",
    desc: "Validated across clinical dataset trials",
  },
  {
    value: "3-in-1",
    label: "Multimodal Fusion",
    desc: "Combines Vitals, ECG & Echo scans",
  },
  {
    value: "14",
    label: "Digital Twin Vitals",
    desc: "Real-time physiological risk simulation",
  },
  {
    value: "Real-time",
    label: "Explainable AI",
    desc: "Instant print-ready specialist reports",
  },
];

export const capabilities = [
  {
    icon: Stethoscope,
    title: "Clinical Assessment",
    description:
      "Evaluates core vitals, blood pressure, lipid panels, fasting glucose, and lifestyle risk factors for accurate diagnostic stratification.",
  },
  {
    icon: Activity,
    title: "ECG Waveform Analysis",
    description:
      "Deep learning neural networks classify voltage waveforms to detect arrhythmias, ischemic changes, and rhythmic anomalies in seconds.",
  },
  {
    icon: HeartPulse,
    title: "Echocardiogram Imaging",
    description:
      "Analyzes left ventricular ejection fraction and myocardial wall motion metrics through computer vision feature extraction.",
  },
  {
    icon: Layers,
    title: "Digital Twin Simulation",
    description:
      "Constructs an interactive physiological model allowing clinicians and patients to simulate treatment interventions live.",
  },
  {
    icon: FileText,
    title: "Standardized AI Reports",
    description:
      "Generates structured, print-ready diagnostic summaries complete with confidence scores, risk vectors, and doctor notes.",
  },
  {
    icon: Calendar,
    title: "Specialist Consultation",
    description:
      "Seamlessly share digital screening dossiers and schedule follow-up appointments with board-certified cardiologists.",
  },
];

export const reasons = [
  {
    icon: Brain,
    title: "Multimodal AI Architecture",
    description:
      "Engineered to synthesize tabular clinical vitals, continuous ECG signal arrays, and echocardiography metrics simultaneously.",
  },
  {
    icon: ShieldCheck,
    title: "Clinical Decision Support",
    description:
      "Empowers healthcare providers with explainable AI outputs, confidence intervals, and actionable risk stratification.",
  },
  {
    icon: UserCheck,
    title: "Personalized Patient Care",
    description:
      "Models individual patient trajectories to deliver tailored lifestyle recommendations, treatment targets, and proactive care plans.",
  },
];

export const faqs = [
  {
    question: "What diagnostic data does CardioAI evaluate?",
    answer:
      "CardioAI processes three core diagnostic streams: clinical vitals (blood pressure, cholesterol, glucose, BMI), 12-lead ECG waveforms, and echocardiogram ejection metrics to deliver a comprehensive multimodal assessment.",
  },
  {
    question: "Is CardioAI intended to replace a medical doctor?",
    answer:
      "No. CardioAI is an advanced Clinical Decision Support System (CDSS) built to assist healthcare professionals with screening, risk stratification, and patient monitoring. Final clinical decisions must be made by qualified physicians.",
  },
  {
    question: "How does the Digital Twin simulation work?",
    answer:
      "The Digital Twin models an individual patient's baseline physiology. Users can dynamically adjust variables such as blood pressure, BMI, physical activity, and medication adherence to forecast 10-year risk trajectories live.",
  },
  {
    question: "Can generated AI reports be exported or shared?",
    answer:
      "Yes. The platform generates standardized, print-ready PDF clinical summaries that can be attached to Electronic Health Records (EHR) or shared during specialist consultations.",
  },
];

export const heroStreams = [
  {
    title: "Clinical Vitals",
    label: "Blood pressure, lipid panels & biomarkers evaluated",
  },
  {
    title: "ECG Waveform Analysis",
    label: "Deep learning classification complete",
  },
  {
    title: "Echo Motion Scan",
    label: "Ejection fraction & wall motion reviewed",
  },
];

export const twinMetrics = [
  { label: "Systolic Blood Pressure", value: "132 mmHg", percent: "62%" },
  { label: "Body Mass Index (BMI)", value: "24.5", percent: "44%" },
  { label: "Physical Exercise Frequency", value: "4 days/week", percent: "70%" },
  { label: "Medication Adherence Rate", value: "95%", percent: "88%" },
];

export const reportMetrics = [
  { label: "Risk Profile", value: "Moderate" },
  { label: "Risk Score", value: "24%" },
  { label: "AI Confidence", value: "91%" },
];