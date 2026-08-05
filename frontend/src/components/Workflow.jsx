import { motion } from "framer-motion";
import {
  UserPlus,
  Activity,
  HeartPulse,
  ClipboardList,
  BrainCircuit,
  FileText,
} from "lucide-react";

const steps = [
  {
    icon: UserPlus,
    title: "Patient Intake",
    description: "Capture core demographic vitals, family history, and primary clinical background.",
  },
  {
    icon: ClipboardList,
    title: "Clinical Vitals",
    description: "Input blood pressure, lipid panel, fasting glucose, and BMI biomarker metrics.",
  },
  {
    icon: Activity,
    title: "ECG Waveforms",
    description: "Upload continuous 12-lead ECG signals for automated neural network wave analysis.",
  },
  {
    icon: HeartPulse,
    title: "Echo Imaging",
    description: "Incorporate echocardiogram ejection fraction and myocardial wall motion scans.",
  },
  {
    icon: BrainCircuit,
    title: "Multimodal Fusion",
    description: "AI engines fuse tabular vitals, wave signals, and imaging metrics into a single unified risk score.",
  },
  {
    icon: FileText,
    title: "Clinical Report",
    description: "Generate structured, print-ready diagnostic summaries with specialist recommendations.",
  },
];

function Workflow() {
  return (
    <section className="py-20 md:py-28 bg-[var(--bg-secondary)] border-b border-[var(--border-color)]">
      <div className="cardio-container">
        {/* Heading */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <span className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-[var(--accent-melanzane-light)] border border-[var(--accent-melanzane-border)] text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] shadow-sm">
            End-to-End Clinical Flow
          </span>
          <h2 className="mt-4 text-3xl font-extrabold tracking-tight text-[var(--text-primary)] sm:text-4xl md:text-5xl">
            From Patient Intake to Clinical Report
          </h2>
          <p className="mt-4 text-base leading-relaxed text-[var(--text-secondary)] sm:text-lg">
            A practical, streamlined workflow for evaluating cardiovascular risk factors and generating explainable diagnostic insights.
          </p>
        </div>

        {/* Workflow Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 items-stretch">
          {steps.map((step, index) => {
            const Icon = step.icon;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.06 }}
                className="group rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] p-8 shadow-sm hover:shadow-xl hover:border-[var(--accent-melanzane-border)] transition-all duration-300 flex flex-col justify-between h-full"
              >
                <div>
                  <div className="flex items-center justify-between w-full mb-6">
                    <div className="w-14 h-14 rounded-2xl bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] border border-[var(--accent-melanzane-border)] flex items-center justify-center transition-transform group-hover:scale-110 group-hover:bg-[var(--accent-melanzane)] group-hover:text-white">
                      <Icon size={26} />
                    </div>
                    <span className="px-2.5 py-1 rounded-full bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] font-extrabold text-xs flex items-center justify-center border border-[var(--accent-melanzane-border)] shadow-sm">
                      0{index + 1}
                    </span>
                  </div>

                  <h3 className="text-xl font-extrabold text-[var(--text-primary)] mb-3 group-hover:text-[var(--accent-melanzane)] transition-colors leading-snug">
                    {step.title}
                  </h3>

                  <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
                    {step.description}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

export default Workflow;
