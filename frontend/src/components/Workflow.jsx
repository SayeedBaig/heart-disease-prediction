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
    title: "1. Patient Intake",
    description: "Capture core demographic vitals and medical background.",
  },
  {
    icon: ClipboardList,
    title: "2. Clinical Vitals",
    description: "Input blood pressure, cholesterol, and fasting glucose metrics.",
  },
  {
    icon: Activity,
    title: "3. ECG Signal",
    description: "Upload ECG waveforms for automated deep learning wave analysis.",
  },
  {
    icon: HeartPulse,
    title: "4. Echo Imaging",
    description: "Incorporate echocardiogram ejection signals and wall motion scans.",
  },
  {
    icon: BrainCircuit,
    title: "5. Multimodal Fusion",
    description: "AI engines fuse all available diagnostic streams in seconds.",
  },
  {
    icon: FileText,
    title: "6. Clinical Report",
    description: "Generate print-ready diagnostic summaries & consultation charts.",
  },
];

function Workflow() {
  return (
    <section className="p-8 md:p-12 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-[28px]">
      <div className="cardio-container max-w-7xl mx-auto">
        {/* Heading */}
        <div className="text-center max-w-3xl mx-auto mb-16 md:mb-20">
          <span className="caption-small uppercase tracking-widest text-[var(--accent-primary)] font-bold">
            End-to-End Clinical Flow
          </span>
          <h2 className="h2-semibold text-[var(--text-primary)] mt-2 text-2xl md:text-3xl lg:text-4xl font-extrabold tracking-tight">
            From Patient Intake to Clinical Report
          </h2>
          <p className="body-regular mt-3 text-sm md:text-base text-[var(--text-secondary)] leading-relaxed">
            A practical, streamlined workflow for evaluating cardiovascular risk factors and generating explainable insights.
          </p>
        </div>

        {/* Workflow Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {steps.map((step, index) => {
            const Icon = step.icon;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.08 }}
                className="cardio-card p-6 md:p-7 flex flex-col items-start rounded-[28px] group hover:border-[var(--accent-primary-border)] transition-all duration-300 shadow-sm hover:shadow-md"
              >
                <div className="flex items-center justify-between w-full mb-5">
                  <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center transition-transform group-hover:scale-105">
                    <Icon size={24} />
                  </div>
                  <span className="w-7 h-7 rounded-full bg-[var(--accent-primary-light)] text-[var(--accent-primary)] font-extrabold text-xs flex items-center justify-center border border-[var(--accent-primary-border)]">
                    {index + 1}
                  </span>
                </div>

                <h3 className="text-base md:text-lg font-bold text-[var(--text-primary)] mb-2 group-hover:text-[var(--accent-primary)] transition-colors">
                  {step.title}
                </h3>

                <p className="body-regular text-xs md:text-sm leading-relaxed text-[var(--text-secondary)]">
                  {step.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

export default Workflow;
