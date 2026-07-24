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
    title: "Register patient",
    description: "Create a patient record and capture the information needed for assessment.",
  },
  {
    icon: Activity,
    title: "Add ECG recording",
    description: "Upload an ECG image or signal file for waveform analysis.",
  },
  {
    icon: HeartPulse,
    title: "Add echocardiography",
    description: "Include an echo study when imaging is available.",
  },
  {
    icon: ClipboardList,
    title: "Enter clinical data",
    description: "Record vital signs, laboratory indicators, and lifestyle factors.",
  },
  {
    icon: BrainCircuit,
    title: "Generate assessment",
    description: "Combine available inputs into a consolidated cardiovascular risk result.",
  },
  {
    icon: FileText,
    title: "Review and report",
    description: "Examine findings, save assessment history, and generate reports.",
  },
];

function Workflow() {
  return (
    <section className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">

        {/* Heading */}

        <div className="text-center mb-20">

          <h2 className="text-4xl font-bold text-slate-900">
            From patient record to clinical report
          </h2>

          <p className="mt-4 text-gray-600 max-w-3xl mx-auto leading-7">
            A practical workflow for documenting cardiovascular assessments and
            reviewing results in one place.
          </p>

        </div>

        <div className="relative">

          {/* Connector Line */}

          <div className="hidden lg:block absolute top-12 left-0 w-full h-0.5 bg-blue-100 rounded-full"></div>

          <div className="grid lg:grid-cols-6 md:grid-cols-3 sm:grid-cols-2 gap-8 relative">

            {steps.map((step, index) => {

              const Icon = step.icon;

              return (

                <motion.div
                  key={index}
                  transition={{
                    duration: 0.25,
                  }}
                  className="relative bg-white rounded-2xl shadow-lg border border-gray-100 p-5 text-center"
                >

                  {/* Step Number */}

                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-sm shadow-md">

                    {String(index + 1).padStart(2, "0")}

                  </div>

                  {/* Icon */}

                  <div className="w-16 h-16 mx-auto mt-4 rounded-xl bg-blue-50 flex items-center justify-center">

                    <Icon
                      size={32}
                      className="text-blue-600"
                    />

                  </div>

                  {/* Title */}

                  <h3 className="mt-5 text-lg font-bold text-slate-900">

                    {step.title}

                  </h3>

                  {/* Description */}

                  <p className="mt-3 text-sm text-gray-500 leading-6">

                    {step.description}

                  </p>

                </motion.div>

              );

            })}

          </div>

        </div>

      </div>
    </section>
  );
}

export default Workflow;
