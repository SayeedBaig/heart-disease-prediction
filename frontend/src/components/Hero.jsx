import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  HeartPulse,
  Brain,
  ShieldCheck,
  CheckCircle,
} from "lucide-react";

import heart from "../assets/heart.png";
import AnimatedBackground from "./AnimatedBackground";

function Hero() {
  return (
   <section className="hero-background relative min-h-[92vh] pt-24 pb-16 flex items-center overflow-hidden">
      <AnimatedBackground />
      
      <div className="relative z-10 max-w-7xl mx-auto px-8 grid lg:grid-cols-[1.15fr_0.85fr] gap-16 items-center">
        {/* LEFT SECTION */}

        <motion.div
          initial={{ opacity: 0, x: -60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
        >
          {/* Badge */}
          <div className="inline-flex items-center gap-2 bg-[var(--accent-melanzane-light)] text-[var(--accent-melanzane)] border border-[var(--accent-melanzane-border)] px-4 py-2 rounded-full font-semibold text-xs mb-6">
            <Activity size={16} />
            Multi-Modal Cardiovascular Intelligence
          </div>

          {/* Heading */}
          <h1 className="h1-large max-w-2xl">
            Early Detection of{" "}
            <span className="text-[var(--accent-melanzane)]">Heart Disease</span>{" "}
            Powered by AI
          </h1>

          {/* Description */}
          <p className="body-regular mt-6 max-w-xl text-base md:text-lg">
            Combine Clinical Data, ECG signals, and Echocardiography using advanced AI to deliver fast, accurate, and explainable cardiovascular risk predictions with digital twin insights.
          </p>

          {/* Buttons */}
          <div className="flex gap-4 mt-8 flex-wrap">
            <Link to="/get-started" className="btn-primary btn-lg">
              Get Started
            </Link>
          </div>

          <div className="flex flex-wrap gap-6 mt-8 text-xs font-semibold text-[var(--text-secondary)]">
            <div className="flex items-center gap-2">
              <ShieldCheck className="text-emerald-500" size={18} />
              Secure AI
            </div>
            <div className="flex items-center gap-2">
              <Activity className="text-[var(--accent-melanzane)]" size={18} />
              Real-time Analysis
            </div>
            <div className="flex items-center gap-2">
              <HeartPulse className="text-red-500" size={18} />
              Explainable AI
            </div>
          </div>

          {/* Feature List */}
          <div className="grid grid-cols-2 gap-4 mt-8 text-xs font-semibold text-[var(--text-primary)]">
            <div className="flex items-center gap-2.5">
              <CheckCircle className="text-emerald-500" size={18} />
              Clinical AI
            </div>
            <div className="flex items-center gap-2.5">
              <CheckCircle className="text-emerald-500" size={18} />
              ECG Waveform Analysis
            </div>
            <div className="flex items-center gap-2.5">
              <CheckCircle className="text-emerald-500" size={18} />
              Echo Motion Analysis
            </div>
            <div className="flex items-center gap-2.5">
              <CheckCircle className="text-emerald-500" size={18} />
              Digital Twin Simulation
            </div>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-3 gap-4 mt-10 max-w-xl">
            <div className="cardio-card p-4 text-center">
              <h2 className="text-2xl font-extrabold text-[var(--accent-melanzane)]">96%</h2>
              <p className="caption-small mt-1">Prediction Confidence</p>
            </div>
            <div className="cardio-card p-4 text-center">
              <h2 className="text-2xl font-extrabold text-[var(--accent-melanzane)]">3 AI</h2>
              <p className="caption-small mt-1">Clinical · ECG · Echo</p>
            </div>
            <div className="cardio-card p-4 text-center">
              <h2 className="text-2xl font-extrabold text-[var(--accent-melanzane)]">100%</h2>
              <p className="caption-small mt-1">Explainable AI Reports</p>
            </div>
          </div>

        </motion.div>

        {/* RIGHT SECTION */}

        <motion.div
  initial={{ opacity: 0, x: 60 }}
  animate={{ opacity: 1, x: 0 }}
  transition={{ duration: 1 }}
  className="relative flex justify-center items-center"
>

        {/* Blue Glow */}

<motion.div
  className="absolute
             w-[650px]
             h-[650px]
             rounded-full
             bg-gradient-to-r
             from-blue-300
             via-cyan-200
             to-blue-400
             blur-[140px]
             opacity-25"
  animate={{
    scale: [1, 1.08, 1],
    opacity: [0.18, 0.28, 0.18],
  }}
  transition={{
    duration: 6,
    repeat: Infinity,
    ease: "easeInOut",
  }}
/>

{/* White Center Glow */}

<motion.div
  className="absolute
             w-[420px]
             h-[420px]
             rounded-full
             bg-white
             blur-[120px]"
  animate={{
    opacity: [0.55, 0.75, 0.55],
  }}
  transition={{
    duration: 5,
    repeat: Infinity,
    ease: "easeInOut",
  }}
/>
<div
  className="absolute
             bottom-12
             w-[260px]
             h-[70px]
             bg-black/20
             blur-3xl
             rounded-full"
/>

          {/* Heart */}

          <motion.img
            src={heart}
            alt="Heart"
            className="relative w-[420px] md:w-[500px] xl:w-[580px] z-10"
            animate={{
  y: [0, -6, 0],
  scale: [1, 1.03, 1, 1.05, 1],
  rotate: [0, 1, 0, -1, 0],
}}

transition={{
  duration: 2.2,
  repeat: Infinity,
  ease: "easeInOut",
}}
          />

          {/* ECG Card */}

          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ repeat: Infinity, duration: 3 }}
            className="absolute top-8 left-2 bg-white/80 backdrop-blur-xl border border-white/70 rounded-3xl shadow-2xl px-5 py-4"
          >

            <div className="flex items-center gap-3">

              <HeartPulse className="text-red-500" />

              <div>

                <p className="text-gray-500 text-sm">
                  ECG Analysis
                </p>

                <h3 className="font-bold text-green-600">
                  Completed
                </h3>

              </div>

            </div>

          </motion.div>

          {/* Risk Card */}

          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ repeat: Infinity, duration: 4 }}
            className="absolute bottom-24 left-4 bg-white/80 backdrop-blur-xl border border-white/70 rounded-3xl shadow-2xl px-5 py-4"
          >

            <div className="flex items-center gap-3">

              <ShieldCheck className="text-green-600" />

              <div>

                <p className="text-gray-500 text-sm">
                  Risk Prediction
                </p>

                <h3 className="font-bold text-green-600">
                  Low Risk
                </h3>

                <p className="text-blue-600 text-sm">
                  31.4%
                </p>

              </div>

            </div>

          </motion.div>

          {/* AI Confidence */}

          <motion.div
            animate={{ y: [0, -10, 0] }}
            transition={{ repeat: Infinity, duration: 5 }}
            className="absolute bottom-8 right-2 bg-white/80 backdrop-blur-xl border border-white/70 rounded-3xl shadow-2xl px-5 py-4"
          >

            <div className="flex items-center gap-3">

              <Brain className="text-blue-600" />

              <div>

                <p className="text-gray-500 text-sm">
                  AI Confidence
                </p>

                <h3 className="font-bold text-blue-600">
                  96%
                </h3>

              </div>

            </div>

          </motion.div>

        </motion.div>

      </div>
    </section>
  );
}

export default Hero;
