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

          <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full font-medium mb-6">
            <Activity size={18} />
            AI Powered Healthcare
          </div>

          {/* Heading */}

          <h1 className="text-5xl md:text-6xl xl:text-7xl
tracking-tight font-extrabold leading-snug text-slate-900 max-w-2xl">
            Early Detection of
            <br />
            <span className="text-blue-600">
              Heart Disease
            </span>
            <br />
            <span className="text-slate-900">
              Powered by AI
            </span>
          </h1>

          {/* Description */}

          <p className="mt-6 text-lg text-slate-600 leading-8 max-w-xl">
           Combine Clinical Data, ECG signals, and Echocardiography using advanced AI to deliver fast, accurate, and explainable cardiovascular risk predictions with digital twin insights.
          </p>

          {/* Buttons */}

          <div className="flex gap-5 mt-10 flex-wrap">

            <div className="flex gap-5 mt-10 flex-wrap">

              <Link
                to="/get-started"
                className="bg-blue-600 hover:bg-blue-700 text-white px-7 py-4 rounded-xl font-semibold shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl"
              >
                Get Started
              </Link>

            </div>
          </div>
          <div className="flex flex-wrap gap-6 mt-8 text-sm text-slate-600">

  <div className="flex items-center gap-2">
    <ShieldCheck className="text-green-600" size={18}/>
    Secure AI
  </div>

  <div className="flex items-center gap-2">
    <Activity className="text-blue-600" size={18}/>
    Real-time Analysis
  </div>

  <div className="flex items-center gap-2">
    <HeartPulse className="text-red-500" size={18}/>
    Explainable AI
  </div>

</div>

          {/* Feature List */}

          <div className="grid grid-cols-2 gap-5 mt-10">

            <div className="flex items-center gap-3">
              <CheckCircle className="text-green-600" />
              Clinical AI
            </div>

            <div className="flex items-center gap-3">
              <CheckCircle className="text-green-600" />
              ECG Analysis
            </div>

            <div className="flex items-center gap-3">
              <CheckCircle className="text-green-600" />
              Echo Analysis
            </div>

            <div className="flex items-center gap-3">
              <CheckCircle className="text-green-600" />
              Digital Twin
            </div>

          </div>

          {/* Statistics */}

<div className="grid grid-cols-3 gap-4 mt-12 max-w-xl">
  <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-5 hover:shadow-xl transition-all duration-300">
    <h2 className="text-3xl font-bold text-blue-600">96%</h2>
    <p className="text-slate-500 mt-2">
      Prediction Confidence
    </p>
  </div>

  <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-5 hover:shadow-xl transition-all duration-300">
    <h2 className="text-3xl font-bold text-blue-600">
      3 AI
    </h2>
    <p className="text-slate-500 mt-2">
      Clinical • ECG • Echo
    </p>
  </div>

  <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-3 hover:shadow-xl transition-all duration-300">
    <h2 className="text-3xl font-bold text-blue-600">
      Explainable
    </h2>
    <p className="text-slate-500 mt-2">
      AI Powered Reports
    </p>
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
