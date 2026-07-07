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

function Hero() {
  return (
    <section className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-white flex items-center overflow-hidden">
      <div className="max-w-7xl mx-auto px-8 grid lg:grid-cols-2 gap-20 items-center">

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

          <h1 className="text-5xl lg:text-6xl font-extrabold leading-snug text-slate-900 max-w-2xl">
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
            Analyze ECG, Echocardiography and Clinical Data using advanced
            Artificial Intelligence to estimate cardiovascular risk with
            explainable predictions and digital twin simulations.
          </p>

          {/* Buttons */}

          <div className="flex gap-5 mt-10 flex-wrap">

            <Link
              to="/diagnose"
              className="bg-blue-600 hover:bg-blue-700 text-white px-7 py-4 rounded-xl font-semibold shadow-lg transition-all duration-300 hover:-translate-y-1 hover:shadow-xl"
            >
              Start Diagnosis
            </Link>

            <button className="border border-slate-300 hover:border-blue-600 hover:text-blue-600 px-7 py-4 rounded-xl font-semibold transition-all duration-300">
              Explore Features →
            </button>

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

          <div className="grid grid-cols-3 gap-8 mt-14">

            <div>
              <h2 className="text-4xl font-bold text-blue-600">
                96%
              </h2>
              <p className="text-gray-500 mt-1">
                Prediction Confidence
              </p>
            </div>

            <div>
              <h2 className="text-4xl font-bold text-blue-600">
                3 AI
              </h2>
              <p className="text-gray-500 mt-1">
                Clinical • ECG • Echo
              </p>
            </div>

            <div>
              <h2 className="text-4xl font-bold text-blue-600">
                Explainable
              </h2>
              <p className="text-gray-500 mt-1">
                AI Reports
              </p>
            </div>

          </div>

        </motion.div>

        {/* RIGHT SECTION */}

        <motion.div
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1 }}
          className="relative flex justify-center"
        >

          {/* Glow */}

          <div className="absolute w-[620px] h-[620px] rounded-full bg-blue-100 blur-[120px] opacity-50"></div>

          {/* Heart */}

          <motion.img
            src={heart}
            alt="Heart"
            className="relative w-[450px] lg:w-[560px] z-10"
            animate={{
              y: [0, -10, 0],
              scale: [1, 1.02, 1],
            }}
            transition={{
              repeat: Infinity,
              duration: 4,
            }}
          />

          {/* ECG Card */}

          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ repeat: Infinity, duration: 3 }}
            className="absolute top-8 left-2 bg-white rounded-2xl shadow-xl p-5"
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
            className="absolute bottom-24 left-4 bg-white rounded-2xl shadow-xl p-5"
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
            className="absolute bottom-8 right-2 bg-white rounded-2xl shadow-xl p-5"
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