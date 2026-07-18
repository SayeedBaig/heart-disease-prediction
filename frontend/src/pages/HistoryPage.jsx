import { useEffect, useState } from "react";

import Navbar from "../components/Navbar";
import PageBackground from "../components/PageBackground";
import api from "../services/api";
import { motion } from "framer-motion";
import {
  Activity,
  HeartPulse,
  Heart,
  CalendarDays,
  BadgeCheck,
} from "lucide-react";

function HistoryPage() {
  const patientId = localStorage.getItem("selected_patient_id");

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!patientId) {
        setLoading(false);
        return;
      }

      try {
        const response = await api.get(
  `/history/patient/${patientId}`
);  

       setHistory(response.data || []);
      } catch (err) {
        console.error("Failed to load history:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [patientId]);

  return (
    <>
      <Navbar />

      <div className="relative min-h-screen overflow-hidden p-10">
        <PageBackground />

        <div className="relative z-10 max-w-4xl mx-auto">

          {/* Heading */}

          <div className="mb-10">
            <h1 className="text-5xl font-bold text-slate-900">
              Prediction History
            </h1>

            <p className="text-lg text-gray-500 mt-3">
              Review all previous AI-generated cardiovascular predictions.
            </p>
          </div>

          {/* Loading */}

          {loading ? (
            <div className="bg-white rounded-3xl shadow-xl p-12 text-center">

              <div className="animate-spin rounded-full h-12 w-12 border-b-4 border-blue-600 mx-auto"></div>

              <p className="mt-5 text-gray-600 text-lg">
                Loading prediction history...
              </p>

            </div>

          ) : history.length === 0 ? (

            /* Empty State */

            <div className="bg-white rounded-3xl shadow-xl p-16 text-center">

              <div className="text-7xl">
                📈
              </div>

              <h2 className="text-3xl font-bold mt-6">
                No Predictions Yet
              </h2>

              <p className="text-gray-500 mt-4 text-lg">
                Your AI prediction history will appear here after completing
                your first diagnosis.
              </p>

            </div>

          ) : (

            <div className="space-y-8">

              {history.map((item) => {

                const risk = item.risk_level?.toLowerCase();

                return (

                 <motion.div
  key={item.prediction_id}
  initial={{ opacity: 0, y: 25 }}
  animate={{ opacity: 1, y: 0 }}
  whileHover={{ y: -6 }}
  transition={{ duration: 0.4 }}
  className="
    bg-white/90
    backdrop-blur-md
    rounded-3xl
    shadow-[0_15px_40px_rgba(37,99,235,0.12)]
    hover:shadow-[0_20px_50px_rgba(37,99,235,0.18)]
    transition-all
    duration-300
    border
    border-slate-100
    p-8
  "
>
  {/* Header */}

  <div className="flex justify-between items-start">

    <div>

      <span
        className={`
          inline-flex
          items-center
          px-5
          py-2.5
          rounded-full
          text-base
          font-bold
          ${
           risk === "low"
? "bg-green-100 text-green-700 shadow-md shadow-green-200"
              : risk === "medium"
? "bg-yellow-100 text-yellow-700 shadow-md shadow-yellow-200"
              : "bg-red-100 text-red-700 shadow-md shadow-red-200"
          }
        `}
      >
        {risk === "low"
          ? "🟢 LOW RISK"
          : risk === "medium"
          ? "🟡 MEDIUM RISK"
          : "🔴 HIGH RISK"}
      </span>

      <h2 className="text-6xl font-extrabold text-slate-900 mt-6">
        {Number(item.risk_percentage).toFixed(1)}%
      </h2>

      <p className="text-gray-500 mt-2">
        Overall Cardiovascular Risk
      </p>

      {/* Progress Bar */}

      <div className="mt-6 w-72 h-3 bg-slate-200 rounded-full overflow-hidden">

        <motion.div
  initial={{ width: 0 }}
  animate={{
    width: `${Math.max(Number(item.risk_percentage), 30)}%`,
  }}
  transition={{
    duration: 1,
    ease: "easeOut",
  }}
  className={`
    h-full rounded-full
    ${
      risk === "low"
        ? "bg-green-500"
        : risk === "medium"
        ? "bg-yellow-500"
        : "bg-red-500"
    }
  `}
/>

      </div>

    </div>

    <div className="text-right">

      <p className="text-xs uppercase tracking-wider text-gray-400">
  Prediction ID
</p>

<h3 className="text-3xl font-bold text-slate-800">
  #{item.prediction_id ?? item.id}
</h3>

      <div className="flex justify-end items-center gap-2 mt-4 text-gray-500">

        <CalendarDays size={16} />

        <span>
          {new Date(item.created_at).toLocaleDateString()}
        </span>

      </div>

    </div>

  </div>

  <div className="border-t border-slate-200 my-8"></div>

  {/* AI Modules */}

  <div className="grid md:grid-cols-3 gap-5">

    <motion.div
      whileHover={{ scale: 1.05 }}
      className="
bg-blue-50
rounded-2xl
p-6
transition-all
duration-300
hover:shadow-lg
"
    >

      <Activity className="text-blue-600 mb-4" size={30} />

      <p className="text-gray-500 text-sm">
        Clinical AI
      </p>

      <h3 className="text-blue-700 text-3xl font-bold mt-3">
        {item.clinical_level}
      </h3>

    </motion.div>

    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-red-50 rounded-2xl p-6"
    >

      <HeartPulse className="text-red-500 mb-4" size={30} />

      <p className="text-gray-500 text-sm">
        ECG Analysis
      </p>

      <h3 className="text-red-600 text-3xl font-bold mt-3">
        {item.ecg_level}
      </h3>

    </motion.div>

    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-green-50 rounded-2xl p-6"
    >

      <Heart className="text-green-600 mb-4" size={30} />

      <p className="text-gray-500 text-sm">
        Echo Analysis
      </p>

      <h3 className="text-green-600 text-3xl font-bold mt-3">
        {item.echo_level}
      </h3>

    </motion.div>

  </div>

  {/* Footer */}

  <div className="flex justify-between items-center mt-10">

    <div>

      <p className="text-sm text-gray-400">
        Generated on
      </p>

      <p className="font-medium text-gray-600">
        {new Date(item.created_at).toLocaleString()}
      </p>

    </div>

    <div className="flex items-center gap-2 bg-blue-100 text-blue-700 px-5 py-3 rounded-full font-semibold">

      <BadgeCheck size={18} />

      AI Verified Prediction

    </div>

  </div>

</motion.div>

                );
              })}

            </div>

          )}

        </div>
      </div>
    </>
  );
}

export default HistoryPage;