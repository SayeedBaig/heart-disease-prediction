import { Link } from "react-router-dom";
import { motion } from "framer-motion";

function Hero() {
  return (
    <section className="min-h-[85vh] flex flex-col justify-center items-center text-center bg-gradient-to-b from-blue-950 to-blue-800 text-white px-6">

      <motion.h1
        initial={{ opacity: 0, y: -40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-6xl font-bold mb-6"
      >
        ❤️ CardioAI
      </motion.h1>

      <motion.h2
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="text-3xl font-semibold mb-4"
      >
        AI Powered Heart Disease Prediction
      </motion.h2>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="text-xl max-w-3xl mb-10"
      >
        Analyze ECG, Echocardiography and Clinical Data using
        Artificial Intelligence to predict cardiovascular risk.
      </motion.p>

      <Link
        to="/diagnose"
        className="bg-red-500 hover:bg-red-600 px-8 py-4 rounded-xl text-xl font-semibold transition"
      >
        Start Diagnosis
      </Link>

    </section>
  );
}

export default Hero;