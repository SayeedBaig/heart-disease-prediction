import { motion } from "framer-motion";

function FeatureCard({ icon, title, description }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="bg-white rounded-2xl shadow-xl p-8 text-center"
    >
      <div className="text-5xl mb-4">{icon}</div>

      <h2 className="text-2xl font-bold text-blue-900 mb-3">
        {title}
      </h2>

      <p className="text-gray-600">
        {description}
      </p>
    </motion.div>
  );
}

export default FeatureCard;