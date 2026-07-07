import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

function FeatureCard({ icon: Icon, title, description }) {
  return (
    <motion.div
      whileHover={{
        y: -10,
        scale: 1.02,
      }}
      transition={{ duration: 0.25 }}
      className="group bg-white rounded-3xl border border-gray-100 shadow-lg hover:shadow-2xl p-8 transition-all duration-300"
    >
      {/* Icon */}

      <div className="w-16 h-16 rounded-2xl bg-blue-100 flex items-center justify-center mb-6 group-hover:bg-blue-600 transition">

        <Icon
          size={32}
          className="text-blue-600 group-hover:text-white transition"
        />

      </div>

      {/* Title */}

      <h3 className="text-2xl font-bold text-slate-900 mb-4">
        {title}
      </h3>

      {/* Description */}

      <p className="text-gray-600 leading-7 mb-6">
        {description}
      </p>

      {/* Link */}

      <div className="flex items-center gap-2 text-blue-600 font-semibold cursor-pointer">

        Learn More

        <ArrowRight
          size={18}
          className="group-hover:translate-x-2 transition"
        />

      </div>
    </motion.div>
  );
}

export default FeatureCard;