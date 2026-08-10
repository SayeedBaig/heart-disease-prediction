import { motion } from "framer-motion";

export default function EnergyField({ isDark }) {
  const stroke = isDark
    ? "rgba(59,130,246,0.22)"
    : "rgba(59,130,246,0.12)";

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        zIndex: 2,
      }}
    >
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 700 560"
        style={{ overflow: "visible" }}
      >
        <defs>
          <filter id="energyBlur">
            <feGaussianBlur stdDeviation="4" />
          </filter>

          <radialGradient id="energyGlow">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.28" />
            <stop offset="70%" stopColor="#3B82F6" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Background Glow */}
        <ellipse
          cx="510"
          cy="285"
          rx="180"
          ry="150"
          fill="url(#energyGlow)"
        />

        {/* Energy Arc 1 */}
        <motion.path
          d="M330 300 C390 180 630 180 690 300"
          stroke={stroke}
          strokeWidth="2"
          fill="none"
          filter="url(#energyBlur)"
          animate={{ opacity: [0.15, 0.45, 0.15] }}
          transition={{ duration: 4, repeat: Infinity }}
        />

        {/* Energy Arc 2 */}
        <motion.path
          d="M340 330 C410 220 610 220 680 330"
          stroke={stroke}
          strokeWidth="2"
          fill="none"
          filter="url(#energyBlur)"
          animate={{ opacity: [0.1, 0.4, 0.1] }}
          transition={{
            duration: 5,
            repeat: Infinity,
            delay: 0.5,
          }}
        />

        {/* Energy Arc 3 */}
        <motion.path
          d="M350 360 C420 260 600 260 670 360"
          stroke={stroke}
          strokeWidth="2"
          fill="none"
          filter="url(#energyBlur)"
          animate={{ opacity: [0.12, 0.35, 0.12] }}
          transition={{
            duration: 6,
            repeat: Infinity,
            delay: 1,
          }}
        />
      </svg>
    </div>
  );
}