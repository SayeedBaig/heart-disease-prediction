import { motion } from "framer-motion";

function PageBackground() {
  return (
    <>
      {/* Base Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-100 via-blue-50 to-cyan-50" />

      {/* Blue Orb */}
      <motion.div
        className="absolute -top-24 -left-24 w-[500px] h-[500px] rounded-full bg-blue-300/20 blur-[120px]"
        animate={{
          x: [0, 60, 0],
          y: [0, 40, 0],
        }}
        transition={{
          duration: 16,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Cyan Orb */}
      <motion.div
        className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-cyan-300/20 blur-[140px]"
        animate={{
          x: [0, -70, 0],
          y: [0, -50, 0],
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Purple Orb */}
      <motion.div
        className="absolute top-1/2 left-1/3 w-[400px] h-[400px] rounded-full bg-indigo-300/15 blur-[130px]"
        animate={{
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* ECG Watermark */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.04]"
        preserveAspectRatio="none"
      >
        <path
          d="
            M0 350
            L250 350
            L270 330
            L285 390
            L300 260
            L320 350
            L340 350
            L360 320
            L380 350
            L2000 350
          "
          stroke="#2563eb"
          strokeWidth="3"
          fill="none"
        />
      </svg>
    </>
  );
}

export default PageBackground;