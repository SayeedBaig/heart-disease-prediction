import { motion } from "framer-motion";

const particles = [
  { size: 8, left: "8%", top: "18%", duration: 14 },
  { size: 12, left: "18%", top: "72%", duration: 18 },
  { size: 6, left: "30%", top: "40%", duration: 12 },
  { size: 10, left: "42%", top: "82%", duration: 16 },
  { size: 14, left: "56%", top: "20%", duration: 15 },
  { size: 8, left: "68%", top: "55%", duration: 17 },
  { size: 12, left: "82%", top: "30%", duration: 13 },
  { size: 7, left: "92%", top: "78%", duration: 19 },
  { size: 10, left: "24%", top: "12%", duration: 16 },
  { size: 9, left: "76%", top: "12%", duration: 14 },
];

function AnimatedBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">

      {/* Blue Orb */}
      <motion.div
        className="absolute -top-32 -left-24 w-[650px] h-[650px] rounded-full bg-blue-400/20 blur-[170px]"
        animate={{
          x: [0, 120, 0],
          y: [0, 60, 0],
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Cyan Orb */}
      <motion.div
        className="absolute top-1/4 right-0 w-[550px] h-[550px] rounded-full bg-cyan-300/20 blur-[170px]"
        animate={{
          x: [0, -80, 0],
          y: [0, 80, 0],
        }}
        transition={{
          duration: 16,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Purple Orb */}
      <motion.div
        className="absolute bottom-0 left-1/3 w-[600px] h-[600px] rounded-full bg-purple-300/15 blur-[180px]"
        animate={{
          x: [0, 100, 0],
          y: [0, -70, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Floating Particles */}
      {particles.map((particle, index) => (
        <motion.div
          key={index}
          className="absolute rounded-full bg-white/70 shadow-[0_0_12px_rgba(59,130,246,0.5)]"
          style={{
            width: particle.size,
            height: particle.size,
            left: particle.left,
            top: particle.top,
          }}
          animate={{
            y: [0, -40, 0],
            x: [0, 12, -8, 0],
            scale: [1, 1.4, 1],
            opacity: [0.15, 0.7, 0.15],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}

    </div>
  );
}

export default AnimatedBackground;