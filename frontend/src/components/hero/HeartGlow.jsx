/**
 * HeartGlow — REFERENCE MATCH
 * Dramatic blue/cyan neon glow behind the heart for dark mode.
 * Multi-layer: outer ambient + inner pulse + bottom platform glow.
 */
import { motion } from "framer-motion";

export default function HeartGlow({ mouseX = 0, mouseY = 0 }) {
  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
      {/* Large outer ambient glow — deep blue */}
      <motion.div
        animate={{ x: mouseX * 16, y: mouseY * 10, scale: [1, 1.06, 1] }}
        transition={{
          x: { type: "spring", stiffness: 50, damping: 18 },
          y: { type: "spring", stiffness: 50, damping: 18 },
          scale: { duration: 4, repeat: Infinity, ease: "easeInOut" },
        }}
        style={{
          width: 460, height: 460,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(37,99,235,0.38) 0%, rgba(59,130,246,0.18) 35%, rgba(0,100,255,0.06) 65%, transparent 80%)",
          filter: "blur(32px)",
        }}
      />
      {/* Middle cyan glow — heartbeat sync */}
      <motion.div
        style={{ position: "absolute", width: 280, height: 280, borderRadius: "50%",
          background: "radial-gradient(circle, rgba(0,183,255,0.22) 0%, rgba(59,130,246,0.10) 50%, transparent 75%)",
          filter: "blur(20px)",
        }}
        animate={{ scale: [1, 1.12, 1], opacity: [0.5, 0.85, 0.5], x: mouseX * 8, y: mouseY * 6 }}
        transition={{
          scale:   { duration: 1.05, repeat: Infinity, ease: [0.4, 0, 0.6, 1], repeatDelay: 0.72 },
          opacity: { duration: 1.05, repeat: Infinity, ease: [0.4, 0, 0.6, 1], repeatDelay: 0.72 },
          x: { type: "spring", stiffness: 50, damping: 18 },
          y: { type: "spring", stiffness: 50, damping: 18 },
        }}
      />
      {/* Top specular highlight — lighter blue from above */}
      <div style={{
        position: "absolute", width: 200, height: 120, top: "10%",
        borderRadius: "50%",
        background: "radial-gradient(ellipse, rgba(99,163,255,0.15) 0%, transparent 70%)",
        filter: "blur(16px)",
      }}/>
    </div>
  );
}
