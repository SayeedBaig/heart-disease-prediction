/**
 * AIRing — REFERENCE MATCH
 * Bright neon cyan/blue concentric elliptical rings below the heart.
 * High visibility on dark background. Pulsing glow animation.
 */
import { motion } from "framer-motion";

export default function AIRing() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        bottom: -10,
        left: "50%",
        transform: "translateX(-50%)",
        width: 380,
        height: 100,
        pointerEvents: "none",
      }}
    >
      {/* Bright neon glow base */}
      <div style={{
        position: "absolute",
        left: "50%", top: "55%",
        transform: "translate(-50%,-50%)",
        width: 340, height: 40,
        borderRadius: "50%",
        background: "radial-gradient(ellipse, rgba(0,183,255,0.35) 0%, rgba(59,130,246,0.12) 55%, transparent 80%)",
        filter: "blur(12px)",
      }}/>

      <svg width="380" height="100" viewBox="0 0 380 100" overflow="visible" style={{ display: "block" }}>
        <defs>
          <filter id="neon-glow" x="-30%" y="-200%" width="160%" height="500%">
            <feGaussianBlur stdDeviation="2.5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* Outer ring — bright cyan */}
        <motion.ellipse
          cx="190" cy="58" rx="168" ry="28"
          fill="none" stroke="rgba(0,200,255,0.70)" strokeWidth="1.6"
          filter="url(#neon-glow)"
          animate={{ opacity: [0.55, 0.85, 0.55] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
        />
        {/* Ring 2 */}
        <motion.ellipse
          cx="190" cy="58" rx="128" ry="21"
          fill="none" stroke="rgba(59,130,246,0.65)" strokeWidth="1.4"
          filter="url(#neon-glow)"
          animate={{ opacity: [0.50, 0.80, 0.50] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.3 }}
        />
        {/* Ring 3 */}
        <motion.ellipse
          cx="190" cy="58" rx="88" ry="15"
          fill="none" stroke="rgba(99,163,255,0.60)" strokeWidth="1.2"
          filter="url(#neon-glow)"
          animate={{ opacity: [0.45, 0.75, 0.45] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.6 }}
        />
        {/* Inner ring */}
        <motion.ellipse
          cx="190" cy="58" rx="50" ry="9"
          fill="none" stroke="rgba(147,197,253,0.55)" strokeWidth="1.0"
          filter="url(#neon-glow)"
          animate={{ opacity: [0.40, 0.70, 0.40] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.9 }}
        />

        {/* Glowing dots on outer ring */}
        {[0, 60, 120, 180, 240, 300].map((deg, i) => {
          const rad = (deg * Math.PI) / 180;
          const x = 190 + 168 * Math.cos(rad);
          const y = 58  +  28 * Math.sin(rad);
          return (
            <motion.circle key={i} cx={x} cy={y} r={3}
              fill="rgba(0,210,255,0.9)"
              filter="url(#neon-glow)"
              animate={{ opacity: [0.3, 1, 0.3], r: [2.5, 3.5, 2.5] }}
              transition={{ duration: 2, repeat: Infinity, delay: i * 0.35, ease: "easeInOut" }}
            />
          );
        })}

        {/* Center bright dot */}
        <motion.circle cx="190" cy="58" r="5"
          fill="rgba(0,200,255,0.9)" filter="url(#neon-glow)"
          animate={{ r: [4, 6, 4], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
        />
      </svg>
    </div>
  );
}
