/**
 * Particles — small glowing blue/white particles that orbit and fade around the heart.
 * Framer Motion driven for smooth GPU-accelerated animation.
 */
import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

const NUM_PARTICLES = 18;

function generateParticle(index) {
  const angle = (index / NUM_PARTICLES) * Math.PI * 2;
  const radius = 100 + Math.random() * 60;
  const size = 2 + Math.random() * 3;
  const duration = 6 + Math.random() * 8;
  const delay = Math.random() * 4;
  const isWhite = Math.random() > 0.5;

  return {
    id: index,
    angle,
    radius,
    size,
    duration,
    delay,
    isWhite,
    orbitSpeed: 0.3 + Math.random() * 0.4,
  };
}

function Particle({ particle, mouseX, mouseY }) {
  const x = Math.cos(particle.angle) * particle.radius;
  const y = Math.sin(particle.angle) * particle.radius * 0.6; // elliptical orbit

  const color = particle.isWhite
    ? "rgba(255,255,255,0.9)"
    : "rgba(99,179,255,0.9)";

  const glowColor = particle.isWhite
    ? "rgba(255,255,255,0.4)"
    : "rgba(59,130,246,0.5)";

  return (
    <motion.div
      className="absolute rounded-full pointer-events-none"
      style={{
        width: particle.size,
        height: particle.size,
        background: color,
        boxShadow: `0 0 ${particle.size * 3}px ${particle.size}px ${glowColor}`,
        left: "50%",
        top: "50%",
        marginLeft: -particle.size / 2,
        marginTop: -particle.size / 2,
      }}
      animate={{
        x: [x, x * 1.05, x * 0.95, x],
        y: [y, y * 0.92, y * 1.06, y],
        opacity: [0, 0.8, 1, 0.6, 0],
        scale: [0, 1, 1.2, 0.8, 0],
      }}
      transition={{
        duration: particle.duration,
        delay: particle.delay,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    />
  );
}

export default function Particles({ mouseX = 0, mouseY = 0 }) {
  const [particles] = useState(() =>
    Array.from({ length: NUM_PARTICLES }, (_, i) => generateParticle(i))
  );

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden z-10" aria-hidden="true">
      {particles.map((p) => (
        <Particle key={p.id} particle={p} mouseX={mouseX} mouseY={mouseY} />
      ))}
    </div>
  );
}
