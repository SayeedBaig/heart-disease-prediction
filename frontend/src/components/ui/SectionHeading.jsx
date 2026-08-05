/**
 * SectionHeading — centered heading group: Badge + Title + Subtitle.
 * Keeps consistent spacing between eyebrow, heading, and description.
 */
import { motion } from "framer-motion";
import Badge from "./Badge";

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-50px" },
  transition: { duration: 0.5 },
};

export default function SectionHeading({
  badge,
  title,
  subtitle,
  align = "center",
  className = "",
}) {
  const alignClass =
    align === "center" ? "text-center mx-auto" : "text-left";

  return (
    <motion.div
      {...fadeUp}
      className={`max-w-3xl ${alignClass} ${className}`}
    >
      {badge && <Badge>{badge}</Badge>}
      <h2
        className="mt-4 text-3xl font-extrabold tracking-tight text-[var(--text-primary)] sm:text-4xl md:text-5xl"
        style={{
          fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)",
        }}
      >
        {title}
      </h2>
      {subtitle && (
        <p className="mt-4 text-base leading-relaxed text-[var(--text-secondary)] sm:text-lg">
          {subtitle}
        </p>
      )}
    </motion.div>
  );
}