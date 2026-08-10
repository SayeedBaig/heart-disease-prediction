/**
 * FloatingMetricCard — Premium WHITE card matching reference image exactly.
 *
 * Card variants:
 *  "heart-rate"    → red heart icon, 72 BPM, mini ECG sparkline, green dot "Normal"
 *  "risk-level"    → blue shield icon, "Moderate" in orange, gradient progress bar + 58%
 *  "ai-confidence" → blue brain icon, 96.4% in dark bold, green dot "High Confidence"
 *  "ecg-status"    → blue ECG icon, "Normal" in green, "Sinus Rhythm" + green dot
 */
import { motion } from "framer-motion";

/* ── Tiny inline SVG icons (no external dep) ── */
function HeartIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="#ef4444">
      <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.5 2a2.5 2.5 0 0 1 5 0v1a2 2 0 0 0 2 2 2.5 2.5 0 0 1 0 5 2 2 0 0 0-2 2v.5a2.5 2.5 0 0 1-5 0V12a2 2 0 0 0-2-2 2.5 2.5 0 0 1 0-5 2 2 0 0 0 2-2V2z" />
      <path d="M12 12v4" />
    </svg>
  );
}

function ECGIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3,12 6,12 8,6 10,18 12,10 14,14 16,12 21,12" />
    </svg>
  );
}

/* ── Mini ECG sparkline for Heart Rate card ── */
function MiniECGSparkline() {
  return (
    <svg
      viewBox="0 0 80 24"
      width="80"
      height="24"
      fill="none"
      className="my-1"
    >
      <motion.path
        d="M0 12 L12 12 L16 6 L20 18 L24 12 L40 12 L44 4 L48 20 L52 12 L68 12 L72 8 L76 16 L80 12"
        stroke="#f87171"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.4, delay: 0.8, ease: "easeOut" }}
      />
    </svg>
  );
}

/* ── Gradient progress bar for Risk Level card ── */
function RiskProgressBar({ percent = 58 }) {
  return (
    <div className="mt-2">
      <div
        className="relative h-2 rounded-full overflow-hidden"
        style={{ background: "rgba(0,0,0,0.08)" }}
      >
        <motion.div
          className="absolute top-0 left-0 h-full rounded-full"
          style={{
            background: "linear-gradient(90deg, #22c55e 0%, #eab308 55%, #f97316 100%)",
            width: `${percent}%`,
          }}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 1.2, delay: 1.0, ease: "easeOut" }}
        />
      </div>
      <div className="mt-1 text-right text-[11px] font-semibold" style={{ color: "#6b7280" }}>
        {percent}%
      </div>
    </div>
  );
}

/* ── Green status dot + label ── */
function StatusDot({ label, color = "#22c55e" }) {
  return (
    <div className="flex items-center gap-1.5 mt-2">
      <span
        className="w-2 h-2 rounded-full inline-block"
        style={{ background: color }}
      />
      <span className="text-[11px] font-semibold" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

/* ── Card layout configs ── */
const cardConfigs = {
  "heart-rate": {
    position: { top: "2%", left: "-8%" },
    float: { y: [0, -10, 0], x: [0, -3, 0] },
    floatDur: 3.8,
    delay: 0.5,
  },
  "risk-level": {
    position: { top: "2%", right: "-8%" },
    float: { y: [0, -12, 0], x: [0, 3, 0] },
    floatDur: 4.2,
    delay: 0.7,
  },
  "ai-confidence": {
    position: { bottom: "14%", left: "-10%" },
    float: { y: [0, 9, 0], x: [0, -3, 0] },
    floatDur: 4.6,
    delay: 0.9,
  },
  "ecg-status": {
    position: { bottom: "14%", right: "-8%" },
    float: { y: [0, 7, 0], x: [0, 4, 0] },
    floatDur: 3.5,
    delay: 1.1,
  },
};

export default function FloatingMetricCard({ variant = "heart-rate" }) {
  const cfg = cardConfigs[variant];

  return (
    <motion.div
      className="absolute z-20 pointer-events-none"
      style={{ ...cfg.position }}
      initial={{ opacity: 0, scale: 0.8, y: 10 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: cfg.float.y,
        x: cfg.float.x,
      }}
      transition={{
        opacity: { duration: 0.5, delay: cfg.delay },
        scale: { duration: 0.5, delay: cfg.delay },
        y: {
          duration: cfg.floatDur,
          repeat: Infinity,
          ease: "easeInOut",
          delay: cfg.delay,
        },
        x: {
          duration: cfg.floatDur + 0.8,
          repeat: Infinity,
          ease: "easeInOut",
          delay: cfg.delay + 0.4,
        },
      }}
    >
      {/* White premium card */}
      <div
        className="rounded-2xl p-4"
        style={{
          background: "rgba(255,255,255,0.95)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          boxShadow: "0 4px 24px rgba(0,0,0,0.10), 0 1px 4px rgba(0,0,0,0.06)",
          border: "1px solid rgba(255,255,255,0.8)",
          minWidth: 156,
        }}
      >
        {variant === "heart-rate" && <HeartRateContent />}
        {variant === "risk-level" && <RiskLevelContent />}
        {variant === "ai-confidence" && <AIConfidenceContent />}
        {variant === "ecg-status" && <ECGStatusContent />}
      </div>
    </motion.div>
  );
}

/* ─────────────── Card Contents ─────────────── */

function HeartRateContent() {
  return (
    <>
      <div className="flex items-center gap-2 mb-1">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(239,68,68,0.1)" }}
        >
          <HeartIcon />
        </div>
        <span
          className="text-[11px] font-semibold tracking-wide"
          style={{ color: "#6b7280" }}
        >
          Heart Rate
        </span>
      </div>
      <div className="mt-1">
        <span
          className="text-[28px] font-extrabold leading-none tracking-tight"
          style={{ color: "#111827" }}
        >
          72
        </span>
        <span
          className="text-base font-bold ml-1.5"
          style={{ color: "#374151" }}
        >
          BPM
        </span>
      </div>
      {/* Mini ECG sparkline */}
      <MiniECGSparkline />
      <StatusDot label="Normal" color="#22c55e" />
    </>
  );
}

function RiskLevelContent() {
  return (
    <>
      <div className="flex items-center gap-2 mb-1">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(59,130,246,0.1)" }}
        >
          <ShieldIcon />
        </div>
        <span
          className="text-[11px] font-semibold tracking-wide"
          style={{ color: "#6b7280" }}
        >
          Risk Level
        </span>
      </div>
      <div
        className="text-[26px] font-extrabold leading-tight mt-1"
        style={{ color: "#f97316" }}
      >
        Moderate
      </div>
      <RiskProgressBar percent={58} />
    </>
  );
}

function AIConfidenceContent() {
  return (
    <>
      <div className="flex items-center gap-2 mb-1">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(59,130,246,0.1)" }}
        >
          <BrainIcon />
        </div>
        <span
          className="text-[11px] font-semibold tracking-wide"
          style={{ color: "#6b7280" }}
        >
          AI Confidence
        </span>
      </div>
      <div
        className="text-[28px] font-extrabold leading-none tracking-tight mt-1"
        style={{ color: "#111827" }}
      >
        96.4%
      </div>
      <StatusDot label="High Confidence" color="#22c55e" />
    </>
  );
}

function ECGStatusContent() {
  return (
    <>
      <div className="flex items-center gap-2 mb-1">
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "rgba(59,130,246,0.1)" }}
        >
          <ECGIcon />
        </div>
        <span
          className="text-[11px] font-semibold tracking-wide"
          style={{ color: "#6b7280" }}
        >
          ECG Status
        </span>
      </div>
      <div
        className="text-[26px] font-extrabold leading-tight mt-1"
        style={{ color: "#22c55e" }}
      >
        Normal
      </div>
      <div className="flex items-center gap-1.5 mt-1">
        <span
          className="text-[12px] font-medium"
          style={{ color: "#6b7280" }}
        >
          Sinus Rhythm
        </span>
        <span
          className="w-1.5 h-1.5 rounded-full inline-block ml-1"
          style={{ background: "#22c55e" }}
        />
      </div>
    </>
  );
}
