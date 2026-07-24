import { useState, useMemo, useCallback, useRef, useEffect, useLayoutEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RotateCcw, Activity, Heart, Brain, TrendingUp, Flame,
  AlertTriangle, X, ArrowLeft, Scale, FileText, HeartPulse,
  Clock, Shield, Play, ChevronDown, Cpu, Upload, Download,
} from "lucide-react";
import { analyzePatientData, downloadPatientReport } from "../services/portalService";

/* ─────────────────────────── HELPERS ──────────────────────────── */
const clamp = (v, mn, mx) => Math.min(Math.max(Number(v) || 0, mn), mx);

const calcBMI = (h, w) => {
  const hm = clamp(h, 50, 300) / 100;
  const wk = clamp(w, 10, 500);
  if (!hm || !wk) return null;
  return (wk / (hm * hm)).toFixed(1);
};

const bmiCat = (b) => {
  if (!b) return { label: "—", color: "#94a3b8" };
  const n = +b;
  if (n < 18.5) return { label: "Underweight", color: "#f59e0b" };
  if (n < 25)   return { label: "Normal",      color: "#10b981" };
  if (n < 30)   return { label: "Overweight",  color: "#f59e0b" };
  return               { label: "Obese",        color: "#ef4444" };
};

const calcHeartAge = (d) => {
  let age = clamp(d.age, 18, 100) || 35, delta = 0;
  if (clamp(d.systolic, 0, 300) >= 140) delta += 8;
  if (clamp(d.systolic, 0, 300) >= 160) delta += 5;
  if (+d.cholesterol === 3) delta += 7;
  if (d.smoking === "1") delta += 10;
  if (d.alcohol === "1") delta += 4;
  if (d.active === "0") delta += 6;
  if (clamp(d.weight, 0, 500) > 90) delta += 3;
  return Math.round(age + delta);
};

const calcRisk = (d) => {
  let s = 12;
  if (clamp(d.age, 0, 120) >= 55) s += 18;
  if (clamp(d.systolic, 0, 300) >= 140) s += 21;
  if (clamp(d.diastolic, 0, 200) >= 90) s += 11;
  if (+d.cholesterol === 3) s += 14;
  if (d.smoking === "1") s += 11;
  if (d.active === "0") s += 6;
  if (d.alcohol === "1") s += 4;
  return Math.min(s, 95);
};

/* ─────────────────────────── SLIDER CSS ───────────────────────── */
const CSS = `
  .dt-slider{-webkit-appearance:none;appearance:none;width:100%;height:3px;
    border-radius:99px;outline:none;cursor:pointer;background:transparent;}
  .dt-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
    width:26px;height:26px;border-radius:50%;background:#14b8a6;
    border:3.5px solid #fff;box-shadow:0 3px 10px rgba(20,184,166,.5),0 0 0 1px rgba(20,184,166,.2);
    cursor:pointer;transition:transform .15s,box-shadow .15s;}
  .dt-slider::-webkit-slider-thumb:hover{transform:scale(1.2);
    box-shadow:0 5px 18px rgba(20,184,166,.65),0 0 0 3px rgba(20,184,166,.2);}
  .dt-slider::-webkit-slider-thumb:active{transform:scale(1.3);}
  .dt-slider::-moz-range-thumb{width:24px;height:24px;border-radius:50%;
    background:#14b8a6;border:3.5px solid #fff;
    box-shadow:0 3px 10px rgba(20,184,166,.5);cursor:pointer;}
  .dt-slider::-moz-range-track{background:transparent;height:3px;}
`;
if (typeof document !== "undefined" && !document.getElementById("dt-css")) {
  const s = document.createElement("style");
  s.id = "dt-css"; s.textContent = CSS;
  document.head.appendChild(s);
}

/* ─────────────────────────── SLIDER ───────────────────────────── */
function Slider({ label, value, onChange, min, max, step = 1, unit = "" }) {
  const num  = Number(value) || min;
  const pct  = ((num - min) / (max - min)) * 100;
  const teal = "#14b8a6";
  return (
    <div style={{ paddingBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <span style={{ color: "#475569", fontWeight: 600, fontSize: 16 }}>{label}</span>
        {value !== "" && (
          <motion.span
            key={num}
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
            style={{
              fontSize: 14, fontWeight: 700, color: teal,
              background: "rgba(20,184,166,.1)", border: "1px solid rgba(20,184,166,.25)",
              borderRadius: 99, padding: "3px 14px",
            }}
          >
            {num}{unit}
          </motion.span>
        )}
      </div>
      <div style={{ position: "relative", paddingTop: 28 }}>
        {value !== "" && (
          <motion.div
            key={num}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{
              position: "absolute", top: 0,
              left: `clamp(0px, calc(${pct}% - 22px), calc(100% - 44px))`,
              background: teal, color: "#fff",
              fontSize: 12, fontWeight: 700,
              padding: "2px 9px", borderRadius: 99,
              whiteSpace: "nowrap", pointerEvents: "none",
              boxShadow: "0 3px 8px rgba(20,184,166,.4)",
            }}
          >
            {num}{unit}
          </motion.div>
        )}
        <input
          type="range" className="dt-slider"
          min={min} max={max} step={step}
          value={value === "" ? min : num}
          onChange={e => onChange(e.target.value)}
          style={{ background: `linear-gradient(to right,${teal} ${pct}%,#e2e8f0 ${pct}%)` }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
          <span style={{ fontSize: 12, color: "#cbd5e1" }}>{min}{unit}</span>
          <span style={{ fontSize: 12, color: "#cbd5e1" }}>{max}{unit}</span>
        </div>
      </div>
    </div>
  );
}

/* ─────────────────────────── CHIP TOGGLE ──────────────────────── */
function ChipToggle({ label, value, onChange, options }) {
  return (
    <div>
      <div style={{ color: "#475569", fontWeight: 600, fontSize: 16, marginBottom: 14 }}>{label}</div>
      <div style={{ display: "flex", gap: 10 }}>
        {options.map(opt => (
          <motion.button
            key={opt.value} onClick={() => onChange(opt.value)}
            whileTap={{ scale: 0.95 }}
            style={{
              flex: 1, padding: "12px 0", borderRadius: 14,
              fontSize: 15, fontWeight: 600, border: "1.5px solid",
              cursor: "pointer", transition: "all .2s",
              background: value === opt.value ? (opt.color || "#2563eb") : "#f8fafc",
              color: value === opt.value ? "#fff" : "#64748b",
              borderColor: value === opt.value ? (opt.color || "#2563eb") : "#e2e8f0",
            }}
          >
            {opt.label}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────── CATEGORY SELECT ──────────────────── */
function CategorySelect({ label, value, onChange, categories }) {
  return (
    <div>
      <div style={{ color: "#475569", fontWeight: 600, fontSize: 16, marginBottom: 14 }}>{label}</div>
      <div style={{ display: "flex", gap: 10 }}>
        {categories.map(cat => (
          <motion.button
            key={cat.value} onClick={() => onChange(String(cat.value))}
            whileTap={{ scale: 0.94 }}
            style={{
              flex: 1, padding: "12px 0", borderRadius: 14,
              fontSize: 15, fontWeight: 600, border: "1.5px solid",
              cursor: "pointer", transition: "all .2s",
              background: value === String(cat.value) ? cat.bg : "#f8fafc",
              color: value === String(cat.value) ? cat.text : "#64748b",
              borderColor: value === String(cat.value) ? cat.border : "#e2e8f0",
            }}
          >
            {cat.label}
          </motion.button>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────── FILE UPLOAD ──────────────────────── */
function FileUpload({ label, icon: Icon, file, onFile, accept, color }) {
  const ref = useRef();
  return (
    <div>
      <div style={{ color: "#475569", fontWeight: 600, fontSize: 16, marginBottom: 14 }}>{label}</div>
      <motion.button
        whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.98 }}
        onClick={() => ref.current?.click()}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: 14,
          padding: "18px 20px", borderRadius: 16,
          border: `2px dashed ${file ? color : "#e2e8f0"}`,
          background: file ? `${color}08` : "#f8fafc",
          cursor: "pointer",
        }}
      >
        <Icon size={22} color={file ? color : "#94a3b8"} />
        <div style={{ flex: 1, textAlign: "left" }}>
          <div style={{ color: file ? color : "#94a3b8", fontWeight: 600, fontSize: 15 }}>
            {file ? file.name : `Upload ${label}`}
          </div>
          {!file && <div style={{ color: "#cbd5e1", fontSize: 13, marginTop: 2 }}>Click to browse</div>}
        </div>
        {file && (
          <motion.button
            initial={{ scale: 0 }} animate={{ scale: 1 }}
            onClick={e => { e.stopPropagation(); onFile(null); }}
            style={{ background: "#fee2e2", borderRadius: "50%", padding: 4, border: "none", cursor: "pointer" }}
          >
            <X size={12} color="#ef4444" />
          </motion.button>
        )}
      </motion.button>
      <input ref={ref} type="file" accept={accept} style={{ display: "none" }}
        onChange={e => onFile(e.target.files?.[0] || null)} />
    </div>
  );
}

/* ─────────────────────────── ACCORDION SECTION ────────────────── */
function AccordionSection({ title, icon: Icon, color, isOpen, onToggle, children }) {
  return (
    <motion.div
      style={{
        background: "#fff", borderRadius: 24,
        border: "1px solid #f1f5f9",
        boxShadow: "0 2px 12px rgba(0,0,0,0.05)",
        overflow: "hidden",
      }}
      whileHover={{ boxShadow: "0 4px 24px rgba(0,0,0,0.08)" }}
      transition={{ duration: 0.2 }}
    >
      <button
        onClick={onToggle}
        style={{
          width: "100%", display: "flex", alignItems: "center",
          justifyContent: "space-between",
          padding: "24px 32px",
          background: isOpen ? `${color}05` : "#fff",
          border: "none", cursor: "pointer",
          borderBottom: isOpen ? `1px solid ${color}15` : "1px solid transparent",
          transition: "all .25s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 48, height: 48, borderRadius: 16,
            background: isOpen ? `${color}15` : "#f8fafc",
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "background .2s",
          }}>
            <Icon size={22} color={isOpen ? color : "#94a3b8"} />
          </div>
          <span style={{
            color: "#0f172a", fontWeight: 700,
            fontSize: 22, letterSpacing: "-0.01em",
          }}>
            {title}
          </span>
        </div>
        <motion.div animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.25 }}>
          <ChevronDown size={22} color="#94a3b8" />
        </motion.div>
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ padding: "32px 32px 36px" }}>
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ─────────────────────────── METRIC CARD ──────────────────────── */
function MetricCard({ label, value, sub, icon: Icon, color, progress, suffix = "" }) {
  return (
    <motion.div
      whileHover={{ y: -4, boxShadow: "0 16px 40px rgba(0,0,0,0.1)" }}
      transition={{ duration: 0.2 }}
      style={{
        background: "#fff", borderRadius: 24,
        border: "1px solid #f1f5f9",
        boxShadow: "0 2px 12px rgba(0,0,0,0.05)",
        padding: "28px 28px 24px",
        display: "flex", flexDirection: "column", gap: 16,
        cursor: "default",
      }}
    >
      {/* Top row */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div style={{
          width: 52, height: 52, borderRadius: 16,
          background: `${color}12`,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Icon size={24} color={color} />
        </div>
        <span style={{
          fontSize: 13, fontWeight: 700, color,
          background: `${color}10`, border: `1px solid ${color}25`,
          borderRadius: 99, padding: "4px 12px",
        }}>
          {sub}
        </span>
      </div>

      {/* Value */}
      <div>
        <motion.div
          key={String(value)}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          style={{
            fontSize: 48, fontWeight: 800,
            color: "#0f172a", letterSpacing: "-0.04em",
            lineHeight: 1,
          }}
        >
          {value}{suffix}
        </motion.div>
        <div style={{ color: "#94a3b8", fontSize: 15, marginTop: 8, fontWeight: 500 }}>
          {label}
        </div>
      </div>

      {/* Progress */}
      {progress != null && (
        <div style={{ background: "#f1f5f9", borderRadius: 99, height: 6, overflow: "hidden" }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(progress, 100)}%` }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            style={{ height: "100%", background: color, borderRadius: 99 }}
          />
        </div>
      )}
    </motion.div>
  );
}

/* ─────────────────────────── MAIN ─────────────────────────────── */
const blank = {
  age: "", height: "", weight: "",
  systolic: "", diastolic: "",
  cholesterol: "", glucose: "",
  smoking: "", alcohol: "", active: "",
};

/* ─── Download simulation report ───────────────────────────────── */
function DownloadReportButton({ data }) {
  const [status, setStatus] = useState("idle"); // idle | loading | done | error

  async function handleDownload() {
    setStatus("loading");
    try {
      const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
      if (!patient.patient_id) {
        setStatus("error");
        return;
      }
      const result = await analyzePatientData(data, { ecg: null, echo: null }, patient);
      const predId  = result.prediction_id;
      await downloadPatientReport(predId);
      setStatus("done");
      setTimeout(() => setStatus("idle"), 3000);
    } catch {
      setStatus("error");
      setTimeout(() => setStatus("idle"), 3000);
    }
  }

  const label =
    status === "loading" ? "Generating report…" :
    status === "done"    ? "Downloaded ✓" :
    status === "error"   ? "Failed — try again" :
    "Download Simulation Report";

  return (
    <button
      onClick={handleDownload}
      disabled={status === "loading"}
      style={{
        marginTop: 20,
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 8,
        padding: "13px 20px",
        borderRadius: 14,
        border: "1.5px solid #e2e8f0",
        background: status === "done"  ? "#f0fdf4" :
                    status === "error" ? "#fff1f2" : "#f8fafc",
        color:  status === "done"  ? "#16a34a" :
                status === "error" ? "#dc2626" : "#334155",
        fontWeight: 700,
        fontSize: 14,
        cursor: status === "loading" ? "not-allowed" : "pointer",
        transition: "all .2s",
        opacity: status === "loading" ? 0.7 : 1,
      }}
    >
      <Download size={16} />
      {label}
    </button>
  );
}

export default function DigitalTwin({ onBack, initialData = {} }) {
  const [data, setData]   = useState(() => ({ ...blank, ...initialData }));
  const [files, setFiles] = useState({ ecg: null, echo: null });
  const [open, setOpen]   = useState("body");
  const [sim, setSim]     = useState(false);

  const upd    = useCallback((k, v) => setData(p => ({ ...p, [k]: v })), []);
  const toggle = s => setOpen(p => p === s ? null : s);
  const reset  = () => { setData({ ...blank }); setFiles({ ecg: null, echo: null }); };

  const B     = useMemo(() => calcBMI(data.height, data.weight), [data.height, data.weight]);
  const bc    = useMemo(() => bmiCat(B), [B]);
  const risk  = useMemo(() => calcRisk(data), [data]);
  const hs    = useMemo(() => Math.max(5, 100 - risk), [risk]);
  const hAge  = useMemo(() => calcHeartAge(data), [data]);
  const bpHigh = clamp(data.systolic, 0, 300) >= 140;

  const rl = risk < 25 ? { label: "Low",      color: "#10b981" }
           : risk < 50 ? { label: "Moderate",  color: "#f59e0b" }
           : risk < 75 ? { label: "Elevated",  color: "#f97316" }
           :             { label: "High",       color: "#ef4444" };

  const conf = useMemo(() => {
    let c = 40;
    if (data.age) c += 8;
    if (data.height && data.weight) c += 8;
    if (data.systolic) c += 12;
    if (data.cholesterol) c += 10;
    if (data.smoking !== "") c += 7;
    if (data.active  !== "") c += 7;
    if (files.ecg)  c += 15;
    if (files.echo) c += 15;
    return Math.min(c, 98);
  }, [data, files]);

  useEffect(() => {
    if (!sim) return;
    const t = setTimeout(() => setSim(false), 3000);
    return () => clearTimeout(t);
  }, [sim]);

  /* Force light background — overrides the dark cardio-shell theme */
  useLayoutEffect(() => {
    const prev = document.body.style.background;
    const prevRoot = document.getElementById("root")?.style.background;
    document.body.style.background = "#f8fafc";
    document.body.style.color = "#0f172a";
    if (document.getElementById("root")) {
      document.getElementById("root").style.background = "#f8fafc";
    }
    return () => {
      document.body.style.background = prev;
      document.body.style.color = "";
      if (document.getElementById("root")) {
        document.getElementById("root").style.background = prevRoot || "";
      }
    };
  }, []);

  return (
    <div style={{
      width: "100%",
      minHeight: "100vh",
      background: "#f8fafc",
      color: "#0f172a",
      fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
      overflowX: "hidden",
    }}>

      {/* ── TOOLBAR */}
      <div style={{
        position: "sticky", top: 0, zIndex: 30,
        background: "#fff",
        borderBottom: "1px solid #f1f5f9",
        boxShadow: "0 1px 12px rgba(0,0,0,0.05)",
        padding: "0 40px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 76,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          {onBack && (
            <motion.button
              whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}
              onClick={onBack}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "10px 18px", borderRadius: 14,
                border: "1.5px solid #e2e8f0",
                background: "#f8fafc", color: "#475569",
                fontWeight: 600, fontSize: 15, cursor: "pointer",
              }}
            >
              <ArrowLeft size={16} /> Back
            </motion.button>
          )}
          <div>
            <h1 style={{
              margin: 0, color: "#0f172a",
              fontWeight: 800, fontSize: 30,
              letterSpacing: "-0.03em",
            }}>
              Digital Twin
            </h1>
            <p style={{ margin: 0, color: "#94a3b8", fontSize: 14, marginTop: 2 }}>
              Interactive cardiac simulation · Live AI analysis
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Live risk badge */}
          <motion.div
            key={rl.label}
            initial={{ scale: 0.9 }} animate={{ scale: 1 }}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 18px", borderRadius: 99,
              background: `${rl.color}10`,
              color: rl.color, fontWeight: 700, fontSize: 15,
              border: `1.5px solid ${rl.color}30`,
            }}
          >
            <motion.div
              style={{ width: 8, height: 8, borderRadius: "50%", background: rl.color }}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.4, repeat: Infinity }}
            />
            {rl.label} Risk · {risk}%
          </motion.div>

          <motion.button
            whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
            onClick={reset}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "10px 20px", borderRadius: 14,
              border: "1.5px solid #e2e8f0",
              background: "#f8fafc", color: "#475569",
              fontWeight: 600, fontSize: 15, cursor: "pointer",
            }}
          >
            <RotateCcw size={16} /> Reset
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
            onClick={() => setSim(true)}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "10px 24px", borderRadius: 14, border: "none",
              background: sim ? "#10b981" : "#2563eb",
              color: "#fff", fontWeight: 700, fontSize: 15, cursor: "pointer",
              boxShadow: sim
                ? "0 4px 16px rgba(16,185,129,.4)"
                : "0 4px 16px rgba(37,99,235,.35)",
            }}
          >
            {sim
              ? <><motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}><Cpu size={16} /></motion.div> Running…</>
              : <><Play size={16} /> Simulate</>
            }
          </motion.button>
        </div>
      </div>

      {/* ── TWO-COLUMN BODY */}
      <div style={{
        width: "92%",
        maxWidth: 1600,
        margin: "0 auto",
        padding: "40px 0 60px",
        display: "flex",
        gap: 32,
        alignItems: "flex-start",
      }}>

        {/* ── LEFT: ACCORDION INPUTS (65%) */}
        <div style={{ flex: "0 0 65%", display: "flex", flexDirection: "column", gap: 20 }}>

          {/* Body Measurements */}
          <AccordionSection title="Body Measurements" icon={Scale} color="#2563eb"
            isOpen={open === "body"} onToggle={() => toggle("body")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 40 }}>
              <Slider label="Age"    value={data.age}    onChange={v => upd("age", v)}    min={1}   max={100} unit=" yrs" />
              <Slider label="Height" value={data.height} onChange={v => upd("height", v)} min={100} max={220} unit=" cm"  />
              <Slider label="Weight" value={data.weight} onChange={v => upd("weight", v)} min={30}  max={180} unit=" kg"  />
            </div>
            <AnimatePresence>
              {B && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  style={{
                    marginTop: 28, display: "flex",
                    alignItems: "center", justifyContent: "space-between",
                    padding: "16px 24px", borderRadius: 16,
                    background: `${bc.color}10`, border: `1px solid ${bc.color}20`,
                  }}
                >
                  <span style={{ color: "#475569", fontWeight: 600, fontSize: 16 }}>
                    BMI (auto-calculated)
                  </span>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ color: "#0f172a", fontWeight: 800, fontSize: 28 }}>{B}</span>
                    <span style={{
                      background: `${bc.color}15`, color: bc.color,
                      fontWeight: 700, fontSize: 14,
                      borderRadius: 99, padding: "4px 14px",
                    }}>
                      {bc.label}
                    </span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </AccordionSection>

          {/* Clinical Markers */}
          <AccordionSection title="Clinical Markers" icon={Activity} color="#10b981"
            isOpen={open === "clinical"} onToggle={() => toggle("clinical")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 40, marginBottom: 32 }}>
              <div>
                <Slider label="Systolic BP"  value={data.systolic}  onChange={v => upd("systolic", v)}  min={80}  max={200} unit=" mmHg" />
                <AnimatePresence>
                  {bpHigh && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                      style={{
                        display: "flex", alignItems: "center", gap: 8,
                        marginTop: 10, padding: "10px 16px", borderRadius: 12,
                        background: "#fee2e2", color: "#ef4444",
                        fontWeight: 600, fontSize: 14,
                      }}
                    >
                      <AlertTriangle size={14} /> High blood pressure detected
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <Slider label="Diastolic BP" value={data.diastolic} onChange={v => upd("diastolic", v)} min={40} max={130} unit=" mmHg" />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>
              <CategorySelect label="Cholesterol Level" value={data.cholesterol} onChange={v => upd("cholesterol", v)}
                categories={[
                  { value: "1", label: "Normal", bg: "#dcfce7", text: "#16a34a", border: "#86efac" },
                  { value: "2", label: "Above",  bg: "#fef9c3", text: "#b45309", border: "#fde68a" },
                  { value: "3", label: "High",   bg: "#fee2e2", text: "#dc2626", border: "#fca5a5" },
                ]}
              />
              <CategorySelect label="Glucose Level" value={data.glucose} onChange={v => upd("glucose", v)}
                categories={[
                  { value: "1", label: "Normal", bg: "#dcfce7", text: "#16a34a", border: "#86efac" },
                  { value: "2", label: "Above",  bg: "#fef9c3", text: "#b45309", border: "#fde68a" },
                  { value: "3", label: "High",   bg: "#fee2e2", text: "#dc2626", border: "#fca5a5" },
                ]}
              />
            </div>
          </AccordionSection>

          {/* Lifestyle */}
          <AccordionSection title="Lifestyle" icon={Flame} color="#f59e0b"
            isOpen={open === "lifestyle"} onToggle={() => toggle("lifestyle")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 28 }}>
              <ChipToggle label="Smoking" value={data.smoking} onChange={v => upd("smoking", v)}
                options={[
                  { value: "0", label: "Non-smoker", color: "#10b981" },
                  { value: "1", label: "Smoker",     color: "#ef4444" },
                ]}
              />
              <ChipToggle label="Alcohol Consumption" value={data.alcohol} onChange={v => upd("alcohol", v)}
                options={[
                  { value: "0", label: "None",   color: "#10b981" },
                  { value: "1", label: "Drinks", color: "#f59e0b" },
                ]}
              />
              <ChipToggle label="Physical Activity" value={data.active} onChange={v => upd("active", v)}
                options={[
                  { value: "1", label: "Active",   color: "#2563eb" },
                  { value: "0", label: "Inactive", color: "#94a3b8" },
                ]}
              />
            </div>
          </AccordionSection>

          {/* Medical Files */}
          <AccordionSection title="Medical Files" icon={FileText} color="#8b5cf6"
            isOpen={open === "files"} onToggle={() => toggle("files")}
          >
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <FileUpload label="ECG Recording"  icon={Activity}   file={files.ecg}
                accept=".png,.jpg,.jpeg,.csv"          color="#2563eb"
                onFile={f => setFiles(p => ({ ...p, ecg: f }))} />
              <FileUpload label="Echocardiogram" icon={HeartPulse} file={files.echo}
                accept=".mp4,.avi,.mov,.mkv"           color="#8b5cf6"
                onFile={f => setFiles(p => ({ ...p, echo: f }))} />
            </div>
          </AccordionSection>
        </div>

        {/* ── RIGHT: LIVE RESULTS (35%) */}
        <div style={{ flex: "0 0 calc(35% - 32px)", position: "sticky", top: 96 }}>

          {/* Panel header */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <div style={{
                width: 10, height: 10, borderRadius: "50%",
                background: rl.color,
              }}>
                <motion.div
                  style={{ width: "100%", height: "100%", borderRadius: "50%", background: rl.color }}
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1.4, repeat: Infinity }}
                />
              </div>
              <span style={{ color: "#475569", fontWeight: 700, fontSize: 13, letterSpacing: "0.08em" }}>
                LIVE RESULTS
              </span>
              <span style={{
                marginLeft: "auto", color: "#94a3b8",
                fontSize: 13, fontWeight: 500,
              }}>
                {conf}% confidence
              </span>
            </div>
            <div style={{ height: 2, borderRadius: 99, background: "#f1f5f9" }}>
              <motion.div
                animate={{ width: `${conf}%` }}
                transition={{ duration: 0.8 }}
                style={{ height: "100%", borderRadius: 99, background: "#2563eb" }}
              />
            </div>
          </div>

          {/* Simulation banner */}
          <AnimatePresence>
            {sim && (
              <motion.div
                initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                style={{
                  display: "flex", alignItems: "center", gap: 14,
                  padding: "18px 22px", borderRadius: 18, marginBottom: 20,
                  background: "linear-gradient(135deg,#eff6ff,#f0fdf4)",
                  border: "1.5px solid #bfdbfe",
                }}
              >
                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                  <Cpu size={22} color="#2563eb" />
                </motion.div>
                <div>
                  <div style={{ fontWeight: 700, color: "#0f172a", fontSize: 16 }}>AI Simulation Running</div>
                  <div style={{ color: "#64748b", fontSize: 13 }}>Calculating interactions…</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Metric cards */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <MetricCard
              label="Heart Health Score"
              value={hs} suffix="/100"
              sub={hs >= 75 ? "Good" : hs >= 50 ? "Fair" : "At Risk"}
              icon={Heart}
              color={hs >= 75 ? "#10b981" : hs >= 50 ? "#f59e0b" : "#ef4444"}
              progress={hs}
            />
            <MetricCard
              label="Cardiovascular Risk"
              value={risk} suffix="%"
              sub={rl.label}
              icon={Shield}
              color={rl.color}
              progress={risk}
            />
            <MetricCard
              label="AI Confidence"
              value={conf} suffix="%"
              sub={conf >= 80 ? "High" : conf >= 60 ? "Medium" : "Low"}
              icon={Brain}
              color="#8b5cf6"
              progress={conf}
            />
            <MetricCard
              label="Current BMI"
              value={B || "—"}
              sub={bc.label}
              icon={Scale}
              color={bc.color}
              progress={B ? Math.min((+B / 40) * 100, 100) : null}
            />
            <MetricCard
              label="Estimated Heart Age"
              value={data.age ? hAge : "—"}
              suffix={data.age ? " yrs" : ""}
              sub={data.age && hAge > +data.age ? `+${hAge - +data.age} yrs` : "Healthy"}
              icon={Clock}
              color={data.age && hAge > +data.age + 5 ? "#ef4444" : "#2563eb"}
            />
            <MetricCard
              label="Prediction Mode"
              value={files.ecg || files.echo ? "Multi" : "Clinical"}
              sub={files.ecg && files.echo ? "Full" : files.ecg || files.echo ? "Enhanced" : "Basic"}
              icon={TrendingUp}
              color="#14b8a6"
              progress={files.ecg && files.echo ? 100 : files.ecg || files.echo ? 65 : 30}
            />
          </div>

          {/* Download simulation report */}
          <DownloadReportButton data={data} />
        </div>
      </div>
    </div>
  );
}
