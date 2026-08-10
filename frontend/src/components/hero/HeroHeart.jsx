/**
 * HeroHeart — fully theme-aware premium heart visualization.
 *
 * LIGHT MODE: white cards, subtle blue glow, light ECG, light rings
 * DARK MODE:  dark glassmorphic cards, neon glow, bright ECG, neon rings
 *
 * Layout: 640×460 container
 *   Heart 320×320 centered, 4 cards at absolute corners
 */
import { useRef, useState, useCallback } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { useTheme } from "./useTheme";
import heartImg from "../../assets/heart.png";
import EnergyField from "./EnergyField";


/* ─── Theme tokens ─── */
const T = {
  light: {
    card:       { bg: "rgba(255,255,255,0.93)", border: "rgba(59,130,246,0.14)", shadow: "0 4px 24px rgba(59,130,246,0.09),0 1px 6px rgba(0,0,0,0.07)" },
    label:      { color: "#9ca3af" },
    value:      { color: "#111827" },
    valueAlt:   { orange: "#f97316", green: "#16a34a" },
    statusDot:  "#16a34a",
    statusText: "#16a34a",
    subtitle:   "#6b7280",
    sparkline:  "#f87171",
    riskBar:    "rgba(0,0,0,0.07)",
    heartFilter: ["drop-shadow(0 0 24px rgba(37,99,235,0.38))","drop-shadow(0 0 50px rgba(37,99,235,0.14))","drop-shadow(0 18px 44px rgba(0,0,0,0.12))"].join(" "),
    iconBg:     { heart: "rgba(239,68,68,0.10)", blue: "rgba(59,130,246,0.10)" },
    iconStroke: "#3b82f6",
  },
  dark: {
    card:       { bg: "rgba(8,14,38,0.75)", border: "rgba(59,130,246,0.30)", shadow: "0 4px 32px rgba(0,0,80,0.45),inset 0 1px 0 rgba(99,163,255,0.10)" },
    label:      { color: "rgba(147,197,253,0.65)" },
    value:      { color: "#ffffff" },
    valueAlt:   { orange: "#fb923c", green: "#4ade80" },
    statusDot:  "#4ade80",
    statusText: "#4ade80",
    subtitle:   "rgba(255,255,255,0.45)",
    sparkline:  "#f87171",
    riskBar:    "rgba(255,255,255,0.08)",
    heartFilter: ["drop-shadow(0 0 40px rgba(0,140,255,0.72))","drop-shadow(0 0 80px rgba(37,99,235,0.35))","drop-shadow(0 24px 60px rgba(0,0,80,0.50))"].join(" "),
    iconBg:     { heart: "rgba(239,68,68,0.15)", blue: "rgba(59,130,246,0.18)" },
    iconStroke: "rgba(99,163,255,1)",
  },
};

/* ─── SVG Icons ─── */
const HeartIcon = ({ color }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill={color}><path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z"/></svg>
);
const ShieldIcon = ({ stroke }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
);
const BrainIcon = ({ stroke }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
    <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
  </svg>
);
const ECGLineIcon = ({ stroke }) => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3,12 6,12 8,4 10,20 12,10 14,14 16,12 21,12"/></svg>
);

/* ─── Mini ECG sparkline ─── */
const MiniSparkline = ({ color }) => (
  <svg viewBox="0 0 76 18" width="76" height="18" fill="none" style={{ display:"block", margin:"5px 0" }}>
    <motion.polyline
      points="0,9 10,9 14,3 17,14 20,9 33,9 37,2 41,16 45,9 57,9 61,5 65,12 69,9 76,9"
      stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none"
      initial={{ pathLength:0, opacity:0 }} animate={{ pathLength:1, opacity:1 }}
      transition={{ duration:1.3, delay:1.0, ease:"easeOut" }}
    />
  </svg>
);

/* ─── Risk gradient bar ─── */
const RiskBar = ({ trackBg }) => (
  <div style={{ marginTop:8 }}>
    <div style={{ height:8, borderRadius:99, overflow:"hidden", background:trackBg }}>
      <motion.div
        style={{ height:"100%", borderRadius:99, background:"linear-gradient(90deg,#22c55e 0%,#eab308 50%,#f97316 88%)" }}
        initial={{ width:0 }} animate={{ width:"58%" }}
        transition={{ duration:1.2, delay:1.1, ease:"easeOut" }}
      />
    </div>
    <p style={{ textAlign:"right", fontSize:10, fontWeight:700, marginTop:3, color:"inherit", opacity:0.5 }}>58%</p>
  </div>
);

/* ─── Ambient glow (theme-aware) ─── */
function HeartGlow({ mouseX, mouseY, isDark }) {
  const outerGlow = isDark
    ? "radial-gradient(circle, rgba(37,99,235,0.40) 0%, rgba(59,130,246,0.18) 35%, rgba(0,100,255,0.06) 65%, transparent 80%)"
    : "radial-gradient(circle, rgba(59,130,246,0.18) 0%, rgba(99,163,255,0.07) 45%, transparent 72%)";
  const innerGlow = isDark
    ? "radial-gradient(circle, rgba(0,183,255,0.24) 0%, rgba(59,130,246,0.10) 50%, transparent 75%)"
    : "radial-gradient(circle, rgba(59,130,246,0.14) 0%, rgba(99,163,255,0.04) 55%, transparent 75%)";

  return (
    <div style={{ position:"absolute", inset:0, pointerEvents:"none", zIndex:0, display:"flex", alignItems:"center", justifyContent:"center" }}>
      <motion.div
        animate={{ x:mouseX*14, y:mouseY*10, scale:[1,1.06,1] }}
        transition={{ x:{type:"spring",stiffness:50,damping:18}, y:{type:"spring",stiffness:50,damping:18}, scale:{duration:4,repeat:Infinity,ease:"easeInOut"} }}
        style={{ width:620, height:620, borderRadius:"50%", background:outerGlow, filter:`blur(${isDark?32:24}px)` }}
      />
      <motion.div
        style={{ position:"absolute", width:260, height:260, borderRadius:"50%", background:innerGlow, filter:`blur(${isDark?18:14}px)` }}
        animate={{ scale:[1,1.12,1], opacity:[isDark?0.5:0.35, isDark?0.85:0.55, isDark?0.5:0.35], x:mouseX*8, y:mouseY*6 }}
        transition={{ scale:{duration:1.05,repeat:Infinity,ease:[0.4,0,0.6,1],repeatDelay:0.72}, opacity:{duration:1.05,repeat:Infinity,ease:[0.4,0,0.6,1],repeatDelay:0.72}, x:{type:"spring",stiffness:50,damping:18}, y:{type:"spring",stiffness:50,damping:18} }}
      />
    </div>
  );
}

/* ─── Concentric rings (theme-aware) ─── */
function AIRing({ isDark }) {
  const c = isDark
    ? { outer:"rgba(0,200,255,0.72)", r2:"rgba(59,130,246,0.65)", r3:"rgba(99,163,255,0.58)", inner:"rgba(147,197,253,0.50)", dot:"rgba(0,210,255,0.95)", center:"rgba(0,200,255,0.90)", glow:2.5 }
    : { outer:"rgba(59,130,246,0.45)", r2:"rgba(59,130,246,0.35)", r3:"rgba(99,163,255,0.28)", inner:"rgba(147,197,253,0.22)", dot:"rgba(59,130,246,0.80)", center:"rgba(59,130,246,0.70)", glow:1.5 };

  return (
    <div
        aria-hidden="true"
        style={{
            position:"relative",
            width:460,
            height:140,
            pointerEvents:"none"
        }}
    >
      <div style={{ position:"absolute", left:"50%", top:"55%", transform:"translate(-50%,-50%)", width:340, height:40, borderRadius:"50%",
        background:`radial-gradient(ellipse, ${isDark?"rgba(0,183,255,0.30)":"rgba(59,130,246,0.15)"} 0%, ${isDark?"rgba(59,130,246,0.10)":"rgba(59,130,246,0.04)"} 55%, transparent 80%)`,
        filter:`blur(${isDark?12:8}px)` }} />
      <motion.div
      animate={{ rotate: 0 }}
      transition={{
          duration: 40,
          repeat: Infinity,
          ease: "linear",
      }}
  >


      <svg width="380" height="100" viewBox="0 0 460 140" overflow="visible" style={{ display:"block" }}>
        <defs>
          <filter id="ring-glow-h" x="-30%" y="-200%" width="160%" height="500%">
            <feGaussianBlur stdDeviation={c.glow} result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        {[
          
 { rx:205, ry:34, color:c.outer, w:1.6 },

 { rx:170, ry:28, color:c.r2, w:1.4 },

 { rx:135, ry:22, color:c.r3, w:1.2 },

 { rx:100, ry:17, color:c.inner, w:1.1 },

 { rx:70, ry:12, color:c.inner, w:1.0 },

 { rx:40, ry:7, color:c.inner, w:0.8 },
]
        .map((r,i) => (
          <motion.ellipse key={i} cx="230" cy="72" rx={r.rx} ry={r.ry}
            fill="none" stroke={r.color} strokeWidth={r.w} filter="url(#ring-glow-h)"
            animate={{ opacity:[0.55,0.88,0.55] }}
            transition={{ duration:2.5, repeat:Infinity, ease:"easeInOut", delay:i*0.3 }}
          />
        ))}
        {[0,60,120,180,240,300].map((deg,i) => {
          const rad=(deg*Math.PI)/180; const x = 230 + 205 * Math.cos(rad);const y = 72 + 34 * Math.sin(rad);
          return (
            <motion.circle key={i} cx={x} cy={y} r={3} fill={c.dot} filter="url(#ring-glow-h)"
              animate={{ opacity:[0.3,1,0.3] }} transition={{ duration:2,repeat:Infinity,delay:i*0.35,ease:"easeInOut" }} />
          );
        })}
        <motion.circle cx="230" cy="72" r="5" fill={c.center} filter="url(#ring-glow-h)"
          animate={{ r:[4,6,4], opacity:[0.7,1,0.7] }} transition={{ duration:1.5,repeat:Infinity,ease:"easeInOut" }} />
      </svg>
      </motion.div>

    </div>
  );
}

/* ─── Glass/White card shell ─── */
function Card({ children, t, style, delay, floatY, floatDur }) {
  return (
    <motion.div
      initial={{ opacity:0, scale:0.85, y:10 }}
      animate={{ opacity:1, scale:1, y:0 }}
      transition={{ duration:0.5, delay, ease:[0.16,1,0.3,1] }}
      style={{ position:"absolute", zIndex:20, pointerEvents:"none", ...style }}
    >
      <motion.div
        animate={{ y:floatY }}
        transition={{ duration:floatDur, repeat:Infinity, ease:"easeInOut" }}
        style={{ background:t.card.bg, backdropFilter:"blur(20px)", WebkitBackdropFilter:"blur(20px)", borderRadius:14, border:`1px solid ${t.card.border}`, boxShadow:t.card.shadow, padding:"12px 14px", width:160 }}
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

const LABEL = (t) => ({ fontSize:9, fontWeight:800, letterSpacing:"0.08em", textTransform:"uppercase", color:t.label.color });
const CHIP  = (bg) => ({ width:22, height:22, borderRadius:7, background:bg, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 });

/* ─── Main Export ─── */
export default function HeroHeart() {
  const theme = useTheme();
  const isDark = theme === "dark";
  const t = isDark ? T.dark : T.light;

  const containerRef = useRef(null);
  const [mouse, setMouse] = useState({ x:0, y:0 });
  const rawX = useMotionValue(0); const rawY = useMotionValue(0);
  const springX = useSpring(rawX, { stiffness:60, damping:20 });
  const springY = useSpring(rawY, { stiffness:60, damping:20 });
  const rotateY = useTransform(springX, [-1,1], [-7,7]);
  const rotateX = useTransform(springY, [-1,1], [5,-5]);

  const handleMouseMove = useCallback((e) => {
    const rect = containerRef.current?.getBoundingClientRect(); if (!rect) return;
    const nx = (e.clientX-rect.left-rect.width/2)/(rect.width/2);
    const ny = (e.clientY-rect.top-rect.height/2)/(rect.height/2);
    rawX.set(nx); rawY.set(ny); setMouse({ x:nx, y:ny });
  }, [rawX, rawY]);
  const handleMouseLeave = useCallback(() => { rawX.set(0); rawY.set(0); setMouse({x:0,y:0}); }, [rawX,rawY]);

  const W=700, H=560, HW=500;
  const heartLeft=(W-HW)/2, heartTop=(H-HW)/2-24;

  return (
    <motion.div ref={containerRef} onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}
      initial={{ opacity:0 }} animate={{ opacity:1 }} transition={{ duration:0.8, delay:0.2 }}
      style={{ position:"relative", width:W, height:H, userSelect:"none", flexShrink:0 }}
    >
      {/* Ambient glow */}
      <div style={{ position:"absolute", left:heartLeft, top:heartTop, width:HW, height:HW }}>
        <HeartGlow mouseX={mouse.x} mouseY={mouse.y} isDark={isDark} />
      </div>
      <EnergyField isDark={isDark} />
     
      {/* Heart with 3D tilt + float */}
      <motion.div style={{ position:"absolute", left:heartLeft, top:heartTop, width:HW, height:HW, zIndex:10, perspective:"900px" }}>
        <motion.div style={{ rotateX, rotateY, transformStyle:"preserve-3d", width:"100%", height:"100%" }}>
          <motion.div animate={{ y:[0,-14,0] }} transition={{ duration:4.5, repeat:Infinity, ease:"easeInOut" }} style={{ width:"100%", height:"100%", position:"relative" }}>
            <motion.img src={heartImg} alt="CardioAI 3D anatomical heart"
              style={{ width:"100%", height:"100%", objectFit:"contain", display:"block", filter:t.heartFilter }}
              animate={{ scale:[1,1.035,1.01,1.042,1] }}
              transition={{ duration:1.05, repeat:Infinity, repeatDelay:0.7, ease:[0.4,0,0.6,1], times:[0,0.13,0.28,0.42,1] }}
            />
          </motion.div>
        </motion.div>
      </motion.div>

      {/* Concentric rings */}
      <div
    style={{
        position:"absolute",
        left: heartLeft + HW / 2+80,
        top: heartTop + HW - 65,
        transform:"translateX(-50%)",
        zIndex:5
    }}
>
        <AIRing isDark={isDark} />
      </div>

      {/* ── Heart Rate card (top-left) ── */}
      <Card t={t} style={{ top:28, left:0 }} delay={0.5} floatY={[0,-10,0]} floatDur={3.8}>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
          <div style={CHIP(t.iconBg.heart)}><HeartIcon color="#ef4444"/></div>
          <span style={LABEL(t)}>Heart Rate</span>
        </div>
        <div style={{ display:"flex", alignItems:"baseline", gap:3 }}>
          <span style={{ fontSize:34, fontWeight:800, color:t.value.color, lineHeight:1 }}>72</span>
          <span style={{ fontSize:14, fontWeight:700, color:t.value.color, opacity:0.7 }}>BPM</span>
        </div>
        <MiniSparkline color={t.sparkline} />
        <div style={{ display:"flex", alignItems:"center", gap:5, marginTop:4 }}>
          <span style={{ width:6, height:6, borderRadius:"50%", background:t.statusDot, display:"inline-block", flexShrink:0 }}/>
          <span style={{ fontSize:10, fontWeight:700, color:t.statusText }}>Normal</span>
        </div>
      </Card>

      {/* ── Risk Level card (top-right) ── */}
      <Card t={t} style={{ top:28, right:0 }} delay={0.65} floatY={[0,-12,0]} floatDur={4.4}>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
          <div style={CHIP(t.iconBg.blue)}><ShieldIcon stroke={t.iconStroke}/></div>
          <span style={LABEL(t)}>Risk Level</span>
        </div>
        <p style={{ fontSize:22, fontWeight:800, color:t.valueAlt.orange, lineHeight:1.2, marginTop:2 }}>Moderate</p>
        <RiskBar trackBg={t.riskBar} />
      </Card>

      {/* ── AI Confidence card (bottom-left) ── */}
      <Card t={t} style={{ bottom:50, left:0 }} delay={0.8} floatY={[0,9,0]} floatDur={4.1}>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
          <div style={CHIP(t.iconBg.blue)}><BrainIcon stroke={t.iconStroke}/></div>
          <span style={LABEL(t)}>AI Confidence</span>
        </div>
        <p style={{ fontSize:30, fontWeight:800, color:t.value.color, lineHeight:1.1, marginTop:2 }}>96.4%</p>
        <div style={{ display:"flex", alignItems:"center", gap:5, marginTop:6 }}>
          <span style={{ width:6, height:6, borderRadius:"50%", background:t.statusDot, display:"inline-block", flexShrink:0 }}/>
          <span style={{ fontSize:10, fontWeight:700, color:t.statusText }}>High Confidence</span>
        </div>
      </Card>

      {/* ── ECG Status card (bottom-right) ── */}
      <Card t={t} style={{ bottom:50, right:0 }} delay={0.95} floatY={[0,8,0]} floatDur={3.5}>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginBottom:4 }}>
          <div style={CHIP(t.iconBg.blue)}><ECGLineIcon stroke={t.iconStroke}/></div>
          <span style={LABEL(t)}>ECG Status</span>
        </div>
        <p style={{ fontSize:22, fontWeight:800, color:t.valueAlt.green, lineHeight:1.2, marginTop:2 }}>Normal</p>
        <div style={{ display:"flex", alignItems:"center", gap:6, marginTop:5 }}>
          <span style={{ fontSize:11, fontWeight:500, color:t.subtitle }}>Sinus Rhythm</span>
          <span style={{ width:5, height:5, borderRadius:"50%", background:t.statusDot, display:"inline-block" }}/>
        </div>
      </Card>
    </motion.div>
  );
}
