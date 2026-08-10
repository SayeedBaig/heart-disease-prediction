/**
 * ECGBackground — theme-aware animated ECG waveform.
 * Light mode: subtle blue ~12% opacity
 * Dark mode:  bright blue ~22% opacity
 */
import { useEffect, useRef } from "react";
import { useTheme } from "./useTheme";

function drawECGCycle(ctx, x, baseY, unitW) {
  ctx.moveTo(x, baseY);
  ctx.lineTo(x + unitW * 0.15, baseY);
  ctx.bezierCurveTo(x + unitW * 0.17, baseY - unitW * 0.05, x + unitW * 0.22, baseY - unitW * 0.08, x + unitW * 0.24, baseY);
  ctx.lineTo(x + unitW * 0.30, baseY);
  ctx.lineTo(x + unitW * 0.34, baseY + unitW * 0.05);
  ctx.lineTo(x + unitW * 0.38, baseY - unitW * 0.30);
  ctx.lineTo(x + unitW * 0.42, baseY + unitW * 0.08);
  ctx.lineTo(x + unitW * 0.48, baseY);
  ctx.bezierCurveTo(x + unitW * 0.56, baseY, x + unitW * 0.60, baseY - unitW * 0.11, x + unitW * 0.65, baseY - unitW * 0.09);
  ctx.bezierCurveTo(x + unitW * 0.70, baseY - unitW * 0.07, x + unitW * 0.73, baseY, x + unitW * 0.78, baseY);
  ctx.lineTo(x + unitW, baseY);
}

export default function ECGBackground() {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const offsetRef = useRef(0);
  const theme     = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = 0, H = 0;

    const resize = () => { W = canvas.offsetWidth; H = canvas.offsetHeight; canvas.width = W; canvas.height = H; };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const UNIT = 180, SPEED = 0.9;

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      offsetRef.current = (offsetRef.current + SPEED) % UNIT;

      const isDark = document.documentElement.classList.contains("dark");
      const rows = [
        { yRatio: 0.28, opacity: isDark ? 0.22 : 0.12, scale: 1.0 },
        { yRatio: 0.72, opacity: isDark ? 0.18 : 0.09, scale: 0.9 },
      ];

      rows.forEach(({ yRatio, opacity, scale }) => {
        const baseY = H * yRatio;
        const uW    = UNIT * scale;
        let   x     = -(UNIT) + (offsetRef.current * scale);

        ctx.beginPath();
        ctx.strokeStyle = `rgba(59,130,246,${opacity})`;
        ctx.lineWidth   = 1.8;
        ctx.lineJoin    = "round";
        ctx.lineCap     = "round";
        while (x < W + uW) { drawECGCycle(ctx, x, baseY, uW); x += uW; }
        ctx.stroke();
      });

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => { cancelAnimationFrame(animRef.current); ro.disconnect(); };
  }, [theme]); // re-init when theme changes

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 1 }}
      aria-hidden="true"
    />
  );
}
