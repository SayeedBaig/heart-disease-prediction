import { useEffect, useRef } from "react";

export default function BeatingHeart3D() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let animationFrameId;
    let width = (canvas.width = canvas.parentElement.clientWidth || 360);
    let height = (canvas.height = canvas.parentElement.clientHeight || 360);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };

    window.addEventListener("resize", handleResize);

    // Generate 3D point cloud for heart shape
    const points = [];
    const numPoints = 650;

    for (let i = 0; i < numPoints; i++) {
      // Heart 3D parametric equations
      const t = Math.PI * (2 * Math.random() - 1);
      
      const x = 16 * Math.pow(Math.sin(t), 3);
      const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
      const z = (Math.random() - 0.5) * 12;

      points.push({ x, y, z });
    }

    let angleY = 0;
    let time = 0;

    const render = () => {
      time += 0.04;
      angleY += 0.008;

      // Heartbeat pulse rhythm equation (systole & diastole pulse)
      const beatCycle = time % (Math.PI * 2);
      const pulseScale = 1 + 0.12 * Math.sin(beatCycle) * Math.exp(-Math.cos(beatCycle * 2) * 0.5);

      ctx.clearRect(0, 0, width, height);

      // Center offset
      const cx = width / 2;
      const cy = height / 2 - 10;
      const baseScale = Math.min(width, height) / 48;

      // Background subtle radial glow
      const glowGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, width / 2);
      const isDark = document.documentElement.dataset.theme === "dark";
      glowGrad.addColorStop(0, isDark ? "rgba(57, 6, 43, 0.4)" : "rgba(57, 6, 43, 0.12)");
      glowGrad.addColorStop(1, "transparent");
      ctx.fillStyle = glowGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, width / 2, 0, Math.PI * 2);
      ctx.fill();

      // Sort points by Z distance for proper depth rendering
      const rotatedPoints = points.map((p) => {
        // Rotate around Y axis
        const cosY = Math.cos(angleY);
        const sinY = Math.sin(angleY);
        
        const rx = p.x * cosY - p.z * sinY;
        const rz = p.x * sinY + p.z * cosY;

        // Apply heartbeat scale factor
        const sx = rx * baseScale * pulseScale;
        const sy = p.y * baseScale * pulseScale;
        const sz = rz * baseScale * pulseScale;

        return { x: cx + sx, y: cy + sy, z: sz, rz };
      });

      rotatedPoints.sort((a, b) => b.rz - a.rz);

      // Render heart nodes & subtle cardiac wireframe links
      for (let i = 0; i < rotatedPoints.length; i++) {
        const pt = rotatedPoints[i];
        const depthAlpha = Math.max(0.25, (pt.rz + 25) / 50);
        const ptRadius = Math.max(1.2, (pt.rz + 25) / 14);

        // Render point
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, ptRadius, 0, Math.PI * 2);
        
        const melanzaneColor = isDark 
          ? `rgba(230, 115, 200, ${depthAlpha})`
          : `rgba(57, 6, 43, ${depthAlpha * 0.95})`;

        ctx.fillStyle = melanzaneColor;
        ctx.shadowBlur = ptRadius * 3;
        ctx.shadowColor = isDark ? "#e673c8" : "#39062b";
        ctx.fill();

        // Connect nearby points to form cardiac mesh lines
        if (i % 3 === 0) {
          const nextPt = rotatedPoints[(i + 1) % rotatedPoints.length];
          const dist = Math.hypot(pt.x - nextPt.x, pt.y - nextPt.y);
          if (dist < 45) {
            ctx.beginPath();
            ctx.moveTo(pt.x, pt.y);
            ctx.lineTo(nextPt.x, nextPt.y);
            ctx.strokeStyle = isDark
              ? `rgba(230, 115, 200, ${depthAlpha * 0.25})`
              : `rgba(57, 6, 43, ${depthAlpha * 0.18})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      // Reset shadow blur
      ctx.shadowBlur = 0;

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className="relative w-full h-[360px] md:h-[420px] flex items-center justify-center">
      <canvas ref={canvasRef} className="w-full h-full cursor-pointer" />
      <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-white/70 dark:bg-black/70 backdrop-blur-md px-3 py-1 rounded-full text-[11px] font-semibold text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800">
        3D Cardiac Pulse Simulation
      </div>
    </div>
  );
}
