/**
 * useTheme — watches document.documentElement for .dark class changes.
 * Returns "dark" | "light" reactively so components can adapt.
 */
import { useState, useEffect } from "react";

export function useTheme() {
  const [isDark, setIsDark] = useState(
    () => document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    const root = document.documentElement;

    const observer = new MutationObserver(() => {
      setIsDark(root.classList.contains("dark"));
    });

    observer.observe(root, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, []);

  return isDark ? "dark" : "light";
}
