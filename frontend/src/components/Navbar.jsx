import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Activity, ArrowLeft, Sun, Moon } from "lucide-react";

export default function Navbar({ onBack, backLabel, breadcrumb }) {
  const location = useLocation();
  const navigate = useNavigate();

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("cardio-theme") || "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("cardio-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const handleBack = () => {
    const path = location.pathname;
    let parentRoute = null;

    if (path === "/patient/digital-twin/report") {
      parentRoute = "/patient/digital-twin";
    } else if (path.startsWith("/patient/") && path !== "/patient/dashboard") {
      parentRoute = "/patient/dashboard";
    } else if (
      path.startsWith("/doctor/") &&
      !["/doctor/dashboard", "/doctor/login", "/doctor/register"].includes(path)
    ) {
      parentRoute = "/doctor/dashboard";
    } else if (["/digital-twin", "/reports", "/appointments"].includes(path)) {
      parentRoute = localStorage.getItem("access_token") ? "/patient/dashboard" : "/";
    }

    if (onBack) {
      onBack();
      if (parentRoute && path !== parentRoute) {
        navigate(parentRoute);
      }
      return;
    }

    if (parentRoute) {
      navigate(parentRoute);
    } else if (path !== "/") {
      navigate("/");
    }
  };

  const isHome = location.pathname === "/" || location.pathname === "";
  const isAuthOrDashboard = [
    "/patient/login",
    "/patient/signup",
    "/patient/register",
    "/patient/dashboard",
    "/doctor/login",
    "/doctor/register",
    "/doctor/dashboard",
  ].includes(location.pathname);

  return (
    <header className="sticky top-0 z-50 bg-[var(--card-bg)]/90 backdrop-blur-md border-b border-[var(--border-color)] transition-colors duration-200">
      <div className="cardio-container flex items-center justify-between gap-3 py-2.5">
        
        {/* Left Section: Back Button + Brand Logo + Breadcrumbs */}
        <div className="flex min-w-0 items-center gap-2 sm:gap-3">
          {!isHome && (
            <button
              onClick={handleBack}
              className="btn-secondary px-3 py-1.5 text-xs font-medium rounded-lg flex items-center gap-1.5 shrink-0"
              title="Go back"
            >
              <ArrowLeft size={14} />
              <span className="hidden sm:inline">{backLabel || "Back"}</span>
            </button>
          )}

          <Link to="/" className="flex shrink-0 items-center gap-2 sm:gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-[#39062B] text-white flex items-center justify-center shadow transition-all group-hover:shadow-md group-hover:scale-105">
              <Activity size={17} className="text-white" />
            </div>

            <div className="flex flex-col leading-none">
              <span className="text-sm font-bold tracking-tight text-[var(--text-primary)] font-display" style={{ fontFamily: "var(--font-display, 'Plus Jakarta Sans', sans-serif)" }}>
                CardioAI
              </span>
              <span className="text-[10px] text-[var(--text-muted)] font-medium hidden sm:block">
                Heart Intelligence Platform
              </span>
            </div>
          </Link>

          {breadcrumb && !isAuthOrDashboard && (
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-[var(--text-muted)] ml-2 pl-3 border-l border-[var(--border-color)]">
              <span className="opacity-50">/</span>
              <span className="font-medium text-[var(--text-secondary)]">{breadcrumb}</span>
            </div>
          )}
        </div>

        {/* Right Section */}
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          <nav className="hidden lg:flex items-center gap-5 text-xs font-medium text-[var(--text-secondary)]">
            <Link
              to="/"
              className={`transition-colors hover:text-[var(--accent-melanzane)] ${
                isHome ? "text-[var(--accent-melanzane)] font-semibold" : ""
              }`}
            >
              Home
            </Link>
          </nav>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            className="w-8 h-8 rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)] hover:border-[var(--accent-melanzane-border)] transition-all flex items-center justify-center"
            title={`Switch to ${theme === "light" ? "Dark" : "Light"} mode`}
            aria-label="Toggle theme"
          >
            {theme === "light" ? (
              <Moon size={15} className="text-[var(--text-secondary)]" />
            ) : (
              <Sun size={15} className="text-amber-400" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
}

