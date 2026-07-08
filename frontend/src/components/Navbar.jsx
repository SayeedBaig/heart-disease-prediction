import { Link, useLocation } from "react-router-dom";
import { Activity } from "lucide-react";

function Navbar() {
  const location = useLocation();

  const navItem = (path, label) => (
    <Link
      to={path}
      className={`
        relative font-medium transition-all duration-300
        ${
          location.pathname === path
            ? "text-blue-600"
            : "text-slate-700 hover:text-blue-600"
        }
        after:absolute after:left-0 after:-bottom-1
        after:h-0.5 after:w-0 after:bg-blue-600
        after:transition-all after:duration-300
        hover:after:w-full
      `}
    >
      {label}
    </Link>
  );

  return (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 h-18 flex items-center justify-between">

        {/* Logo */}

        <Link to="/" className="flex items-center gap-3 group">

          <div
            className="
              w-11 h-11
              rounded-xl
              bg-blue-600
              flex items-center justify-center
              text-white
              transition-transform duration-300
              group-hover:scale-110
            "
          >
            <Activity size={22} />
          </div>

          <div>
            <h1 className="text-2xl font-bold text-slate-900">
              CardioAI
            </h1>

            <p className="text-xs text-slate-500">
              Heart Disease Prediction
            </p>
          </div>

        </Link>

        {/* Navigation */}

        <nav className="hidden md:flex gap-10">

          {navItem("/", "Home")}
          {navItem("/register", "Diagnose")}
          {navItem("/reports", "Reports")}

        </nav>

        {/* CTA Button */}

        <Link
  to="/register"
          className="
            rounded-xl
            bg-blue-600
            px-6
            py-3
            text-white
            font-semibold
            shadow-lg
            transition-all
            duration-300
            hover:bg-blue-700
            hover:-translate-y-1
            hover:shadow-xl
          "
        >
          Start Diagnosis
        </Link>

      </div>
    </header>
  );
}

export default Navbar;