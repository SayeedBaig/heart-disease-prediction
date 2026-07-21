import { Link, useLocation, useNavigate } from "react-router-dom";
import { Activity, Stethoscope } from "lucide-react";

function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();

  const scrollToSection = (sectionId) => {
    // If already on the landing page
    if (location.pathname === "/") {
      const section = document.getElementById(sectionId);

      if (section) {
        section.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    } else {
      // Navigate to home first, then scroll
      navigate("/");

      setTimeout(() => {
        const section = document.getElementById(sectionId);

        if (section) {
          section.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }, 200);
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/60 bg-white/80 backdrop-blur-xl shadow-sm">

      <div className="max-w-7xl mx-auto flex items-center justify-between px-8 py-4">

        {/* Logo */}

        <Link
          to="/"
          className="flex items-center gap-3 group"
        >
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg transition group-hover:scale-105">
            <Activity
              size={22}
              className="text-white"
            />
          </div>

          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
              CardioAI
            </h1>

            <p className="text-xs text-slate-500">
              AI Heart Disease Prediction Platform
            </p>
          </div>
        </Link>

        {/* Center Navigation */}

        <nav className="hidden lg:flex items-center gap-10">

          <button
            onClick={() => navigate("/")}
            className="relative font-semibold text-slate-700 hover:text-blue-600 transition"
          >
            Home
          </button>

          <button
            onClick={() => scrollToSection("workflow")}
            className="relative font-semibold text-slate-700 hover:text-blue-600 transition"
          >
            Workflow
          </button>

          <button
            onClick={() => scrollToSection("features")}
            className="relative font-semibold text-slate-700 hover:text-blue-600 transition"
          >
            Features
          </button>

          <button
            onClick={() => scrollToSection("about")}
            className="relative font-semibold text-slate-700 hover:text-blue-600 transition"
          >
            About
          </button>

        </nav>

        {/* Right Side */}

        <div className="flex items-center gap-4">

  <Link
    to="/patient/login"
    className="rounded-xl bg-gradient-to-r from-green-600 to-emerald-500 px-6 py-3 font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-xl"
  >
    Patient Portal
  </Link>

  <Link
    to="/doctor/login"
    className="flex items-center gap-2 rounded-xl border border-slate-300 px-5 py-2.5 font-semibold text-slate-700 transition-all duration-300 hover:border-blue-500 hover:text-blue-600 hover:bg-blue-50"
  >
    <Stethoscope size={18} />
    Doctor Portal
  </Link>

</div>
      </div>

    </header>
  );
}

export default Navbar;