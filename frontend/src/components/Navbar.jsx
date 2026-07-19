import { Link, useLocation } from "react-router-dom";
import { Activity, Stethoscope } from "lucide-react";

function Navbar() {
  const location = useLocation();

  const links = [
    { path: "/", label: "Home" },
    { path: "/#about", label: "About" },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/60 bg-white/80 backdrop-blur-xl shadow-sm">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-8 py-4">

        {/* Logo */}

        <Link to="/" className="flex items-center gap-3 group">

          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg transition group-hover:scale-105">

            <Activity size={22} className="text-white" />

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

          {links.map((item) => (
            <Link
              key={item.label}
              to={item.path}
              className={`relative text-[15px] font-semibold transition-all duration-300 ${
                location.pathname === item.path
                  ? "text-blue-600"
                  : "text-slate-700 hover:text-blue-600"
              }`}
            >
              {item.label}

              <span
                className={`absolute left-0 -bottom-2 h-0.5 bg-blue-600 transition-all duration-300 ${
                  location.pathname === item.path
                    ? "w-full"
                    : "w-0 group-hover:w-full"
                }`}
              />
            </Link>
          ))}

        </nav>

        {/* Right Side */}

        <div className="flex items-center gap-4">

          <Link
            to="/doctor/login"
            className="flex items-center gap-2 rounded-xl border border-slate-300 px-5 py-2.5 font-semibold text-slate-700 transition-all duration-300 hover:border-blue-500 hover:text-blue-600 hover:bg-blue-50"
          >
            <Stethoscope size={18} />
            Doctor Login
          </Link>

          <Link
            to="/register"
            className="rounded-xl bg-gradient-to-r from-blue-600 to-cyan-500 px-6 py-3 font-semibold text-white shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-xl"
          >
            Start Diagnosis
          </Link>

        </div>

      </div>
    </header>
  );
}

export default Navbar;