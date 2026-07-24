import { useNavigate } from "react-router-dom";
import { Stethoscope, UserRound } from "lucide-react";

function RoleSelection() {
  const navigate = useNavigate();

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-cyan-100 flex items-center justify-center p-6">
      <section className="w-full max-w-3xl rounded-3xl bg-white p-10 shadow-2xl">
        <p className="text-center font-semibold text-blue-600">CardioAI</p>
        <h1 className="mt-2 text-center text-4xl font-bold text-slate-900">
          Continue to sign in or register
        </h1>
        <p className="mt-3 text-center text-slate-500">
          Choose the portal you want to access.
        </p>

        <div className="mt-10 grid gap-6 md:grid-cols-2">
          <button
            onClick={() => navigate("/doctor/login")}
            className="rounded-2xl border border-blue-200 bg-blue-50 p-8 text-left transition hover:border-blue-500"
          >
            <Stethoscope className="text-blue-600" size={40} />
            <h2 className="mt-5 text-2xl font-bold text-slate-900">Doctor Portal</h2>
            <p className="mt-2 text-slate-600">
              Manage patients, appointments, diagnoses, reports, and history.
            </p>
          </button>

          <button
            onClick={() => navigate("/patient/login")}
            className="rounded-2xl border border-emerald-200 bg-emerald-50 p-8 text-left transition hover:border-emerald-500"
          >
            <UserRound className="text-emerald-600" size={40} />
            <h2 className="mt-5 text-2xl font-bold text-slate-900">Patient Portal</h2>
            <p className="mt-2 text-slate-600">
              Track appointments, reports, health insights, and your profile.
            </p>
          </button>
        </div>
      </section>
    </main>
  );
}

export default RoleSelection;
