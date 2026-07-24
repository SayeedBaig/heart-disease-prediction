import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  CalendarDays,
  FileText,
  HeartPulse,
  Bot,
  User,
  Activity,
} from "lucide-react";

function PatientDashboard() {
  const navigate = useNavigate();
  useEffect(() => {
  const token = localStorage.getItem("access_token");

  if (!token) {
    navigate("/patient/login", { replace: true });
  }
}, [navigate]);

  const patient =
    JSON.parse(localStorage.getItem("patient")) || {};

  const prediction =
    JSON.parse(localStorage.getItem("prediction_result")) || {};

  const cards = [
    {
      title: "My Appointments",
      description: "Manage appointments",
      icon: CalendarDays,
      route: "/patient/appointments",
      color: "bg-blue-500",
    },
    {
      title: "My Reports",
      description: "View diagnosis reports",
      icon: FileText,
      route: "/patient/reports",
      color: "bg-green-500",
    },
    {
      title: "Health Insights",
      description: "Heart health summary",
      icon: HeartPulse,
      route: "/patient/insights",
      color: "bg-red-500",
    },
    {
      title: "AI Health Assistant",
      description: "Ask medical questions",
      icon: Bot,
      route: "/patient/assistant",
      color: "bg-purple-500",
    },
    {
      title: "My Profile",
      description: "Manage profile",
      icon: User,
      route: "/patient/profile",
      color: "bg-orange-500",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-100 p-8">

      {/* Welcome */}

      <div className="bg-gradient-to-r from-blue-600 to-cyan-500 rounded-3xl text-white p-8 mb-8 shadow-lg">

        <h1 className="text-4xl font-bold">
          Welcome, {patient.full_name || "Patient"}
        </h1>

        <p className="mt-3 text-lg text-blue-100">
          Welcome back to CardioAI Patient Portal
        </p>

      </div>

      {/* Statistics */}

      <div className="grid md:grid-cols-4 gap-6 mb-8">

        <div className="bg-white rounded-2xl shadow-md p-6">
          <p className="text-slate-500">Risk Level</p>

          <h2 className="text-3xl font-bold text-red-600 mt-3">
            {prediction?.fusion?.final_level || "N/A"}
          </h2>
        </div>

        <div className="bg-white rounded-2xl shadow-md p-6">
          <p className="text-slate-500">Risk Percentage</p>

          <h2 className="text-3xl font-bold text-blue-600 mt-3">
            {prediction?.fusion?.risk_percentage || 0}%
          </h2>
        </div>

        <div className="bg-white rounded-2xl shadow-md p-6">
          <p className="text-slate-500">Health Status</p>

          <h2 className="text-3xl font-bold text-green-600 mt-3">
            Stable
          </h2>
        </div>

        <div className="bg-white rounded-2xl shadow-md p-6">
          <p className="text-slate-500">AI Analysis</p>

          <h2 className="text-3xl font-bold text-purple-600 mt-3">
            Complete
          </h2>
        </div>

      </div>

      {/* Quick Actions */}

      <h2 className="text-2xl font-bold mb-6">
        Quick Actions
      </h2>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">

        {cards.map((card) => {
          const Icon = card.icon;

          return (
            <div
              key={card.title}
              onClick={() => navigate(card.route)}
              className="bg-white rounded-2xl shadow-md p-6 transition cursor-pointer"
            >
              <div
                className={`w-14 h-14 ${card.color} rounded-xl flex items-center justify-center mb-5`}
              >
                <Icon className="text-white" size={28} />
              </div>

              <h3 className="text-xl font-bold">
                {card.title}
              </h3>

              <p className="text-slate-500 mt-2">
                {card.description}
              </p>
            </div>
          );
        })}

      </div>

      {/* Recent Activity */}

      <div className="bg-white rounded-2xl shadow-md p-6 mt-10">

        <div className="flex items-center gap-3 mb-5">
          <Activity className="text-blue-600" />
          <h2 className="text-2xl font-bold">
            Recent Activity
          </h2>
        </div>

        <ul className="space-y-3 text-slate-600">
          <li>AI Heart Disease Prediction Completed</li>
          <li>Latest Medical Report Generated</li>
          <li>Health Insights Updated</li>
          <li>AI Assistant Available for Questions</li>
        </ul>

      </div>

    </div>
  );
}

export default PatientDashboard;
