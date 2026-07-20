import { useNavigate } from "react-router-dom";

import Sidebar from "../components/dashboard/Sidebar";
import Header from "../components/dashboard/Header";
import StatCard from "../components/dashboard/StatCard";
import QuickActionCard from "../components/dashboard/QuickActionCard";
import RecentActivity from "../components/dashboard/RecentActivity";

import {
  Users,
  HeartPulse,
  FileText,
  Brain,
  UserPlus,
  ClipboardList,
} from "lucide-react";

function DoctorDashboard() {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen bg-slate-100">

      {/* Sidebar */}

      <Sidebar />

      {/* Main Content */}

      <div className="flex-1 flex flex-col">

        {/* Header */}

        <Header />

        {/* Dashboard Content */}

        <main className="flex-1 p-10">

          {/* Statistics */}

          <section>

            <h2 className="text-2xl font-bold text-slate-800 mb-6">
              Dashboard Overview
            </h2>

            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">

              <StatCard
                title="Registered Patients"
                value="128"
                subtitle="Total Patients"
                icon={Users}
                color="blue"
              />

              <StatCard
                title="Today's Diagnoses"
                value="24"
                subtitle="Completed Today"
                icon={HeartPulse}
                color="red"
              />

              <StatCard
                title="Reports Generated"
                value="61"
                subtitle="Available Reports"
                icon={FileText}
                color="green"
              />

              <StatCard
                title="AI Confidence"
                value="96.8%"
                subtitle="Prediction Accuracy"
                icon={Brain}
                color="purple"
              />

            </div>

          </section>

          {/* Quick Actions */}

          <section className="mt-12">

            <div className="flex items-center justify-between mb-6">

              <h2 className="text-2xl font-bold text-slate-800">
                Quick Actions
              </h2>

              <p className="text-slate-500">
                Frequently used doctor tools
              </p>

            </div>

            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">

              <QuickActionCard
  title="Register Patient"
  description="Create a new patient profile before diagnosis."
  icon={UserPlus}
  color="blue"
  onClick={() => navigate("/patients/register")}
/>

              <QuickActionCard
                title="Diagnose Patient"
                description="Start AI-powered heart disease prediction."
                icon={HeartPulse}
                color="red"
                onClick={() => navigate("/diagnose")}
              />

              <QuickActionCard
                title="Patient History"
                description="Review previous diagnosis records."
                icon={ClipboardList}
                color="green"
                onClick={() => navigate("/history")}
              />

              <QuickActionCard
                title="Reports"
                description="Download and manage generated reports."
                icon={FileText}
                color="purple"
                onClick={() => navigate("/reports")}
              />

            </div>

          </section>

          {/* Recent Activity */}

          <section className="mt-12">
                        <RecentActivity />

          </section>

          {/* Footer */}

          <footer className="mt-12">

            <div className="rounded-2xl bg-gradient-to-r from-blue-600 to-cyan-500 p-8 text-white shadow-lg">

              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">

                <div>

                  <h2 className="text-2xl font-bold">
                    CardioAI Doctor Portal
                  </h2>

                  <p className="mt-2 text-blue-100">
                    AI-powered cardiovascular disease prediction platform
                    designed to assist doctors with faster and smarter
                    clinical decision making.
                  </p>

                </div>

                <div className="grid grid-cols-2 gap-8 text-center">

                  <div>

                    <h3 className="text-3xl font-bold">
                      96.8%
                    </h3>

                    <p className="text-blue-100">
                      AI Accuracy
                    </p>

                  </div>

                  <div>

                    <h3 className="text-3xl font-bold">
                      24/7
                    </h3>

                    <p className="text-blue-100">
                      System Available
                    </p>

                  </div>

                </div>

              </div>

            </div>

          </footer>

        </main>

      </div>

    </div>
  );
}

export default DoctorDashboard;