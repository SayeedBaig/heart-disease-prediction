import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  UserPlus,
  HeartPulse,
  ClipboardList,
  FileText,
  Brain,
  CalendarDays,
} from "lucide-react";

const menuItems = [
  {
    name: "Dashboard",
    path: "/doctor/dashboard",
    icon: LayoutDashboard,
  },
  {
    name: "Register Patient",
    path: "/patients/register",
    icon: UserPlus,
  },
  {
    name: "Diagnose Patient",
    path: "/diagnose",
    icon: HeartPulse,
  },
  {
    name: "Patient History",
    path: "/history",
    icon: ClipboardList,
  },
  {
    name: "Reports",
    path: "/reports",
    icon: FileText,
  },
  {
    name: "Appointment Management",
    path: "/appointment-management",
    icon: CalendarDays,
  },
  {
    name:"AI Health Assisant",
    path: "/ai-health-assistant",
    icon:Brain,
  }
  
];

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-white border-r shadow-sm">
      <div className="p-6 border-b">
        <h1 className="text-3xl font-bold text-blue-600">
          CardioAI
        </h1>

        <p className="text-sm text-gray-500 mt-1">
          Doctor Portal
        </p>
      </div>

      <nav className="p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition ${
                  isActive
                    ? "bg-blue-100 text-blue-600 font-semibold"
                    : "text-gray-700 hover:bg-gray-100"
                }`
              }
            >
              <Icon size={20} />
              {item.name}
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}