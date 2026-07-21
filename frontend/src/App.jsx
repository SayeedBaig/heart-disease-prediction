import { Routes, Route } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import DoctorLogin from "./pages/DoctorLogin";
import DoctorRegister from "./pages/DoctorRegister";
import DoctorDashboard from "./pages/DoctorDashboard";
import DoctorNotes from "./pages/DoctorNotes";
import AppointmentManagement from "./pages/AppointmentManagement";
import AIHealthAssistant from "./pages/AIHealthAssistant";
import PatientRegister from "./pages/PatientRegister";

import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";
import Reports from "./pages/Reports";
import HistoryPage from "./pages/HistoryPage";
import Appointments from "./pages/Appointments";
import PatientLogin from "./pages/PatientLogin";
import PatientDashboard from "./pages/PatientDashboard";
import PatientAppointments from "./pages/PatientAppointments";
import PatientReports from "./pages/PatientReports";
import HealthInsights from "./pages/HealthInsights";
import PatientAssistant from "./pages/PatientAssistant";
import PatientProfile from "./pages/PatientProfile";
function App() {
  return (
    <Routes>

<Route path="/" element={<LandingPage />} />

<Route path="/doctor/login" element={<DoctorLogin />} />

<Route path="/doctor/register" element={<DoctorRegister />} />

<Route path="/doctor/dashboard" element={<DoctorDashboard />} />

<Route path="/patients/register" element={<PatientRegister />} />

<Route path="/diagnose" element={<DiagnosePage />} />

<Route path="/results" element={<ResultsPage />} />

<Route path="/reports" element={<Reports />} />

<Route path="/history" element={<HistoryPage />} />

<Route path="/appointments" element={<Appointments />} />

<Route path="/doctor-notes" element={<DoctorNotes />} />


<Route path="/appointment-management" element={<AppointmentManagement />}/>

<Route path="/ai-health-assistant" element={<AIHealthAssistant />}/>

<Route path="/patient/login" element={<PatientLogin />} />

<Route path="/patient/dashboard" element={<PatientDashboard />} />

<Route path="/patient/appointments" element={<PatientAppointments />} />

<Route path="/patient/reports" element={<PatientReports />} />

<Route path="/patient/insights" element={<HealthInsights />} />

<Route path="/patient/assistant" element={<PatientAssistant />} />

<Route path="/patient/profile" element={<PatientProfile />} />

</Routes>
  );
}

export default App;