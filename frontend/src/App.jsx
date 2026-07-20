import { Routes, Route } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import DoctorLogin from "./pages/DoctorLogin";
import DoctorRegister from "./pages/DoctorRegister";
import DoctorDashboard from "./pages/DoctorDashboard";

import PatientRegister from "./pages/PatientRegister";

import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";
import Reports from "./pages/Reports";
import HistoryPage from "./pages/HistoryPage";
import Appointments from "./pages/Appointments";

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

</Routes>
  );
}

export default App;