import { Routes, Route } from "react-router-dom";
import DoctorLogin from "./pages/DoctorLogin";
import LandingPage from "./pages/LandingPage";
import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";
import Reports from "./pages/Reports";
import RegisterPage from "./pages/RegisterPage";
import Dashboard from "./pages/Dashboard";
import HistoryPage from "./pages/HistoryPage";
import DoctorDashboard from "./pages/DoctorDashboard";
import Patients from "./pages/Patients";
import Appointments from "./pages/Appointments";

function App() {
  return (
    <Routes>
  <Route path="/" element={<LandingPage />} />
<Route path="/register" element={<RegisterPage />} />
<Route path="/dashboard" element={<Dashboard />} />
<Route path="/diagnose" element={<DiagnosePage />} />
<Route path="/results" element={<ResultsPage />} />
<Route path="/reports" element={<Reports />} />
<Route path="/history" element={<HistoryPage />} />
<Route path="/doctor/login" element={<DoctorLogin />} />
<Route path="/doctor/dashboard" element={<DoctorDashboard />}/>
<Route path="/doctor/patients"element={<Patients />}/>
<Route path="/appointments" element={<Appointments />}/>
</Routes>
  );
}

export default App;