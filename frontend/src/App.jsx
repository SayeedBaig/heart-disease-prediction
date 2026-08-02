import { Navigate, Route, Routes } from "react-router-dom";
import PatientPortal from "./pages/PatientPortal";
import DoctorPortal from "./pages/DoctorPortal";
import LandingPage from "./pages/LandingPage";
import ExploreCapabilities from "./pages/ExploreCapabilities";
import RoleSelection from "./pages/RoleSelection";
import DoctorLogin from "./pages/DoctorLogin";
import DoctorRegister from "./pages/DoctorRegister";
import PatientLogin from "./pages/PatientLogin";
import PatientSignup from "./pages/PatientSignup";
import DigitalTwin from "./pages/DigitalTwin";
import Reports from "./pages/Reports";
import Appointments from "./pages/Appointments";
import ResultsPage from "./pages/ResultsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/explore-capabilities" element={<ExploreCapabilities />} />
      <Route path="/get-started" element={<RoleSelection />} />
      <Route path="/role-selection" element={<RoleSelection />} />
      
      {/* Doctor Authentication Routes */}
      <Route path="/doctor/login" element={<DoctorLogin />} />
      <Route path="/doctor/register" element={<DoctorRegister />} />
      <Route path="/doctor/*" element={<DoctorPortal />} />

      {/* Patient Authentication & Workspace Routes */}
      <Route path="/patient/login" element={<PatientLogin />} />
      <Route path="/patient/signup" element={<PatientSignup />} />
      <Route path="/patient/register" element={<PatientSignup />} />
      <Route path="/patient/*" element={<PatientPortal />} />

      {/* Additional Features */}
      <Route path="/digital-twin" element={<DigitalTwin />} />
      <Route path="/reports" element={<Reports />} />
      <Route path="/appointments" element={<Appointments />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/diagnose" element={<Navigate to="/patient/intake" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;


