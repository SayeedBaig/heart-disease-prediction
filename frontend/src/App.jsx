import { Navigate, Route, Routes } from "react-router-dom";
import PatientPortal from "./pages/PatientPortal";
import DoctorPortal from "./pages/DoctorPortal";

function App() {
  return (
    <Routes>
      <Route path="*" element={<PatientPortal />} />
      <Route path="/" element={<PatientPortal />} />
      <Route path="/get-started" element={<PatientPortal />} />
      <Route path="/patient/*" element={<PatientPortal />} />
      <Route path="/doctor/*" element={<DoctorPortal />} />
      <Route path="/diagnose" element={<Navigate to="/patient/intake" replace />} />
    </Routes>
  );
}

export default App;
