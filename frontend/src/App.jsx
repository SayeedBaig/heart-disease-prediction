import { Routes, Route } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";
import Reports from "./pages/Reports";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/diagnose" element={<DiagnosePage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/reports" element={<Reports />} />
    </Routes>
  );
}

export default App;