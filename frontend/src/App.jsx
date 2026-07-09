import { Routes, Route } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";
import Reports from "./pages/Reports";
import RegisterPage from "./pages/RegisterPage";
import Dashboard from "./pages/Dashboard";
import HistoryPage from "./pages/HistoryPage";
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
</Routes>
  );
}

export default App;