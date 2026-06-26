import { Routes, Route } from "react-router-dom";

import LandingPage from "./pages/LandingPage";
import DiagnosePage from "./pages/DiagnosePage";
import ResultsPage from "./pages/ResultsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/diagnose" element={<DiagnosePage />} />
      <Route path="/results" element={<ResultsPage />} />
    </Routes>
  );
}

export default App;