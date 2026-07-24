import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";

import Navbar from "../components/Navbar";
import PageBackground from "../components/PageBackground";
import api from "../services/api";

import {
  TrendingUp,
  Search,
  Activity,
} from "lucide-react";

export default function HistoryPage() {
  const patientId = localStorage.getItem("patient_id");

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(Boolean(patientId));

  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState("all");
  const [sortBy, setSortBy] = useState("newest");

  // -------------------------------
  // Helper Functions
  // -------------------------------

  const getRiskColor = (risk) => {
    switch (risk?.toLowerCase()) {
      case "low":
        return "bg-green-100 text-green-700";

      case "medium":
        return "bg-yellow-100 text-yellow-700";

      case "high":
        return "bg-red-100 text-red-700";

      default:
        return "bg-blue-100 text-blue-700";
    }
  };

  const formatDate = (date) => {
    return new Date(date).toLocaleString("en-IN", {
      dateStyle: "medium",
      timeStyle: "short",
    });
  };

  // -------------------------------
  // Fetch History
  // -------------------------------

  useEffect(() => {
  if (!patientId) return;

  const fetchHistory = async () => {
    try {
      const res = await api.get(`/patients/${patientId}/predictions`);


setHistory(res.data.predictions || []);
    } catch (error) {
      console.error("History Fetch Error:", error);
    } finally {
      setLoading(false);
    }
  };

  fetchHistory();
}, [patientId]);

  // -------------------------------
  // Statistics
  // -------------------------------

  const stats = useMemo(() => {
    return {
      total: history.length,

      low: history.filter(
        (item) => item.risk_level?.toLowerCase() === "low"
      ).length,

      medium: history.filter(
        (item) => item.risk_level?.toLowerCase() === "medium"
      ).length,

      high: history.filter(
        (item) => item.risk_level?.toLowerCase() === "high"
      ).length,

      average:
        history.length > 0
          ? (
              history.reduce(
                (sum, item) =>
                  sum + Number(item.risk_percentage || 0),
                0
              ) / history.length
            ).toFixed(1)
          : "0",
    };
  }, [history]);

  // -------------------------------
  // Search + Filter + Sort
  // -------------------------------

  const filteredHistory = useMemo(() => {
    let data = [...history];

    // Search by Prediction ID OR Risk Level
    if (searchTerm.trim()) {
      const search = searchTerm.toLowerCase();

      data = data.filter(
        (item) =>
          item.prediction_id.toString().includes(search) ||
          item.risk_level?.toLowerCase().includes(search)
      );
    }

    // Risk Filter

    if (riskFilter !== "all") {
      data = data.filter(
        (item) =>
          item.risk_level?.toLowerCase() ===
          riskFilter.toLowerCase()
      );
    }

    // Sorting

    switch (sortBy) {
      case "oldest":
        data.sort(
          (a, b) =>
            new Date(a.created_at) -
            new Date(b.created_at)
        );
        break;

      case "highest":
        data.sort(
          (a, b) =>
            Number(b.risk_percentage) -
            Number(a.risk_percentage)
        );
        break;

      case "lowest":
        data.sort(
          (a, b) =>
            Number(a.risk_percentage) -
            Number(b.risk_percentage)
        );
        break;

      default:
        data.sort(
          (a, b) =>
            new Date(b.created_at) -
            new Date(a.created_at)
        );
    }

    return data;
  }, [history, searchTerm, riskFilter, sortBy]);

  // -------------------------------
  // Patient Not Found
  // -------------------------------

  if (!patientId) {
    return (
      <>
        <Navbar />

        <div className="relative min-h-screen">
          <PageBackground />

          <div className="relative z-10 flex justify-center items-center h-screen">

            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-3xl shadow-xl p-10 text-center max-w-md"
            >
              <Activity
                size={60}
                className="mx-auto text-blue-600"
              />

              <h2 className="text-3xl font-bold mt-5">
                Patient Not Found
              </h2>

              <p className="text-gray-500 mt-3">
                Please register a patient before viewing
                prediction history.
              </p>

            </motion.div>

          </div>
        </div>
      </>
    );
  }

  // -------------------------------
  // Main UI
  // -------------------------------

  return (
    <>
      <Navbar />

      <div className="relative min-h-screen">
        <PageBackground />

        <div className="relative z-10 max-w-7xl mx-auto px-8 py-10">

          <motion.div
            initial={{ opacity: 0, y: -25 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-5xl font-extrabold text-slate-900">
              Prediction History
            </h1>

            <p className="text-lg text-slate-500 mt-3">
              Review all previous AI-generated cardiovascular
              predictions and monitor patient risk trends.
            </p>

          </motion.div>

          {loading ? (

            <div className="flex justify-center items-center py-24">

              <div className="animate-spin rounded-full h-14 w-14 border-4 border-blue-600 border-t-transparent"></div>

            </div>

          ) : (<div>

  {/* ===========================
      Statistics Cards
  =========================== */}

  <div className="grid md:grid-cols-5 gap-6 mt-10">

    {/* Total Predictions */}

    <motion.div
      transition={{ duration: 0.2 }}
      className="bg-white rounded-3xl shadow-lg p-6"
    >
      <p className="text-gray-500">
        Total Predictions
      </p>

      <h2 className="text-4xl font-bold text-blue-700 mt-3">
        {stats.total}
      </h2>
    </motion.div>

    {/* Low Risk */}

    <motion.div
      transition={{ duration: 0.2 }}
      className="bg-green-50 rounded-3xl shadow-lg p-6"
    >
      <p className="text-green-700">
        Low Risk
      </p>

      <h2 className="text-4xl font-bold text-green-600 mt-3">
        {stats.low}
      </h2>
    </motion.div>

    {/* Medium Risk */}

    <motion.div
      transition={{ duration: 0.2 }}
      className="bg-yellow-50 rounded-3xl shadow-lg p-6"
    >
      <p className="text-yellow-700">
        Medium Risk
      </p>

      <h2 className="text-4xl font-bold text-yellow-600 mt-3">
        {stats.medium}
      </h2>
    </motion.div>

    {/* High Risk */}

    <motion.div
      transition={{ duration: 0.2 }}
      className="bg-red-50 rounded-3xl shadow-lg p-6"
    >
      <p className="text-red-700">
        High Risk
      </p>

      <h2 className="text-4xl font-bold text-red-600 mt-3">
        {stats.high}
      </h2>
    </motion.div>

    {/* Average Risk */}

    <motion.div
      transition={{ duration: 0.2 }}
      className="bg-indigo-50 rounded-3xl shadow-lg p-6"
    >
      <div className="flex items-center gap-2">

        <TrendingUp
          size={22}
          className="text-indigo-700"
        />

        <p className="text-indigo-700">
          Average Risk
        </p>

      </div>

      <h2 className="text-4xl font-bold text-indigo-700 mt-3">
        {stats.average}%
      </h2>
    </motion.div>

  </div>

  {/* ===========================
      Search & Filter
  =========================== */}

  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.2 }}
    className="bg-white rounded-3xl shadow-lg p-6 mt-8"
  >

    <div className="grid md:grid-cols-3 gap-4">

      {/* Search */}

      <div className="relative">

        <Search
          size={20}
          className="absolute left-4 top-3.5 text-gray-400"
        />

        <input
          type="text"
          placeholder="Search Prediction ID or Risk..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-12 pr-4 py-3 border rounded-xl outline-none focus:ring-2 focus:ring-blue-500"
        />

      </div>

      {/* Risk Filter */}

      <select
        value={riskFilter}
        onChange={(e) => setRiskFilter(e.target.value)}
        className="border rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500"
      >

        <option value="all">
          All Risk Levels
        </option>

        <option value="low">
          Low Risk
        </option>

        <option value="medium">
          Medium Risk
        </option>

        <option value="high">
          High Risk
        </option>

      </select>

      {/* Sort */}

      <select
        value={sortBy}
        onChange={(e) => setSortBy(e.target.value)}
        className="border rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500"
      >

        <option value="newest">
          Newest First
        </option>

        <option value="oldest">
          Oldest First
        </option>

        <option value="highest">
          Highest Risk
        </option>

        <option value="lowest">
          Lowest Risk
        </option>

      </select>

    </div>

  </motion.div>

  {/* ===========================
      Prediction List
  =========================== */}

  <div className="mt-8">

    {filteredHistory.length === 0 ? (

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="bg-white rounded-3xl shadow-lg p-12 text-center"
      >

        <Activity
          size={70}
          className="mx-auto text-blue-500"
        />

        <h2 className="text-3xl font-bold mt-5">
          No Predictions Found
        </h2>

        <p className="text-gray-500 mt-3">
          No prediction records match your search or filters.
        </p>

      </motion.div>

    ) : (

      <div className="space-y-6">    {filteredHistory.map((item, index) => (

          <motion.div
            key={item.prediction_id}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.4,
              delay: index * 0.08,
            }}
            className="bg-white rounded-3xl shadow-lg border border-gray-100 p-8 transition-all"
          >

            {/* Header */}

            <div className="flex justify-between items-start flex-wrap gap-4">

              <div>

                <h2 className="text-2xl font-bold text-slate-800">
                 Prediction #{item.prediction_id}
                </h2>

                <p className="text-gray-500 mt-2">
                  {formatDate(item.created_at)}
                </p>

              </div>

              <span
                className={`px-4 py-2 rounded-full font-semibold ${getRiskColor(
                  item.risk_level
                )}`}
              >
                {item.risk_level}
              </span>

            </div>
             {/* Analysis Summary */}

<div className="grid md:grid-cols-3 gap-4 mt-8">

  <div className="bg-slate-50 rounded-2xl p-4">
    <p className="text-xs uppercase tracking-wide text-gray-500">Clinical</p>
    <h4 className="text-xl font-bold mt-2 text-slate-800">
     {item.clinical_level || "N/A"}
    </h4>
  </div>

  <div className="bg-slate-50 rounded-2xl p-4">
    <p className="text-xs uppercase tracking-wide text-gray-500">ECG</p>
    <h4 className="text-xl font-bold mt-2 text-slate-800">
      {item.ecg_level || "N/A"}
    </h4>
  </div>

  <div className="bg-slate-50 rounded-2xl p-4">
    <p className="text-xs uppercase tracking-wide text-gray-500">Echo</p>
    <h4 className="text-xl font-bold mt-2 text-slate-800">
      {item.echo_level || "N/A"}
    </h4>
  </div>

</div>

            {/* Risk Percentage */}

            <div className="mt-8">

              <div className="flex justify-between items-center">

                <h3 className="text-lg font-semibold text-slate-700">
                  Risk Percentage
                </h3>

                <span className="text-3xl font-bold text-blue-700">
                  {Number(item.risk_percentage).toFixed(1)}%
                </span>

              </div>
             

              {/* Progress Bar */}

              <div className="w-full h-3 bg-gray-200 rounded-full mt-4 overflow-hidden">

                <motion.div
                  initial={{ width: 0 }}
                  animate={{
                    width: `${Number(item.risk_percentage)}%`,
                  }}
                  transition={{
                    duration: 1,
                  }}
                  className={`h-full rounded-full ${
                    item.risk_level?.toLowerCase() === "high"
                      ? "bg-red-500"
                      : item.risk_level?.toLowerCase() === "medium"
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                />

              </div>

            </div>

            {/* Optional Explanation */}

           {item.explanation && (
  <div className="mt-8">

    <h4 className="font-semibold text-slate-700 mb-2">
      AI Explanation
    </h4>

    <p className="text-gray-600 leading-7">
      {item.explanation}
    </p>

    {/* Action Buttons */}

    <div className="mt-8 flex flex-wrap gap-4">

      <button className="px-5 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700">
        View Report
      </button>

      <button className="px-5 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700">
        Download PDF
      </button>

      <button
        onClick={() => window.print()}
        className="px-5 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-800"
      >
        Print
      </button>

    </div>

  </div>
)}

          </motion.div>

        ))}

      </div>

    )}

  </div>

</div>

)}

        </div>
      </div>
    </>
  );
}
