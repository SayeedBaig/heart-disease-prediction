import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Activity, ShieldCheck, ArrowRight } from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";
import { emailPatientReport, getPatientReport, viewPatientReportPdf } from "../services/portalService";

function PatientReports() {
  const navigate = useNavigate();
  const patient = JSON.parse(localStorage.getItem("cardio-patient") || "{}");
  const [loading, setLoading] = useState(Boolean(patient.patient_id));
  const [error, setError] = useState("");
  const [reports, setReports] = useState([]);
  const [mailingReportId, setMailingReportId] = useState(null);
  const [mailStatus, setMailStatus] = useState("");
  const [summarizingReportId, setSummarizingReportId] = useState(null);
  const [reportSummary, setReportSummary] = useState("");

  useEffect(() => {
    if (!patient.patient_id) {
      return undefined;
    }

    const loadReports = async () => {
      try {
        const response = await api.get(
          `/patients/${patient.patient_id}/predictions`
        );
        setReports(response.data.predictions || []);
        setError("");
      } catch (requestError) {
        setError(
          requestError.response?.data?.detail || "Unable to load your reports."
        );
      } finally {
        setLoading(false);
      }
    };

    loadReports();
    const intervalId = window.setInterval(loadReports, 15000);
    return () => window.clearInterval(intervalId);
  }, [patient.patient_id]);

  const displayError = patient.patient_id
    ? error
    : "Please sign in to view your reports.";

  return (
    <div className="cardio-shell">
      <Navbar breadcrumb="Health Reports" />

      <main className="cardio-container py-8 flex-1 w-full" style={{ maxWidth: "72rem", marginInline: "auto" }}>
        {/* Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)] gap-4">
          <div>
            <span className="caption-small text-[var(--accent-primary)] uppercase font-bold tracking-wider">
              Diagnostic History
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              My Health Reports
            </h1>
            <p className="body-regular text-xs mt-1">
              View all your heart disease prediction and screening records.
            </p>
          </div>

          <div className="cardio-card p-3 flex items-center gap-2.5 shrink-0 rounded-xl">
            <div className="w-8 h-8 rounded-lg bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center font-bold">
              <FileText size={16} />
            </div>
            <div>
              <span className="caption-small uppercase text-[10px] font-bold block">Total Reports</span>
              <span className="text-base font-bold text-[var(--text-primary)]">{reports.length}</span>
            </div>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="cardio-card p-12 text-center text-xs text-[var(--text-muted)]">
            Loading reports...
          </div>
        )}

        {/* Error */}
        {!loading && displayError && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-xs font-semibold mb-6">
            {displayError}
          </div>
        )}

        {/* Empty State */}
        {!loading && !displayError && reports.length === 0 && (
          <div className="cardio-card p-12 text-center space-y-3 max-w-lg mx-auto">
            <div className="w-12 h-12 rounded-2xl bg-[var(--accent-primary-light)] text-[var(--accent-primary)] flex items-center justify-center mx-auto">
              <Activity size={24} />
            </div>
            <h2 className="section-title text-base text-[var(--text-primary)]">
              No Reports Available
            </h2>
            <p className="body-regular text-xs">
              Your prediction reports will appear here after an assessment or diagnosis.
            </p>
          </div>
        )}

        {/* Reports Table */}
        {!loading && !displayError && reports.length > 0 && (
          <div className="cardio-card p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[var(--bg-secondary)] border-b border-[var(--border-color)] text-[var(--text-muted)] uppercase font-semibold">
                    <th className="p-4">Date</th>
                    <th className="p-4">Risk Level</th>
                    <th className="p-4">Risk %</th>
                    <th className="p-4">Clinical</th>
                    <th className="p-4">ECG</th>
                    <th className="p-4">Echo</th>
                    <th className="p-4 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-color)]">
                  {reports.map((report) => (
                    <tr
                      key={report.prediction_id}
                      className="hover:bg-[var(--card-hover)] transition-colors"
                    >
                      <td className="p-4 font-semibold text-[var(--text-primary)]">
                        {new Date(report.created_at).toLocaleDateString()}
                      </td>
                      <td className="p-4">
                        <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                          report.risk_level === "Low" ? "bg-emerald-500/10 text-emerald-500" :
                          report.risk_level === "Moderate" ? "bg-amber-500/10 text-amber-500" :
                          "bg-red-500/10 text-red-500"
                        }`}>
                          {report.risk_level || "Unknown"}
                        </span>
                      </td>
                      <td className="p-4 font-bold text-[var(--text-primary)]">
                        {report.risk_percentage}%
                      </td>
                      <td className="p-4 text-[var(--text-secondary)]">
                        {report.clinical_level || "N/A"}
                      </td>
                      <td className="p-4 text-[var(--text-secondary)]">
                        {report.ecg_level || "N/A"}
                      </td>
                      <td className="p-4 text-[var(--text-secondary)]">
                        {report.echo_level || "N/A"}
                      </td>
                      <td className="p-4 text-center">
                        <div className="flex justify-center gap-2">
                          <button
                            onClick={() => viewPatientReportPdf(report.prediction_id).catch((requestError) => {
                              setError(requestError.response?.data?.detail || "Unable to open this report.");
                            })}
                            className="btn-primary text-xs py-1.5 px-3 rounded-lg"
                          >
                            View
                          </button>
                          <button
                            disabled={mailingReportId === report.prediction_id}
                            onClick={async () => {
                              setMailStatus("");
                              setError("");
                              setMailingReportId(report.prediction_id);
                              try {
                                await emailPatientReport(report.prediction_id);
                                setMailStatus("Report sent to your registered email address.");
                              } catch (requestError) {
                                setError(requestError.response?.data?.detail || "Unable to email this report.");
                              } finally {
                                setMailingReportId(null);
                              }
                            }}
                            className="btn-secondary text-xs py-1.5 px-3 rounded-lg disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            {mailingReportId === report.prediction_id ? "Sending..." : "Mail"}
                          </button>
                          <button
                            disabled={summarizingReportId === report.prediction_id}
                            onClick={async () => {
                              setReportSummary("");
                              setError("");
                              setSummarizingReportId(report.prediction_id);
                              try {
                                const reportData = await getPatientReport(report.prediction_id);
                                const summary = reportData.summary || reportData.details || "No written summary is available for this report yet.";
                                setReportSummary(summary);
                              } catch (requestError) {
                                setError(requestError.response?.data?.detail || "Unable to summarize this report.");
                              } finally {
                                setSummarizingReportId(null);
                              }
                            }}
                            className="btn-secondary text-xs py-1.5 px-3 rounded-lg disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            {summarizingReportId === report.prediction_id ? "Loading..." : "Summarize"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {mailStatus && (
          <div className="mt-5 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3 text-xs font-semibold text-emerald-600">
            {mailStatus}
          </div>
        )}

        {reportSummary && (
          <div className="mt-5 rounded-xl border border-[var(--accent-primary-border)] bg-[var(--accent-primary-light)] p-5 text-sm text-[var(--text-secondary)]">
            <div className="mb-2 text-xs font-bold uppercase tracking-wide text-[var(--accent-primary)]">Report Summary</div>
            <p className="leading-6">{reportSummary}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default PatientReports;
