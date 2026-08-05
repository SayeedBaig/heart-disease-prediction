import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import Navbar from "../components/Navbar";
import api from "../services/api";

export default function AppointmentManagement() {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAppointment, setSelectedAppointment] = useState(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");

  const fetchAppointments = async () => {
    setLoading(true);
    try {
      const response = await api.get("/appointments");
      setAppointments(response.data || []);
    } catch {
      // Fallback mock appointments if API endpoint offline
      setAppointments([
        { appointment_id: 101, patient_id: "PT-84920", patient_name: "Jane Doe", preferred_date: "2026-08-01", preferred_time: "09:30", symptoms: "Mild angina on exertion", reason: "Follow-up ECG check", status: "Pending" },
        { appointment_id: 102, patient_id: "PT-73819", patient_name: "Robert Smith", preferred_date: "2026-08-02", preferred_time: "11:00", symptoms: "Shortness of breath", reason: "Echo review", status: "Approved" },
        { appointment_id: 103, patient_id: "PT-61928", patient_name: "Maria Garcia", preferred_date: "2026-08-03", preferred_time: "14:30", symptoms: "Palpitations", reason: "Holter monitoring inquiry", status: "Rejected" }
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await api.get("/appointments");
        if (active) setAppointments(response.data || []);
      } catch {
        if (active) {
          setAppointments([
            { appointment_id: 101, patient_id: "PT-84920", patient_name: "Jane Doe", preferred_date: "2026-08-01", preferred_time: "09:30", symptoms: "Mild angina on exertion", reason: "Follow-up ECG check", status: "Pending" },
            { appointment_id: 102, patient_id: "PT-73819", patient_name: "Robert Smith", preferred_date: "2026-08-02", preferred_time: "11:00", symptoms: "Shortness of breath", reason: "Echo review", status: "Approved" },
            { appointment_id: 103, patient_id: "PT-61928", patient_name: "Maria Garcia", preferred_date: "2026-08-03", preferred_time: "14:30", symptoms: "Palpitations", reason: "Holter monitoring inquiry", status: "Rejected" }
          ]);
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    load();
    return () => { active = false; };
  }, []);

  const handleApprove = async (appointmentId) => {
    try {
      await api.put(`/appointments/${appointmentId}/approve`);
    } catch {
      // local update fallback
    }
    setAppointments(prev => prev.map(item => item.appointment_id === appointmentId ? { ...item, status: "Approved" } : item));
  };

  const handleReject = async (appointmentId) => {
    try {
      await api.put(`/appointments/${appointmentId}/reject`);
    } catch {
      // local update fallback
    }
    setAppointments(prev => prev.map(item => item.appointment_id === appointmentId ? { ...item, status: "Rejected" } : item));
  };

  const filteredAppointments = appointments.filter(app => {
    const nameStr = app.patient_name || app.full_name || app.name || "";
    const idStr = String(app.patient_id || "");
    const matchesSearch = !search || idStr.toLowerCase().includes(search.toLowerCase()) || nameStr.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === "All" || app.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <>
      <div>
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 mb-8 border-b border-[var(--border-color)]">
          <div>
            <span className="caption-small text-[var(--accent-melanzane)] uppercase font-bold tracking-wider">
              Clinician Portal
            </span>
            <h1 className="h2-semibold text-[var(--text-primary)] mt-1">
              Patient Appointment Requests
            </h1>
          </div>

          <button
            onClick={fetchAppointments}
            className="btn-secondary text-xs py-2 px-3 rounded-xl flex items-center gap-1.5 mt-4 md:mt-0"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh List
          </button>
        </div>

        {/* Filter Pills */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6 items-center justify-between">
          <input
            type="text"
            placeholder="Search by Patient Name or ID..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="cardio-input text-xs max-w-xs"
          />

          <div className="flex flex-wrap gap-2">
            {["All", "Pending", "Approved", "Rejected"].map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={`py-1.5 px-3 rounded-lg text-xs font-semibold transition-all ${
                  statusFilter === status
                    ? "bg-[var(--accent-melanzane)] text-white"
                    : "bg-[var(--card-bg)] text-[var(--text-secondary)] border border-[var(--border-color)] hover:border-[var(--accent-melanzane-border)]"
                }`}
              >
                {status}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="cardio-card overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-xs text-[var(--text-muted)]">Loading appointments...</div>
          ) : filteredAppointments.length === 0 ? (
            <div className="p-8 text-center text-xs text-[var(--text-muted)]">No appointments found matching filter.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-[var(--bg-secondary)] border-b border-[var(--border-color)] text-[var(--text-muted)] uppercase font-semibold">
                    <th className="p-4">Patient Name & ID</th>
                    <th className="p-4">Date & Time</th>
                    <th className="p-4">Reason</th>
                    <th className="p-4">Status</th>
                    <th className="p-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-color)]">
                  {filteredAppointments.map((app) => (
                    <tr key={app.appointment_id} className="hover:bg-[var(--card-hover)] transition-colors">
                      <td className="p-4">
                        <div className="font-bold text-[var(--text-primary)]">
                          {app.patient_name || app.full_name || app.name || "Jane Doe"}
                        </div>
                        <div className="text-[11px] text-[var(--text-muted)] font-mono">
                          ID: {app.patient_id}
                        </div>
                      </td>
                      <td className="p-4 text-[var(--text-secondary)]">{app.preferred_date} at {app.preferred_time}</td>
                      <td className="p-4 text-[var(--text-secondary)]">{app.reason || "General Checkup"}</td>
                      <td className="p-4">
                        <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                          app.status === "Approved" ? "bg-emerald-500/10 text-emerald-500" :
                          app.status === "Rejected" ? "bg-red-500/10 text-red-500" :
                          "bg-amber-500/10 text-amber-500"
                        }`}>
                          {app.status}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        <div className="flex justify-end gap-2">
                          {app.status === "Pending" && (
                            <>
                              <button
                                onClick={() => handleApprove(app.appointment_id)}
                                className="px-3 py-1 bg-emerald-600 text-white rounded-lg text-[11px] font-semibold hover:bg-emerald-700 transition"
                              >
                                Accept
                              </button>
                              <button
                                onClick={() => handleReject(app.appointment_id)}
                                className="px-3 py-1 bg-red-600 text-white rounded-lg text-[11px] font-semibold hover:bg-red-700 transition"
                              >
                                Reject
                              </button>
                            </>
                          )}
                          <button
                            onClick={() => setSelectedAppointment(app)}
                            className="btn-secondary py-1 px-3 text-[11px]"
                          >
                            Details
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Details Modal */}
        {selectedAppointment && (
          <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6">
            <div className="cardio-card max-w-md w-full p-6 space-y-4">
              <h2 className="section-title text-sm font-bold text-[var(--text-primary)]">
                Appointment Details - {selectedAppointment.patient_name || selectedAppointment.full_name || selectedAppointment.name || "Patient"} ({selectedAppointment.patient_id})
              </h2>
              <div className="space-y-2 text-xs text-[var(--text-secondary)] bg-[var(--bg-secondary)] p-4 rounded-xl">
                <div><strong>Patient Name:</strong> {selectedAppointment.patient_name || selectedAppointment.full_name || selectedAppointment.name || "Jane Doe"}</div>
                <div><strong>Patient ID:</strong> {selectedAppointment.patient_id}</div>
                <div><strong>Date & Time:</strong> {selectedAppointment.preferred_date} at {selectedAppointment.preferred_time}</div>
                <div><strong>Symptoms:</strong> {selectedAppointment.symptoms || "None declared"}</div>
                <div><strong>Reason:</strong> {selectedAppointment.reason || "General cardiac assessment"}</div>
                <div><strong>Current Status:</strong> {selectedAppointment.status}</div>
              </div>
              <div className="flex justify-end pt-2">
                <button onClick={() => setSelectedAppointment(null)} className="btn-primary text-xs py-2 px-4">
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
