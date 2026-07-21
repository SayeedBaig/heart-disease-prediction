function RecentActivity() {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-800">
          Recent Activity
        </h2>
      </div>

      <div className="flex flex-col items-center justify-center py-16 text-center">
        <h3 className="text-xl font-semibold text-slate-700">
          No Recent Activity
        </h3>

        <p className="mt-3 text-slate-500 max-w-md">
          Patient registrations, diagnoses, reports, and appointments
          will appear here once data is available.
        </p>
      </div>
    </div>
  );
}

export default RecentActivity;