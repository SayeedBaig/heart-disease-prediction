function RecentActivity({ activities = [] }) {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-800">
          Recent Activity
        </h2>
      </div>

      {activities.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <h3 className="text-xl font-semibold text-slate-700">
            No Recent Activity
          </h3>

          <p className="mt-3 text-slate-500 max-w-md">
            Patient registrations, diagnoses, reports, and appointments
            will appear here once data is available.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {activities.map((activity, index) => (
            <div
              key={index}
              className="border border-slate-200 rounded-xl p-4 hover:bg-slate-50 transition"
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-semibold text-slate-800">
                    {activity.title}
                  </h3>

                  <p className="text-sm text-slate-600 mt-1">
                    {activity.description}
                  </p>

                  <span className="inline-block mt-2 px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-700">
                    {activity.type}
                  </span>
                </div>

                <div className="text-xs text-slate-500 whitespace-nowrap">
                  {new Date(activity.timestamp).toLocaleString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default RecentActivity;