function RecentActivity() {
  const activities = [
    {
      patient: "John Doe",
      status: "Diagnosis Completed",
      risk: "Low Risk",
      time: "10 mins ago",
    },
    {
      patient: "Sarah Lee",
      status: "Report Generated",
      risk: "High Risk",
      time: "35 mins ago",
    },
    {
      patient: "Michael Brown",
      status: "Patient Registered",
      risk: "-",
      time: "1 hour ago",
    },
    {
      patient: "Emily Davis",
      status: "Diagnosis Completed",
      risk: "Medium Risk",
      time: "2 hours ago",
    },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">

      <div className="flex justify-between items-center mb-6">

        <h2 className="text-2xl font-bold text-slate-800">
          Recent Activity
        </h2>

        <button className="text-blue-600 font-semibold hover:underline">
          View All
        </button>

      </div>

      <div className="overflow-x-auto">

        <table className="w-full">

          <thead>

            <tr className="text-left border-b">

              <th className="py-3 text-slate-500">Patient</th>

              <th className="py-3 text-slate-500">Status</th>

              <th className="py-3 text-slate-500">Risk</th>

              <th className="py-3 text-slate-500">Time</th>

            </tr>

          </thead>

          <tbody>

            {activities.map((item, index) => (

              <tr
                key={index}
                className="border-b last:border-none hover:bg-slate-50 transition"
              >

                <td className="py-4 font-semibold text-slate-800">
                  {item.patient}
                </td>

                <td className="py-4">
                  {item.status}
                </td>

                <td className="py-4">

                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium
                    ${
                      item.risk === "Low Risk"
                        ? "bg-green-100 text-green-700"
                        : item.risk === "Medium Risk"
                        ? "bg-yellow-100 text-yellow-700"
                        : item.risk === "High Risk"
                        ? "bg-red-100 text-red-700"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {item.risk}
                  </span>

                </td>

                <td className="py-4 text-slate-500">
                  {item.time}
                </td>

              </tr>

            ))}

          </tbody>

        </table>

      </div>

    </div>
  );
}

export default RecentActivity;