function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = "blue",
}) {
  const colors = {
    blue: "from-blue-500 to-cyan-500",
    green: "from-green-500 to-emerald-500",
    purple: "from-purple-500 to-indigo-500",
    red: "from-red-500 to-pink-500",
  };

  return (
    <div className="group bg-white rounded-2xl p-6 shadow-sm border border-slate-200 hover:shadow-xl hover:-translate-y-1 transition-all duration-300">

      <div className="flex items-center justify-between">

        <div>

          <p className="text-sm font-medium text-slate-500">
            {title}
          </p>

          <h2 className="mt-2 text-4xl font-bold text-slate-800">
            {value}
          </h2>

          <p className="mt-2 text-sm text-slate-400">
            {subtitle}
          </p>

        </div>

        <div
          className={`h-16 w-16 rounded-2xl bg-gradient-to-br ${colors[color]} flex items-center justify-center shadow-lg`}
        >
          <Icon
            size={30}
            className="text-white"
          />
        </div>

      </div>

    </div>
  );
}

export default StatCard;