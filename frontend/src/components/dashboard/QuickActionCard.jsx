import { ArrowRight } from "lucide-react";

function QuickActionCard({
  title,
  description,
  icon: Icon,
  color = "blue",
  onClick,
}) {
  const colors = {
    blue: "from-blue-500 to-cyan-500",
    green: "from-green-500 to-emerald-500",
    red: "from-red-500 to-pink-500",
    purple: "from-purple-500 to-indigo-500",
  };

  return (
    <div
      onClick={onClick}
      className="group cursor-pointer bg-white rounded-2xl p-6 border border-slate-200 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300"
    >
      <div
        className={`w-14 h-14 rounded-xl bg-gradient-to-br ${colors[color]} flex items-center justify-center shadow-md`}
      >
        <Icon size={28} className="text-white" />
      </div>

      <h3 className="mt-5 text-xl font-bold text-slate-800">
        {title}
      </h3>

      <p className="mt-2 text-slate-500 text-sm leading-6">
        {description}
      </p>

      <div className="mt-6 flex items-center text-blue-600 font-semibold group-hover:translate-x-2 transition">
        Open
        <ArrowRight size={18} className="ml-2" />
      </div>
    </div>
  );
}

export default QuickActionCard;