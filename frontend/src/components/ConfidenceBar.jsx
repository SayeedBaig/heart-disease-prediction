function ConfidenceBar({ score }) {
  const percentage = Math.round(score * 100);

  let color = "bg-green-500";

  if (percentage < 50) {
    color = "bg-red-500";
  } else if (percentage < 75) {
    color = "bg-yellow-500";
  }

  return (
    <div className="mt-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-slate-700">
          AI Confidence
        </span>

        <span className="text-sm font-bold text-blue-600">
          {percentage}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden">
        <div
          className={`${color} h-full rounded-full transition-all duration-700 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Footer */}
      <div className="flex justify-between mt-2 text-xs text-slate-500">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  );
}

export default ConfidenceBar;