function ConfidenceBar({ score }) {
  const percentage = (score * 100).toFixed(2);

  return (
    <div className="mt-2">
      <div className="flex justify-between text-sm font-medium mb-1">
        <span>Confidence</span>
        <span>{percentage}%</span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className="bg-blue-600 h-3 rounded-full transition-all duration-500"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
}

export default ConfidenceBar;