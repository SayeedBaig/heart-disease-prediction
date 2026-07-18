function ProgressBar({ step }) {
  const steps = [
    "Patient",
    "Health",
    "Lifestyle",
    "ECG",
    "Echo",
  ];

  return (
    <div className="max-w-5xl mx-auto mb-10 px-4">

      {/* Step Labels */}
      <div className="flex justify-between mb-3 text-sm font-semibold text-gray-600">
        {steps.map((label, index) => (
          <span
            key={index}
            className={`w-20 text-center ${
              step >= index + 1
                ? "text-blue-900"
                : "text-gray-400"
            }`}
          >
            {label}
          </span>
        ))}
      </div>

      {/* Progress Circles */}
      <div className="flex items-center">

        {steps.map((_, index) => (
          <div
            key={index}
            className="flex items-center flex-1 last:flex-none"
          >
            <div
              className={`w-11 h-11 rounded-full flex items-center justify-center text-white font-bold transition-all duration-300 ${
                step >= index + 1
                  ? "bg-blue-900"
                  : "bg-gray-300"
              }`}
            >
              {index + 1}
            </div>

            {index !== steps.length - 1 && (
              <div
                className={`flex-1 h-1 transition-all duration-300 ${
                  step > index + 1
                    ? "bg-blue-900"
                    : "bg-gray-300"
                }`}
              />
            )}
          </div>
        ))}

      </div>

    </div>
  );
}

export default ProgressBar;