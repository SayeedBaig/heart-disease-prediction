function ProgressBar({ step }) {
  return (
    <div className="max-w-xl mx-auto mb-8">

      <div className="flex justify-between text-sm font-semibold text-gray-600 mb-2">
        <span>Patient</span>
        <span>Health</span>
        <span>Lifestyle</span>
      </div>

      <div className="flex items-center">

        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold
          ${step >= 1 ? "bg-blue-900" : "bg-gray-300"}`}
        >
          1
        </div>

        <div
          className={`flex-1 h-1
          ${step >= 2 ? "bg-blue-900" : "bg-gray-300"}`}
        ></div>

        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold
          ${step >= 2 ? "bg-blue-900" : "bg-gray-300"}`}
        >
          2
        </div>

        <div
          className={`flex-1 h-1
          ${step >= 3 ? "bg-blue-900" : "bg-gray-300"}`}
        ></div>

        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold
          ${step >= 3 ? "bg-blue-900" : "bg-gray-300"}`}
        >
          3
        </div>

      </div>

    </div>
  );
}

export default ProgressBar;