function StepThree({
  formData,
  handleChange,
  prevStep,
  errors,
  handleSubmit,
  loading,
  error,
}) {
  return (
    <div className="max-w-xl mx-auto bg-white shadow-lg rounded-xl p-8">

      <h2 className="text-3xl font-bold text-blue-900 mb-6">
        Lifestyle Information
      </h2>

      {/* Smoking */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Smoking
        </label>

        <select
          name="smoke"
          value={formData.smoke}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select</option>
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>

        {errors.smoke && (
          <p className="text-red-500 text-sm mt-1">
            {errors.smoke}
          </p>
        )}
      </div>

      {/* Alcohol */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Alcohol Consumption
        </label>

        <select
          name="alco"
          value={formData.alco}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select</option>
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>

        {errors.alco && (
          <p className="text-red-500 text-sm mt-1">
            {errors.alco}
          </p>
        )}
      </div>

      {/* Physical Activity */}

      <div className="mb-6">
        <label className="block mb-2 font-semibold">
          Physically Active
        </label>

        <select
          name="active"
          value={formData.active}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select</option>
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>

        {errors.active && (
          <p className="text-red-500 text-sm mt-1">
            {errors.active}
          </p>
        )}
      </div>

      {/* Error */}

      {error && (
        <div className="mb-4 rounded-lg border border-red-300 bg-red-50 p-4">
          <p className="text-red-700 font-medium">
            {error}
          </p>
        </div>
      )}

      {/* Buttons */}

      <div className="flex justify-between">

        <button
          onClick={prevStep}
          className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition"
        >
          Back
        </button>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-8 py-3 rounded-lg text-white font-semibold shadow-lg transition-all duration-300 ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 hover:shadow-xl"
          }`}
        >
          {loading ? "🔄 Running AI Prediction..." : "🫀 Predict Risk"}
        </button>

      </div>

    </div>
  );
}

export default StepThree;