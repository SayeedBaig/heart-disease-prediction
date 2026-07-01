function StepThree({
  formData,
  handleChange,
  handleImageChange,
  handleCsvChange,
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

      {/* ECG IMAGE */}

      <div className="mb-6">
        <label className="block mb-2 font-semibold text-blue-900">
          Upload ECG Image
        </label>

        <input
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff"
          onChange={handleImageChange}
          className="w-full border rounded-lg p-3"
        />

        {formData.ecgImage && (
          <p className="text-green-700 mt-2 text-sm">
            Selected Image: {formData.ecgImage.name}
          </p>
        )}
      </div>

      {/* ECG CSV */}

      <div className="mb-8">
        <label className="block mb-2 font-semibold text-blue-900">
          Upload ECG CSV
        </label>

        <input
          type="file"
          accept=".csv"
          onChange={handleCsvChange}
          className="w-full border rounded-lg p-3"
        />

        {formData.ecgCsv && (
          <p className="text-green-700 mt-2 text-sm">
            Selected CSV: {formData.ecgCsv.name}
          </p>
        )}

        <p className="text-gray-500 text-sm mt-2">
          Upload either an ECG Image or an ECG CSV file.
        </p>
      </div>

      {/* Error */}

      {error && (
        <p className="text-red-600 font-semibold mb-4">
          {error}
        </p>
      )}

      {/* Buttons */}

      <div className="flex justify-between">

        <button
          onClick={prevStep}
          className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600"
        >
          Back
        </button>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-8 py-3 rounded-lg text-white transition ${
            loading
              ? "bg-gray-500 cursor-not-allowed"
              : "bg-blue-900 hover:bg-blue-800"
          }`}
        >
          {loading ? "Predicting..." : "Predict Risk"}
        </button>

      </div>

    </div>
  );
}

export default StepThree;