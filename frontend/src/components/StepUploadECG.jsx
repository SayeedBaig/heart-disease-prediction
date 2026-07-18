function StepUploadECG({
  formData,
  handleECGFileChange,
  nextStep,
  prevStep,
}) {
  return (
    <div className="max-w-2xl mx-auto bg-white rounded-2xl shadow-xl p-10">

      <div className="text-center mb-10">

        <div className="text-6xl mb-4">
          ❤️
        </div>

        <h2 className="text-4xl font-bold text-blue-900">
          Upload ECG
        </h2>

        <p className="text-gray-600 mt-3">
          Upload an ECG image or CSV signal for AI analysis.
        </p>

      </div>

      <div className="border-2 border-dashed border-blue-300 rounded-xl p-8">

        <input
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.csv"
          onChange={handleECGFileChange}
          className="w-full"
        />

        {formData.ecgFile && (
          <div className="mt-5 p-4 rounded-lg bg-green-50 border border-green-300">
            <p className="text-green-700 font-semibold">
              ✅ {formData.ecgFile.name}
            </p>
          </div>
        )}

        <p className="text-sm text-gray-500 mt-4">
          Supported Formats: PNG, JPG, JPEG, BMP, TIFF and CSV
        </p>

      </div>

      <div className="flex justify-between mt-10">

        <button
          onClick={prevStep}
          className="bg-gray-500 hover:bg-gray-600 text-white px-8 py-3 rounded-xl transition"
        >
          ← Back
        </button>

        <button
          onClick={nextStep}
          className="bg-blue-700 hover:bg-blue-800 text-white px-8 py-3 rounded-xl transition"
        >
          Continue →
        </button>

      </div>

    </div>
  );
}

export default StepUploadECG;