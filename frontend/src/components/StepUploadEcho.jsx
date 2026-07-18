function StepUploadEcho({
  formData,
  handleEchoFileChange,
  prevStep,
  handleSubmit,
  loading,
  error,
}) {
  return (
    <div className="max-w-2xl mx-auto bg-white rounded-2xl shadow-xl p-10">

      <div className="text-center mb-10">

        <div className="text-6xl mb-4">
          🫀
        </div>

        <h2 className="text-4xl font-bold text-blue-900">
          Upload Echo
        </h2>

        <p className="text-gray-600 mt-3">
          Upload an echocardiography video for AI analysis.
        </p>

      </div>

      <div className="border-2 border-dashed border-blue-300 rounded-xl p-8">

        <input
          type="file"
          accept=".mp4,.avi,.mov,.mkv"
          onChange={handleEchoFileChange}
          className="w-full"
        />

        {formData.echoFile && (
          <div className="mt-5 p-4 rounded-lg bg-green-50 border border-green-300">

            <p className="text-green-700 font-semibold">
              ✅ {formData.echoFile.name}
            </p>

          </div>
        )}

        <p className="text-sm text-gray-500 mt-4">
          Supported Formats: MP4, AVI, MOV, MKV
        </p>

        <p className="text-xs text-gray-400 mt-2">
          This step is optional. If you don't have an echo video, you can still continue.
        </p>

      </div>

      {error && (
        <div className="mt-6 rounded-lg border border-red-300 bg-red-50 p-4">
          <p className="text-red-700 font-medium">
            {error}
          </p>
        </div>
      )}

      <div className="flex justify-between mt-10">

        <button
          onClick={prevStep}
          className="bg-gray-500 hover:bg-gray-600 text-white px-8 py-3 rounded-xl transition"
        >
          ← Back
        </button>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`px-8 py-3 rounded-xl text-white font-semibold transition ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-700 hover:bg-blue-800"
          }`}
        >
          {loading ? "🔄 Running AI Prediction..." : "🫀 Predict Risk"}
        </button>

      </div>

    </div>
  );
}

export default StepUploadEcho;