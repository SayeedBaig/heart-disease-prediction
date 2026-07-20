function DoctorReportCard({ report }) {
  if (!report) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <p>Loading Doctor Report...</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-md p-6 mb-8">

      <h2 className="text-3xl font-bold text-blue-700 mb-6">
        👨‍⚕️ Doctor Report
      </h2>
      <div className="flex gap-4 mb-6">

  <button
    onClick={() => {
      const predictionId = localStorage.getItem("prediction_id");

window.open(
  `http://localhost:8000/reports/${predictionId}/doctor/pdf`,
  "_blank"
);
    }}
    className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg font-semibold"
  >
    📥 Download Doctor PDF
  </button>
  
    <button
  onClick={async () => {
    const predictionId = localStorage.getItem("prediction_id");

    try {
      const response = await fetch(
        `http://localhost:8000/reports/${predictionId}/doctor/email`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to send email");
      }

      const data = await response.json();
      alert(data.message || "Doctor report emailed successfully!");
    } catch (err) {
      console.error(err);
      alert("Failed to send doctor report email.");
    }
  }}
  className="bg-purple-600 hover:bg-purple-700 text-white px-5 py-2 rounded-lg font-semibold"
>
  📧 Email Doctor Report
</button>

</div>

      {/* Final Prediction */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          🫀 Final Prediction
        </h3>

        <p>
          <strong>Risk Level:</strong>{" "}
          {report.final_prediction?.final_level ?? "N/A"}
        </p>

        <p>
          <strong>Risk Percentage:</strong>{" "}
          {report.final_prediction?.risk_percentage ?? "N/A"}%
        </p>
      </div>

      <hr className="my-6" />

      {/* Clinical Analysis */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          🩺 Clinical Analysis
        </h3>

        <p>
          <strong>Level:</strong>{" "}
          {report.clinical_analysis?.level ?? "N/A"}
        </p>

        <p>
          <strong>Confidence:</strong>{" "}
          {report.clinical_analysis?.score != null
            ? `${(report.clinical_analysis.score * 100).toFixed(2)}%`
            : "N/A"}
        </p>

        <p className="mt-2">
          {report.clinical_analysis?.reason ?? "No clinical explanation available."}
        </p>
      </div>

      <hr className="my-6" />

      {/* ECG Analysis */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          📈 ECG Analysis
        </h3>

        <p>
          <strong>Level:</strong>{" "}
          {report.ecg_analysis?.level ?? "N/A"}
        </p>

        <p>
          <strong>Confidence:</strong>{" "}
          {report.ecg_analysis?.score != null
            ? `${(report.ecg_analysis.score * 100).toFixed(2)}%`
            : "N/A"}
        </p>

        <p className="mt-2">
          {report.ecg_analysis?.reason ??
            "ECG not available. Please upload an ECG for a detailed assessment."}
        </p>
      </div>

      <hr className="my-6" />

      {/* Echo Analysis */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          🫁 Echo Analysis
        </h3>

        <p>
          <strong>Level:</strong>{" "}
          {report.echo_analysis?.level ?? "N/A"}
        </p>

        <p>
          <strong>Confidence:</strong>{" "}
          {report.echo_analysis?.score != null
            ? `${(report.echo_analysis.score * 100).toFixed(2)}%`
            : "N/A"}
        </p>

        <p className="mt-2">
          {report.echo_analysis?.reason ??
            "Echo not available. Please upload an Echo video for a more accurate diagnosis."}
        </p>
      </div>

      <hr className="my-6" />

      {/* AI Recommendation */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          🤖 AI Recommendation
        </h3>

        <p>
          <strong>Risk Level:</strong>{" "}
          {report.ai_recommendation?.final_level ?? "N/A"}
        </p>

        <p>
          <strong>Risk Percentage:</strong>{" "}
          {report.ai_recommendation?.risk_percentage ?? "N/A"}%
        </p>

        <p className="mt-3">
          {report.ai_recommendation?.explanation ??
            "No AI recommendation available."}
        </p>

        {report.ai_recommendation?.details?.length > 0 && (
          <ul className="list-disc ml-6 mt-4 space-y-2">
            {report.ai_recommendation.details.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        )}
      </div>

      <hr className="my-6" />

      {/* Medical Explanation */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          📚 Medical Explanation
        </h3>

        <p>
          <strong>Summary:</strong>
        </p>

        <p className="mb-4">
          {report.medical_explanation?.summary ??
            "No summary available."}
        </p>

        <p>
          <strong>Details:</strong>
        </p>

        <p className="mb-4">
          {report.medical_explanation?.details ??
            "No detailed explanation available."}
        </p>

        {report.medical_explanation?.lifestyle_suggestions?.length > 0 && (
          <>
            <h4 className="font-semibold mb-2">
              Lifestyle Suggestions
            </h4>

            <ul className="list-disc ml-6 space-y-2">
              {report.medical_explanation.lifestyle_suggestions.map(
                (item, index) => (
                  <li key={index}>{item}</li>
                )
              )}
            </ul>
          </>
        )}
      </div>

      <hr className="my-6" />

      {/* Digital Twin */}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">
          🧬 Digital Twin
        </h3>

        <p>
          <strong>Baseline Risk:</strong>{" "}
          {report.digital_twin?.baseline_risk != null
            ? `${(report.digital_twin.baseline_risk * 100).toFixed(1)}%`
            : "N/A"}
        </p>

        {report.digital_twin?.simulations?.length > 0 ? (
          <>
            <h4 className="font-semibold mt-4 mb-2">
              Simulations
            </h4>

            <table className="table-auto border w-full">
              <thead>
                <tr className="bg-gray-100">
                  <th className="border p-2">Scenario</th>
                  <th className="border p-2">Risk</th>
                  <th className="border p-2">Change</th>
                </tr>
              </thead>

              <tbody>
                {report.digital_twin.simulations.map((sim, index) => (
                  <tr key={index}>
                    <td className="border p-2">{sim.scenario}</td>
                    <td className="border p-2">{sim.risk}</td>
                    <td className="border p-2">{sim.change}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="mt-3">
            No Digital Twin simulations available.
          </p>
        )}
      </div>

      <hr className="my-6" />

      {/* Supporting References */}

      <div>
        <h3 className="text-xl font-semibold mb-4">
          📖 Supporting References
        </h3>

        {report.supporting_references?.length > 0 ? (
          <div className="space-y-4">
            {report.supporting_references.map((ref, index) => (
              <div
                key={index}
                className="border rounded-lg p-4 bg-gray-50 shadow-sm"
              >
                <p className="font-semibold text-blue-700">
                  {ref.source ?? "Unknown Source"}
                </p>

                <p className="text-sm text-gray-600">
                  Page {ref.page ?? "-"} • Score{" "}
                  {ref.score != null ? ref.score.toFixed(2) : "N/A"}
                </p>

                <p className="mt-3 text-gray-700">
                  {ref.text
                    ? ref.text.length > 250
                      ? ref.text.substring(0, 250) + "..."
                      : ref.text
                    : "No preview available."}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p>No supporting references available.</p>
        )}
      </div>

    </div>
  );
}

export default DoctorReportCard;