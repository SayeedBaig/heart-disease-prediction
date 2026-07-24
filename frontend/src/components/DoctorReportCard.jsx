
function DoctorReportCard({ report }) {
  if (!report) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8">
        <p className="text-center text-xl font-semibold text-blue-600 animate-pulse">
          Loading Doctor Report...
        </p>
      </div>
    );
  }

  const predictionId = localStorage.getItem("prediction_id");

  const handleDownloadPDF = () => {
    window.open(
      `http://localhost:8000/reports/${predictionId}/doctor/pdf`,
      "_blank"
    );
  };

  const handleEmailReport = async () => {
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
    } catch (error) {
      console.error(error);
      alert("Failed to send doctor report email.");
    }
  };

  return (
    <div className="bg-white rounded-3xl shadow-xl p-8">

      {/* ================= Header ================= */}

      <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-6 mb-10">

        <div>
          <h1 className="text-4xl font-bold text-blue-700">
            👨‍⚕️ Doctor Report
          </h1>

          <p className="text-gray-500 mt-2">
            Comprehensive AI-generated medical assessment
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">

          <button
            onClick={handleDownloadPDF}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-semibold transition shadow-md"
          >
            📥 Download PDF
          </button>

          <button
            onClick={handleEmailReport}
            className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-xl font-semibold transition shadow-md"
          >
            📧 Email Report
          </button>

        </div>

      </div>

      {/* ================= Overall Prediction ================= */}

      <div className="bg-gradient-to-r from-blue-600 to-cyan-500 rounded-3xl text-white p-8 mb-10 shadow-lg">

        <h2 className="text-3xl font-bold mb-8">
          Overall AI Prediction
        </h2>

        <div className="grid md:grid-cols-3 gap-8">

          <div>

            <p className="uppercase text-blue-100 tracking-wide">
              Risk Level
            </p>

            <h3 className="text-5xl font-extrabold mt-3">
              {report.final_prediction?.final_level || "N/A"}
            </h3>

          </div>

          <div>

            <p className="uppercase text-blue-100 tracking-wide">
              Risk Percentage
            </p>

            <h3 className="text-5xl font-extrabold mt-3">
              {report.final_prediction?.risk_percentage || "N/A"}%
            </h3>

          </div>
          <div>

    <p className="uppercase text-blue-100 tracking-wide">
        Recommendation
    </p>

    <h3 className="text-3xl font-bold mt-3">
        Routine Check-up
    </h3>

</div>

        </div>
        

      </div>
            {/* ================= Analysis Results ================= */}

      <div className="mb-10">

        <h2 className="text-3xl font-bold text-slate-800 mb-8">
          🩺 Analysis Results
        </h2>

        <div className="grid lg:grid-cols-3 gap-6">

          {/* Clinical */}

          <div className="bg-white border rounded-2xl shadow-lg p-8 transition-all duration-300">
            <div className="flex justify-between items-center mb-4">

              <h3 className="text-xl font-bold text-blue-700">
                Clinical
              </h3>

              <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full text-sm font-semibold">
                {report.clinical_analysis?.level || "N/A"}
              </span>

            </div>

            <p className="text-sm font-semibold mb-2">
              Confidence
            </p>

            <div className="w-full bg-gray-200 rounded-full h-3">

              <div
                className="bg-blue-600 h-3 rounded-full"
                style={{
                  width: `${(report.clinical_analysis?.score || 0) * 100}%`,
                }}
              />

            </div>

            <p className="text-lg font-bold text-gray-700 mt-2">
              {((report.clinical_analysis?.score || 0) * 100).toFixed(1)}%
            </p>

            <p className="mt-5 text-gray-700 leading-7">
              {report.clinical_analysis?.reason ||
                "No clinical explanation available."}
            </p>

          </div>

          {/* ECG */}

          <div className="bg-white border rounded-2xl shadow-lg p-8 transition-all duration-300">

            <div className="flex justify-between items-center mb-4">

              <h3 className="text-xl font-bold text-green-700">
                ECG
              </h3>

              <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-semibold">
                {report.ecg_analysis?.level || "N/A"}
              </span>

            </div>

            <p className="text-sm font-semibold mb-2">
              Confidence
            </p>

            <div className="w-full bg-gray-200 rounded-full h-3">

              <div
                className="bg-green-600 h-3 rounded-full"
                style={{
                  width: `${(report.ecg_analysis?.score || 0) * 100}%`,
                }}
              />

            </div>

            <p className="text-lg font-bold text-gray-700 mt-2">
              {((report.ecg_analysis?.score || 0) * 100).toFixed(1)}%
            </p>

            <p className="mt-5 text-gray-700 leading-7">
              {report.ecg_analysis?.reason ||
                "No ECG explanation available."}
            </p>

          </div>

          {/* Echo */}

          <div className="bg-white border rounded-2xl shadow-lg p-8 transition-all duration-300">
            <div className="flex justify-between items-center mb-4">

              <h3 className="text-xl font-bold text-purple-700">
                Echo
              </h3>

              <span className="bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-sm font-semibold">
                {report.echo_analysis?.level || "N/A"}
              </span>

            </div>

            <p className="text-sm font-semibold mb-2">
              Confidence
            </p>

            <div className="w-full bg-gray-200 rounded-full h-3">

              <div
                className="bg-purple-600 h-3 rounded-full"
                style={{
                  width: `${(report.echo_analysis?.score || 0) * 100}%`,
                }}
              />

            </div>

            <p className="text-lg font-bold text-gray-700 mt-2">
              {((report.echo_analysis?.score || 0) * 100).toFixed(1)}%
            </p>

            <p className="mt-5 text-gray-700 leading-7">
              {report.echo_analysis?.reason ||
                "No Echo explanation available."}
            </p>

          </div>

        </div>

      </div>

      {/* ================= AI Recommendation ================= */}

      <div className="bg-blue-50 rounded-3xl border-l-4 border-blue-600 p-8 mb-10">

        <h2 className="text-3xl font-bold text-blue-800 mb-6">
          🤖 AI Recommendation
        </h2>

        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 p-8">

          <p className="text-gray-700 leading-8 text-lg">
            {report.ai_recommendation?.explanation ||
              "No AI recommendation available."}
          </p>

          {report.ai_recommendation?.details?.length > 0 && (

            <ul className="mt-6 space-y-3">

              {report.ai_recommendation.details.map((item, index) => (

                <li
                  key={index}
                  className="bg-blue-50 rounded-xl p-3"
                >
                  ✅ {item}
                </li>

              ))}

            </ul>

          )}

        </div>

      </div>

      {/* ================= Medical Explanation ================= */}

      <div className="bg-white rounded-3xl shadow-lg p-8 mb-10">

        <h2 className="text-3xl font-bold text-slate-800 mb-6">
          📚 Medical Explanation
        </h2>

        <div className="space-y-6">

          <div className="bg-slate-50 rounded-2xl border shadow-md p-8">

            <h3 className="font-bold text-lg mb-3">
              Summary
            </h3>

            <p className="leading-8">
              {report.medical_explanation?.summary || "N/A"}
            </p>

          </div>

          <div className="bg-slate-50 rounded-2xl border shadow-md p-8">

            <h3 className="font-bold text-lg mb-3">
              Details
            </h3>

            <p className="leading-8">
              {report.medical_explanation?.details || "N/A"}
            </p>

          </div>

        </div>

      </div>
            {/* ================= Digital Twin ================= */}

      <div className="bg-gradient-to-r from-indigo-50 to-cyan-50 rounded-3xl shadow-lg p-8 mb-10">

        <h2 className="text-3xl font-bold text-indigo-700 mb-8">
          🧬 Digital Twin
        </h2>

        <div className="bg-white rounded-2xl shadow p-6 mb-6">

          <p className="text-sm uppercase tracking-wide text-gray-500">
            Baseline Risk
          </p>

          <h3 className="text-5xl font-extrabold text-indigo-700 mt-3">
            {report.digital_twin?.baseline_risk != null
              ? `${(report.digital_twin.baseline_risk * 100).toFixed(1)}%`
              : "N/A"}
          </h3>

        </div>

        {report.digital_twin?.simulations?.length > 0 ? (

          <div className="overflow-x-auto">

            <table className="w-full bg-white rounded-2xl overflow-hidden shadow">

              <thead className="bg-indigo-600 text-white">

                <tr>
                  <th className="text-left px-6 py-4">Scenario</th>
                  <th className="text-left px-6 py-4">Risk</th>
                  <th className="text-left px-6 py-4">Improvement</th>
                </tr>

              </thead>

              <tbody>

                {report.digital_twin.simulations.map((sim, index) => (

                  <tr
                    key={index}
                    className="border-b hover:bg-gray-50"
                  >
                    <td className="px-6 py-4">
                      {sim.scenario}
                    </td>

                    <td className="px-6 py-4">
                      {sim.risk}
                    </td>

                    <td className="px-6 py-4 text-green-600 font-semibold">
                      {sim.change}%
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>

        ) : (

          <div className="bg-white rounded-2xl shadow-md p-8 text-center">

    <h3 className="text-2xl font-bold text-indigo-700 mb-3">
        No Digital Twin Simulation Available
    </h3>

    <p className="text-gray-600">
        Digital Twin data was not returned by the AI engine.
    </p>

    <p className="text-gray-500 mt-2">
        Future simulations will appear here once available.
    </p>

</div>
        )}

      </div>

      {/* ================= Supporting References ================= */}

      <div className="bg-white rounded-3xl shadow-lg p-8">

        <h2 className="text-3xl font-bold text-slate-800 mb-8">
          📖 Supporting References
        </h2>

        {report.supporting_references?.length > 0 ? (

          <div className="space-y-6">

            {report.supporting_references.map((ref, index) => (

              <div
                key={index}
                className="bg-slate-50 rounded-2xl border-l-4 border-blue-600 shadow p-6"
              >

                <div className="flex justify-between flex-wrap gap-3">

                  <h3 className="text-xl font-bold text-blue-700">
                    {ref.source || "Unknown Source"}
                  </h3>

                  <span className="text-gray-500 text-sm">
                    Page {ref.page ?? "-"} • Score{" "}
                    {ref.score != null ? ref.score.toFixed(2) : "N/A"}
                  </span>

                </div>

                <p className="mt-5 leading-8 text-gray-700">
                  {ref.text || "No preview available."}
                </p>

              </div>

            ))}

          </div>

        ) : (

          <div className="bg-slate-50 rounded-2xl border shadow-md p-8 text-center">

    <h3 className="text-xl font-bold text-blue-700 mb-3">
        No Supporting References
    </h3>

    <p className="text-gray-600">
        No medical references were returned by the AI model.
    </p>

    <p className="text-gray-500 mt-2">
        References from PubMed, WHO and AHA guidelines will appear here when available.
    </p>

</div>

        )}

      </div>

    </div>
  );
}

export default DoctorReportCard;