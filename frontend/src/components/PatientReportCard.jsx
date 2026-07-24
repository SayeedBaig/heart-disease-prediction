function PatientReportCard({ report }) {
  if (!report) {
    return (
      <div className="bg-white rounded-3xl shadow-xl p-8">
        <p className="text-center text-xl font-semibold text-green-600 animate-pulse">
          Loading Patient Report...
        </p>
      </div>
    );
  }

  const predictionId = localStorage.getItem("prediction_id");

  const handleDownloadPDF = () => {
    window.open(
      `http://localhost:8000/reports/${predictionId}/patient/pdf`,
      "_blank"
    );
  };

  const handleEmailReport = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/reports/${predictionId}/patient/email`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to send email");
      }

      const data = await response.json();

      alert(data.message || "Patient report emailed successfully!");
    } catch (err) {
      console.error(err);
      alert("Failed to send patient report email.");
    }
  };

  return (
    <div className="bg-white rounded-3xl shadow-xl p-8">

      {/* Header */}

      <div className="flex flex-col lg:flex-row lg:justify-between lg:items-center gap-6 mb-10">

        <div>

          <h2 className="text-4xl font-bold text-green-700">
            👤 Patient Report
          </h2>

          <p className="text-gray-500 mt-2">
            A simple explanation of your heart health assessment.
          </p>

        </div>

        <div className="flex flex-col sm:flex-row gap-4">

          <button
            onClick={handleDownloadPDF}
            className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-xl font-semibold shadow-md transition-all duration-300"
          >
            📥 Download PDF
          </button>

          <button
            onClick={handleEmailReport}
            className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-xl font-semibold shadow-md transition-all duration-300"
          >
            📧 Email Report
          </button>

        </div>

      </div>

      {/* Heart Health Summary */}

      <div className="bg-gradient-to-r from-green-600 to-emerald-500 rounded-3xl text-white shadow-lg transition-all duration-300 p-8 mb-10">

        <h2 className="text-3xl font-bold mb-8">
          ❤️ Your Heart Health
        </h2>

        <div className="grid md:grid-cols-2 gap-8">

          <div>

            <p className="uppercase text-green-100 tracking-wide">
              Risk Level
            </p>

            <h2 className="text-5xl font-extrabold mt-3">
            {report.risk_level || "N/A"}
            </h2>

          </div>

          <div>

            <p className="uppercase text-green-100 tracking-wide">
              Risk Percentage
            </p>

            <h2 className="text-5xl font-extrabold mt-3">
             {report.risk_percentage ?? "N/A"}%
            </h2>

          </div>

        </div>

      </div>


           {/* ================= Summary ================= */}

      <div className="bg-white rounded-3xl shadow-lg p-8 mb-8">

        <h2 className="text-3xl font-bold text-slate-800 mb-6">
          📋 What We Found
        </h2>

        <div className="bg-slate-50 rounded-2xl border shadow-md p-6">

          <p className="text-lg leading-9 text-gray-700">
         {report.summary || "No summary available."}
          </p>

        </div>

      </div>

      {/* ================= Lifestyle ================= */}

      <div className="bg-green-50 rounded-3xl shadow-lg p-8 mb-8">

        <h2 className="text-3xl font-bold text-green-700 mb-6">
          🌱 Healthy Lifestyle Tips
        </h2>

        <div className="grid md:grid-cols-2 gap-5">

          {report.lifestyle_recommendations?.length > 0 ? (

            report.lifestyle_recommendations.map((item, index) => (

              <div
                key={index}
                className="bg-white rounded-2xl shadow-md border-l-4 border-green-500 p-5 transition-all"
              >
                ✅ {item}
              </div>

            ))

          ) : (

            <div className="bg-white rounded-xl p-5">
            No lifestyle recommendations are available at this time.
            </div>

          )}

        </div>

      </div>

      {/* ================= Follow-up ================= */}

      <div className="bg-blue-50 rounded-3xl shadow-lg p-8 mb-8">

        <h2 className="text-3xl font-bold text-blue-700 mb-6">
          🩺 Follow-up Advice
        </h2>

        <div className="space-y-4">

          {report.follow_up_advice?.length > 0 ? (

            report.follow_up_advice.map((item, index) => (

              <div
                key={index}
                className="bg-white rounded-2xl shadow-md border-l-4 border-blue-500 p-5 transition-all"
              >
                📌 {item}
              </div>

            ))

          ) : (

            <div className="bg-white rounded-xl p-5">
              No follow-up advice is available at this time.
            </div>

          )}

        </div>

      </div>

    </div>
  );
}

export default PatientReportCard;