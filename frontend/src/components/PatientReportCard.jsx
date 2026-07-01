function PatientReportCard({ report }) {
  if (!report) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <p>Loading Patient Report...</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <h2 className="text-2xl font-bold text-green-700 mb-4">
        🩺 Patient Report
      </h2>

      <p>
        <strong>Risk Level:</strong> {report.risk_level}
      </p>

      <p>
        <strong>Risk Percentage:</strong>{" "}
        {report.risk_percentage}%
      </p>

      <hr className="my-4" />

      <h3 className="text-xl font-semibold">
        Summary
      </h3>

      <p className="mt-2">
        {report.summary}
      </p>

      <hr className="my-4" />

      <h3 className="text-xl font-semibold">
        Lifestyle Recommendations
      </h3>

      <ul className="list-disc ml-6 mt-2">
        {report.lifestyle_recommendations?.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>

      <hr className="my-4" />

      <h3 className="text-xl font-semibold">
        Follow-up Advice
      </h3>

      <ul className="list-disc ml-6 mt-2">
        {report.follow_up_advice?.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export default PatientReportCard;