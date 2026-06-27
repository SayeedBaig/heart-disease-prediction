function ResultCard({ result }) {
  return (
    <div className="max-w-5xl mx-auto mt-10 bg-white shadow-lg rounded-xl p-8">

      <h2 className="text-3xl font-bold text-blue-900 mb-8">
        Heart Disease Prediction Report
      </h2>

      {/* Fusion Result */}

      <div className="mb-6">
        <h3 className="text-xl font-bold text-green-700">
          Final Prediction
        </h3>

        <p>
          <strong>Risk Level:</strong> {result.fusion.final_level}
        </p>

        <p>
          <strong>Risk Percentage:</strong> {result.fusion.risk_percentage}%
        </p>
      </div>

      <hr className="my-6"/>

      {/* Clinical */}

      <div className="mb-6">
        <h3 className="text-xl font-bold">
          Clinical Analysis
        </h3>

        <p><strong>Risk:</strong> {result.clinical.level}</p>

        <p><strong>Confidence:</strong> {(result.clinical.score*100).toFixed(2)}%</p>

        <p className="mt-2">
          {result.clinical.reason}
        </p>
      </div>

      <hr className="my-6"/>

      {/* ECG */}

      <div className="mb-6">
        <h3 className="text-xl font-bold">
          ECG Analysis
        </h3>

        <p><strong>Risk:</strong> {result.ecg.level}</p>

        <p><strong>Confidence:</strong> {(result.ecg.score*100).toFixed(2)}%</p>

        <p className="mt-2">
          {result.ecg.reason}
        </p>
      </div>

      <hr className="my-6"/>

      {/* Echo */}

      <div className="mb-6">
        <h3 className="text-xl font-bold">
          Echo Analysis
        </h3>

        <p><strong>Risk:</strong> {result.echo.level}</p>

        <p><strong>Confidence:</strong> {(result.echo.score*100).toFixed(2)}%</p>

        <p className="mt-2">
          {result.echo.reason}
        </p>
      </div>

      <hr className="my-6"/>

      {/* AI Explanation */}

      <div>
        <h3 className="text-xl font-bold">
          AI Recommendation
        </h3>

        <p className="mb-4">
          {result.rag.explanation}
        </p>

        <ul className="list-disc ml-6">
          {result.rag.details.map((item,index)=>(
            <li key={index}>{item}</li>
          ))}
        </ul>
      </div>

    </div>
  );
}

export default ResultCard;