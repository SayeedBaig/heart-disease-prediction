function ExplanationCard({ explanation }) {
  if (!explanation) return null;

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mt-8 shadow-sm">

      <h2 className="text-2xl font-bold text-blue-700 mb-6">
        📚 Medical Explanation
      </h2>

      <p className="mb-4">
        <strong>Status:</strong> {explanation.status}
      </p>

      <div className="mb-6">
        <h3 className="font-semibold text-lg mb-2">
          Summary
        </h3>

        <p>{explanation.explanation.summary}</p>
      </div>

      <div className="mb-6">
        <h3 className="font-semibold text-lg mb-2">
          Detailed Explanation
        </h3>

        <p>{explanation.explanation.details}</p>
      </div>

      <div className="mb-6">
        <h3 className="font-semibold text-lg mb-2">
          Query Used
        </h3>

        <p className="italic text-gray-700">
          {explanation.query}
        </p>
      </div>

      <div>
        <h3 className="font-semibold text-lg mb-3">
          Retrieved References
        </h3>

        <ul className="space-y-3">
          {explanation.chunks.map((chunk, index) => (
            <li
              key={index}
              className="bg-white rounded-lg p-4 border"
            >
              <p>
                <strong>Source:</strong> {chunk.source}
              </p>

              <p>
                <strong>Page:</strong> {chunk.page}
              </p>

              <p>
                <strong>Category:</strong> {chunk.category}
              </p>

              <p>
                <strong>Similarity Score:</strong>{" "}
                {chunk.score.toFixed(2)}
              </p>
            </li>
          ))}
        </ul>
      </div>

    </div>
  );
}

export default ExplanationCard;