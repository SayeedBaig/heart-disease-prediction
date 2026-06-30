function RecommendationCard({ prediction }) {
  if (!prediction) return null;

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">

      <h3 className="text-2xl font-bold text-gray-800 mb-4">
        🤖 AI Recommendation
      </h3>

      <p className="leading-8 text-gray-700 mb-6">
        {prediction.rag.explanation}
      </p>

      <ul className="space-y-3">
        {prediction.rag.details.map((item, index) => (
          <li
            key={index}
            className="flex items-start gap-3 text-gray-700"
          >
            <span>✅</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>

    </div>
  );
}

export default RecommendationCard;