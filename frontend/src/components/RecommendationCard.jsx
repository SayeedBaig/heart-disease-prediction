import { Brain, CheckCircle2 } from "lucide-react";

function RecommendationCard({ prediction }) {
  if (!prediction) return null;

  return (
    <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-8 mb-8 transition-all duration-300">

      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center shadow-lg">
          <Brain className="text-white" size={28} />
        </div>

        <div>
          <h3 className="text-2xl font-bold text-slate-900">
            AI Recommendation
          </h3>

          <p className="text-slate-500">
            Personalized insights generated using Explainable AI
          </p>
        </div>
      </div>

      {/* AI Explanation */}

      <div className="bg-blue-50 border border-blue-100 rounded-2xl p-5 mb-8">
        <p className="leading-8 text-slate-700">
          {prediction.rag.explanation}
        </p>
      </div>

      {/* Recommendation List */}

      <div className="space-y-4">

        {prediction.rag.details.map((item, index) => (
          <div
            key={index}
            className="flex items-start gap-4 bg-slate-50 rounded-2xl p-4 border border-slate-100 transition-all duration-300 hover:bg-blue-50 hover:border-blue-200"
          >
            <div className="mt-1">
              <CheckCircle2
                className="text-green-600"
                size={22}
              />
            </div>

            <p className="text-slate-700 leading-7">
              {item}
            </p>
          </div>
        ))}

      </div>

    </div>
  );
}

export default RecommendationCard;