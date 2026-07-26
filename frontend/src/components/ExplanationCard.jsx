import {
  BookOpen,
  FileText,
  Search,
  Database,
  CheckCircle2,
} from "lucide-react";

function ExplanationCard({ explanation }) {
  if (!explanation) return null;

  return (
    <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-8 mt-8 transition-all duration-300">

      {/* Header */}

      <div className="flex items-center gap-4 mb-8">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-600 to-blue-500 flex items-center justify-center shadow-lg">
          <BookOpen className="text-white" size={28} />
        </div>

        <div>
          <h2 className="text-2xl font-bold text-slate-900">
            Medical Explanation
          </h2>

          <p className="text-slate-500">
            AI generated explanation with retrieved medical evidence
          </p>
        </div>
      </div>

      {/* Status */}

      <div className="flex items-center gap-3 bg-green-50 border border-green-200 rounded-2xl p-4 mb-8">
        <CheckCircle2 className="text-green-600" size={22} />

        <div>
          <p className="text-sm text-slate-500">Status</p>
          <p className="font-semibold text-green-700">
            {explanation.status}
          </p>
        </div>
      </div>

      {/* Summary */}

      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 mb-6">
        <div className="flex items-center gap-3 mb-3">
          <FileText className="text-blue-600" />
          <h3 className="text-xl font-semibold">
            Summary
          </h3>
        </div>

        <p className="leading-8 text-slate-700">
          {explanation.explanation.summary}
        </p>
      </div>

      {/* Detailed Explanation */}

      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 mb-6">
        <div className="flex items-center gap-3 mb-3">
          <BookOpen className="text-indigo-600" />
          <h3 className="text-xl font-semibold">
            Detailed Explanation
          </h3>
        </div>

        <p className="leading-8 text-slate-700">
          {explanation.explanation.details}
        </p>
      </div>

      {/* Query */}

      <div className="bg-blue-50 rounded-2xl border border-blue-200 p-6 mb-8">
        <div className="flex items-center gap-3 mb-3">
          <Search className="text-blue-600" />
          <h3 className="text-xl font-semibold">
            Query Used
          </h3>
        </div>

        <p className="italic text-slate-700">
          {explanation.query}
        </p>
      </div>

      {/* References */}

      <div>

        <div className="flex items-center gap-3 mb-5">
          <Database className="text-blue-600" />
          <h3 className="text-2xl font-bold text-slate-900">
            Retrieved References
          </h3>
        </div>

        <div className="space-y-5">

          {explanation.chunks.map((chunk, index) => (

            <div
              key={index}
              className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm transition-all duration-300"
            >

              <div className="grid md:grid-cols-2 gap-4">

                <div>
                  <p className="text-sm text-slate-500">
                    Source
                  </p>

                  <p className="font-semibold text-slate-800">
                    {chunk.source}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-slate-500">
                    Page
                  </p>

                  <p className="font-semibold text-slate-800">
                    {chunk.page}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-slate-500">
                    Category
                  </p>

                  <p className="font-semibold text-slate-800">
                    {chunk.category}
                  </p>
                </div>

                <div>
                  <p className="text-sm text-slate-500">
                    Similarity Score
                  </p>

                  <p className="font-semibold text-blue-600">
                    {chunk.score.toFixed(2)}
                  </p>
                </div>

              </div>

            </div>

          ))}

        </div>

      </div>

    </div>
  );
}

export default ExplanationCard;