import { Dna, TrendingDown, HeartPulse } from "lucide-react";

function TwinSimulationCard({ digitalTwin }) {
  if (!digitalTwin) return null;

  return (
    <div className="bg-white rounded-3xl border border-slate-200 shadow-lg p-8 mt-8 transition-all duration-300">

      {/* Header */}

      <div className="flex items-center gap-4 mb-8">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center shadow-lg">
          <Dna className="text-white" size={28} />
        </div>

        <div>
          <h2 className="text-2xl font-bold text-slate-900">
            Digital Twin Simulation
          </h2>

          <p className="text-slate-500">
            AI simulations showing how lifestyle changes may influence cardiovascular risk.
          </p>
        </div>
      </div>

      {/* Current Risk */}

      <div className="bg-purple-50 border border-purple-200 rounded-2xl p-6 mb-8">

        <div className="flex items-center justify-between">

          <div className="flex items-center gap-3">
            <HeartPulse className="text-purple-600" size={28} />

            <div>
              <p className="text-sm text-slate-500">
                Current Estimated Risk
              </p>

              <h3 className="text-3xl font-bold text-purple-700">
                {(digitalTwin.baseline_risk * 100).toFixed(1)}%
              </h3>
            </div>
          </div>

        </div>

      </div>

      {/* Simulations */}

      <div className="grid md:grid-cols-2 gap-6">

        {digitalTwin.simulations.map((simulation, index) => (

          <div
            key={index}
            className="bg-slate-50 border border-slate-200 rounded-2xl p-6 transition-all duration-300"
          >

            <div className="flex items-center gap-3 mb-5">

              <div className="w-12 h-12 rounded-xl bg-purple-100 flex items-center justify-center">
                <TrendingDown className="text-purple-600" />
              </div>

              <div>
                <h3 className="text-lg font-bold text-slate-900">
                  {simulation.scenario}
                </h3>

                <p className="text-sm text-slate-500">
                  Lifestyle Simulation
                </p>
              </div>

            </div>

            <div className="space-y-4">

              <div className="flex justify-between">
                <span className="text-slate-600">
                  Predicted Risk
                </span>

                <span className="font-bold text-blue-600">
                  {(simulation.risk * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between">
                <span className="text-slate-600">
                  Risk Improvement
                </span>

                <span className="font-bold text-green-600">
                  {simulation.change}%
                </span>
              </div>

            </div>

          </div>

        ))}

      </div>

    </div>
  );
}

export default TwinSimulationCard;