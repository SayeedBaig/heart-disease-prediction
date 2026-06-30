function TwinSimulationCard({ digitalTwin }) {
  if (!digitalTwin) return null;

  return (
    <div className="bg-purple-50 border border-purple-200 rounded-xl p-6 mt-8 shadow-sm">

      <h2 className="text-2xl font-bold text-purple-700 mb-6">
        🧬 Digital Twin Simulation
      </h2>

      <div className="mb-6">
        <p className="text-lg">
          <strong>Current Risk:</strong>{" "}
          {(digitalTwin.baseline_risk * 100).toFixed(1)}%
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-4">

        {digitalTwin.simulations.map((simulation, index) => (
          <div
            key={index}
            className="bg-white border rounded-lg p-4 shadow-sm"
          >
            <h3 className="font-bold text-lg text-purple-700 mb-3">
              {simulation.scenario}
            </h3>

            <p>
              <strong>Predicted Risk:</strong>{" "}
              {(simulation.risk * 100).toFixed(1)}%
            </p>

            <p>
              <strong>Risk Improvement:</strong>{" "}
              {simulation.change}%
            </p>
          </div>
        ))}

      </div>

    </div>
  );
}

export default TwinSimulationCard;