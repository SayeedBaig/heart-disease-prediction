function StepTwo({
  formData,
  handleChange,
  nextStep,
  prevStep,
  errors,
}) {
  return (
    <div className="max-w-xl mx-auto bg-white shadow-lg rounded-xl p-8">

      <h2 className="text-3xl font-bold text-blue-900 mb-6">
        Health Information
      </h2>

      {/* Systolic Blood Pressure */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Systolic Blood Pressure
        </label>

        <input
          type="number"
          name="ap_hi"
          value={formData.ap_hi}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        />

        {errors.ap_hi && (
          <p className="text-red-500 text-sm mt-1">
            {errors.ap_hi}
          </p>
        )}
      </div>

      {/* Diastolic Blood Pressure */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Diastolic Blood Pressure
        </label>

        <input
          type="number"
          name="ap_lo"
          value={formData.ap_lo}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        />

        {errors.ap_lo && (
          <p className="text-red-500 text-sm mt-1">
            {errors.ap_lo}
          </p>
        )}
      </div>

      {/* Cholesterol */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Cholesterol
        </label>

        <select
          name="cholesterol"
          value={formData.cholesterol}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select</option>
          <option value="1">Normal</option>
          <option value="2">Above Normal</option>
          <option value="3">Well Above Normal</option>
        </select>

        {errors.cholesterol && (
          <p className="text-red-500 text-sm mt-1">
            {errors.cholesterol}
          </p>
        )}
      </div>

      {/* Glucose */}

      <div className="mb-8">
        <label className="block mb-2 font-semibold">
          Glucose
        </label>

        <select
          name="gluc"
          value={formData.gluc}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select</option>
          <option value="1">Normal</option>
          <option value="2">Above Normal</option>
          <option value="3">Well Above Normal</option>
        </select>

        {errors.gluc && (
          <p className="text-red-500 text-sm mt-1">
            {errors.gluc}
          </p>
        )}
      </div>

      <div className="flex justify-between">

        <button
          onClick={prevStep}
          className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600"
        >
          Back
        </button>

        <button
          onClick={nextStep}
          className="bg-blue-900 text-white px-6 py-3 rounded-lg hover:bg-blue-800"
        >
          Next
        </button>

      </div>

    </div>
  );
}

export default StepTwo;