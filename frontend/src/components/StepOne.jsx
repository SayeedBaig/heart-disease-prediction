function StepOne({
  formData,
  handleChange,
  nextStep,
  errors,
}) {
  return (
    <div className="max-w-xl mx-auto bg-white shadow-lg rounded-xl p-8">

      <h2 className="text-3xl font-bold text-blue-900 mb-6">
        Patient Information
      </h2>

      {/* Age */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Age
        </label>

        <input
          type="number"
          name="age"
          value={formData.age}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        />

        {errors.age && (
          <p className="text-red-500 text-sm mt-1">
            {errors.age}
          </p>
        )}
      </div>

      {/* Gender */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Gender
        </label>

        <select
          name="gender"
          value={formData.gender}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        >
          <option value="">Select Gender</option>
          <option value="1">Female</option>
          <option value="2">Male</option>
        </select>

        {errors.gender && (
          <p className="text-red-500 text-sm mt-1">
            {errors.gender}
          </p>
        )}
      </div>

      {/* Height */}

      <div className="mb-4">
        <label className="block mb-2 font-semibold">
          Height (cm)
        </label>

        <input
          type="number"
          name="height"
          value={formData.height}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        />

        {errors.height && (
          <p className="text-red-500 text-sm mt-1">
            {errors.height}
          </p>
        )}
      </div>

      {/* Weight */}

      <div className="mb-8">
        <label className="block mb-2 font-semibold">
          Weight (kg)
        </label>

        <input
          type="number"
          name="weight"
          value={formData.weight}
          onChange={handleChange}
          className="w-full border rounded-lg p-3"
        />

        {errors.weight && (
          <p className="text-red-500 text-sm mt-1">
            {errors.weight}
          </p>
        )}
      </div>

      <button
        onClick={nextStep}
        className="w-full bg-blue-900 text-white py-3 rounded-lg hover:bg-blue-800"
      >
        Next
      </button>

    </div>
  );
}

export default StepOne;


