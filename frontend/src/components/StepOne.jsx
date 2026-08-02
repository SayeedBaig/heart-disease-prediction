import {
  User,
  CalendarDays,
  BadgeInfo,
} from "lucide-react";

function StepOne({
  formData,
  handleChange,
  nextStep,
  errors,
}) {
  const patientId = localStorage.getItem("patient_id");
  const patientName = localStorage.getItem("patient_name");
  const patientGender = localStorage.getItem("patient_gender");
  const patientDOB = localStorage.getItem("patient_dob");

  return (
    <div className="max-w-5xl mx-auto">

      {/* Patient Summary */}

      <div className="bg-white rounded-3xl shadow-lg p-8 mb-8">

        <div className="flex items-center gap-3 mb-6">

          <div className="w-14 h-14 rounded-xl bg-blue-100 flex items-center justify-center">
            <User className="text-blue-600" size={28} />
          </div>

          <div>
            <h2 className="text-3xl font-bold text-slate-800">
              Patient Summary
            </h2>

            <p className="text-slate-500">
              Registered patient information
            </p>
          </div>

        </div>

        <div className="grid md:grid-cols-2 gap-6">

          <div className="rounded-2xl bg-slate-50 p-5">
            <p className="text-sm text-slate-500">Patient ID</p>
            <h3 className="text-xl font-bold text-blue-600">
              {patientId}
            </h3>
          </div>

          <div className="rounded-2xl bg-slate-50 p-5">
            <p className="text-sm text-slate-500">Patient Name</p>
            <h3 className="text-xl font-semibold">
              {patientName}
            </h3>
          </div>

          <div className="rounded-2xl bg-slate-50 p-5 flex items-center gap-3">
            <BadgeInfo className="text-blue-600" />
            <div>
              <p className="text-sm text-slate-500">Gender</p>
              <h3 className="font-semibold">
                {patientGender}
              </h3>
            </div>
          </div>

          <div className="rounded-2xl bg-slate-50 p-5 flex items-center gap-3">
            <CalendarDays className="text-blue-600" />
            <div>
              <p className="text-sm text-slate-500">
                Date of Birth
              </p>
              <h3 className="font-semibold">
                {patientDOB}
              </h3>
            </div>
          </div>

        </div>

      </div>

      {/* Clinical Measurements */}

      <div className="bg-white rounded-3xl shadow-lg p-8">

        <h2 className="text-2xl font-bold text-slate-800 mb-2">
          Clinical Measurements
        </h2>

        <p className="text-slate-500 mb-8">
          Enter the patient's current measurements.
        </p>

        <div className="grid md:grid-cols-3 gap-6">

          {/* Age */}

          <div>

            <label className="block mb-2 font-semibold">
              Age
            </label>

            <input
              type="number"
              name="age"
              value={formData.age}
              onChange={handleChange}
              className="w-full rounded-xl border p-3"
            />

            {errors.age && (
              <p className="text-red-500 text-sm mt-1">
                {errors.age}
              </p>
            )}

          </div>

          {/* Height */}

          <div>

            <label className="block mb-2 font-semibold">
              Height (cm)
            </label>

            <input
              type="number"
              name="height"
              value={formData.height}
              onChange={handleChange}
              className="w-full rounded-xl border p-3"
            />

            {errors.height && (
              <p className="text-red-500 text-sm mt-1">
                {errors.height}
              </p>
            )}

          </div>

          {/* Weight */}

          <div>

            <label className="block mb-2 font-semibold">
              Weight (kg)
            </label>

            <input
              type="number"
              name="weight"
              value={formData.weight}
              onChange={handleChange}
              className="w-full rounded-xl border p-3"
            />

            {errors.weight && (
              <p className="text-red-500 text-sm mt-1">
                {errors.weight}
              </p>
            )}

          </div>

        </div>

        <button
          onClick={nextStep}
          className="mt-10 w-full rounded-xl bg-blue-600 py-4 text-white font-semibold hover:bg-blue-700 transition"
        >
          Continue →
        </button>

      </div>

    </div>
  );
}

export default StepOne;