import {
  User,
  Mail,
  Phone,
  CalendarDays,
} from "lucide-react";

function PatientForm({
  formData,
  setFormData,
  onSubmit,
  loading,
}) {

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <form
      onSubmit={onSubmit}
      className="bg-white rounded-3xl shadow-xl p-10"
    >

      <div className="flex items-center gap-3 mb-8">

        <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">

          <User className="text-blue-600" />

        </div>

        <div>

          <h2 className="text-2xl font-bold text-slate-800">
            Patient Information
          </h2>

          <p className="text-slate-500">
            Enter the patient's registration details.
          </p>

        </div>

      </div>

      <div className="grid md:grid-cols-2 gap-6">

        {/* Full Name */}

        <div>

          <label className="block mb-2 font-medium text-slate-700">
            Full Name
          </label>

          <div className="relative">

            <User
              size={18}
              className="absolute left-3 top-3.5 text-slate-400"
            />

            <input
              type="text"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              placeholder="Enter full name"
              required
              className="w-full rounded-xl border border-slate-300 pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

          </div>

        </div>

        {/* Email */}

        <div>

          <label className="block mb-2 font-medium text-slate-700">
            Email
          </label>

          <div className="relative">

            <Mail
              size={18}
              className="absolute left-3 top-3.5 text-slate-400"
            />

            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="example@email.com"
              required
              className="w-full rounded-xl border border-slate-300 pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

          </div>

        </div>

        {/* Phone */}

        <div>

          <label className="block mb-2 font-medium text-slate-700">
            Phone
          </label>

          <div className="relative">

            <Phone
              size={18}
              className="absolute left-3 top-3.5 text-slate-400"
            />

            <input
              type="text"
              name="phone"
              value={formData.phone}
              onChange={handleChange}
              placeholder="Enter phone number"
              required
              className="w-full rounded-xl border border-slate-300 pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

          </div>

        </div>

        {/* Gender */}

        <div>

          <label className="block mb-2 font-medium text-slate-700">
            Gender
          </label>

          <select
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            required
            className="w-full rounded-xl border border-slate-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select Gender</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>

        </div>

        {/* Date of Birth */}

        <div>

          <label className="block mb-2 font-medium text-slate-700">
            Date of Birth
          </label>

          <div className="relative">

            <CalendarDays
              size={18}
              className="absolute left-3 top-3.5 text-slate-400"
            />

            <input
              type="date"
              name="date_of_birth"
              value={formData.date_of_birth}
              onChange={handleChange}
              required
              className="w-full rounded-xl border border-slate-300 pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />

          </div>

        </div>

      </div>

      <div className="mt-10 flex justify-end">

        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-semibold transition disabled:opacity-60"
        >
          {loading ? "Registering..." : "Register Patient"}
        </button>

      </div>

    </form>
  );
}

export default PatientForm;