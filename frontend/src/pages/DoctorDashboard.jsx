function DoctorDashboard() {
  const doctor = JSON.parse(localStorage.getItem("doctor"));

  return (
    <div className="min-h-screen bg-slate-100 p-10">
      <h1 className="text-4xl font-bold">
        Doctor Dashboard
      </h1>

      <div className="mt-8 bg-white rounded-xl shadow p-6">

        <h2 className="text-2xl font-semibold">
          Welcome
        </h2>

        <p className="mt-4">
          {doctor?.full_name || "Doctor"}
        </p>

        <p>{doctor?.email}</p>

        <p>{doctor?.hospital}</p>

      </div>
    </div>
  );
}

export default DoctorDashboard;