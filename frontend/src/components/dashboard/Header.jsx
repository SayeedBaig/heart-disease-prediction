import { Bell, Search, UserCircle2 } from "lucide-react";

function Header() {
  const doctor = JSON.parse(localStorage.getItem("doctor"));

  const hours = new Date().getHours();

  let greeting = "Good Evening";

  if (hours < 12) {
    greeting = "Good Morning";
  } else if (hours < 17) {
    greeting = "Good Afternoon";
  }

  return (
    <header className="bg-white border-b border-slate-200 px-10 py-6">

      <div className="flex items-center justify-between">

        {/* Left */}

        <div>

          <h1 className="text-3xl font-bold text-slate-800">
            {greeting}, Dr. {doctor?.full_name?.split(" ")[0] || "Doctor"}
          </h1>

          <p className="text-slate-500 mt-2">
            Welcome back to CardioAI. Let's make healthcare smarter today.
          </p>

        </div>

        {/* Right */}

        <div className="flex items-center gap-5">

          {/* Search */}

          <div className="hidden lg:flex items-center gap-3 rounded-xl border border-slate-200 px-4 py-2">

            <Search
              size={18}
              className="text-slate-500"
            />

            <input
              type="text"
              placeholder="Search patients..."
              className="outline-none text-sm bg-transparent w-56"
            />

          </div>

          {/* Notification */}

          <button className="relative rounded-xl bg-slate-100 p-3 hover:bg-blue-100 transition">

            <Bell
              size={20}
              className="text-slate-700"
            />

            <span className="absolute top-2 right-2 h-2 w-2 rounded-full bg-red-500"></span>

          </button>

          {/* Doctor Profile */}

          <div className="flex items-center gap-3 rounded-xl border border-slate-200 px-4 py-2">

            <UserCircle2
              size={36}
              className="text-blue-600"
            />

            <div>

              <h3 className="font-semibold text-slate-800">
                Dr. {doctor?.full_name || "Doctor"}
              </h3>

              <p className="text-sm text-slate-500">
                Cardiologist
              </p>

            </div>

          </div>

        </div>

      </div>

    </header>
  );
}

export default Header;
