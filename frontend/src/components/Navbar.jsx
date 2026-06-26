import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="flex justify-between items-center px-10 py-5 bg-blue-950 text-white shadow-lg">

      {/* Logo */}
      <h1 className="text-3xl font-bold">
        ❤️ CardioAI
      </h1>

      {/* Navigation Links */}
      <div className="space-x-8 text-lg">

        <Link to="/" className="hover:text-blue-300">
          Home
        </Link>

        <Link to="/diagnose" className="hover:text-blue-300">
          Diagnose
        </Link>

      </div>

    </nav>
  );
}

export default Navbar;