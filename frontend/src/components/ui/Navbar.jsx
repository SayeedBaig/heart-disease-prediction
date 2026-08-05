import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, HeartPulse, Moon, Sun } from "lucide-react";
import { Container } from "./Container";
import { Button } from "./Button";

export function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const isHome = location.pathname === "/";

  // Dark mode state
  const [theme, setTheme] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("theme") || "light";
    }
    return "light";
  });

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === "light" ? "dark" : "light");
  };

  const handleScrollToFeatures = (e) => {
    e.preventDefault();
    if (!isHome) {
      navigate("/");
      setTimeout(() => {
        document.getElementById("capabilities")?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } else {
      document.getElementById("capabilities")?.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <Container>
        <div className="flex h-14 items-center justify-between">
          <div className="flex items-center gap-4 md:gap-8">
            {!isHome && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => navigate(-1)} 
                className="mr-2 h-8 w-8 text-muted-foreground hover:text-foreground cursor-pointer"
                title="Go Back"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
            <Link to="/" className="flex items-center space-x-2 group cursor-pointer" title="Return to Home">
              <div className="h-8 w-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center group-hover:scale-105 transition-transform">
                <HeartPulse className="h-5 w-5" />
              </div>
              <span className="inline-block font-extrabold text-lg tracking-tight">CardioAI</span>
            </Link>
            <nav className="hidden md:flex gap-6">
              <button
                onClick={handleScrollToFeatures}
                className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
              >
                Features
              </button>
              <a
                href="https://github.com/"
                target="_blank"
                rel="noreferrer"
                className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
              >
                Documentation
              </a>
            </nav>
          </div>
          
          <div className="flex items-center gap-2">
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={toggleTheme}
              className="mr-2 h-9 w-9 text-muted-foreground hover:text-foreground cursor-pointer"
              title="Toggle Theme"
            >
              {theme === "light" ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </Button>
            <Link to="/role-selection">
              <Button variant="ghost" size="sm" className="cursor-pointer">Login</Button>
            </Link>
            <Link to="/get-started">
              <Button size="sm" className="cursor-pointer">Get Started</Button>
            </Link>
          </div>
        </div>
      </Container>
    </header>
  );
}
