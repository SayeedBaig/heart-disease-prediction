import { Navbar } from "./Navbar";

export function AppLayout({ children, className = "" }) {
  return (
    <div className={`relative flex min-h-screen flex-col ${className}`}>
      <Navbar />
      <main className="flex-1">
        {children}
      </main>
    </div>
  );
}
