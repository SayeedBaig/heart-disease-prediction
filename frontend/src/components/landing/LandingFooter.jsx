/**
 * LandingFooter — page footer with brand, links, and clinical note.
 * Preserves: #capabilities, #workflow, #digital-twin anchor links,
 * /get-started route, copyright year.
 */
import { Link } from "react-router-dom";
import Container from "../ui/Container";

export default function LandingFooter() {
  return (
    <footer className="border-t border-[var(--border-color)] bg-[var(--card-bg)] text-xs text-[var(--text-muted)]">
      <Container>
        <div className="grid gap-10 md:grid-cols-4">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2.5 text-sm font-bold text-[var(--text-primary)]">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[var(--accent-melanzane)] text-xs font-black text-white">
                AI
              </div>
              <span>CardioAI Intelligence Platform</span>
            </div>
            <p className="mt-4 max-w-md text-xs leading-relaxed text-[var(--text-secondary)] sm:text-sm">
              AI powered cardiovascular screening, explainable reports, and Digital Twin simulation for clinical decision support.
            </p>
          </div>

          {/* Platform links */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wide text-[var(--text-primary)]">
              Platform
            </h3>
            <div className="mt-4 space-y-2.5 text-xs">
              <a href="#capabilities" className="block text-left hover:text-[var(--accent-melanzane)]">
                Capabilities
              </a>
              <a href="#workflow" className="block text-left hover:text-[var(--accent-melanzane)]">
                Clinical Workflow
              </a>
              <a href="#digital-twin" className="block text-left hover:text-[var(--accent-melanzane)]">
                Digital Twin
              </a>
              <Link to="/get-started" className="block hover:text-[var(--accent-melanzane)]">
                Access Portals
              </Link>
            </div>
          </div>

          {/* Clinical note */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wide text-[var(--text-primary)]">
              Clinical Note
            </h3>
            <p className="mt-4 text-xs leading-relaxed text-[var(--text-secondary)]">
              Designed strictly for clinical decision support and patient screening. Not a replacement for formal physician diagnosis.
            </p>
          </div>
        </div>
      </Container>

      <div className="border-t border-[var(--border-color)] py-5 text-center text-xs text-[var(--text-muted)]">
        <Container>
          © {new Date().getFullYear()} CardioAI Platform. All rights reserved.
        </Container>
      </div>
    </footer>
  );
}