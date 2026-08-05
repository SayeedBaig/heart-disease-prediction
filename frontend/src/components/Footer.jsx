function Footer() {
  return (
    <footer className="bg-[var(--bg-secondary)] border-t border-[var(--border-color)] text-center py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-2">
        <h2 className="text-base font-bold text-[var(--text-primary)] flex items-center justify-center gap-2">
          <span className="w-6 h-6 rounded-lg bg-[var(--accent-melanzane)] text-white flex items-center justify-center text-xs font-black">C</span>
          CardioAI Platform
        </h2>
        <p className="caption-small text-[var(--text-secondary)]">
          Multi-Modal Artificial Intelligence System for Cardiovascular Disease Prediction & Decision Support
        </p>
        <p className="caption-small text-[var(--text-muted)] pt-2 border-t border-[var(--border-subtle)] mt-4">
          © {new Date().getFullYear()} CardioAI Platform. All rights reserved.
        </p>
      </div>
    </footer>
  );
}

export default Footer;