/**
 * Badge — small eyebrow/pill label used above section headings.
 * Renders as an inline-flex pill with accent border + light background.
 */
export default function Badge({ children, className = "" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border border-[var(--accent-melanzane-border)] bg-[var(--accent-melanzane-light)] px-3.5 py-1.5 text-xs font-bold uppercase tracking-wider text-[var(--accent-melanzane)] shadow-sm ${className}`}
    >
      {children}
    </span>
  );
}