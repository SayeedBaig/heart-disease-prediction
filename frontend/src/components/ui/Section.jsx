export function Section({ children, className = "", noPadding = false }) {
  return (
    <section className={`${noPadding ? "" : "py-12 md:py-16 lg:py-20"} ${className}`}>
      {children}
    </section>
  );
}