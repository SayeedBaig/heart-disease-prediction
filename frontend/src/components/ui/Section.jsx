export function Section({ children, className = "", noPadding = false, style, id }) {
  return (
    <section id={id} className={`${noPadding ? "" : "py-12 md:py-16 lg:py-20"} ${className}`} style={style}>
      {children}
    </section>
  );
}