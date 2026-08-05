export function Container({ children, className = "" }) {
  return (
    <div className={`max-w-[1280px] w-full mx-auto px-4 md:px-8 ${className}`}>
      {children}
    </div>
  );
}