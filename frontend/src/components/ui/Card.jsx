export function Card({ children, className = "", noPadding = false }) {
  return (
    <div
      className={`bg-card text-card-foreground border border-border shadow-sm rounded-xl flex flex-col overflow-hidden ${
        noPadding ? "" : "p-6"
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = "" }) {
  return (
    <div className={`flex flex-col space-y-1.5 mb-4 ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = "" }) {
  return (
    <h3 className={`font-semibold leading-none tracking-tight ${className}`}>
      {children}
    </h3>
  );
}

export function CardDescription({ children, className = "" }) {
  return (
    <p className={`text-sm text-muted-foreground ${className}`}>
      {children}
    </p>
  );
}

export function CardContent({ children, className = "" }) {
  return <div className={`flex-1 ${className}`}>{children}</div>;
}

export function CardFooter({ children, className = "" }) {
  return (
    <div className={`flex items-center pt-4 mt-auto ${className}`}>
      {children}
    </div>
  );
}