export function LoadingState({ text = "Loading...", className = "" }) {
  return (
    <div className={`flex flex-col items-center justify-center space-y-4 p-12 ${className}`}>
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-muted border-t-primary" />
      <p className="text-sm text-muted-foreground">{text}</p>
    </div>
  );
}
