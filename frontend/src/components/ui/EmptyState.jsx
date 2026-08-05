import { Button } from "./Button";

export function EmptyState({ icon, title, description, actionText, onAction, className = "" }) {
  return (
    <div className={`flex flex-col items-center justify-center text-center p-8 border border-dashed border-border rounded-xl bg-secondary/20 ${className}`}>
      {icon && (
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary text-muted-foreground mb-4">
          {icon}
        </div>
      )}
      <h3 className="mt-2 text-sm font-semibold text-foreground">{title}</h3>
      <p className="mt-1 text-sm text-muted-foreground max-w-sm">
        {description}
      </p>
      {actionText && onAction && (
        <div className="mt-6">
          <Button onClick={onAction} variant="outline" size="sm">
            {actionText}
          </Button>
        </div>
      )}
    </div>
  );
}
