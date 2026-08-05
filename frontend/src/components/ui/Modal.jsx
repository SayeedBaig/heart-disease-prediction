import { useEffect } from "react";
import { Button } from "./Button";

export function Modal({ isOpen, onClose, title, children, actions }) {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div 
        className="fixed inset-0 bg-background/80 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />
      
      <div className="relative z-50 w-full max-w-lg rounded-lg border border-border bg-background p-6 shadow-lg duration-200 sm:rounded-xl">
        <div className="flex flex-col space-y-1.5 text-center sm:text-left mb-4">
          <h2 className="text-lg font-semibold leading-none tracking-tight">{title}</h2>
        </div>
        
        <div className="py-4">
          {children}
        </div>
        
        {(actions || onClose) && (
          <div className="flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2 mt-6">
            {actions || (
              <Button variant="primary" onClick={onClose}>
                Close
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
