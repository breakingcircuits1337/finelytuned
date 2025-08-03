import React from "react";
import clsx from "clsx";
import { AlertCircle } from "lucide-react";

export function Alert({ className = "", children }) {
  return (
    <div
      className={clsx(
        "flex items-center gap-2 bg-yellow-50 text-yellow-800 px-4 py-3 rounded border border-yellow-200",
        className
      )}
    >
      <AlertCircle className="w-5 h-5 text-yellow-400 shrink-0" />
      <span>{children}</span>
    </div>
  );
}