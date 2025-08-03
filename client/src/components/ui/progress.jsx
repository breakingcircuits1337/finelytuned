import React from "react";
import clsx from "clsx";

export function Progress({ value = 0, max = 100, className = "", ...props }) {
  return (
    <div className={clsx("w-full bg-gray-200 rounded-full h-3", className)}>
      <div
        className="bg-indigo-500 h-3 rounded-full transition-all duration-300"
        style={{ width: `${Math.min(value, max)}%` }}
      />
    </div>
  );
}