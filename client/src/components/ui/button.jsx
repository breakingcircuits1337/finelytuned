import React from "react";
import clsx from "clsx";

export function Button({ className = "", ...props }) {
  return (
    <button
      {...props}
      className={clsx(
        "px-4 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 shadow disabled:opacity-50 disabled:cursor-not-allowed font-medium transition",
        className
      )}
    />
  );
}