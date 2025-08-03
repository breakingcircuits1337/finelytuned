import React from "react";
import clsx from "clsx";

export function Input({ className = "", ...props }) {
  return (
    <input
      {...props}
      className={clsx(
        "block w-full rounded border border-gray-300 px-3 py-2 focus:outline-none focus:border-indigo-500",
        className
      )}
    />
  );
}