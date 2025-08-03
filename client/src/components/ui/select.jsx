import React from "react";
import clsx from "clsx";

export function Select({ className = "", children, ...props }) {
  return (
    <select
      {...props}
      className={clsx(
        "block w-full rounded border border-gray-300 px-3 py-2 bg-white focus:outline-none focus:border-indigo-500",
        className
      )}
    >
      {children}
    </select>
  );
}