import React from "react";
import clsx from "clsx";

export function Textarea({ className = "", ...props }) {
  return (
    <textarea
      {...props}
      className={clsx(
        "block w-full rounded border border-gray-300 px-3 py-2 resize-vertical focus:outline-none focus:border-indigo-500",
        className
      )}
    />
  );
}