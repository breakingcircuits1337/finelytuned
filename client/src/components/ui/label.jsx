import React from "react";
import clsx from "clsx";

export function Label({ className = "", ...props }) {
  return (
    <label
      {...props}
      className={clsx("block text-sm font-semibold text-gray-700 mb-1", className)}
    />
  );
}