import React from "react";
import clsx from "clsx";

export function Toggle({ checked = false, onChange, className = "", ...props }) {
  return (
    <input
      type="checkbox"
      checked={checked}
      onChange={onChange}
      className={clsx(
        "form-checkbox h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500",
        className
      )}
      {...props}
    />
  );
}