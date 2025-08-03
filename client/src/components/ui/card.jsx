import React from "react";
import clsx from "clsx";

export function Card({ className = "", ...props }) {
  return (
    <div
      {...props}
      className={clsx(
        "bg-white rounded-lg shadow border border-gray-200 p-6",
        className
      )}
    />
  );
}