import React, { useState } from "react";
import clsx from "clsx";

export function Tabs({ defaultValue, children, className = "" }) {
  const [value, setValue] = useState(defaultValue);
  const tabTriggers = [];
  const tabContents = [];

  React.Children.forEach(children, child => {
    if (child?.type === TabsList) {
      React.Children.forEach(child.props.children, trigger => {
        if (trigger?.type === TabsTrigger) {
          tabTriggers.push(React.cloneElement(trigger, {
            isActive: value === trigger.props.value,
            onClick: () => setValue(trigger.props.value)
          }));
        }
      });
    }
    if (child?.type === TabsContent) {
      tabContents.push(child);
    }
  });

  return (
    <div className={className}>
      <div className="flex space-x-2">{tabTriggers}</div>
      {tabContents.map(child =>
        value === child.props.value ? child : null
      )}
    </div>
  );
}

export function TabsList({ children }) {
  return <div>{children}</div>;
}

export function TabsTrigger({ value, children, isActive, onClick }) {
  return (
    <button
      type="button"
      className={clsx(
        "px-4 py-2 rounded-t bg-gray-100 border-b-2 font-medium",
        isActive
          ? "border-indigo-500 text-indigo-700 bg-white"
          : "border-transparent text-gray-500 hover:text-indigo-600"
      )}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children }) {
  return <div className="py-4">{children}</div>;
}