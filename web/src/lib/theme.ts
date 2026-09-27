"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

function current(): Theme {
  if (typeof document === "undefined") return "light";
  const forced = document.documentElement.dataset.theme;
  if (forced === "light" || forced === "dark") return forced;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** The effective theme (saved choice, else the OS setting), kept in sync across components. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>("light");

  useEffect(() => {
    setThemeState(current());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setThemeState(current());
    mq.addEventListener("change", onChange);
    window.addEventListener("sailab-theme", onChange);
    return () => {
      mq.removeEventListener("change", onChange);
      window.removeEventListener("sailab-theme", onChange);
    };
  }, []);

  const setTheme = (t: Theme) => {
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem("sailab-theme", t);
    } catch {
      /* private mode: the choice just isn't remembered */
    }
    window.dispatchEvent(new Event("sailab-theme"));
  };
  return [theme, setTheme];
}
