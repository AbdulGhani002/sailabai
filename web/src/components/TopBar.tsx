"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "@/lib/theme";

const LINKS = [
  { href: "/", label: "Map" },
  { href: "/results", label: "Results" },
  { href: "/about", label: "About" },
];

export function TopBar() {
  const path = usePathname();
  const [theme, setTheme] = useTheme();
  return (
    <header className="topbar">
      <div className="brand">
        <strong>SailabAI</strong>
        <span>Flood digital twin · Chenab, Ravi and Sutlej around Multan</span>
      </div>
      <nav className="nav" aria-label="Main">
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} aria-current={path === l.href ? "page" : undefined}>
            {l.label}
          </Link>
        ))}
      </nav>
      <button
        className="icon-btn"
        type="button"
        aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      >
        {theme === "dark" ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
            <circle cx="12" cy="12" r="4.5" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
            <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" />
          </svg>
        )}
      </button>
    </header>
  );
}

export function Notice({ disclaimer, synthetic }: { disclaimer: string; synthetic?: boolean }) {
  return (
    <div className="notice" role="note">
      <span>
        <span className="i">ⓘ</span> {disclaimer} For flood experts and planners; follow NDMA, PDMA and PMD for official
        warnings.
      </span>
      {synthetic && (
        <span className="badge" title="Everything on this page comes from the synthetic demo world, not real observations">
          <span className="dot" aria-hidden /> Synthetic demo data
        </span>
      )}
    </div>
  );
}
