const nf0 = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** 1234567 -> "1.2M", 12345 -> "12.3k", 950 -> "950" */
export function compact(n: number): string {
  if (!Number.isFinite(n)) return "–";
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return nf0.format(n);
}

export function whole(n: number): string {
  return Number.isFinite(n) ? nf0.format(n) : "–";
}

export function pct(p: number): string {
  return Number.isFinite(p) ? `${Math.round(p * 100)}%` : "–";
}

export function cusecs(q: number): string {
  return `${compact(q)} cusecs`;
}

export function horizonLabel(name: string, leadDays?: number): string {
  if (name === "now") return "Now";
  if (name === "next") return leadDays ? `Next pass (+${leadDays.toFixed(1)} d)` : "Next pass";
  const d = Number(name.replace("d", ""));
  return `+${d} day${d === 1 ? "" : "s"}`;
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
}

export function longDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/** Pakistan time (UTC+5) for pass and issue times. */
export function pkTime(iso: string): string {
  const d = new Date(new Date(iso).getTime() + 5 * 3600 * 1000);
  return `${d.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" })}, ${d
    .toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })} PKT`;
}
