import type { NextConfig } from "next";

// The browser talks to this Next.js server only; /api and /tiles are forwarded to FastAPI.
const API_URL = process.env.SAILAB_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_URL}/api/:path*` },
      { source: "/tiles/:path*", destination: `${API_URL}/tiles/:path*` },
    ];
  },
};

export default nextConfig;
