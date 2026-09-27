import type { Metadata, Viewport } from "next";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import { TopBar } from "@/components/TopBar";

export const metadata: Metadata = {
  title: "SailabAI flood twin",
  description:
    "Flood digital twin for the Chenab, Ravi and Sutlej around Multan: today's flood from radar and 1 to 7 day forecasts with confidence. Experimental, not an official warning.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

// Apply the saved theme before first paint so the page does not flash.
const themeScript = `try{var t=localStorage.getItem('sailab-theme');if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t}}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <TopBar />
        {children}
      </body>
    </html>
  );
}
