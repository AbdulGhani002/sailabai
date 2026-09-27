"use client";

import {
  type GeoJSONSource,
  Map as MLMap,
  type MapLayerMouseEvent,
  NavigationControl,
  type RasterTileSource,
  ScaleControl,
  setWorkerUrl,
} from "maplibre-gl";

// served from public/ (see scripts/copy-maplibre-worker.mjs)
if (typeof window !== "undefined") setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
import { useEffect, useRef } from "react";
import type { Meta } from "@/lib/api";
import type { Theme } from "@/lib/theme";

// OpenFreeMap: free basemaps, commercial use allowed, no key.
const STYLES: Record<Theme, string> = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};

type Props = {
  meta: Meta;
  tileUrl: string | null;
  theme: Theme;
  selectedGauge: string | null;
  onGaugeSelect: (id: string) => void;
};

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function absolute(url: string): string {
  return url.startsWith("http") ? url : `${window.location.origin}${url}`;
}

function bboxPolygon([w, s, e, n]: Meta["bbox"]): GeoJSON.Feature {
  return {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
  };
}

export function FloodMap({ meta, tileUrl, theme, selectedGauge, onGaugeSelect }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const tileRef = useRef(tileUrl);
  const gaugeRef = useRef(selectedGauge);
  const selectRef = useRef(onGaugeSelect);
  const styleTheme = useRef<Theme>(theme);
  tileRef.current = tileUrl;
  gaugeRef.current = selectedGauge;
  selectRef.current = onGaugeSelect;

  useEffect(() => {
    if (!container.current) return;
    const [w, s, e, n] = meta.bbox;
    const map = new MLMap({
      container: container.current,
      style: STYLES[theme],
      bounds: [[w, s], [e, n]],
      fitBoundsOptions: { padding: 24 },
      attributionControl: { compact: true },
      maxZoom: 13,
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-right");
    map.on("style.load", () => addOverlays(map));
    // the container may still be sizing when the map is created: fit the study area once loaded
    map.once("load", () => {
      map.resize();
      map.fitBounds([[w, s], [e, n]], { padding: 24, animate: false });
    });
    map.on("click", "gauges", (ev: MapLayerMouseEvent) => {
      const id = ev.features?.[0]?.properties?.id;
      if (typeof id === "string") selectRef.current(id);
    });
    map.on("mouseenter", "gauges", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "gauges", () => (map.getCanvas().style.cursor = ""));
    // the panel below the map loads later and shrinks it; keep the canvas matched to its container
    const resizer = new ResizeObserver(() => map.resize());
    resizer.observe(container.current);
    mapRef.current = map;
    if (process.env.NODE_ENV !== "production") (window as unknown as { __sailabMap?: MLMap }).__sailabMap = map;
    return () => {
      resizer.disconnect();
      map.remove();
      mapRef.current = null;
    };
    // the map is created once per dataset; theme and tiles are updated below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta.cube]);

  function addOverlays(map: MLMap) {
    const layers = map.getStyle().layers ?? [];
    const firstSymbol = layers.find((l) => l.type === "symbol")?.id;
    // reuse a font the basemap's glyph server actually has
    const withFont = layers.find((l) => l.type === "symbol" && (l.layout as Record<string, unknown> | undefined)?.["text-font"]);
    const font = ((withFont?.layout as Record<string, unknown> | undefined)?.["text-font"] as string[] | undefined) ?? ["Noto Sans Regular"];
    const ink2 = cssVar("--ink-2");
    const surface = cssVar("--surface");
    const url = tileRef.current;

    if (!map.getSource("flood")) {
      map.addSource("flood", {
        type: "raster",
        tiles: url ? [absolute(url)] : [],
        tileSize: 256,
        bounds: meta.bbox,
        attribution: "Flood layers: SailabAI (experimental)",
      });
      map.addLayer(
        { id: "flood", type: "raster", source: "flood", layout: { visibility: url ? "visible" : "none" },
          paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } },
        firstSymbol,
      );
    }
    map.addSource("aoi", { type: "geojson", data: bboxPolygon(meta.bbox) });
    map.addLayer({ id: "aoi", type: "line", source: "aoi",
      paint: { "line-color": ink2, "line-width": 1, "line-dasharray": [4, 3], "line-opacity": 0.7 } });
    map.addSource("rivers", { type: "geojson", data: meta.rivers as GeoJSON.FeatureCollection });
    map.addLayer({ id: "rivers", type: "line", source: "rivers",
      paint: { "line-color": cssVar("--muted"), "line-width": 1.2, "line-opacity": 0.8 } }, firstSymbol);
    map.addSource("places", { type: "geojson", data: meta.places as GeoJSON.FeatureCollection });
    map.addLayer({ id: "places-dot", type: "circle", source: "places",
      paint: { "circle-radius": ["match", ["get", "kind"], "city", 4, 3], "circle-color": cssVar("--ink"),
        "circle-stroke-color": surface, "circle-stroke-width": 1.5 } });
    map.addLayer({ id: "places-label", type: "symbol", source: "places",
      layout: { "text-field": ["get", "name"], "text-size": ["match", ["get", "kind"], "city", 13, 11.5],
        "text-offset": [0, 0.9], "text-anchor": "top", "text-font": font },
      paint: { "text-color": cssVar("--ink"), "text-halo-color": surface, "text-halo-width": 1.5 } });
    map.addSource("breaches", { type: "geojson", data: meta.breaches as GeoJSON.FeatureCollection });
    map.addLayer({ id: "breaches", type: "circle", source: "breaches",
      paint: { "circle-radius": 5, "circle-color": cssVar("--series-2"), "circle-stroke-color": surface,
        "circle-stroke-width": 2 } });
    map.addSource("gauges", { type: "geojson", data: meta.gauges as GeoJSON.FeatureCollection });
    map.addLayer({ id: "gauges", type: "circle", source: "gauges",
      paint: {
        "circle-radius": ["case", ["==", ["get", "id"], gaugeRef.current ?? ""], 8, 5.5],
        "circle-color": ["case", ["==", ["get", "country"], "IN"], surface, cssVar("--series-1")],
        "circle-stroke-color": cssVar("--series-1"),
        "circle-stroke-width": 2,
      } });
    map.addLayer({ id: "gauges-label", type: "symbol", source: "gauges", minzoom: 7,
      layout: { "text-field": ["get", "name"], "text-size": 11, "text-offset": [0, -1.2], "text-anchor": "bottom",
        "text-font": font },
      paint: { "text-color": cssVar("--ink-2"), "text-halo-color": surface, "text-halo-width": 1.5 } });
  }

  // flood layer: swap tiles when the run, horizon, view or theme changes
  useEffect(() => {
    // isStyleLoaded() is false while tiles load, so check for our source instead; if it does not
    // exist yet, addOverlays() picks up the current tile URL when the style finishes loading.
    const map = mapRef.current;
    const source = map?.getSource("flood") as RasterTileSource | undefined;
    if (!map || !source) return;
    if (tileUrl) {
      source.setTiles([absolute(tileUrl)]);
      map.setLayoutProperty("flood", "visibility", "visible");
    } else {
      map.setLayoutProperty("flood", "visibility", "none");
    }
  }, [tileUrl]);

  // basemap follows the theme; overlays are re-added on style.load
  useEffect(() => {
    const map = mapRef.current;
    if (!map || styleTheme.current === theme) return;
    styleTheme.current = theme;
    map.setStyle(STYLES[theme]);
  }, [theme]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("gauges")) return;
    map.setPaintProperty("gauges", "circle-radius", ["case", ["==", ["get", "id"], selectedGauge ?? ""], 8, 5.5]);
  }, [selectedGauge]);

  useEffect(() => {
    const map = mapRef.current;
    const src = map?.getSource("breaches") as GeoJSONSource | undefined;
    src?.setData(meta.breaches as GeoJSON.FeatureCollection);
  }, [meta.breaches]);

  return <div ref={container} className="map" aria-label="Flood map" role="region" />;
}
