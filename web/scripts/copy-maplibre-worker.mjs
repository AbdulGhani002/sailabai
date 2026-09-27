// MapLibre GL 6 runs its tile work in an ES-module web worker that imports a shared chunk next to
// it. Bundlers cannot always resolve that worker URL (Turbopack in dev serves HTML instead), so we
// serve both files from public/ and point MapLibre at them with setWorkerUrl().
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const dist = dirname(require.resolve("maplibre-gl/dist/maplibre-gl.mjs"));
const out = join(process.cwd(), "public", "maplibre");
mkdirSync(out, { recursive: true });
for (const file of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) {
  copyFileSync(join(dist, file), join(out, file));
}
console.log(`copied MapLibre worker to ${out}`);
