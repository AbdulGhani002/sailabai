#!/usr/bin/env bash
# The whole SailabAI demo: synthetic world -> models -> calibration -> scores -> twin.
#   ./scripts/demo.sh           (full, about an hour on an RTX 4050)
#   ./scripts/demo.sh --quick   (small models, a few minutes; checks the pipeline)
set -euo pipefail
E1=12; E2=14; SEEDS="--seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4"
if [[ "${1:-}" == "--quick" ]]; then E1=2; E2=2; SEEDS="--seeds 0"; fi
step() { echo; echo "== $1"; }
step "synthetic world";                 sailab synthetic generate --name demo
step "Model 1 (U-Net)";                 sailab train mapping --epochs "$E1"
step "baselines";                       sailab train baselines
step "Model 2 (UNet-TT ensemble)";      sailab train forecast $SEEDS --epochs "$E2"
step "calibration (2023 validation)";   sailab calibrate
step "mapping scores (validation)";     sailab evaluate mapping --split val
step "forecast scores (validation)";    sailab evaluate forecast --split val
step "twin replay of the 2025 flood";   sailab twin replay 2025-08-15 2025-09-20
echo; echo "Done. Start the API (sailab api serve) and the dashboard (cd web && npm run dev)."
