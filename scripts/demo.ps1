# The whole SailabAI demo on one laptop: synthetic world -> models -> calibration -> scores -> twin.
# About an hour on an RTX 4050. Run from the repo root with the venv active:
#   .\scripts\demo.ps1            (full)
#   .\scripts\demo.ps1 -Quick     (small models, a few minutes; for checking the pipeline)
param([switch]$Quick)
$ErrorActionPreference = "Stop"

function Step($text, [scriptblock]$block) {
    Write-Host "`n== $text" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) { throw "failed: $text" }
}

$epochs1 = if ($Quick) { 2 } else { 12 }
$epochs2 = if ($Quick) { 2 } else { 14 }
$seeds = if ($Quick) { @(0) } else { @(0, 1, 2, 3, 4) }
$seedArgs = $seeds | ForEach-Object { "--seeds"; "$_" }

Step "synthetic world" { sailab synthetic generate --name demo }
Step "Model 1 (U-Net)" { sailab train mapping --epochs $epochs1 }
Step "baselines" { sailab train baselines }
Step "Model 2 (UNet-TT ensemble)" { sailab train forecast @seedArgs --epochs $epochs2 }
Step "calibration on the 2023 validation flood" { sailab calibrate }
Step "mapping scores (validation)" { sailab evaluate mapping --split val }
Step "forecast scores (validation; tunes thresholds)" { sailab evaluate forecast --split val }
Step "twin replay of the 2025 flood" { sailab twin replay 2025-08-15 2025-09-20 }
Write-Host "`nDone. Start the API (sailab api serve) and the dashboard (cd web; npm run dev)." -ForegroundColor Green
Write-Host "The synthetic 2025 test is scored with: sailab evaluate forecast --split test (once per cube)."
