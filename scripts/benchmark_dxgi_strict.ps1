param(
    [string]$BaselinePath = "target/perf/dxgi-baseline.csv",
    [string]$TargetLabel = "ci-primary",
    [int]$WarmupFrames = 30,
    [int]$MeasureFrames = 400,
    [int]$Rounds = 5,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 100.0,
    [switch]$SaveBaselineOnly
)

$commonArgs = @(
    "--backends", "dxgi",
    "--warmup", $WarmupFrames,
    "--frames", $MeasureFrames,
    "--rounds", $Rounds,
    "--target-label", $TargetLabel,
    "--regression-metrics", "p50,p95,p99",
    "--max-duplicate-pct", $MaxDuplicatePct
)

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

if ($SaveBaselineOnly) {
    cargo run --release --example benchmark -- @($commonArgs + @("--save-baseline", $BaselinePath))
    exit $LASTEXITCODE
}

if (-not (Test-Path $BaselinePath)) {
    Write-Error "Baseline file not found: $BaselinePath"
    exit 1
}

cargo run --release --example benchmark -- @(
    $commonArgs + @(
        "--baseline", $BaselinePath,
        "--max-regression-pct", $MaxRegressionPct
    )
)
exit $LASTEXITCODE
