param(
    [string]$BaselinePath = "target/perf/dxgi-region-baseline.csv",
    [string]$TargetLabel = "dxgi-region-center-1280x720",
    [string]$RegionSize = "1280x720",
    [int]$WarmupFrames = 60,
    [int]$MeasureFrames = 600,
    [int]$Rounds = 6,
    [double]$SampleIntervalMs = 0.0,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 100.0,
    [switch]$SaveBaselineOnly
)

$commonArgs = @(
    "--backends", "dxgi",
    "--region-center", $RegionSize,
    "--warmup", $WarmupFrames,
    "--frames", $MeasureFrames,
    "--rounds", $Rounds,
    "--sample-interval-ms", $SampleIntervalMs,
    "--target-label", $TargetLabel,
    "--regression-metrics", "avg,p50,p95,p99",
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
