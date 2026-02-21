param(
    [string]$BaselinePath = "target/perf/gdi-baseline.csv",
    [string]$TargetLabel = "ci-primary-gdi",
    [int]$WarmupFrames = 40,
    [int]$MeasureFrames = 500,
    [int]$Rounds = 5,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 100.0,
    [switch]$SaveBaselineOnly
)

$commonArgs = @(
    "--backends", "gdi",
    "--warmup", $WarmupFrames,
    "--frames", $MeasureFrames,
    "--rounds", $Rounds,
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
