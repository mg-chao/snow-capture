param(
    [string]$BaselinePath = "target/perf/wgc-baseline.csv",
    [string]$TargetLabel = "ci-primary",
    [int]$WarmupFrames = 30,
    [int]$MeasureFrames = 400,
    [int]$Rounds = 5,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 100.0,
    [switch]$SaveBaselineOnly
)

$commonArgs = @(
    "--backends", "wgc",
    "--warmup", $WarmupFrames,
    "--frames", $MeasureFrames,
    "--rounds", $Rounds,
    "--target-label", $TargetLabel,
    "--regression-metrics", "p50,p95,p99",
    "--max-duplicate-pct", $MaxDuplicatePct
)

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

