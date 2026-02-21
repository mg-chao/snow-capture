param(
    [string]$BaselinePath = "target/perf/wgc-region-duplicate-shortcircuit-ab-baseline.csv",
    [string]$TargetLabel = "wgc-region-duplicate-shortcircuit",
    [string]$RegionSize = "960x540",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 900,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 15.0,
    [double]$MaxDuplicatePct = 100.0,
    [switch]$SaveBaselineOnly
)

$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$warmupFramesArg = $WarmupFrames.ToString($invariantCulture)
$measureFramesArg = $MeasureFrames.ToString($invariantCulture)
$roundsArg = $Rounds.ToString($invariantCulture)
$sampleIntervalMsArg = $SampleIntervalMs.ToString($invariantCulture)
$maxRegressionPctArg = $MaxRegressionPct.ToString($invariantCulture)
$maxDuplicatePctArg = $MaxDuplicatePct.ToString($invariantCulture)

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-WgcDuplicateShortCircuitBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableOptimization
    )

    $oldDisableShortCircuit = $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT

    try {
        if ($DisableOptimization) {
            $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT -ErrorAction SilentlyContinue
        }

        $cargoArgs = @(
            "run",
            "--release",
            "--example",
            "benchmark",
            "--"
        ) + $BenchmarkArgs

        $cargoProcess = Start-Process -FilePath "cargo.exe" -ArgumentList $cargoArgs -NoNewWindow -Wait -PassThru
        return [int]$cargoProcess.ExitCode
    }
    finally {
        if ($null -ne $oldDisableShortCircuit) {
            $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT = $oldDisableShortCircuit
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT -ErrorAction SilentlyContinue
        }
    }
}

$commonArgs = @(
    "--backends", "wgc",
    "--region-center", $RegionSize,
    "--warmup", $warmupFramesArg,
    "--frames", $measureFramesArg,
    "--rounds", $roundsArg,
    "--sample-interval-ms", $sampleIntervalMsArg,
    "--target-label", $TargetLabel,
    "--regression-metrics", "avg,p95,p99",
    "--max-duplicate-pct", $maxDuplicatePctArg
)

Write-Host "Collecting baseline with region duplicate short-circuit disabled..."
$baselineExitCode = Invoke-WgcDuplicateShortCircuitBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableOptimization $true
if ($baselineExitCode -ne 0) {
    exit $baselineExitCode
}

if ($SaveBaselineOnly) {
    Write-Host "Saved disabled-optimization baseline to $BaselinePath"
    exit 0
}

if (-not (Test-Path $BaselinePath)) {
    Write-Error "Baseline file not found after baseline run: $BaselinePath"
    exit 1
}

Write-Host "Running optimized benchmark and regression guard..."
$optimizedExitCode = Invoke-WgcDuplicateShortCircuitBenchmark -BenchmarkArgs (
    $commonArgs + @(
        "--baseline", $BaselinePath,
        "--max-regression-pct", $maxRegressionPctArg
    )
) -DisableOptimization $false

exit $optimizedExitCode
