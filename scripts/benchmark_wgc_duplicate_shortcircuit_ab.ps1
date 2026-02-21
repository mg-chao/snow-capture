param(
    [string]$BaselinePath = "target/perf/wgc-duplicate-shortcircuit-ab-baseline.csv",
    [string]$TargetLabel = "wgc-window-duplicate-shortcircuit",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 900,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 4.0,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 100.0,
    [int]$WorkloadWidth = 960,
    [int]$WorkloadHeight = 540,
    [int]$WorkloadX = 180,
    [int]$WorkloadY = 160,
    [int]$WorkloadIntervalMs = 120,
    [switch]$SaveBaselineOnly
)

$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
$warmupFramesArg = $WarmupFrames.ToString($invariantCulture)
$measureFramesArg = $MeasureFrames.ToString($invariantCulture)
$roundsArg = $Rounds.ToString($invariantCulture)
$sampleIntervalMsArg = $SampleIntervalMs.ToString($invariantCulture)
$maxRegressionPctArg = $MaxRegressionPct.ToString($invariantCulture)
$maxDuplicatePctArg = $MaxDuplicatePct.ToString($invariantCulture)
$workloadWidthArg = $WorkloadWidth.ToString($invariantCulture)
$workloadHeightArg = $WorkloadHeight.ToString($invariantCulture)
$workloadXArg = $WorkloadX.ToString($invariantCulture)
$workloadYArg = $WorkloadY.ToString($invariantCulture)
$workloadIntervalMsArg = $WorkloadIntervalMs.ToString($invariantCulture)

$workloadScript = Join-Path $PSScriptRoot "start_animated_region_window.ps1"
if (-not (Test-Path $workloadScript)) {
    Write-Error "Workload script not found: $workloadScript"
    exit 1
}

$workloadInfoPath = "target/perf/wgc-duplicate-shortcircuit-workload.json"
if (Test-Path $workloadInfoPath) {
    Remove-Item $workloadInfoPath -Force
}

$workloadArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-STA",
    "-File", $workloadScript,
    "-OutputPath", $workloadInfoPath,
    "-X", $workloadXArg,
    "-Y", $workloadYArg,
    "-Width", $workloadWidthArg,
    "-Height", $workloadHeightArg,
    "-IntervalMs", $workloadIntervalMsArg
)

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -PassThru

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-WgcDuplicateShortCircuitBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableOptimization
    )

    $oldDisableOptimization = $env:SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_SHORTCIRCUIT
    $oldDisableImmediateStaleReturn = $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN

    try {
        # Keep immediate stale-return disabled for both A/B runs so this
        # benchmark isolates duplicate short-circuit behavior.
        $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = "1"

        if ($DisableOptimization) {
            $env:SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_SHORTCIRCUIT = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_SHORTCIRCUIT -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableOptimization) {
            $env:SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_SHORTCIRCUIT = $oldDisableOptimization
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_SHORTCIRCUIT -ErrorAction SilentlyContinue
        }

        if ($null -ne $oldDisableImmediateStaleReturn) {
            $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = $oldDisableImmediateStaleReturn
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN -ErrorAction SilentlyContinue
        }
    }
}

try {
    $deadline = (Get-Date).AddSeconds(20)
    while (-not (Test-Path $workloadInfoPath)) {
        if ($workloadProcess.HasExited) {
            Write-Error "Animated workload exited before writing metadata (exit code: $($workloadProcess.ExitCode))"
            exit 1
        }
        if ((Get-Date) -ge $deadline) {
            Write-Error "Timed out waiting for animated workload metadata at $workloadInfoPath"
            exit 1
        }
        Start-Sleep -Milliseconds 50
    }

    $workloadInfo = Get-Content $workloadInfoPath -Raw | ConvertFrom-Json
    $windowHandleValue = [uint64]$workloadInfo.hwnd
    if ($windowHandleValue -eq 0) {
        Write-Error "Animated workload reported an invalid window handle in $workloadInfoPath"
        exit 1
    }

    $windowHandle = "0x{0:X}" -f $windowHandleValue
    $effectiveTargetLabel = "{0}-{1}x{2}" -f $TargetLabel, $workloadInfo.width, $workloadInfo.height

    $commonArgs = @(
        "--backends", "wgc",
        "--window-handle", $windowHandle,
        "--warmup", $warmupFramesArg,
        "--frames", $measureFramesArg,
        "--rounds", $roundsArg,
        "--sample-interval-ms", $sampleIntervalMsArg,
        "--target-label", $effectiveTargetLabel,
        "--regression-metrics", "avg,p50,p95,p99",
        "--max-duplicate-pct", $maxDuplicatePctArg
    )

    Write-Host "Benchmark window handle: $windowHandle"

    Write-Host "Collecting baseline with duplicate short-circuit disabled..."
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
}
finally {
    if ($workloadProcess -and -not $workloadProcess.HasExited) {
        $null = $workloadProcess.CloseMainWindow()
        Start-Sleep -Milliseconds 300
        if (-not $workloadProcess.HasExited) {
            Stop-Process -Id $workloadProcess.Id -Force
        }
    }
}
