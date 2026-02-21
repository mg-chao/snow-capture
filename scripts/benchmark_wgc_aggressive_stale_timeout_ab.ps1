param(
    [string]$BaselinePath = "target/perf/wgc-aggressive-stale-timeout-ab-baseline.csv",
    [string]$TargetLabel = "wgc-window-aggressive-stale-timeout",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 800,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 16.0,
    [double]$MaxRegressionPct = 0.0,
    [double]$MaxDuplicatePct = 100.0,
    [int]$WorkloadWidth = 960,
    [int]$WorkloadHeight = 540,
    [int]$WorkloadX = 180,
    [int]$WorkloadY = 160,
    [int]$WorkloadIntervalMs = 8,
    [switch]$SaveBaselineOnly
)

$workloadScript = Join-Path $PSScriptRoot "start_animated_region_window.ps1"
if (-not (Test-Path $workloadScript)) {
    Write-Error "Workload script not found: $workloadScript"
    exit 1
}

$workloadInfoPath = "target/perf/wgc-aggressive-stale-timeout-workload.json"
if (Test-Path $workloadInfoPath) {
    Remove-Item $workloadInfoPath -Force
}

$workloadArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-STA",
    "-File", $workloadScript,
    "-OutputPath", $workloadInfoPath,
    "-X", $WorkloadX,
    "-Y", $WorkloadY,
    "-Width", $WorkloadWidth,
    "-Height", $WorkloadHeight,
    "-IntervalMs", $WorkloadIntervalMs
)

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -PassThru

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-WgcStaleTimeoutBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableAggressiveStaleTimeout
    )

    $oldDisableAggressiveStaleTimeout = $env:SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT
    $oldDisableImmediateStaleReturn = $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN

    try {
        # Keep immediate-stale behavior fixed so the benchmark isolates
        # the stale-timeout profile only.
        $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = "1"

        if ($DisableAggressiveStaleTimeout) {
            $env:SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableAggressiveStaleTimeout) {
            $env:SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT = $oldDisableAggressiveStaleTimeout
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT -ErrorAction SilentlyContinue
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

    $workloadCaptureWidth = [int]$workloadInfo.width
    $workloadCaptureHeight = [int]$workloadInfo.height
    if ($workloadCaptureWidth -le 0 -or $workloadCaptureHeight -le 0) {
        Write-Error "Animated workload reported invalid dimensions (${workloadCaptureWidth}x${workloadCaptureHeight}) in $workloadInfoPath"
        exit 1
    }

    $windowHandle = "0x{0:X}" -f $windowHandleValue
    $effectiveTargetLabel = "{0}-{1}x{2}" -f $TargetLabel, $workloadCaptureWidth, $workloadCaptureHeight

    $commonArgs = @(
        "--backends", "wgc",
        "--window-handle", $windowHandle,
        "--warmup", $WarmupFrames,
        "--frames", $MeasureFrames,
        "--rounds", $Rounds,
        "--sample-interval-ms", $SampleIntervalMs,
        "--target-label", $effectiveTargetLabel,
        "--regression-metrics", "avg,p50,p95,p99",
        "--max-duplicate-pct", $MaxDuplicatePct
    )

    Write-Host "Benchmark window handle: $windowHandle"

    Write-Host "Collecting baseline with aggressive stale timeout disabled..."
    $baselineExitCode = Invoke-WgcStaleTimeoutBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableAggressiveStaleTimeout $true
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
    $optimizedExitCode = Invoke-WgcStaleTimeoutBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $BaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableAggressiveStaleTimeout $false

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
