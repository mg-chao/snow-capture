param(
    [string]$BaselinePath = "target/perf/dxgi-window-low-latency-ab-baseline.csv",
    [string]$TargetLabel = "dxgi-window-low-latency",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 600,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 0.0,
    [double]$MaxDuplicatePct = 70.0,
    [long]$LowLatencyMaxPixels = 1048576,
    [int]$WorkloadWidth = 1600,
    [int]$WorkloadHeight = 900,
    [int]$WorkloadX = 180,
    [int]$WorkloadY = 120,
    [int]$WorkloadIntervalMs = 8,
    [switch]$SaveBaselineOnly
)

$workloadScript = Join-Path $PSScriptRoot "start_animated_region_window.ps1"
if (-not (Test-Path $workloadScript)) {
    Write-Error "Workload script not found: $workloadScript"
    exit 1
}

$workloadInfoPath = "target/perf/dxgi-window-low-latency-workload.json"
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

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -PassThru

function Invoke-DxgiWindowBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$ForceLegacyLowLatency,
        [long]$AdaptiveMaxPixels
    )

    $oldDisableWindowLowLatency = $env:SNOW_CAPTURE_DXGI_DISABLE_WINDOW_LOW_LATENCY_REGION
    $oldForceWindowLowLatency = $env:SNOW_CAPTURE_DXGI_FORCE_WINDOW_LOW_LATENCY_REGION
    $oldWindowLowLatencyMaxPixels = $env:SNOW_CAPTURE_DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS

    try {
        Remove-Item Env:SNOW_CAPTURE_DXGI_DISABLE_WINDOW_LOW_LATENCY_REGION -ErrorAction SilentlyContinue
        if ($ForceLegacyLowLatency) {
            $env:SNOW_CAPTURE_DXGI_FORCE_WINDOW_LOW_LATENCY_REGION = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_FORCE_WINDOW_LOW_LATENCY_REGION -ErrorAction SilentlyContinue
        }

        if ($AdaptiveMaxPixels -gt 0) {
            $env:SNOW_CAPTURE_DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS = [string]$AdaptiveMaxPixels
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableWindowLowLatency) {
            $env:SNOW_CAPTURE_DXGI_DISABLE_WINDOW_LOW_LATENCY_REGION = $oldDisableWindowLowLatency
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_DISABLE_WINDOW_LOW_LATENCY_REGION -ErrorAction SilentlyContinue
        }
        if ($null -ne $oldForceWindowLowLatency) {
            $env:SNOW_CAPTURE_DXGI_FORCE_WINDOW_LOW_LATENCY_REGION = $oldForceWindowLowLatency
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_FORCE_WINDOW_LOW_LATENCY_REGION -ErrorAction SilentlyContinue
        }
        if ($null -ne $oldWindowLowLatencyMaxPixels) {
            $env:SNOW_CAPTURE_DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS = $oldWindowLowLatencyMaxPixels
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS -ErrorAction SilentlyContinue
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
    $windowHandle = "0x{0:X}" -f ([uint64]$workloadInfo.hwnd)
    $effectiveTargetLabel = "{0}-{1}x{2}" -f $TargetLabel, $workloadInfo.width, $workloadInfo.height

    $commonArgs = @(
        "--backends", "dxgi",
        "--window-handle", $windowHandle,
        "--warmup", $WarmupFrames,
        "--frames", $MeasureFrames,
        "--rounds", $Rounds,
        "--sample-interval-ms", $SampleIntervalMs,
        "--target-label", $effectiveTargetLabel,
        "--regression-metrics", "avg,p50,p95",
        "--max-duplicate-pct", $MaxDuplicatePct
    )

    Write-Host "Benchmark window handle: $windowHandle"
    Write-Host "Adaptive low-latency max pixels: $LowLatencyMaxPixels"

    Write-Host "Collecting baseline with forced legacy low-latency path..."
    $baselineExitCode = Invoke-DxgiWindowBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -ForceLegacyLowLatency $true -AdaptiveMaxPixels $LowLatencyMaxPixels
    if ($baselineExitCode -ne 0) {
        exit $baselineExitCode
    }

    if ($SaveBaselineOnly) {
        Write-Host "Saved forced-low-latency baseline to $BaselinePath"
        exit 0
    }

    if (-not (Test-Path $BaselinePath)) {
        Write-Error "Baseline file not found after baseline run: $BaselinePath"
        exit 1
    }

    Write-Host "Running adaptive low-latency benchmark and regression guard..."
    $optimizedExitCode = Invoke-DxgiWindowBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $BaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -ForceLegacyLowLatency $false -AdaptiveMaxPixels $LowLatencyMaxPixels

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
