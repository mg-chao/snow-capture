param(
    [string]$BaselinePath = "target/perf/wgc-window-ab-baseline.csv",
    [string]$TargetLabel = "wgc-window-animated",
    [int]$WarmupFrames = 100,
    [int]$MeasureFrames = 900,
    [int]$Rounds = 6,
    [double]$SampleIntervalMs = 4.0,
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

$workloadInfoPath = "target/perf/wgc-window-workload.json"
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

function Invoke-WgcWindowBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableImmediateStaleReturn
    )

    $oldDisableImmediateStaleReturn = $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN

    try {
        if ($DisableImmediateStaleReturn) {
            $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN -ErrorAction SilentlyContinue
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
    $windowHandle = "0x{0:X}" -f ([uint64]$workloadInfo.hwnd)
    $effectiveTargetLabel = "{0}-{1}x{2}" -f $TargetLabel, $workloadInfo.width, $workloadInfo.height

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

    Write-Host "Collecting baseline with immediate stale return disabled..."
    $baselineExitCode = Invoke-WgcWindowBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableImmediateStaleReturn $true
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
    $optimizedExitCode = Invoke-WgcWindowBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $BaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableImmediateStaleReturn $false

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
