param(
    [string]$BaselinePath = "target/perf/dxgi-monitor-ab-baseline.csv",
    [string]$TargetLabel = "dxgi-monitor-animated",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 700,
    [int]$Rounds = 6,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 0.0,
    [double]$MaxDuplicatePct = 80.0,
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

$workloadInfoPath = "target/perf/dxgi-monitor-workload.json"
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

function Invoke-DxgiBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableOptimizations
    )

    $oldDisableDuplicateFastpath = $env:SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH

    try {
        if ($DisableOptimizations) {
            $env:SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableDuplicateFastpath) {
            $env:SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH = $oldDisableDuplicateFastpath
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH -ErrorAction SilentlyContinue
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
        "--warmup", $WarmupFrames,
        "--frames", $MeasureFrames,
        "--rounds", $Rounds,
        "--sample-interval-ms", $SampleIntervalMs,
        "--target-label", $effectiveTargetLabel,
        "--regression-metrics", "avg,p50,p95,p99",
        "--max-duplicate-pct", $MaxDuplicatePct
    )

    Write-Host "Animated workload hwnd: $windowHandle"

    Write-Host "Collecting baseline with duplicate fastpath disabled..."
    $baselineExitCode = Invoke-DxgiBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableOptimizations $true
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
    $optimizedExitCode = Invoke-DxgiBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $BaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableOptimizations $false

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
