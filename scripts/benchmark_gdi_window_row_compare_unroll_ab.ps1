param(
    [string]$BaselinePath = "target/perf/gdi-window-row-compare-unroll-ab-baseline.csv",
    [string]$TargetLabel = "gdi-window-row-compare-unroll",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 700,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 60.0,
    [int]$WorkloadWidth = 960,
    [int]$WorkloadHeight = 540,
    [int]$WorkloadBoxSize = 420,
    [int]$WorkloadX = 180,
    [int]$WorkloadY = 160,
    [int]$WorkloadIntervalMs = 8,
    [switch]$SaveBaselineOnly
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedBaselinePath = if ([System.IO.Path]::IsPathRooted($BaselinePath)) {
    $BaselinePath
} else {
    Join-Path $repoRoot $BaselinePath
}

$workloadScript = Join-Path $PSScriptRoot "start_animated_region_window.ps1"
if (-not (Test-Path $workloadScript)) {
    Write-Error "Workload script not found: $workloadScript"
    exit 1
}

$workloadInfoPath = Join-Path $repoRoot "target/perf/gdi-window-row-compare-unroll-workload.json"
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
    "-BoxSize", $WorkloadBoxSize,
    "-IntervalMs", $WorkloadIntervalMs
)

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -WorkingDirectory $repoRoot -PassThru

$baselineDir = Split-Path -Parent $resolvedBaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-GdiBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableUnroll
    )

    $oldDisableUnroll = $env:SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL

    try {
        if ($DisableUnroll) {
            $env:SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL -ErrorAction SilentlyContinue
        }

        $cargoArgs = @(
            "run",
            "--release",
            "--example",
            "benchmark",
            "--"
        ) + $BenchmarkArgs

        $cargoProcess = Start-Process -FilePath "cargo.exe" -ArgumentList $cargoArgs -WorkingDirectory $repoRoot -NoNewWindow -Wait -PassThru
        return [int]$cargoProcess.ExitCode
    }
    finally {
        if ($null -ne $oldDisableUnroll) {
            $env:SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL = $oldDisableUnroll
        } else {
            Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL -ErrorAction SilentlyContinue
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
        "--backends", "gdi",
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

    Write-Host "Collecting baseline with SIMD row-compare unroll disabled..."
    $baselineExitCode = Invoke-GdiBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $resolvedBaselinePath)) -DisableUnroll $true
    if ($baselineExitCode -ne 0) {
        exit $baselineExitCode
    }

    if ($SaveBaselineOnly) {
        Write-Host "Saved baseline to $resolvedBaselinePath"
        exit 0
    }

    if (-not (Test-Path $resolvedBaselinePath)) {
        Write-Error "Baseline file not found after baseline run: $resolvedBaselinePath"
        exit 1
    }

    if ($workloadProcess.HasExited) {
        Write-Error "Animated workload exited before optimized run (exit code: $($workloadProcess.ExitCode))"
        exit 1
    }

    Write-Host "Running optimized benchmark and regression guard..."
    $optimizedExitCode = Invoke-GdiBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $resolvedBaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableUnroll $false

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
