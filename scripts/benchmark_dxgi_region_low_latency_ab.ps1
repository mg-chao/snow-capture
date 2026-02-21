param(
    [string]$BaselinePath = "target/perf/dxgi-region-low-latency-ab-baseline.csv",
    [string]$TargetLabel = "dxgi-region-low-latency",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 700,
    [int]$Rounds = 6,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 0.0,
    [double]$MaxDuplicatePct = 100.0,
    [long]$RegionLowLatencyMaxPixels = 1048576,
    [int]$WorkloadWidth = 960,
    [int]$WorkloadHeight = 540,
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

$workloadInfoPath = Join-Path $repoRoot "target/perf/dxgi-region-low-latency-workload.json"
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

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -WorkingDirectory $repoRoot -PassThru

$baselineDir = Split-Path -Parent $resolvedBaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-DxgiRegionBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$EnableRegionLowLatency,
        [long]$LowLatencyMaxPixels
    )

    $oldEnableRegionLowLatency = $env:SNOW_CAPTURE_DXGI_ENABLE_REGION_LOW_LATENCY
    $oldLowLatencyMaxPixels = $env:SNOW_CAPTURE_DXGI_REGION_LOW_LATENCY_MAX_PIXELS

    try {
        if ($EnableRegionLowLatency) {
            $env:SNOW_CAPTURE_DXGI_ENABLE_REGION_LOW_LATENCY = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_ENABLE_REGION_LOW_LATENCY -ErrorAction SilentlyContinue
        }

        if ($LowLatencyMaxPixels -gt 0) {
            $env:SNOW_CAPTURE_DXGI_REGION_LOW_LATENCY_MAX_PIXELS = [string]$LowLatencyMaxPixels
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_REGION_LOW_LATENCY_MAX_PIXELS -ErrorAction SilentlyContinue
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
        if ($null -ne $oldEnableRegionLowLatency) {
            $env:SNOW_CAPTURE_DXGI_ENABLE_REGION_LOW_LATENCY = $oldEnableRegionLowLatency
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_ENABLE_REGION_LOW_LATENCY -ErrorAction SilentlyContinue
        }
        if ($null -ne $oldLowLatencyMaxPixels) {
            $env:SNOW_CAPTURE_DXGI_REGION_LOW_LATENCY_MAX_PIXELS = $oldLowLatencyMaxPixels
        } else {
            Remove-Item Env:SNOW_CAPTURE_DXGI_REGION_LOW_LATENCY_MAX_PIXELS -ErrorAction SilentlyContinue
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
    $region = "{0},{1},{2},{3}" -f $workloadInfo.x, $workloadInfo.y, $workloadInfo.width, $workloadInfo.height
    $effectiveTargetLabel = "{0}-{1}x{2}" -f $TargetLabel, $workloadInfo.width, $workloadInfo.height

    $commonArgs = @(
        "--backends", "dxgi",
        "--region", $region,
        "--warmup", $WarmupFrames,
        "--frames", $MeasureFrames,
        "--rounds", $Rounds,
        "--sample-interval-ms", $SampleIntervalMs,
        "--target-label", $effectiveTargetLabel,
        "--regression-metrics", "avg,p50,p95,p99",
        "--max-duplicate-pct", $MaxDuplicatePct
    )

    Write-Host "Benchmark region: $region"
    Write-Host "Region low-latency max pixels: $RegionLowLatencyMaxPixels"

    Write-Host "Collecting baseline with region low-latency path disabled..."
    $baselineExitCode = Invoke-DxgiRegionBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $resolvedBaselinePath)) -EnableRegionLowLatency $false -LowLatencyMaxPixels $RegionLowLatencyMaxPixels
    if ($baselineExitCode -ne 0) {
        exit $baselineExitCode
    }

    if ($SaveBaselineOnly) {
        Write-Host "Saved disabled-low-latency baseline to $resolvedBaselinePath"
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
    $optimizedExitCode = Invoke-DxgiRegionBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $resolvedBaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -EnableRegionLowLatency $true -LowLatencyMaxPixels $RegionLowLatencyMaxPixels

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
