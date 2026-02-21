param(
    [string]$BaselinePath = "target/perf/wgc-region-pipeline-ab-baseline.csv",
    [string]$TargetLabel = "wgc-region-pipeline",
    [int]$WarmupFrames = 60,
    [int]$MeasureFrames = 480,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 16.0,
    [double]$MaxRegressionPct = 0.0,
    [double]$MaxDuplicatePct = 100.0,
    [int]$WorkloadWidth = 960,
    [int]$WorkloadHeight = 540,
    [int]$WorkloadX = 180,
    [int]$WorkloadY = 160,
    [int]$WorkloadIntervalMs = 16,
    [switch]$EnableLowLatencySlot,
    [switch]$FullRedraw,
    [switch]$SaveBaselineOnly
)

$workloadScript = Join-Path $PSScriptRoot "start_animated_region_window.ps1"
if (-not (Test-Path $workloadScript)) {
    Write-Error "Workload script not found: $workloadScript"
    exit 1
}

$workloadInfoPath = "target/perf/wgc-region-pipeline-workload.json"
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

if ($FullRedraw) {
    $workloadArgs += "-FullRedraw"
}

$workloadProcess = Start-Process -FilePath powershell.exe -ArgumentList $workloadArgs -PassThru

$baselineDir = Split-Path -Parent $BaselinePath
if ($baselineDir -and -not (Test-Path $baselineDir)) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
}

function Invoke-WgcPipelineBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableOptimizations
    )

    $oldDisableFullSlotMap = $env:SNOW_CAPTURE_WGC_DISABLE_REGION_FULL_SLOT_MAP
    $oldEnableLowLatencySlot = $env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT
    $oldDisableImmediateStale = $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN
    $oldDisableRegionShortCircuit = $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT

    try {
        # Keep benchmark focused on map/copy throughput rather than stale fast-path behavior.
        $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = "1"
        $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT = "1"

        if ($DisableOptimizations) {
            $env:SNOW_CAPTURE_WGC_DISABLE_REGION_FULL_SLOT_MAP = "1"
            Remove-Item Env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT -ErrorAction SilentlyContinue
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_REGION_FULL_SLOT_MAP -ErrorAction SilentlyContinue
            if ($EnableLowLatencySlot) {
                $env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT = "1"
            } else {
                Remove-Item Env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT -ErrorAction SilentlyContinue
            }
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
        if ($null -ne $oldDisableFullSlotMap) {
            $env:SNOW_CAPTURE_WGC_DISABLE_REGION_FULL_SLOT_MAP = $oldDisableFullSlotMap
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_REGION_FULL_SLOT_MAP -ErrorAction SilentlyContinue
        }

        if ($null -ne $oldEnableLowLatencySlot) {
            $env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT = $oldEnableLowLatencySlot
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_ENABLE_REGION_LOW_LATENCY_SLOT -ErrorAction SilentlyContinue
        }

        if ($null -ne $oldDisableImmediateStale) {
            $env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN = $oldDisableImmediateStale
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN -ErrorAction SilentlyContinue
        }

        if ($null -ne $oldDisableRegionShortCircuit) {
            $env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT = $oldDisableRegionShortCircuit
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT -ErrorAction SilentlyContinue
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
        "--backends", "wgc",
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

    Write-Host "Collecting baseline with region pipeline optimizations disabled..."
    $baselineExitCode = Invoke-WgcPipelineBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableOptimizations $true
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
    $optimizedExitCode = Invoke-WgcPipelineBenchmark -BenchmarkArgs (
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
