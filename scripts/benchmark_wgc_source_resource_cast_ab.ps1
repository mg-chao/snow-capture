param(
    [string]$BaselinePath = "target/perf/wgc-source-resource-cast-ab-baseline.csv",
    [string]$TargetLabel = "wgc-window-source-resource-cast",
    [int]$WarmupFrames = 60,
    [int]$MeasureFrames = 600,
    [int]$Rounds = 6,
    [double]$SampleIntervalMs = 16.0,
    [double]$MaxRegressionPct = 3.0,
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

$workloadInfoPath = "target/perf/wgc-source-resource-cast-workload.json"
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

function Invoke-WgcSourceResourceBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableBorrowedResourceFastPaths
    )

    $oldDisableBorrowedSourceResource = $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE
    $oldDisableBorrowedSlotResource = $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE

    try {
        if ($DisableBorrowedResourceFastPaths) {
            $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE = "1"
            $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE -ErrorAction SilentlyContinue
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableBorrowedSourceResource) {
            $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE = $oldDisableBorrowedSourceResource
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE -ErrorAction SilentlyContinue
        }

        if ($null -ne $oldDisableBorrowedSlotResource) {
            $env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE = $oldDisableBorrowedSlotResource
        } else {
            Remove-Item Env:SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE -ErrorAction SilentlyContinue
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
        "--regression-metrics", "avg,p50",
        "--max-duplicate-pct", $MaxDuplicatePct
    )

    Write-Host "Benchmark window handle: $windowHandle"

    Write-Host "Collecting baseline with borrowed resource fast paths disabled..."
    $baselineExitCode = Invoke-WgcSourceResourceBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $BaselinePath)) -DisableBorrowedResourceFastPaths $true
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
    $optimizedExitCode = Invoke-WgcSourceResourceBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $BaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableBorrowedResourceFastPaths $false

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
