param(
    [string]$BaselinePath = "target/perf/gdi-region-span-convert-ab-baseline.csv",
    [string]$TargetLabel = "gdi-region-span-convert",
    [int]$WarmupFrames = 80,
    [int]$MeasureFrames = 800,
    [int]$Rounds = 5,
    [double]$SampleIntervalMs = 8.0,
    [double]$MaxRegressionPct = 5.0,
    [double]$MaxDuplicatePct = 40.0,
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

$workloadInfoPath = Join-Path $repoRoot "target/perf/gdi-region-span-convert-workload.json"
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

function Invoke-GdiRegionBenchmark {
    param(
        [string[]]$BenchmarkArgs,
        [bool]$DisableSpanConvert
    )

    $oldDisableSpanConvert = $env:SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_SPAN_CONVERT

    try {
        if ($DisableSpanConvert) {
            $env:SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_SPAN_CONVERT = "1"
        } else {
            Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_SPAN_CONVERT -ErrorAction SilentlyContinue
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
        if ($null -ne $oldDisableSpanConvert) {
            $env:SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_SPAN_CONVERT = $oldDisableSpanConvert
        } else {
            Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_SPAN_CONVERT -ErrorAction SilentlyContinue
        }
    }
}

try {
    $deadline = (Get-Date).AddSeconds(20)
    $workloadInfo = $null

    while ($null -eq $workloadInfo) {
        if ($workloadProcess.HasExited) {
            Write-Error "Animated workload exited before writing metadata (exit code: $($workloadProcess.ExitCode))"
            exit 1
        }
        if ((Get-Date) -ge $deadline) {
            Write-Error "Timed out waiting for animated workload metadata at $workloadInfoPath"
            exit 1
        }
        if (-not (Test-Path $workloadInfoPath)) {
            Start-Sleep -Milliseconds 50
            continue
        }

        try {
            $candidate = Get-Content $workloadInfoPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            if (
                $null -eq $candidate -or
                $null -eq $candidate.x -or
                $null -eq $candidate.y -or
                $null -eq $candidate.width -or
                $null -eq $candidate.height
            ) {
                throw "Workload metadata is missing required fields."
            }
            if ([int]$candidate.width -le 0 -or [int]$candidate.height -le 0) {
                throw "Workload metadata reported a non-positive capture region."
            }
            $workloadInfo = $candidate
        }
        catch {
            Start-Sleep -Milliseconds 50
        }
    }

    $region = "{0},{1},{2},{3}" -f $workloadInfo.x, $workloadInfo.y, $workloadInfo.width, $workloadInfo.height
    $effectiveTargetLabel = "{0}-{1}x{2}-box{3}" -f $TargetLabel, $workloadInfo.width, $workloadInfo.height, $WorkloadBoxSize

    $commonArgs = @(
        "--backends", "gdi",
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

    Write-Host "Collecting baseline with incremental span conversion disabled..."
    $baselineExitCode = Invoke-GdiRegionBenchmark -BenchmarkArgs ($commonArgs + @("--save-baseline", $resolvedBaselinePath)) -DisableSpanConvert $true
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
    $optimizedExitCode = Invoke-GdiRegionBenchmark -BenchmarkArgs (
        $commonArgs + @(
            "--baseline", $resolvedBaselinePath,
            "--max-regression-pct", $MaxRegressionPct
        )
    ) -DisableSpanConvert $false

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
