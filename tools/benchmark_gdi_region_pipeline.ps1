param(
    [string]$Region = "1280x720",
    [int]$Warmup = 30,
    [int]$Frames = 240,
    [int]$Rounds = 3,
    [double]$MaxRegressionPct = 0.0,
    [string]$BeforeCsv = "docs/compliance/gdi_region_pipeline_before.csv",
    [string]$AfterCsv = "docs/compliance/gdi_region_pipeline_after.csv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $Path
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $script:RepoRoot $Path
}

function Invoke-Benchmark {
    param(
        [string]$BaselinePath = "",
        [string]$SaveBaselinePath = ""
    )

    $args = @(
        "run",
        "--release",
        "--example",
        "benchmark",
        "--",
        "--backends", "gdi",
        "--region-center", $Region,
        "--warmup", $Warmup,
        "--frames", $Frames,
        "--rounds", $Rounds,
        "--regression-metrics", "avg,p50,p95"
    )

    if ($BaselinePath -ne "") {
        $args += @("--baseline", $BaselinePath, "--max-regression-pct", $MaxRegressionPct)
    }
    if ($SaveBaselinePath -ne "") {
        $args += @("--save-baseline", $SaveBaselinePath)
    }

    & cargo @args
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark command failed with exit code $LASTEXITCODE."
    }
}

$hadOriginalToggle = Test-Path Env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION
$originalToggle = $env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION
$resolvedBeforeCsv = Resolve-RepoPath $BeforeCsv
$resolvedAfterCsv = Resolve-RepoPath $AfterCsv

foreach ($path in @($resolvedBeforeCsv, $resolvedAfterCsv)) {
    $directory = Split-Path -Path $path -Parent
    if (-not [string]::IsNullOrWhiteSpace($directory)) {
        [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    }
}

try {
    Push-Location $script:RepoRoot
    try {
        Write-Host "Generating legacy baseline with direct region path disabled..."
        $env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION = "1"
        Invoke-Benchmark -SaveBaselinePath $resolvedBeforeCsv

        Write-Host "Running optimized path with strict regression guard..."
        Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION -ErrorAction SilentlyContinue
        Invoke-Benchmark -BaselinePath $resolvedBeforeCsv -SaveBaselinePath $resolvedAfterCsv

        Write-Host "Completed. Baselines:"
        Write-Host "  legacy:    $resolvedBeforeCsv"
        Write-Host "  optimized: $resolvedAfterCsv"
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($hadOriginalToggle) {
        $env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION = $originalToggle
    }
    else {
        Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION -ErrorAction SilentlyContinue
    }
}
