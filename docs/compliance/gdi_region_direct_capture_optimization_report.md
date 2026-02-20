# GDI Region Direct-Capture Optimization Report

## Scope

This optimization targets the GDI backend hot path for region capture in
`CaptureMode::ScreenRecording`.

Before this change, region capture with GDI always fell back to:

1. full-monitor `BitBlt` into the monitor-sized DIB
2. full-monitor BGRA->RGBA conversion
3. CPU-side crop/copy into the region output frame

That wastes work when the requested region is much smaller than the monitor.

## Implementation Summary

Primary changes:

1. Added a direct GDI region path in `src/platform/windows/gdi.rs`:
   - `GdiResources::capture_region_into_rgba(...)`
   - issues `BitBlt` for only the requested source rectangle
   - writes conversion output directly into the destination frame offset
2. Wired `WindowsMonitorCapturer::capture_region_into(...)` to use the new path:
   - returns `CaptureSampleMetadata` so `CaptureSession` no longer falls back
     to full monitor capture + crop for GDI regions
3. Added runtime guard for A/B benchmarking:
   - env var `SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION=1` forces legacy fallback
4. Improved NT kernel selection robustness in `src/convert/mod.rs`:
   - BGRA NT kernels now choose the best SIMD width that matches destination
     alignment (AVX-512 64B, AVX2 32B, SSSE3 16B), instead of dropping to
     temporal stores when AVX2 alignment is unavailable
5. Added automation script:
   - `tools/benchmark_gdi_region_pipeline.ps1`
   - generates legacy baseline, runs optimized path, and enforces strict
     regression checks (`avg,p50,p95`)

## Benchmark Method

All runs were executed in release mode with the built-in benchmark example:

```powershell
powershell -ExecutionPolicy Bypass -File tools/benchmark_gdi_region_pipeline.ps1
```

The script runs:

1. Legacy baseline (`SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION=1`)
2. Optimized run (direct region path enabled)
3. Strict regression guard against the legacy baseline:
   - `--regression-metrics avg,p50,p95`
   - `--max-regression-pct 0`

Additional monitor safety check:

```powershell
$env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION='1'
cargo run --release --example benchmark -- --backends gdi --warmup 30 --frames 240 --rounds 3 --save-baseline docs/compliance/gdi_monitor_pipeline_before_direct_region.csv
Remove-Item Env:SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION -ErrorAction SilentlyContinue
cargo run --release --example benchmark -- --backends gdi --warmup 30 --frames 240 --rounds 3 --baseline docs/compliance/gdi_monitor_pipeline_before_direct_region.csv --max-regression-pct 5 --regression-metrics avg,p50,p95 --save-baseline docs/compliance/gdi_monitor_pipeline_after_direct_region.csv
```

## Results

### Region target (`region:1280:720:1280:720`)

From:
- `docs/compliance/gdi_region_pipeline_before.csv`
- `docs/compliance/gdi_region_pipeline_after.csv`

- avg: `44.486 ms -> 7.061 ms` (-84.13%)
- p50: `42.504 ms -> 6.942 ms` (-83.67%)
- p95: `53.352 ms -> 7.614 ms` (-85.73%)
- fps: `22.48 -> 141.63`

Regression gate status:
- avg: passed (`max-regression-pct=0`)
- p50: passed (`max-regression-pct=0`)
- p95: passed (`max-regression-pct=0`)

### Primary monitor safety snapshot

From:
- `docs/compliance/gdi_monitor_pipeline_before_direct_region.csv`
- `docs/compliance/gdi_monitor_pipeline_after_direct_region.csv`

- avg: `48.951 ms -> 42.649 ms`
- p50: `45.438 ms -> 41.855 ms`
- p95: `64.887 ms -> 48.631 ms`

Regression gate status:
- avg/p50/p95 passed (`max-regression-pct=5`)

## Validation

- `cargo check` passed
- `cargo test --lib` passed
- Benchmark regression checks passed with strict thresholds

## Artifacts

- `src/platform/windows/gdi.rs`
- `src/convert/mod.rs`
- `tools/benchmark_gdi_region_pipeline.ps1`
- `docs/compliance/gdi_region_pipeline_before.csv`
- `docs/compliance/gdi_region_pipeline_after.csv`
- `docs/compliance/gdi_monitor_pipeline_before_direct_region.csv`
- `docs/compliance/gdi_monitor_pipeline_after_direct_region.csv`
