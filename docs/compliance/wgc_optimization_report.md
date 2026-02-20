# WGC Capture Optimization Report

## Scope

This change targets WGC throughput/latency in two hot paths:

- monitor/window capture duplicate frames (skip unnecessary GPU copy/readback work)
- region capture (replace full-frame staging copy with GPU sub-rect readback)

## Implementation Summary

Core changes are in `src/platform/windows/wgc.rs`:

1. Enabled WGC dirty-region reporting/rendering at session setup (`SetDirtyRegionMode(ReportAndRender)`).
2. Added a duplicate fast path for recording mode:
   - when WGC reports the same present timestamp, reuse the pending slot and skip submitting a new GPU copy.
3. Added a duplicate fast path for screenshot mode:
   - if output history already matches source dimensions, bypass GPU copy/map and only update metadata.
4. Added GPU dirty-rect copy into staging (`CopySubresourceRegion`) when a slot already contains the immediate previous frame:
   - falls back to `CopyResource` when dirty-copy preconditions are not satisfied.
5. Reworked region capture to a dedicated sub-rect staging path:
   - per-region staging texture (width/height of requested region)
   - GPU-side `CopySubresourceRegion` from source frame into compact staging
   - map/convert only the compact staging texture into destination.
6. Added a region-specific stale-frame acquire path so region capture can return duplicate metadata without depending on full-frame pending slots.

## Benchmark Commands

### Baseline used for regression comparison

```powershell
# Existing baseline copied to docs/compliance/wgc_region_1280_before.csv
```

### Region capture strict regression gate

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --baseline docs/compliance/wgc_region_1280_before.csv --max-regression-pct 0 --regression-metric p50 --max-duplicate-pct 95
```

### Region capture artifact capture

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --save-baseline docs/compliance/wgc_region_1280_after.csv
```

### Primary monitor throughput/freshness snapshot

```powershell
cargo run --release --example benchmark -- --backends wgc --warmup 30 --frames 240 --rounds 3 --save-baseline docs/compliance/wgc_monitor_after.csv --max-duplicate-pct 95
```

## Results

From saved CSV artifacts:

| Scenario | Before avg (ms) | After avg (ms) | Avg improvement | Before p50 (ms) | After p50 (ms) | p50 improvement |
|---|---:|---:|---:|---:|---:|---:|
| WGC region (`1280x720`) | 14.538 | 11.326 | 22.1% faster | 15.440 | 13.207 | 14.5% faster |

Additional snapshot (post-optimization):

| Scenario | avg (ms) | p50 (ms) | fps | duplicate % | fresh_fps |
|---|---:|---:|---:|---:|---:|
| WGC primary monitor | 9.945 | 10.408 | 100.55 | 59.58 | 40.64 |

Strict gate result:

- `p50` regression check against `docs/compliance/wgc_region_1280_before.csv` passed with `--max-regression-pct 0`.
- duplicate budget check passed with `--max-duplicate-pct 95`.

## Artifacts

- `docs/compliance/wgc_region_1280_before.csv`
- `docs/compliance/wgc_region_1280_after.csv`
- `docs/compliance/wgc_monitor_after.csv`
