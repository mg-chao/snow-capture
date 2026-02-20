# DXGI Region/Window Pipeline Optimization Report

## Scope

Validated a new DXGI optimization pass focused on **window and region capture throughput**.

The optimization replaces synchronous per-frame `CopySubresourceRegion + Flush + Map` with a dedicated **multi-slot asynchronous region readback pipeline** plus region-aware dirty-rect handling.

## Implementation Summary

Primary code changes are in `src/platform/windows/duplication.rs`:

1. Added a dedicated DXGI region staging ring (`RegionStagingSlot`) with:
   - per-slot staging texture/resource/query cache
   - rolling write/read slot scheduling in recording mode
   - adaptive query spin-wait before CPU map
2. Added region pipeline lifecycle controls:
   - reset/invalidate on mode switches, access-lost recovery, and target geometry changes
   - isolation from full-monitor staging pipeline to avoid stale cross-path state
3. Added dirty-rect extraction upgrades:
   - reusable DXGI `RECT` scratch buffer to avoid per-frame metadata allocations
   - region dirty-rect intersection/rebasing (`extract_region_dirty_rects`)
   - region unchanged detection when dirty metadata is available
4. Added duplicate fast paths for region/window capture:
   - skip GPU submit for unchanged/duplicate region in recording mode when a pending slot exists
   - skip CPU mapping when destination history is valid and output is duplicate
5. Added tests for region dirty-rect correctness:
   - intersection and coordinate rebasing
   - unchanged-region detection
   - invalid blit rejection

## Benchmark Method

All runs used `examples/benchmark` in release mode.

### 1) Region capture regression guard (same workspace, before/after)

```powershell
cargo run --release --example benchmark -- --backends dxgi --region-center 1280x720 --warmup 20 --frames 200 --rounds 3 --save-baseline docs/compliance/dxgi_region_pipeline_before.csv
cargo run --release --example benchmark -- --backends dxgi --region-center 1280x720 --warmup 20 --frames 200 --rounds 3 --baseline docs/compliance/dxgi_region_pipeline_before.csv --max-regression-pct 0 --regression-metric p50 --save-baseline docs/compliance/dxgi_region_pipeline_after.csv
```

### 2) Window capture A/B with pre-optimization worktree baseline

```powershell
# Pre-optimization baseline (separate worktree at HEAD commit before local changes)
cargo run --release --example benchmark -- --backends dxgi --window-under-cursor --warmup 20 --frames 200 --rounds 3 --save-baseline E:/snow-capture/docs/compliance/dxgi_window_pipeline_before_preopt_cursor.csv

# Optimized code, same HWND from baseline
cargo run --release --example benchmark -- --backends dxgi --window-handle 0x3410b8 --warmup 20 --frames 200 --rounds 3 --baseline docs/compliance/dxgi_window_pipeline_before_preopt_cursor.csv --max-regression-pct 0 --regression-metric p50 --save-baseline docs/compliance/dxgi_window_pipeline_after.csv
```

### 3) Full-monitor safety regression guard

```powershell
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 3 --save-baseline docs/compliance/dxgi_monitor_pipeline_before.csv
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 3 --baseline docs/compliance/dxgi_monitor_pipeline_before.csv --max-regression-pct 5 --regression-metric p50 --save-baseline docs/compliance/dxgi_monitor_pipeline_after.csv
```

## Results

| Scenario | Before avg (ms) | After avg (ms) | Avg delta | Before p50 (ms) | After p50 (ms) | p50 delta |
|---|---:|---:|---:|---:|---:|---:|
| DXGI window (`HWND 0x3410b8`) | 9.103 | 2.639 | **-71.0%** | 7.134 | 1.048 | **-85.3%** |
| DXGI region (`1280x720`) | 6.200 | 6.025 | -2.8% | 2.679 | 2.634 | -1.7% |
| DXGI primary monitor | 31.202 | 9.370 | -70.0% | 6.967 | 5.715 | -18.0% |

Regression checks passed:

- Region p50 guard (`max-regression-pct=0`): **passed**
- Window p50 guard (`max-regression-pct=0`): **passed**
- Monitor p50 guard (`max-regression-pct=5`): **passed**

## Notes

- Duplicate-frame percentage increased significantly for window/region in this workload because the new logic marks frames duplicate when the **captured sub-region** is unchanged (even if other monitor areas changed). This is expected and enables major CPU readback/convert savings in static-window scenarios.
- Conversion still uses the existing unsafe + SIMD pipelines in `convert/`; this change improves how often those kernels are invoked (or skipped) rather than replacing them.

## Artifacts

- `docs/compliance/dxgi_window_pipeline_before_preopt_cursor.csv`
- `docs/compliance/dxgi_window_pipeline_after.csv`
- `docs/compliance/dxgi_region_pipeline_before.csv`
- `docs/compliance/dxgi_region_pipeline_after.csv`
- `docs/compliance/dxgi_monitor_pipeline_before.csv`
- `docs/compliance/dxgi_monitor_pipeline_after.csv`
