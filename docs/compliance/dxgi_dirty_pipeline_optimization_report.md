# DXGI Dirty Pipeline Optimization Report

## Scope

This validation covers a DXGI capture-path optimization focused on damage processing overhead in `ScreenRecording` mode.

## Optimization Summary

Changes were made in `src/platform/windows/duplication.rs`:

1. Added `normalize_dirty_rects_in_place` to clamp, sort, and merge touching/overlapping dirty rectangles before dirty-copy heuristics and readback.
2. Added duplicate fast path that skips `GetFrameDirtyRects` COM metadata queries when `LastPresentTime` indicates no new desktop present.
3. Applied normalized dirty metadata to both monitor and region pipelines so sparse updates stay on the dirty-copy path more often.
4. Added focused unit tests for dirty-rect normalization behavior.

To enforce stricter regression gates, `examples/benchmark.rs` now supports `--regression-metrics <csv>` so multiple latency metrics can be checked in one run.

## Benchmark Commands

### Baseline capture (pre-optimization state)

```powershell
cargo run --release --example benchmark -- --backends dxgi --warmup 20 --frames 200 --rounds 3 --save-baseline docs/compliance/dxgi_dirty_pipeline_monitor_before_controlled.csv
```

### Optimized build + strict multi-metric regression gate

```powershell
cargo run --release --example benchmark -- --backends dxgi --warmup 20 --frames 200 --rounds 3 --baseline docs/compliance/dxgi_dirty_pipeline_monitor_before_controlled.csv --max-regression-pct 0 --regression-metrics avg,p50 --save-baseline docs/compliance/dxgi_dirty_pipeline_monitor_after_controlled.csv
```

## Results

| Scenario | Before avg (ms) | After avg (ms) | Avg improvement | Before p50 (ms) | After p50 (ms) | p50 improvement |
|---|---:|---:|---:|---:|---:|---:|
| DXGI primary monitor | 16.368 | 11.379 | 30.5% faster | 8.092 | 5.998 | 25.9% faster |

Strict regression checks passed with `--max-regression-pct 0 --regression-metrics avg,p50`.

## Artifacts

- `docs/compliance/dxgi_dirty_pipeline_monitor_before_controlled.csv`
- `docs/compliance/dxgi_dirty_pipeline_monitor_after_controlled.csv`
