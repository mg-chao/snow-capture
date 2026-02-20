# DXGI Dirty-Rect Readback Optimization Report

## Scope

Validated a DXGI monitor-capture optimization that avoids unnecessary CPU readback/conversion work in the staged readback pipeline:

- skip readback/mapping when the pending frame is marked duplicate and the destination frame has valid history
- apply dirty-rect-only CPU conversion for sparse updates instead of full-frame conversion
- automatically fall back to full-frame mapping when dirty-rect mapping is not beneficial or fails

## Implementation Summary

Core changes live in `src/platform/windows/duplication.rs`:

1. Added a dirty-rect selection heuristic (`should_use_dirty_copy`) with bounded rect-count and dirty-area thresholds.
2. Extended staged slot readback with strategy flags:
   - `skip_readback` for duplicate frames with reusable destination history
   - `use_dirty_copy` for sparse dirty updates
3. Cached pending-frame duplicate/dirty metadata between pipelined frames so the read slot can use previous-frame change data.
4. Reset pending dirty/duplicate state on duplication recreation, mode switches, and region-capture path transitions.
5. Added focused unit tests for dirty-copy heuristics.

## Benchmark Commands

### Baseline (pre-optimization)

```powershell
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 4 --save-baseline docs/compliance/dxgi_monitor_dirty_before.csv
```

### Optimized build + strict regression checks

```powershell
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 4 --baseline docs/compliance/dxgi_monitor_dirty_before.csv --max-regression-pct 0 --regression-metric avg --save-baseline docs/compliance/dxgi_monitor_dirty_after.csv
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 4 --baseline docs/compliance/dxgi_monitor_dirty_before.csv --max-regression-pct 2 --regression-metric p50
```

## Results (from saved CSV baselines)

| Scenario | Before avg (ms) | After avg (ms) | Avg improvement | Before p50 (ms) | After p50 (ms) | p50 delta |
|---|---:|---:|---:|---:|---:|---:|
| DXGI primary monitor | 18.721 | 8.159 | 56.4% faster | 7.012 | 7.076 | +0.9% |

`avg` regression guard at 0% passed. `p50` guard at 2% passed in follow-up validation.

## Artifacts

- `docs/compliance/dxgi_monitor_dirty_before.csv`
- `docs/compliance/dxgi_monitor_dirty_after.csv`
