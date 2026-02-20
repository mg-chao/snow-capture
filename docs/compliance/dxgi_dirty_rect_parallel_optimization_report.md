# DXGI Dirty-Rect Parallel Conversion Optimization Report

## Objective

Confirm whether DXGI region/window capture still has headroom and, if yes, ship a measurable optimization without regressing other paths.

## Findings

There was still headroom in the CPU dirty-rect conversion stage:

- dirty rectangles were converted one rectangle at a time,
- each rectangle re-entered the generic surface conversion path,
- fragmented damage patterns (many medium/small rectangles) underused CPU cores.

## Optimization Scheme

Implemented a new dirty-rect work scheduler in `src/platform/windows/surface.rs`:

1. Build validated/clipped work items once per map (`DirtyRectWorkItem`) with precomputed source/destination offsets.
2. Keep all bounds/overflow checks, but pay them once per work item before conversion.
3. Add safe parallel execution for disjoint dirty rectangles:
   - verify rectangle non-overlap before parallel execution,
   - dispatch through the existing conversion worker pool,
   - each worker calls the existing unsafe SIMD conversion kernels via `convert_surface_to_rgba_unchecked`.
4. Add runtime kill-switch for A/B benchmarking:
   - env var `SNOW_CAPTURE_DISABLE_DIRTY_RECT_PARALLEL=1` disables the new parallel path.
5. Add guard tests for overlap/disjoint detection in `surface.rs`.

Support changes in `src/convert/mod.rs`:

- exposed `with_conversion_pool(...)` and `should_parallelize_work(...)` to let the DXGI/WGC surface path reuse the same worker-pool policy as the core conversion module.

## Benchmark Method

A controlled moving test window was used to generate stable, non-static desktop damage:

- launcher script: `tools/animated_bench_window.ps1`
- form handle used for window benchmark: `0x10E07F0`
- matching desktop region benchmark: `632,316,1296,759`

### Commands

```powershell
# 1) Start controlled animation window (writes HWND to file)
Start-Process -FilePath "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','E:\snow-capture\tools\animated_bench_window.ps1','-OutputFile','C:\Temp\snow_capture_bench_hwnd.txt')

# 2) Baseline (parallel path disabled)
$env:SNOW_CAPTURE_DISABLE_DIRTY_RECT_PARALLEL='1'
cargo run --release --example benchmark -- --backends dxgi --window-handle 0x10E07F0 --warmup 30 --frames 260 --rounds 4 --save-baseline docs/compliance/dxgi_window_dirty_parallel_disabled.csv
cargo run --release --example benchmark -- --backends dxgi --region 632,316,1296,759 --warmup 30 --frames 260 --rounds 4 --save-baseline docs/compliance/dxgi_region_dirty_parallel_windowarea_disabled.csv

# 3) Optimized path enabled + strict regression guard vs disabled baseline
Remove-Item Env:SNOW_CAPTURE_DISABLE_DIRTY_RECT_PARALLEL -ErrorAction SilentlyContinue
cargo run --release --example benchmark -- --backends dxgi --window-handle 0x10E07F0 --warmup 30 --frames 260 --rounds 4 --baseline docs/compliance/dxgi_window_dirty_parallel_disabled.csv --max-regression-pct 0 --regression-metrics avg,p50,p95 --save-baseline docs/compliance/dxgi_window_dirty_parallel_enabled.csv
cargo run --release --example benchmark -- --backends dxgi --region 632,316,1296,759 --warmup 30 --frames 260 --rounds 4 --baseline docs/compliance/dxgi_region_dirty_parallel_windowarea_disabled.csv --max-regression-pct 0 --regression-metrics avg,p50,p95 --save-baseline docs/compliance/dxgi_region_dirty_parallel_windowarea_enabled.csv

# 4) Full-monitor safety guard (p50 only; robust against long-tail outliers)
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 220 --rounds 3 --baseline docs/compliance/dxgi_monitor_dirty_parallel_disabled.csv --max-regression-pct 15 --regression-metric p50
```

## Results

| Scenario | Disabled avg (ms) | Enabled avg (ms) | Avg delta | Disabled p50 (ms) | Enabled p50 (ms) | p50 delta | Disabled p95 (ms) | Enabled p95 (ms) | p95 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DXGI window (`0x10E07F0`) | 5.705 | 4.061 | **-28.8%** | 1.841 | 1.082 | **-41.2%** | 41.082 | 17.007 | **-58.6%** |
| DXGI region (`632,316,1296,759`) | 5.695 | 1.683 | **-70.4%** | 2.018 | 1.003 | **-50.3%** | 35.016 | 7.050 | **-79.9%** |

Regression guard outcomes:

- Window benchmark regression checks (`avg,p50,p95`, max regression `0%`): **passed**.
- Region benchmark regression checks (`avg,p50,p95`, max regression `0%`): **passed**.
- Full-monitor safety guard (`p50`, max regression `15%`): **passed**.

## Notes

- Duplicate-frame ratios are still high in this workload because only a small moving area changes each frame; this is expected for dirty-rect-optimized paths.
- The optimization improves throughput by changing scheduling/granularity; it reuses the existing unsafe + SIMD conversion kernels rather than replacing them.

## Artifacts

- `docs/compliance/dxgi_window_dirty_parallel_disabled.csv`
- `docs/compliance/dxgi_window_dirty_parallel_enabled.csv`
- `docs/compliance/dxgi_region_dirty_parallel_windowarea_disabled.csv`
- `docs/compliance/dxgi_region_dirty_parallel_windowarea_enabled.csv`
- `docs/compliance/dxgi_monitor_dirty_parallel_disabled.csv`
