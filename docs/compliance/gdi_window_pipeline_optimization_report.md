# GDI Window Capture Pipeline Optimization Report

## Scope

This pass targets **GDI window capture throughput** (`CaptureMode::ScreenRecording`) while preserving fallback correctness and avoiding regressions for monitor capture.

The existing pipeline already had optimized SIMD/unsafe BGRA->RGBA conversion, but the window acquisition stage still spent significant time in repeated API fallbacks each frame.

## Bottleneck Found

Before this change, each window frame always attempted:

1. `PrintWindow(hwnd, ..., flags=2)`
2. `PrintWindow(hwnd, ..., flags=0)`
3. `PrintWindow(hwnd, ..., flags=4)`
4. `GetWindowDC + BitBlt` fallback

Even when one strategy consistently worked (or failed), this probe chain was repeated every frame, causing avoidable overhead.

## Implementation Summary

Primary changes are in `src/platform/windows/gdi.rs`:

1. Added a mode-aware window capture strategy planner:
   - `CaptureMode::ScreenRecording`: prefer `WindowDcBitBlt` first (throughput-first)
   - `CaptureMode::Screenshot`: keep `PrintWindow` first (fidelity-first)
2. Added per-capturer strategy caching for recording mode:
   - `WindowsWindowCapturer` remembers `preferred_path` while running `CaptureMode::ScreenRecording`
   - screenshot mode keeps `PrintWindow` ordering first for fidelity
3. Added persistent cached window DC for the BitBlt path:
   - reuse `GetWindowDC` result across frames
   - release when switching away from BitBlt / refresh / drop
4. Kept full fallback safety:
   - if preferred path fails, remaining paths are tried in order
   - on success, the working path becomes the new preferred path
5. Added unit tests for strategy ordering/dedup logic:
   - screenshot ordering
   - recording ordering
   - preferred-path front-loading without duplicates

The pixel conversion stage remains the existing unsafe + SIMD-accelerated path (`convert_bgra_to_rgba_nt_unchecked` in recording mode).

## Benchmark Method

All runs used `examples/benchmark` in release mode.

### 1) Window baseline (before optimization)

```powershell
cargo run --release --example benchmark -- --backends gdi --window-under-cursor --warmup 30 --frames 300 --rounds 3 --save-baseline docs/compliance/gdi_window_cursor_before.csv
```

### 2) Window after optimization + strict regression guard (same HWND target)

```powershell
cargo run --release --example benchmark -- --backends gdi --window-handle 0xdf10f4 --warmup 30 --frames 300 --rounds 3 --baseline docs/compliance/gdi_window_cursor_before.csv --max-regression-pct 0 --regression-metrics avg,p50,p95 --save-baseline docs/compliance/gdi_window_cursor_after.csv
```

### 3) Monitor safety baseline and regression guard

```powershell
cargo run --release --example benchmark -- --backends gdi --warmup 30 --frames 300 --rounds 3 --save-baseline docs/compliance/gdi_monitor_before.csv
cargo run --release --example benchmark -- --backends gdi --warmup 30 --frames 300 --rounds 3 --baseline docs/compliance/gdi_monitor_before.csv --max-regression-pct 5 --regression-metrics avg,p50,p95 --save-baseline docs/compliance/gdi_monitor_after.csv
```

## Results

| Scenario | Before avg (ms) | After avg (ms) | Avg delta | Before p50 (ms) | After p50 (ms) | p50 delta | Before p95 (ms) | After p95 (ms) | p95 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GDI window (`0xdf10f4`) | 42.107 | 4.624 | **-89.02%** | 41.774 | 4.587 | **-89.02%** | 46.647 | 5.237 | **-88.77%** |
| GDI primary monitor | 43.452 | 43.602 | +0.35% | 42.145 | 42.306 | +0.38% | 49.707 | 49.639 | -0.14% |

Regression checks passed:

- Window avg/p50/p95 guard (`max-regression-pct=0`): **passed**
- Monitor avg/p50/p95 guard (`max-regression-pct=5`): **passed**

## Notes

- The win mainly comes from removing repeated `PrintWindow` probe overhead from the steady-state frame loop.
- The optimization intentionally differentiates screenshot vs recording behavior through `CaptureMode`.
- No monitor-capture performance regression was observed beyond normal benchmark noise.

## Artifacts

- `docs/compliance/gdi_window_cursor_before.csv`
- `docs/compliance/gdi_window_cursor_after.csv`
- `docs/compliance/gdi_monitor_before.csv`
- `docs/compliance/gdi_monitor_after.csv`
- `src/platform/windows/gdi.rs`

