# DXGI Optimization Benchmark Report

## Scope

Validated the DXGI window-capture optimization that replaces full-monitor readback + CPU crop with GPU-side sub-rectangle copy (`CopySubresourceRegion`) and direct region mapping.

## Commands

### Baseline (pre-optimization, detached HEAD worktree)

```powershell
cargo run --release --example benchmark -- --window-handle 0x140e78 --backends dxgi --warmup 20 --frames 200 --rounds 3 --save-baseline docs/compliance/dxgi_window_handle_140e78_before.csv
cargo run --release --example benchmark -- --region-center 1280x720 --backends dxgi --warmup 20 --frames 200 --rounds 3 --save-baseline docs/compliance/dxgi_region_1280_before.csv
```

### Optimized branch

```powershell
cargo run --release --example benchmark -- --window-handle 0x140e78 --backends dxgi --warmup 20 --frames 200 --rounds 3 --baseline docs/compliance/dxgi_window_handle_140e78_before.csv --max-regression-pct 0 --regression-metric p50 --save-baseline docs/compliance/dxgi_window_handle_140e78_after.csv
cargo run --release --example benchmark -- --region-center 1280x720 --backends dxgi --warmup 20 --frames 200 --rounds 3 --baseline docs/compliance/dxgi_region_1280_before.csv --max-regression-pct 0 --regression-metric p50 --save-baseline docs/compliance/dxgi_region_1280_after.csv
cargo run --release --example benchmark -- --backends dxgi --warmup 30 --frames 300 --rounds 4 --baseline docs/compliance/dxgi_monitor_before.csv --max-regression-pct 5 --regression-metric p50
```

## Results

| Scenario | Before p50 (ms) | After p50 (ms) | p50 Improvement | Before avg (ms) | After avg (ms) | Avg Improvement |
|---|---:|---:|---:|---:|---:|---:|
| DXGI window capture (`0x140e78`) | 14.712 | 6.029 | 59.0% faster | 17.669 | 8.834 | 50.0% faster |
| DXGI region capture (`1280x720`) | 12.515 | 1.950 | 84.4% faster | 19.841 | 6.041 | 69.6% faster |

Primary-monitor regression guard passed against the existing baseline with `--max-regression-pct 5` on `p50`.

## Artifacts

- `docs/compliance/dxgi_window_handle_140e78_before.csv`
- `docs/compliance/dxgi_window_handle_140e78_after.csv`
- `docs/compliance/dxgi_region_1280_before.csv`
- `docs/compliance/dxgi_region_1280_after.csv`
- `docs/compliance/dxgi_monitor_before.csv`
