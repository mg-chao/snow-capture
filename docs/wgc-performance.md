# WGC Performance Regression Guard

Use the benchmark example with a persisted baseline CSV so regressions fail fast.

## 1) Refresh baseline (manual)

### Full-monitor WGC baseline

```powershell
cargo run --release --example benchmark -- --backends wgc --warmup 30 --frames 240 --rounds 3 --save-baseline target/perf/wgc-monitor-baseline.csv
```

### Region WGC baseline (centered 1280x720 on primary monitor)

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --save-baseline target/perf/wgc-region-1280x720-baseline.csv
```

You can also benchmark an exact virtual-desktop rectangle with:

```powershell
cargo run --release --example benchmark -- --backends wgc --region 640,360,1280,720 --warmup 30 --frames 240 --rounds 3
```

## 2) Validate candidate changes

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --baseline target/perf/wgc-region-1280x720-baseline.csv --max-regression-pct 3 --regression-metric p50
```

The command exits with a non-zero code when the selected metric regresses more than the allowed threshold.

## Notes

- Baselines are keyed by both `target` and `backend`, so one CSV can store monitor, region, and window runs together.
- `p50` is the default guard because it is stable across runs and tracks steady-state capture throughput.
- Keep `--warmup` at least 30 frames to absorb shader/LUT/JIT warmup costs.
- Tune `--max-regression-pct` per machine if your capture source has large timing variance.
