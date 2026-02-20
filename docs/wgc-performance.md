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

For strict optimization validation (no p50 regression allowed + duplicate budget):

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --baseline target/perf/wgc-region-1280x720-baseline.csv --max-regression-pct 0 --regression-metric p50 --max-duplicate-pct 95
```

## Notes

- Baselines are keyed by both `target` and `backend`, so one CSV can store monitor, region, and window runs together.
- `p50` is the default guard because it is stable across runs and tracks steady-state capture throughput.
- Keep `--warmup` at least 30 frames to absorb shader/LUT/JIT warmup costs.
- Tune `--max-regression-pct` per machine if your capture source has large timing variance.
- Benchmark output now includes:
  - `dup_%`: percentage of duplicate frames reported by backend metadata.
  - `fresh_fps`: effective FPS for non-duplicate frames.
- Optional freshness guard:

```powershell
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --max-duplicate-pct 95
```

Use this when validating low-latency changes so throughput wins do not silently come from excessive duplicate delivery.
