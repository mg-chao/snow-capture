# Capture perf guardrails

This folder contains scripts and benchmark entry points used to validate backend-specific capture performance changes (GDI + WGC + DXGI).

## 1) Deterministic micro-benchmarks (CI-friendly)

Runs an in-process synthetic sparse-damage benchmark that compares:

- legacy two-pass span detection (`row_equal` + `row_diff_span`)
- optimized single-pass span detection (`row_diff_span` only)
- legacy parallel full-row incremental conversion
- optimized parallel sparse-span incremental conversion
- legacy parallel compare-then-diff span scan
- optimized adaptive parallel span scan (single-scan when sampled damage is detected)
- optimized adaptive parallel span scan mode history (reuses prior mode decisions instead of re-probing every frame)

Command:

```powershell
cargo test --release platform::windows::gdi::tests::bench_span_single_scan_vs_legacy_sparse_damage -- --ignored --nocapture
cargo test --release platform::windows::gdi::tests::bench_parallel_span_scan_vs_parallel_row_sparse_damage -- --ignored --nocapture
cargo test --release platform::windows::gdi::tests::bench_parallel_span_auto_vs_compare_then_diff_sparse_damage -- --ignored --nocapture
cargo test --release platform::windows::gdi::tests::bench_parallel_span_auto_vs_compare_then_diff_duplicate_surface -- --ignored --nocapture
cargo test --release platform::windows::wgc::tests::bench_dirty_region_rect_clamp_and_normalize_vs_legacy -- --ignored --nocapture
cargo test --release platform::windows::wgc::tests::bench_region_dirty_dense_fallback_vs_legacy -- --ignored --nocapture
cargo test --release platform::windows::wgc::tests::bench_full_dirty_dense_fallback_vs_legacy -- --ignored --nocapture
cargo test --release platform::windows::surface::tests::bench_trusted_direct_hints_vs_runtime_scan -- --ignored --nocapture
cargo test --release platform::windows::surface::tests::bench_trusted_direct_bgra_batch_kernel_vs_legacy_dispatch -- --ignored --nocapture
cargo test --release platform::windows::duplication::tests::bench_direct_region_dirty_extract_clip_vs_legacy -- --ignored --nocapture
cargo test --release platform::windows::duplication::tests::bench_region_move_apply_vs_full_convert -- --ignored --nocapture
```

The test prints timing and fails if the optimized path regresses materially versus legacy.

## 2) End-to-end display benchmark: GDI span single-scan

This launches an animated WPF workload window, captures the same desktop region with GDI, and runs:

- optimized build
- legacy A/B run (`SNOW_CAPTURE_DISABLE_GDI_SPAN_SINGLE_SCAN=1`)

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/perf/run_gdi_span_single_scan_ab.ps1
```

Useful options:

- `-GuardBaselinePath <csv>`: enforce regression check against a saved baseline
- `-MinImprovementPct <value>`: required p50 improvement vs legacy
- `-MaxDuplicatePct <value>`: duplicate-frame budget guard for workload validity

## 3) End-to-end display benchmark: WGC dirty-region batch fetch

This runs the same animated WPF workload while benchmarking WGC region capture:

- optimized build (dirty regions fetched with WinRT `GetMany` batches)
- legacy A/B run (`SNOW_CAPTURE_WGC_DISABLE_DIRTY_REGION_BATCH_FETCH=1`)

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/perf/run_wgc_dirty_region_batch_ab.ps1
```

Useful options:

- `-GuardBaselinePath <csv>`: enforce regression check against a saved baseline
- `-MinImprovementPct <value>`: required p50 improvement vs legacy
- `-MaxDuplicatePct <value>`: duplicate-frame budget guard for workload validity

## 4) End-to-end benchmark: DXGI window capture A/B

Use the cross-backend benchmark example in DXGI window mode and compare optimized vs legacy toggles:

```powershell
# optimized
cargo run --release --example benchmark -- --backends dxgi --window-under-cursor

# legacy A/B (batch row-kernel dispatch)
$env:SNOW_CAPTURE_DISABLE_DIRTY_RECT_BGRA_BATCH_KERNEL=1
cargo run --release --example benchmark -- --backends dxgi --window-under-cursor
Remove-Item Env:SNOW_CAPTURE_DISABLE_DIRTY_RECT_BGRA_BATCH_KERNEL
```

Optional regression guard with a saved baseline:

```powershell
cargo run --release --example benchmark -- --backends dxgi --window-under-cursor --save-baseline target/perf/dxgi-window-baseline.csv
cargo run --release --example benchmark -- --backends dxgi --window-under-cursor --baseline target/perf/dxgi-window-baseline.csv --max-regression-pct 5 --regression-metrics p50,p95
```

Useful toggles for focused A/B:

- `SNOW_CAPTURE_DISABLE_DIRTY_RECT_TRUSTED_DIRECT=1`: force the pre-batch trusted direct path
- `SNOW_CAPTURE_DISABLE_DIRTY_RECT_BGRA_BATCH_KERNEL=1`: disable BGRA batch row-kernel dispatch
- `SNOW_CAPTURE_DXGI_DISABLE_REGION_MOVE_RECONSTRUCT=1`: disable region move-rect reconstruction and fall back to full readback
- `SNOW_CAPTURE_WGC_DISABLE_FULL_DIRTY_DENSE_FALLBACK=1`: disable WGC full-frame dense dirty fallback
