# Capture perf guardrails

This folder contains scripts and benchmark entry points used to validate backend-specific capture performance changes (GDI + WGC).

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
- Legacy toggle for dense region fallback: `SNOW_CAPTURE_WGC_DISABLE_REGION_DIRTY_DENSE_FALLBACK=1`
- Legacy toggle for A/B: `SNOW_CAPTURE_DISABLE_GDI_SPAN_SINGLE_SCAN=1`

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
