# GDI perf guardrails

This folder contains scripts and benchmark entry points used to validate GDI incremental-capture performance changes.

## 1) Deterministic micro-benchmark (CI-friendly)

Runs an in-process synthetic sparse-damage benchmark that compares:

- legacy two-pass span detection (`row_equal` + `row_diff_span`)
- optimized single-pass span detection (`row_diff_span` only)

Command:

```powershell
cargo test --release platform::windows::gdi::tests::bench_span_single_scan_vs_legacy_sparse_damage -- --ignored --nocapture
```

The test prints timing and fails if the optimized path regresses materially versus legacy.

## 2) End-to-end display benchmark (real desktop workload)

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
