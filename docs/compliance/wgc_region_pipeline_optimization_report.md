# WGC Region Capture Pipeline Optimization Report

## Scope

This change targets the WGC **region capture** hot path in `src/platform/windows/wgc.rs`.

The previous path did:

1. `CopySubresourceRegion` to a single staging texture
2. unconditional `ID3D11DeviceContext::Flush()`
3. immediate `Map()` + CPU conversion

That serializes GPU copy and CPU conversion every frame.

## Optimization Scheme

### 1) Region readback ring (double-buffered)

- Added a dedicated region staging ring (`region_slots`) with two slots.
- Each slot now tracks:
  - staging texture/resource
  - completion query (`D3D11_QUERY_EVENT`)
  - capture metadata (`capture_time`, `present_time_ticks`, duplicate flags)
  - region-local dirty rectangles
- Recording mode now overlaps work:
  - frame N: submit GPU copy into write slot
  - same call: read/convert pending slot from frame N-1

### 2) Query-driven synchronization (reduced flush stalls)

- Region path now uses `End(query)` + adaptive spin polling before map.
- `Flush()` is now conditional (`maybe_flush_region_after_submit`) instead of unconditional.
- This removes a forced CPU/GPU sync point from the steady-state region path.

### 3) Dirty-rectangle intersection for region sub-rects

- Added region-aware dirty extraction (`extract_region_dirty_rects`):
  - intersect WGC dirty rectangles with the requested source blit
  - translate them into local region coordinates
- Added `intersect_dirty_rects` helper and tests.

### 4) Offset-aware dirty conversion path

- Added `surface::map_staging_dirty_rects_to_frame_with_offset` in `src/platform/windows/surface.rs`.
- Region path now applies dirty conversion directly at `(dst_x, dst_y)` in the destination frame.
- Falls back to full-region map/convert if dirty conversion is not usable.

## Benchmark Method

Hardware/OS load affects duplicate-frame ratio, so benchmarks were run with:

- warmup: 30 frames
- measured: 240 frames
- rounds: 3
- duplicate budget gate: `--max-duplicate-pct 95`

Commands:

```powershell
# Before (saved baseline)
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --save-baseline docs/compliance/wgc_region_pipeline_before.csv

# After (saved baseline)
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --save-baseline docs/compliance/wgc_region_pipeline_after.csv

# Regression gate
cargo run --release --example benchmark -- --backends wgc --region-center 1280x720 --warmup 30 --frames 240 --rounds 3 --baseline docs/compliance/wgc_region_pipeline_before.csv --max-regression-pct 5 --regression-metric p50 --max-duplicate-pct 95
```

## Results

From `docs/compliance/wgc_region_pipeline_before.csv` and `docs/compliance/wgc_region_pipeline_after.csv`:

- region target: `region:1280:720:1280:720`
- avg: `14.394 ms -> 11.760 ms` (**18.3% faster**)
- p50: `15.479 ms -> 14.889 ms` (**3.8% faster**)
- fps: `69.47 -> 85.03`

From monitor non-regression snapshots:

- `docs/compliance/wgc_monitor_pipeline_before.csv`
- `docs/compliance/wgc_monitor_pipeline_after.csv`

primary monitor:

- avg: `13.792 ms -> 10.563 ms`
- p50: `15.388 ms -> 12.228 ms`

## Validation

- `cargo test --lib` passes (including new rectangle-intersection tests).
- Region benchmark regression gate (`p50`, 5% threshold) passes.
- Duplicate budget (`--max-duplicate-pct 95`) passes on active scenes; fully idle desktops can legitimately exceed the threshold.

## Artifacts

- `docs/compliance/wgc_region_pipeline_before.csv`
- `docs/compliance/wgc_region_pipeline_after.csv`
- `docs/compliance/wgc_monitor_pipeline_before.csv`
- `docs/compliance/wgc_monitor_pipeline_after.csv`
