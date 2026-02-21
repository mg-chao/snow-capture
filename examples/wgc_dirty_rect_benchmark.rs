use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use snow_capture::frame::DirtyRect;

const BENCH_WIDTH: u32 = 1920;
const BENCH_HEIGHT: u32 = 1080;
const DEFAULT_ROUNDS: usize = 8;
const DEFAULT_ITERATIONS: usize = 15_000;
const DEFAULT_MAX_REGRESSION_PCT: f64 = 3.0;
const DENSE_MERGE_LEGACY_MIN_RECTS: usize = 64;
const DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN: u32 = 96;

#[derive(Clone)]
struct Workload {
    name: &'static str,
    rects: Vec<DirtyRect>,
}

#[derive(Clone, Copy)]
struct CaseTiming {
    legacy: Duration,
    optimized_v1: Duration,
    optimized_v2: Duration,
}

fn clamp_dirty_rect(rect: DirtyRect, width: u32, height: u32) -> Option<DirtyRect> {
    let x = rect.x.min(width);
    let y = rect.y.min(height);
    if x >= width || y >= height {
        return None;
    }

    let max_w = width - x;
    let max_h = height - y;
    let clamped_w = rect.width.min(max_w);
    let clamped_h = rect.height.min(max_h);
    if clamped_w == 0 || clamped_h == 0 {
        return None;
    }

    Some(DirtyRect {
        x,
        y,
        width: clamped_w,
        height: clamped_h,
    })
}

#[inline(always)]
fn dirty_rect_bounds(rect: DirtyRect) -> (u32, u32) {
    (
        rect.x.saturating_add(rect.width),
        rect.y.saturating_add(rect.height),
    )
}

#[inline(always)]
fn intervals_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start < b_end && b_start < a_end
}

#[inline(always)]
fn intervals_touch_or_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start <= b_end && b_start <= a_end
}

#[derive(Clone, Copy)]
struct DirtyRectMergeCandidate {
    rect: DirtyRect,
    right: u32,
    bottom: u32,
}

impl DirtyRectMergeCandidate {
    #[inline(always)]
    fn new(rect: DirtyRect) -> Self {
        let (right, bottom) = dirty_rect_bounds(rect);
        Self {
            rect,
            right,
            bottom,
        }
    }

    #[inline(always)]
    fn can_merge(self, other: Self) -> bool {
        let horizontal_overlap =
            intervals_overlap(self.rect.x, self.right, other.rect.x, other.right);
        let vertical_overlap =
            intervals_overlap(self.rect.y, self.bottom, other.rect.y, other.bottom);
        let horizontal_touch_or_overlap =
            intervals_touch_or_overlap(self.rect.x, self.right, other.rect.x, other.right);
        let vertical_touch_or_overlap =
            intervals_touch_or_overlap(self.rect.y, self.bottom, other.rect.y, other.bottom);

        (horizontal_overlap && vertical_touch_or_overlap)
            || (vertical_overlap && horizontal_touch_or_overlap)
    }

    #[inline(always)]
    fn merge_in_place(&mut self, other: Self) {
        self.rect.x = self.rect.x.min(other.rect.x);
        self.rect.y = self.rect.y.min(other.rect.y);
        self.right = self.right.max(other.right);
        self.bottom = self.bottom.max(other.bottom);
        self.rect.width = self.right.saturating_sub(self.rect.x);
        self.rect.height = self.bottom.saturating_sub(self.rect.y);
    }
}

fn dirty_rects_can_merge(a: DirtyRect, b: DirtyRect) -> bool {
    DirtyRectMergeCandidate::new(a).can_merge(DirtyRectMergeCandidate::new(b))
}

fn merge_dirty_rects(a: DirtyRect, b: DirtyRect) -> DirtyRect {
    let mut merged = DirtyRectMergeCandidate::new(a);
    merged.merge_in_place(DirtyRectMergeCandidate::new(b));
    merged.rect
}

fn normalize_dirty_rects_legacy(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut write = 0usize;
    for read in 0..rects.len() {
        if let Some(clamped) = clamp_dirty_rect(rects[read], width, height) {
            rects[write] = clamped;
            write += 1;
        }
    }
    rects.truncate(write);
    if rects.len() <= 1 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;

        let mut i = 0usize;
        while i < rects.len() {
            let mut j = i + 1;
            while j < rects.len() {
                if dirty_rects_can_merge(rects[i], rects[j]) {
                    rects[i] = merge_dirty_rects(rects[i], rects[j]);
                    rects.swap_remove(j);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    rects.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));
}

fn normalize_dirty_rects_legacy_after_clamp(rects: &mut Vec<DirtyRect>) {
    let mut changed = true;
    while changed {
        changed = false;

        let mut i = 0usize;
        while i < rects.len() {
            let mut j = i + 1;
            while j < rects.len() {
                if dirty_rects_can_merge(rects[i], rects[j]) {
                    rects[i] = merge_dirty_rects(rects[i], rects[j]);
                    rects.swap_remove(j);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    rects.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));
}

#[inline(always)]
unsafe fn remove_dirty_rect_at_unchecked(rects: &mut Vec<DirtyRect>, idx: usize) {
    let len = rects.len();
    let ptr = rects.as_mut_ptr();
    unsafe {
        std::ptr::copy(ptr.add(idx + 1), ptr.add(idx), len - idx - 1);
        rects.set_len(len - 1);
    }
}

fn should_use_legacy_dense_merge(rects: &[DirtyRect]) -> bool {
    if rects.len() < DENSE_MERGE_LEGACY_MIN_RECTS {
        return false;
    }

    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    for rect in rects {
        min_y = min_y.min(rect.y);
        max_y = max_y.max(rect.y.saturating_add(rect.height));
    }

    max_y.saturating_sub(min_y) <= DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN
}

fn normalize_dirty_rects_optimized_v1(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut pending = std::mem::take(rects);
    let mut write = 0usize;
    for read in 0..pending.len() {
        if let Some(clamped) = clamp_dirty_rect(pending[read], width, height) {
            pending[write] = clamped;
            write += 1;
        }
    }
    pending.truncate(write);
    if pending.len() <= 1 {
        *rects = pending;
        return;
    }

    if should_use_legacy_dense_merge(&pending) {
        *rects = pending;
        normalize_dirty_rects_legacy_after_clamp(rects);
        return;
    }

    pending.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));

    rects.reserve(pending.len());
    for rect in pending {
        let mut candidate = rect;
        loop {
            let mut merged_any = false;
            let candidate_bottom = candidate.y.saturating_add(candidate.height);
            let mut idx = 0usize;

            while idx < rects.len() {
                let existing = rects[idx];
                let existing_bottom = existing.y.saturating_add(existing.height);
                if existing_bottom < candidate.y {
                    idx += 1;
                    continue;
                }
                if existing.y > candidate_bottom {
                    break;
                }

                if dirty_rects_can_merge(candidate, existing) {
                    candidate = merge_dirty_rects(candidate, existing);
                    rects.remove(idx);
                    merged_any = true;
                } else {
                    idx += 1;
                }
            }

            if !merged_any {
                break;
            }
        }

        let insert_at = rects
            .binary_search_by(|probe| {
                probe
                    .y
                    .cmp(&candidate.y)
                    .then_with(|| probe.x.cmp(&candidate.x))
            })
            .unwrap_or_else(|pos| pos);
        rects.insert(insert_at, candidate);
    }
}

fn normalize_dirty_rects_optimized_v2(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut pending = std::mem::take(rects);
    let mut write = 0usize;
    for read in 0..pending.len() {
        if let Some(clamped) = clamp_dirty_rect(pending[read], width, height) {
            pending[write] = clamped;
            write += 1;
        }
    }
    pending.truncate(write);
    if pending.len() <= 1 {
        *rects = pending;
        return;
    }

    if should_use_legacy_dense_merge(&pending) {
        *rects = pending;
        normalize_dirty_rects_legacy_after_clamp(rects);
        return;
    }

    pending.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));

    rects.reserve(pending.len());
    for rect in pending {
        let mut candidate = rect;
        loop {
            let mut merged_any = false;
            let candidate_bottom = candidate.y.saturating_add(candidate.height);
            let mut idx = 0usize;
            while idx < rects.len() {
                let existing = rects[idx];
                let existing_bottom = existing.y.saturating_add(existing.height);
                if existing_bottom < candidate.y {
                    idx += 1;
                    continue;
                }
                if existing.y > candidate_bottom {
                    break;
                }

                if dirty_rects_can_merge(candidate, existing) {
                    candidate = merge_dirty_rects(candidate, existing);
                    // SAFETY: `idx` is bounded by the loop condition (`idx < rects.len()`).
                    unsafe { remove_dirty_rect_at_unchecked(rects, idx) };
                    merged_any = true;
                } else {
                    idx += 1;
                }
            }

            if !merged_any {
                break;
            }
        }

        let insert_at = rects
            .binary_search_by(|probe| {
                probe
                    .y
                    .cmp(&candidate.y)
                    .then_with(|| probe.x.cmp(&candidate.x))
            })
            .unwrap_or_else(|pos| pos);
        rects.insert(insert_at, candidate);
    }
}

fn workload_sparse_grid() -> Workload {
    let mut rects = Vec::new();
    for y in (0..BENCH_HEIGHT).step_by(90) {
        for x in (0..BENCH_WIDTH).step_by(120) {
            rects.push(DirtyRect {
                x,
                y,
                width: 24,
                height: 20,
            });
        }
    }
    Workload {
        name: "sparse-grid",
        rects,
    }
}

fn workload_cascading_merges() -> Workload {
    let mut rects = Vec::new();
    for i in 0..200u32 {
        let y = (i % 4) * 2;
        rects.push(DirtyRect {
            x: i * 6,
            y,
            width: 7,
            height: 6,
        });
        rects.push(DirtyRect {
            x: i * 6 + 6,
            y: y + 4,
            width: 8,
            height: 6,
        });
    }
    Workload {
        name: "cascading-merges",
        rects,
    }
}

fn workload_mixed_noise() -> Workload {
    let mut rects = Vec::new();
    let mut state = 0x8f31_d2a4_u64;
    for _ in 0..260 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 16) as u32) % BENCH_WIDTH;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = ((state >> 20) as u32) % BENCH_HEIGHT;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let w = 8 + (((state >> 24) as u32) % 64);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let h = 6 + (((state >> 28) as u32) % 56);
        rects.push(DirtyRect {
            x,
            y,
            width: w,
            height: h,
        });
    }
    Workload {
        name: "mixed-noise",
        rects,
    }
}

fn bench_variant(
    seed_rects: &[DirtyRect],
    rounds: usize,
    iterations: usize,
    normalize: fn(&mut Vec<DirtyRect>, u32, u32),
) -> Duration {
    let mut best = Duration::MAX;
    let mut work = Vec::with_capacity(seed_rects.len());

    for _ in 0..rounds {
        let mut checksum = 0u64;
        let start = Instant::now();
        for _ in 0..iterations {
            work.clear();
            work.extend_from_slice(seed_rects);
            normalize(&mut work, BENCH_WIDTH, BENCH_HEIGHT);
            checksum = checksum.wrapping_add(work.len() as u64);
            if let Some(first) = work.first() {
                checksum = checksum.wrapping_add(first.width as u64);
            }
        }
        black_box(checksum);
        best = best.min(start.elapsed());
    }

    best
}

fn ns_per_iter(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1_000_000_000.0 / iterations as f64
}

fn parse_args() -> Result<(usize, usize, f64)> {
    let mut rounds = DEFAULT_ROUNDS;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut max_regression_pct = DEFAULT_MAX_REGRESSION_PCT;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--rounds" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--rounds requires a value");
                };
                rounds = raw.parse::<usize>()?;
                i += 2;
            }
            "--iterations" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--iterations requires a value");
                };
                iterations = raw.parse::<usize>()?;
                i += 2;
            }
            "--max-regression-pct" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--max-regression-pct requires a value");
                };
                max_regression_pct = raw.parse::<f64>()?;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example wgc_dirty_rect_benchmark -- [options]
  --rounds <n>               Benchmark rounds per workload (default: {DEFAULT_ROUNDS})
  --iterations <n>           Iterations per workload per round (default: {DEFAULT_ITERATIONS})
  --max-regression-pct <f>   Allowed safe-remove slowdown vs unsafe-remove before failing (default: {DEFAULT_MAX_REGRESSION_PCT})"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    if rounds == 0 {
        bail!("--rounds must be >= 1");
    }
    if iterations == 0 {
        bail!("--iterations must be >= 1");
    }
    if max_regression_pct < 0.0 {
        bail!("--max-regression-pct must be >= 0");
    }

    Ok((rounds, iterations, max_regression_pct))
}

fn verify_equivalence(workload: &Workload) -> Result<()> {
    let mut legacy = workload.rects.clone();
    let mut optimized_v1 = workload.rects.clone();
    let mut optimized_v2 = workload.rects.clone();
    normalize_dirty_rects_legacy(&mut legacy, BENCH_WIDTH, BENCH_HEIGHT);
    normalize_dirty_rects_optimized_v1(&mut optimized_v1, BENCH_WIDTH, BENCH_HEIGHT);
    normalize_dirty_rects_optimized_v2(&mut optimized_v2, BENCH_WIDTH, BENCH_HEIGHT);

    if legacy != optimized_v1 {
        bail!(
            "normalized output mismatch for workload `{}` (legacy {} rects vs optimized-v1 {} rects)",
            workload.name,
            legacy.len(),
            optimized_v1.len()
        );
    }

    if legacy != optimized_v2 {
        bail!(
            "normalized output mismatch for workload `{}` (legacy {} rects vs optimized-v2 {} rects)",
            workload.name,
            legacy.len(),
            optimized_v2.len()
        );
    }

    Ok(())
}

fn run_case(workload: &Workload, rounds: usize, iterations: usize) -> CaseTiming {
    let legacy = bench_variant(
        &workload.rects,
        rounds,
        iterations,
        normalize_dirty_rects_legacy,
    );
    let optimized_v1 = bench_variant(
        &workload.rects,
        rounds,
        iterations,
        normalize_dirty_rects_optimized_v1,
    );
    let optimized_v2 = bench_variant(
        &workload.rects,
        rounds,
        iterations,
        normalize_dirty_rects_optimized_v2,
    );
    CaseTiming {
        legacy,
        optimized_v1,
        optimized_v2,
    }
}

fn main() -> Result<()> {
    let (rounds, iterations, max_regression_pct) = parse_args()?;
    let workloads = vec![
        workload_sparse_grid(),
        workload_cascading_merges(),
        workload_mixed_noise(),
    ];

    println!(
        "Running WGC dirty-rect benchmark: rounds={} iterations={} max_regression_pct={:.2}",
        rounds, iterations, max_regression_pct
    );
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "workload", "legacy(ns)", "safe(ns)", "unsafe(ns)", "safe/unsafe", "safe/legacy"
    );

    let mut regressions = Vec::new();
    for workload in &workloads {
        verify_equivalence(workload)?;
        let timing = run_case(workload, rounds, iterations);
        let legacy_ns = ns_per_iter(timing.legacy, iterations);
        let safe_ns = ns_per_iter(timing.optimized_v1, iterations);
        let unsafe_ns = ns_per_iter(timing.optimized_v2, iterations);
        let safe_vs_unsafe = if unsafe_ns > 0.0 {
            safe_ns / unsafe_ns
        } else {
            f64::INFINITY
        };
        let safe_vs_legacy = if legacy_ns > 0.0 {
            safe_ns / legacy_ns
        } else {
            f64::INFINITY
        };
        println!(
            "{:<20} {:>12.1} {:>12.1} {:>12.1} {:>11.2}x {:>11.2}x",
            workload.name, legacy_ns, safe_ns, unsafe_ns, safe_vs_unsafe, safe_vs_legacy
        );

        let delta_pct = if unsafe_ns > 0.0 {
            ((safe_ns - unsafe_ns) / unsafe_ns) * 100.0
        } else {
            0.0
        };
        if delta_pct > max_regression_pct {
            regressions.push(format!(
                "{} regressed by {:.2}% (unsafe {:.1} ns -> safe {:.1} ns)",
                workload.name, delta_pct, unsafe_ns, safe_ns
            ));
        }
    }

    if regressions.is_empty() {
        println!("Regression guard passed.");
        Ok(())
    } else {
        bail!(
            "dirty-rect normalization regression detected:\n{}",
            regressions.join("\n")
        )
    }
}
