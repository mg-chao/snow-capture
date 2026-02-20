use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use snow_capture::backend::CaptureBackendKind;
use snow_capture::frame::Frame;
use snow_capture::{
    CaptureMode, CaptureRegion, CaptureSession, CaptureTarget, MonitorLayout, WindowId,
};

const DEFAULT_WARMUP_FRAMES: usize = 30;
const DEFAULT_MEASURE_FRAMES: usize = 240;
const DEFAULT_ROUNDS: usize = 3;
const DEFAULT_MAX_REGRESSION_PCT: f64 = 10.0;

#[derive(Clone, Debug)]
enum BenchTarget {
    PrimaryMonitor,
    Region(CaptureRegion),
    Window(WindowId),
}

#[derive(Clone, Copy, Debug)]
enum RegressionMetric {
    Avg,
    P50,
    P95,
    P99,
}

impl RegressionMetric {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "avg" | "average" => Some(Self::Avg),
            "p50" | "median" => Some(Self::P50),
            "p95" => Some(Self::P95),
            "p99" => Some(Self::P99),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Avg => "avg",
            Self::P50 => "p50",
            Self::P95 => "p95",
            Self::P99 => "p99",
        }
    }

    fn current_value(self, result: &BenchResult) -> f64 {
        match self {
            Self::Avg => result.avg_ms,
            Self::P50 => result.p50_ms,
            Self::P95 => result.p95_ms,
            Self::P99 => result.p99_ms,
        }
    }

    fn baseline_value(self, entry: &BaselineEntry) -> f64 {
        match self {
            Self::Avg => entry.avg_ms,
            Self::P50 => entry.p50_ms,
            Self::P95 => entry.p95_ms,
            Self::P99 => entry.p99_ms,
        }
    }
}

impl Default for RegressionMetric {
    fn default() -> Self {
        // Median is usually the most stable guard for interactive capture.
        Self::P50
    }
}

#[derive(Clone, Debug)]
struct Config {
    warmup_frames: usize,
    measure_frames: usize,
    rounds: usize,
    backends: Vec<CaptureBackendKind>,
    target: BenchTarget,
    baseline_path: Option<PathBuf>,
    save_baseline_path: Option<PathBuf>,
    max_regression_pct: f64,
    regression_metric: RegressionMetric,
}

#[derive(Clone, Debug)]
struct BenchResult {
    backend: CaptureBackendKind,
    target_label: String,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
    fps: f64,
}

#[derive(Clone, Copy, Debug)]
struct BaselineEntry {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn baseline_key(target: &str, backend: &str) -> String {
    format!("{target}|{backend}")
}

fn backend_name(kind: CaptureBackendKind) -> &'static str {
    match kind {
        CaptureBackendKind::Auto => "auto",
        CaptureBackendKind::DxgiDuplication => "dxgi",
        CaptureBackendKind::WindowsGraphicsCapture => "wgc",
        CaptureBackendKind::Gdi => "gdi",
    }
}

fn parse_backend(token: &str) -> Option<CaptureBackendKind> {
    match token.trim().to_ascii_lowercase().as_str() {
        "dxgi" | "dxgi-duplication" | "duplication" => Some(CaptureBackendKind::DxgiDuplication),
        "wgc" | "windowsgraphicscapture" | "windows-graphics-capture" => {
            Some(CaptureBackendKind::WindowsGraphicsCapture)
        }
        "gdi" => Some(CaptureBackendKind::Gdi),
        "auto" => Some(CaptureBackendKind::Auto),
        _ => None,
    }
}

fn parse_backends(csv: &str) -> Result<Vec<CaptureBackendKind>> {
    let mut out = Vec::new();
    for token in csv.split(',') {
        if token.trim().is_empty() {
            continue;
        }
        let Some(kind) = parse_backend(token) else {
            bail!("unknown backend token in --backends: {token}");
        };
        if kind != CaptureBackendKind::Auto && !out.contains(&kind) {
            out.push(kind);
        }
    }
    if out.is_empty() {
        bail!("--backends resolved to empty backend list");
    }
    Ok(out)
}

fn parse_usize_arg(flag: &str, value: Option<&str>) -> Result<usize> {
    let Some(raw) = value else {
        bail!("{flag} requires a value");
    };
    raw.parse::<usize>()
        .with_context(|| format!("failed to parse {flag} value: {raw}"))
}

fn parse_f64_arg(flag: &str, value: Option<&str>) -> Result<f64> {
    let Some(raw) = value else {
        bail!("{flag} requires a value");
    };
    raw.parse::<f64>()
        .with_context(|| format!("failed to parse {flag} value: {raw}"))
}

fn parse_window_handle(raw: &str) -> Result<isize> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("window handle cannot be empty");
    }

    let parsed_u64 = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16)
            .with_context(|| format!("failed to parse hex window handle: {trimmed}"))?
    } else {
        trimmed
            .parse::<u64>()
            .with_context(|| format!("failed to parse window handle: {trimmed}"))?
    };

    if parsed_u64 > isize::MAX as u64 {
        bail!("window handle out of range for this platform: {trimmed}");
    }
    Ok(parsed_u64 as isize)
}

fn window_under_cursor() -> Result<WindowId> {
    use windows::Win32::Foundation::POINT;
    use windows::Win32::UI::WindowsAndMessaging::{
        GA_ROOT, GetAncestor, GetCursorPos, WindowFromPoint,
    };

    let mut pt = POINT::default();
    unsafe { GetCursorPos(&mut pt) }
        .ok()
        .context("GetCursorPos failed while resolving benchmark window")?;

    let hwnd = unsafe { WindowFromPoint(pt) };
    if hwnd.0.is_null() {
        bail!("no window found under cursor at ({}, {})", pt.x, pt.y);
    }

    let root = unsafe { GetAncestor(hwnd, GA_ROOT) };
    let handle = if root.0.is_null() { hwnd } else { root };
    Ok(WindowId::from_raw_handle(handle.0 as isize))
}

fn parse_region_csv(raw: &str) -> Result<CaptureRegion> {
    let parts: Vec<&str> = raw.split(',').map(|part| part.trim()).collect();
    if parts.len() != 4 {
        bail!("--region expects x,y,width,height (got: {raw})");
    }

    let x = parts[0]
        .parse::<i32>()
        .with_context(|| format!("failed to parse region x: {}", parts[0]))?;
    let y = parts[1]
        .parse::<i32>()
        .with_context(|| format!("failed to parse region y: {}", parts[1]))?;
    let width = parts[2]
        .parse::<u32>()
        .with_context(|| format!("failed to parse region width: {}", parts[2]))?;
    let height = parts[3]
        .parse::<u32>()
        .with_context(|| format!("failed to parse region height: {}", parts[3]))?;
    CaptureRegion::new(x, y, width, height).context("invalid --region dimensions")
}

fn parse_size_2d(raw: &str) -> Result<(u32, u32)> {
    let trimmed = raw.trim();
    let Some((w_raw, h_raw)) = trimmed.split_once('x').or_else(|| trimmed.split_once('X')) else {
        bail!("expected <width>x<height>, got: {raw}");
    };
    let width = w_raw
        .trim()
        .parse::<u32>()
        .with_context(|| format!("failed to parse width in {raw}"))?;
    let height = h_raw
        .trim()
        .parse::<u32>()
        .with_context(|| format!("failed to parse height in {raw}"))?;
    if width == 0 || height == 0 {
        bail!("region dimensions must be > 0: {raw}");
    }
    Ok((width, height))
}

fn centered_primary_region(width: u32, height: u32) -> Result<CaptureRegion> {
    let layout = MonitorLayout::snapshot().context("failed to snapshot monitor layout")?;
    let primary = layout
        .monitors
        .iter()
        .find(|monitor| monitor.monitor.is_primary())
        .or_else(|| layout.monitors.first())
        .context("no monitor available for --region-center")?;

    if primary.width == 0 || primary.height == 0 {
        bail!("primary monitor has zero-sized bounds");
    }

    let fit_width = width.min(primary.width);
    let fit_height = height.min(primary.height);
    let x = i32::try_from(i64::from(primary.x) + i64::from((primary.width - fit_width) / 2))
        .context("centered region x overflowed i32 range")?;
    let y = i32::try_from(i64::from(primary.y) + i64::from((primary.height - fit_height) / 2))
        .context("centered region y overflowed i32 range")?;
    CaptureRegion::new(x, y, fit_width, fit_height)
        .context("failed to build centered primary region")
}

fn target_label(target: &BenchTarget) -> String {
    match target {
        BenchTarget::PrimaryMonitor => "primary-monitor".to_string(),
        BenchTarget::Region(region) => {
            format!(
                "region:{}:{}:{}:{}",
                region.x, region.y, region.width, region.height
            )
        }
        BenchTarget::Window(window) => format!("window:{}", window.stable_id()),
    }
}

fn parse_args() -> Result<Config> {
    let mut warmup_frames = DEFAULT_WARMUP_FRAMES;
    let mut measure_frames = DEFAULT_MEASURE_FRAMES;
    let mut rounds = DEFAULT_ROUNDS;
    let mut backends = vec![
        CaptureBackendKind::DxgiDuplication,
        CaptureBackendKind::WindowsGraphicsCapture,
        CaptureBackendKind::Gdi,
    ];
    let mut target = BenchTarget::PrimaryMonitor;
    let mut baseline_path = None;
    let mut save_baseline_path = None;
    let mut max_regression_pct = DEFAULT_MAX_REGRESSION_PCT;
    let mut regression_metric = RegressionMetric::default();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--warmup" => {
                warmup_frames = parse_usize_arg("--warmup", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--frames" => {
                measure_frames = parse_usize_arg("--frames", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--rounds" => {
                rounds = parse_usize_arg("--rounds", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--backends" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--backends requires a comma-separated value");
                };
                backends = parse_backends(raw)?;
                i += 2;
            }
            "--window-under-cursor" => {
                target = BenchTarget::Window(window_under_cursor()?);
                i += 1;
            }
            "--window-handle" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--window-handle requires a value (decimal or hex, e.g. 0x1234)");
                };
                target = BenchTarget::Window(WindowId::from_raw_handle(parse_window_handle(raw)?));
                i += 2;
            }
            "--region" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--region requires x,y,width,height");
                };
                target = BenchTarget::Region(parse_region_csv(raw)?);
                i += 2;
            }
            "--region-center" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--region-center requires <width>x<height>");
                };
                let (width, height) = parse_size_2d(raw)?;
                target = BenchTarget::Region(centered_primary_region(width, height)?);
                i += 2;
            }
            "--baseline" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--baseline requires a file path");
                };
                baseline_path = Some(PathBuf::from(raw));
                i += 2;
            }
            "--save-baseline" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--save-baseline requires a file path");
                };
                save_baseline_path = Some(PathBuf::from(raw));
                i += 2;
            }
            "--max-regression-pct" => {
                max_regression_pct =
                    parse_f64_arg("--max-regression-pct", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--regression-metric" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--regression-metric requires one of: avg, p50, p95, p99");
                };
                let Some(metric) = RegressionMetric::parse(raw) else {
                    bail!("invalid --regression-metric: {raw}. Use avg, p50, p95, or p99");
                };
                regression_metric = metric;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example benchmark -- [options]
  --warmup <n>               Warmup frames per backend (default: {DEFAULT_WARMUP_FRAMES})
  --frames <n>               Measured frames per backend (default: {DEFAULT_MEASURE_FRAMES})
  --rounds <n>               Benchmark rounds per backend (default: {DEFAULT_ROUNDS})
  --backends <csv>           Backends list, e.g. dxgi,wgc,gdi
  --window-under-cursor      Benchmark window capture for the window under the cursor
  --window-handle <value>    Benchmark window capture for an HWND (decimal or 0xHEX)
  --region <x,y,w,h>         Benchmark region capture in virtual desktop coordinates
  --region-center <WxH>      Benchmark a centered region on the primary monitor
  --baseline <path>          Compare current run to baseline CSV
  --save-baseline <path>     Save current run as baseline CSV
  --max-regression-pct <f>   Allowed metric increase vs baseline (default: {DEFAULT_MAX_REGRESSION_PCT})
  --regression-metric <m>    Metric for regression checks: avg | p50 | p95 | p99 (default: p50)"
                );
                std::process::exit(0);
            }
            other => {
                bail!("unknown argument: {other}");
            }
        }
    }

    if warmup_frames == 0 {
        bail!("--warmup must be >= 1");
    }
    if measure_frames == 0 {
        bail!("--frames must be >= 1");
    }
    if rounds == 0 {
        bail!("--rounds must be >= 1");
    }
    if max_regression_pct < 0.0 {
        bail!("--max-regression-pct must be >= 0");
    }

    Ok(Config {
        warmup_frames,
        measure_frames,
        rounds,
        backends,
        target,
        baseline_path,
        save_baseline_path,
        max_regression_pct,
        regression_metric,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let clamped = p.clamp(0.0, 1.0);
    let idx = ((n - 1) as f64 * clamped).round() as usize;
    sorted[idx]
}

fn capture_target_for(target: &BenchTarget) -> CaptureTarget {
    match target {
        BenchTarget::PrimaryMonitor => CaptureTarget::PrimaryMonitor,
        BenchTarget::Region(region) => CaptureTarget::Region(*region),
        BenchTarget::Window(window) => CaptureTarget::Window(*window),
    }
}

fn run_backend(
    kind: CaptureBackendKind,
    warmup_frames: usize,
    measure_frames: usize,
    rounds: usize,
    bench_target: &BenchTarget,
) -> Result<BenchResult> {
    let mut session = CaptureSession::builder()
        .with_backend_kind(kind)
        .capture_mode(CaptureMode::ScreenRecording)
        .build()
        .with_context(|| {
            format!(
                "failed to initialize {} backend session",
                backend_name(kind)
            )
        })?;
    let target = capture_target_for(bench_target);
    let target_label = target_label(bench_target);
    let mut frame = Frame::empty();

    let total_samples = measure_frames
        .checked_mul(rounds)
        .context("benchmark sample count overflow")?;
    let mut samples_ms = Vec::with_capacity(total_samples);

    for _round in 0..rounds {
        for _ in 0..warmup_frames {
            session
                .capture_frame_into(&target, &mut frame)
                .with_context(|| format!("warmup capture failed for {}", backend_name(kind)))?;
        }

        for _ in 0..measure_frames {
            let t0 = Instant::now();
            session
                .capture_frame_into(&target, &mut frame)
                .with_context(|| format!("capture failed for {}", backend_name(kind)))?;
            samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum_ms: f64 = samples_ms.iter().sum();
    let avg_ms = sum_ms / samples_ms.len() as f64;
    let variance = samples_ms
        .iter()
        .map(|sample| {
            let d = *sample - avg_ms;
            d * d
        })
        .sum::<f64>()
        / samples_ms.len() as f64;
    let stddev_ms = variance.sqrt();

    Ok(BenchResult {
        backend: kind,
        target_label,
        avg_ms,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        min_ms: *sorted.first().unwrap(),
        max_ms: *sorted.last().unwrap(),
        stddev_ms,
        fps: if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 },
    })
}

fn save_baseline(path: &PathBuf, results: &[BenchResult]) -> Result<()> {
    let mut out =
        String::from("target,backend,avg_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,stddev_ms,fps\n");
    for result in results {
        out.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            result.target_label,
            backend_name(result.backend),
            result.avg_ms,
            result.p50_ms,
            result.p95_ms,
            result.p99_ms,
            result.min_ms,
            result.max_ms,
            result.stddev_ms,
            result.fps
        ));
    }
    fs::write(path, out)
        .with_context(|| format!("failed to write baseline file {}", path.display()))
}

fn load_baseline(path: &PathBuf) -> Result<HashMap<String, BaselineEntry>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read baseline file {}", path.display()))?;
    let mut lines = text.lines();
    let header_line = lines
        .next()
        .context("baseline file is empty (missing header row)")?;
    let header: Vec<&str> = header_line.split(',').map(|column| column.trim()).collect();

    let column_index = |name: &str| {
        header
            .iter()
            .position(|column| column.eq_ignore_ascii_case(name))
    };
    let backend_idx =
        column_index("backend").context("baseline header is missing required `backend` column")?;
    let target_idx = column_index("target");
    let avg_idx =
        column_index("avg_ms").context("baseline header is missing required `avg_ms` column")?;
    let p50_idx =
        column_index("p50_ms").context("baseline header is missing required `p50_ms` column")?;
    let p95_idx =
        column_index("p95_ms").context("baseline header is missing required `p95_ms` column")?;
    let p99_idx = column_index("p99_ms");

    let mut out = HashMap::new();
    for (line_offset, line) in lines.enumerate() {
        let line_number = line_offset + 2;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() <= p95_idx || parts.len() <= backend_idx {
            bail!("invalid baseline line {line_number}: {line}");
        }

        let parse_metric = |column_name: &str, index: usize| -> Result<f64> {
            parts
                .get(index)
                .context(format!(
                    "baseline line {line_number} is missing `{column_name}` value"
                ))?
                .trim()
                .parse::<f64>()
                .with_context(|| {
                    format!(
                        "invalid {column_name} in baseline line {line_number}: {}",
                        line
                    )
                })
        };

        let avg_ms = parse_metric("avg_ms", avg_idx)?;
        let p50_ms = parse_metric("p50_ms", p50_idx)?;
        let p95_ms = parse_metric("p95_ms", p95_idx)?;
        let p99_ms = if let Some(index) = p99_idx {
            if index < parts.len() && !parts[index].trim().is_empty() {
                parse_metric("p99_ms", index)?
            } else {
                p95_ms
            }
        } else {
            p95_ms
        };
        let backend = parts[backend_idx].trim();
        if backend.is_empty() {
            bail!("baseline line {line_number} has empty backend value");
        }
        let target = target_idx
            .and_then(|index| parts.get(index))
            .map(|raw| raw.trim())
            .filter(|value| !value.is_empty())
            .unwrap_or("primary-monitor");

        out.insert(
            baseline_key(target, backend),
            BaselineEntry {
                avg_ms,
                p50_ms,
                p95_ms,
                p99_ms,
            },
        );
    }
    Ok(out)
}

fn check_regression(
    baseline: &HashMap<String, BaselineEntry>,
    current: &[BenchResult],
    max_regression_pct: f64,
    metric: RegressionMetric,
) -> Result<()> {
    let mut regressions = Vec::new();
    for result in current {
        let key = baseline_key(&result.target_label, backend_name(result.backend));
        let Some(base) = baseline.get(&key) else {
            continue;
        };
        let base_value = metric.baseline_value(base);
        if base_value <= 0.0 {
            continue;
        }
        let current_value = metric.current_value(result);
        let delta_pct = ((current_value - base_value) / base_value) * 100.0;
        if delta_pct > max_regression_pct {
            regressions.push(format!(
                "{} [{}] {} regressed by {:.2}% (baseline {:.3} ms -> current {:.3} ms, limit {:.2}%)",
                backend_name(result.backend),
                result.target_label,
                metric.as_str(),
                delta_pct,
                base_value,
                current_value,
                max_regression_pct
            ));
        }
    }

    if regressions.is_empty() {
        return Ok(());
    }

    bail!(
        "performance regression detected:\n{}",
        regressions.join("\n")
    )
}

fn print_results(results: &[BenchResult]) {
    println!(
        "{:<32} {:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "target",
        "backend",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "stddev",
        "fps"
    );
    for r in results {
        println!(
            "{:<32} {:<6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.2}",
            r.target_label,
            backend_name(r.backend),
            r.avg_ms,
            r.p50_ms,
            r.p95_ms,
            r.p99_ms,
            r.min_ms,
            r.max_ms,
            r.stddev_ms,
            r.fps
        );
    }
}

fn main() -> Result<()> {
    let config = parse_args()?;
    println!(
        "Running benchmark: target={} warmup={} frames={} rounds={} backends={} regression_metric={}",
        target_label(&config.target),
        config.warmup_frames,
        config.measure_frames,
        config.rounds,
        config
            .backends
            .iter()
            .map(|k| backend_name(*k))
            .collect::<Vec<_>>()
            .join(","),
        config.regression_metric.as_str(),
    );

    let mut results = Vec::with_capacity(config.backends.len());
    for backend in &config.backends {
        println!("Benchmarking {}...", backend_name(*backend));
        let result = run_backend(
            *backend,
            config.warmup_frames,
            config.measure_frames,
            config.rounds,
            &config.target,
        )?;
        results.push(result);
    }
    print_results(&results);

    if let Some(path) = &config.save_baseline_path {
        save_baseline(path, &results)?;
        println!("Saved baseline to {}", path.display());
    }

    if let Some(path) = &config.baseline_path {
        let baseline = load_baseline(path)?;
        check_regression(
            &baseline,
            &results,
            config.max_regression_pct,
            config.regression_metric,
        )?;
        println!(
            "Regression check passed ({}, max allowed regression: {:.2}%)",
            config.regression_metric.as_str(),
            config.max_regression_pct
        );
    }

    Ok(())
}
