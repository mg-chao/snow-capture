use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use snow_capture::backend::CaptureBackendKind;
use snow_capture::frame::Frame;
use snow_capture::{CaptureMode, CaptureSession, CaptureTarget};

const DEFAULT_WARMUP_FRAMES: usize = 30;
const DEFAULT_MEASURE_FRAMES: usize = 240;
const DEFAULT_MAX_REGRESSION_PCT: f64 = 10.0;

#[derive(Clone, Copy, Debug)]
enum RegressionMetric {
    Avg,
    P50,
    P95,
}

impl RegressionMetric {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "avg" | "average" => Some(Self::Avg),
            "p50" | "median" => Some(Self::P50),
            "p95" => Some(Self::P95),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Avg => "avg",
            Self::P50 => "p50",
            Self::P95 => "p95",
        }
    }

    fn current_value(self, result: &BenchResult) -> f64 {
        match self {
            Self::Avg => result.avg_ms,
            Self::P50 => result.p50_ms,
            Self::P95 => result.p95_ms,
        }
    }

    fn baseline_value(self, entry: &BaselineEntry) -> f64 {
        match self {
            Self::Avg => entry.avg_ms,
            Self::P50 => entry.p50_ms,
            Self::P95 => entry.p95_ms,
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
    backends: Vec<CaptureBackendKind>,
    baseline_path: Option<PathBuf>,
    save_baseline_path: Option<PathBuf>,
    max_regression_pct: f64,
    regression_metric: RegressionMetric,
}

#[derive(Clone, Debug)]
struct BenchResult {
    backend: CaptureBackendKind,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    fps: f64,
}

#[derive(Clone, Copy, Debug)]
struct BaselineEntry {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
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

fn parse_args() -> Result<Config> {
    let mut warmup_frames = DEFAULT_WARMUP_FRAMES;
    let mut measure_frames = DEFAULT_MEASURE_FRAMES;
    let mut backends = vec![
        CaptureBackendKind::DxgiDuplication,
        CaptureBackendKind::WindowsGraphicsCapture,
        CaptureBackendKind::Gdi,
    ];
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
            "--backends" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--backends requires a comma-separated value");
                };
                backends = parse_backends(raw)?;
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
                    bail!("--regression-metric requires one of: avg, p50, p95");
                };
                let Some(metric) = RegressionMetric::parse(raw) else {
                    bail!("invalid --regression-metric: {raw}. Use avg, p50, or p95");
                };
                regression_metric = metric;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example benchmark -- [options]
  --warmup <n>               Warmup frames per backend (default: {DEFAULT_WARMUP_FRAMES})
  --frames <n>               Measured frames per backend (default: {DEFAULT_MEASURE_FRAMES})
  --backends <csv>           Backends list, e.g. dxgi,wgc,gdi
  --baseline <path>          Compare current run to baseline CSV
  --save-baseline <path>     Save current run as baseline CSV
  --max-regression-pct <f>   Allowed metric increase vs baseline (default: {DEFAULT_MAX_REGRESSION_PCT})
  --regression-metric <m>    Metric for regression checks: avg | p50 | p95 (default: p50)"
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
    if max_regression_pct < 0.0 {
        bail!("--max-regression-pct must be >= 0");
    }

    Ok(Config {
        warmup_frames,
        measure_frames,
        backends,
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

fn run_backend(
    kind: CaptureBackendKind,
    warmup_frames: usize,
    measure_frames: usize,
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
    let target = CaptureTarget::PrimaryMonitor;
    let mut frame = Frame::empty();

    for _ in 0..warmup_frames {
        session
            .capture_frame_into(&target, &mut frame)
            .with_context(|| format!("warmup capture failed for {}", backend_name(kind)))?;
    }

    let mut samples_ms = Vec::with_capacity(measure_frames);
    for _ in 0..measure_frames {
        let t0 = Instant::now();
        session
            .capture_frame_into(&target, &mut frame)
            .with_context(|| format!("capture failed for {}", backend_name(kind)))?;
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum_ms: f64 = samples_ms.iter().sum();
    let avg_ms = sum_ms / samples_ms.len() as f64;

    Ok(BenchResult {
        backend: kind,
        avg_ms,
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        min_ms: *sorted.first().unwrap(),
        max_ms: *sorted.last().unwrap(),
        fps: if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 },
    })
}

fn save_baseline(path: &PathBuf, results: &[BenchResult]) -> Result<()> {
    let mut out = String::from("backend,avg_ms,p50_ms,p95_ms,min_ms,max_ms,fps\n");
    for result in results {
        out.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            backend_name(result.backend),
            result.avg_ms,
            result.p50_ms,
            result.p95_ms,
            result.min_ms,
            result.max_ms,
            result.fps
        ));
    }
    fs::write(path, out)
        .with_context(|| format!("failed to write baseline file {}", path.display()))
}

fn load_baseline(path: &PathBuf) -> Result<HashMap<String, BaselineEntry>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read baseline file {}", path.display()))?;
    let mut out = HashMap::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx == 0 {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 4 {
            bail!("invalid baseline line {}: {line}", line_idx + 1);
        }
        let avg_ms = parts[1].parse::<f64>().with_context(|| {
            format!("invalid avg_ms in baseline line {}: {}", line_idx + 1, line)
        })?;
        let p50_ms = parts[2].parse::<f64>().with_context(|| {
            format!("invalid p50_ms in baseline line {}: {}", line_idx + 1, line)
        })?;
        let p95_ms = parts[3].parse::<f64>().with_context(|| {
            format!("invalid p95_ms in baseline line {}: {}", line_idx + 1, line)
        })?;
        out.insert(
            parts[0].to_string(),
            BaselineEntry {
                avg_ms,
                p50_ms,
                p95_ms,
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
        let key = backend_name(result.backend).to_string();
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
                "{} {} regressed by {:.2}% (baseline {:.3} ms -> current {:.3} ms, limit {:.2}%)",
                key,
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
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "backend", "avg_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps"
    );
    for r in results {
        println!(
            "{:<6} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.2}",
            backend_name(r.backend),
            r.avg_ms,
            r.p50_ms,
            r.p95_ms,
            r.min_ms,
            r.max_ms,
            r.fps
        );
    }
}

fn main() -> Result<()> {
    let config = parse_args()?;
    println!(
        "Running benchmark: warmup={} frames={} backends={} regression_metric={}",
        config.warmup_frames,
        config.measure_frames,
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
        let result = run_backend(*backend, config.warmup_frames, config.measure_frames)?;
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
