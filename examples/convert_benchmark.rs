use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use snow_capture::convert::{
    SurfaceConversionOptions, SurfacePixelFormat, convert_surface_to_rgba,
};

const DEFAULT_WARMUP_ITERS: usize = 24;
const DEFAULT_MEASURE_ITERS: usize = 240;
const DEFAULT_MAX_REGRESSION_PCT: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
        Self::P50
    }
}

#[derive(Clone, Debug)]
struct Config {
    warmup_iters: usize,
    measure_iters: usize,
    scenario_filter: Option<String>,
    baseline_path: Option<PathBuf>,
    save_baseline_path: Option<PathBuf>,
    max_regression_pct: f64,
    regression_metrics: Vec<RegressionMetric>,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    format: SurfacePixelFormat,
    width: usize,
    height: usize,
    src_pitch: usize,
    dst_pitch: usize,
    options: SurfaceConversionOptions,
}

#[derive(Clone, Debug)]
struct BenchResult {
    scenario: String,
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

fn parse_regression_metrics(csv: &str) -> Result<Vec<RegressionMetric>> {
    let mut metrics = Vec::new();
    for token in csv.split(',') {
        if token.trim().is_empty() {
            continue;
        }
        let Some(metric) = RegressionMetric::parse(token) else {
            bail!(
                "invalid regression metric token `{token}` in --regression-metrics (use avg,p50,p95,p99)"
            );
        };
        if !metrics.contains(&metric) {
            metrics.push(metric);
        }
    }
    if metrics.is_empty() {
        bail!("--regression-metrics resolved to empty metric list");
    }
    Ok(metrics)
}

fn parse_args() -> Result<Config> {
    let mut warmup_iters = DEFAULT_WARMUP_ITERS;
    let mut measure_iters = DEFAULT_MEASURE_ITERS;
    let mut scenario_filter = None;
    let mut baseline_path = None;
    let mut save_baseline_path = None;
    let mut max_regression_pct = DEFAULT_MAX_REGRESSION_PCT;
    let mut regression_metrics = vec![RegressionMetric::default()];

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--warmup" => {
                warmup_iters = parse_usize_arg("--warmup", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--iters" => {
                measure_iters = parse_usize_arg("--iters", args.get(i + 1).map(String::as_str))?;
                i += 2;
            }
            "--scenario" => {
                let Some(raw) = args.get(i + 1) else {
                    bail!("--scenario requires a value (scenario name or `all`)");
                };
                let trimmed = raw.trim();
                if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("all") {
                    scenario_filter = Some(trimmed.to_ascii_lowercase());
                } else {
                    scenario_filter = None;
                }
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
                regression_metrics = vec![metric];
                i += 2;
            }
            "--regression-metrics" => {
                let Some(raw) = args.get(i + 1).map(String::as_str) else {
                    bail!("--regression-metrics requires a comma-separated list (e.g. avg,p50)");
                };
                regression_metrics = parse_regression_metrics(raw)?;
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example convert_benchmark -- [options]
  --warmup <n>               Warmup iterations per scenario (default: {DEFAULT_WARMUP_ITERS})
  --iters <n>                Measured iterations per scenario (default: {DEFAULT_MEASURE_ITERS})
  --scenario <name|all>      Scenario filter (default: all)
  --baseline <path>          Compare current run to baseline CSV
  --save-baseline <path>     Save current run as baseline CSV
  --max-regression-pct <f>   Allowed metric increase vs baseline (default: {DEFAULT_MAX_REGRESSION_PCT})
  --regression-metric <m>    Metric for regression checks: avg | p50 | p95 | p99 (default: p50)
  --regression-metrics <csv> Metrics for regression checks, e.g. avg,p50,p95

Tip: compare batched-row-fence optimization with legacy mode:
  optimized: cargo run --release --example convert_benchmark -- --scenario bgra_row_serial_nt
  legacy (PowerShell): $env:SNOW_CAPTURE_DISABLE_BATCHED_ROW_NT_FENCE=1; cargo run --release --example convert_benchmark -- --scenario bgra_row_serial_nt
  legacy (cmd.exe):    set SNOW_CAPTURE_DISABLE_BATCHED_ROW_NT_FENCE=1 && cargo run --release --example convert_benchmark -- --scenario bgra_row_serial_nt
  legacy (bash):       SNOW_CAPTURE_DISABLE_BATCHED_ROW_NT_FENCE=1 cargo run --release --example convert_benchmark -- --scenario bgra_row_serial_nt"
                );
                std::process::exit(0);
            }
            other => {
                bail!("unknown argument: {other}");
            }
        }
    }

    if warmup_iters == 0 {
        bail!("--warmup must be >= 1");
    }
    if measure_iters == 0 {
        bail!("--iters must be >= 1");
    }
    if !max_regression_pct.is_finite() || max_regression_pct < 0.0 {
        bail!("--max-regression-pct must be a finite value >= 0");
    }
    if let (Some(baseline), Some(save_baseline)) = (&baseline_path, &save_baseline_path)
        && baseline == save_baseline
    {
        bail!("--baseline and --save-baseline must point to different files");
    }

    Ok(Config {
        warmup_iters,
        measure_iters,
        scenario_filter,
        baseline_path,
        save_baseline_path,
        max_regression_pct,
        regression_metrics,
    })
}

fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        return value;
    }
    (value + align - 1) & !(align - 1)
}

fn scenario_catalog() -> Vec<Scenario> {
    let contiguous_4k = Scenario {
        name: "bgra_contiguous_4k",
        format: SurfacePixelFormat::Bgra8,
        width: 3840,
        height: 2160,
        src_pitch: 3840 * 4,
        dst_pitch: 3840 * 4,
        options: SurfaceConversionOptions::default(),
    };

    let row_serial_width = 510usize;
    let row_serial_height = 300usize;
    let row_serial_src_pitch = align_up(row_serial_width * 4, 256);
    let row_serial = Scenario {
        name: "bgra_row_serial_nt",
        format: SurfacePixelFormat::Bgra8,
        width: row_serial_width,
        height: row_serial_height,
        src_pitch: row_serial_src_pitch,
        dst_pitch: row_serial_width * 4,
        options: SurfaceConversionOptions::default(),
    };

    let row_parallel_width = 1919usize;
    let row_parallel_height = 1079usize;
    let row_parallel_src_pitch = align_up(row_parallel_width * 4, 256);
    let row_parallel = Scenario {
        name: "bgra_row_parallel_nt",
        format: SurfacePixelFormat::Bgra8,
        width: row_parallel_width,
        height: row_parallel_height,
        src_pitch: row_parallel_src_pitch,
        dst_pitch: row_parallel_width * 4,
        options: SurfaceConversionOptions::default(),
    };

    vec![contiguous_4k, row_serial, row_parallel]
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

fn source_bytes_per_pixel(format: SurfacePixelFormat) -> usize {
    match format {
        SurfacePixelFormat::Bgra8 | SurfacePixelFormat::Rgba8 => 4usize,
        SurfacePixelFormat::Rgba16Float => 8usize,
    }
}

fn required_surface_bytes(pitch: usize, row_bytes: usize, height: usize) -> usize {
    pitch
        .checked_mul(height.saturating_sub(1))
        .and_then(|base| base.checked_add(row_bytes))
        .expect("surface byte count overflow")
}

fn fill_source_buffer(scenario: Scenario) -> Vec<u8> {
    let row_bytes = scenario.width * source_bytes_per_pixel(scenario.format);
    let len = required_surface_bytes(scenario.src_pitch, row_bytes, scenario.height);
    let mut out = vec![0u8; len];

    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for byte in &mut out {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *byte = (state >> 32) as u8;
    }

    out
}

fn run_scenario(
    scenario: Scenario,
    warmup_iters: usize,
    measure_iters: usize,
) -> Result<BenchResult> {
    let src = fill_source_buffer(scenario);
    let dst_row_bytes = scenario
        .width
        .checked_mul(4)
        .context("destination row byte count overflow")?;
    let dst_len = required_surface_bytes(scenario.dst_pitch, dst_row_bytes, scenario.height);
    let mut dst = vec![0u8; dst_len];

    for _ in 0..warmup_iters {
        let src_slice = black_box(src.as_slice());
        let dst_slice = black_box(dst.as_mut_slice());
        convert_surface_to_rgba(
            scenario.format,
            src_slice,
            scenario.src_pitch,
            dst_slice,
            scenario.dst_pitch,
            scenario.width,
            scenario.height,
            scenario.options,
        );
        black_box(&*dst_slice);
    }

    let mut samples_ms = Vec::with_capacity(measure_iters);
    for _ in 0..measure_iters {
        let src_slice = black_box(src.as_slice());
        let dst_slice = black_box(dst.as_mut_slice());
        let t0 = Instant::now();
        convert_surface_to_rgba(
            scenario.format,
            src_slice,
            scenario.src_pitch,
            dst_slice,
            scenario.dst_pitch,
            scenario.width,
            scenario.height,
            scenario.options,
        );
        black_box(&*dst_slice);
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
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
        scenario: scenario.name.to_string(),
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

    let scenario_idx = column_index("scenario")
        .context("baseline header is missing required `scenario` column")?;
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
        if parts.len() <= p95_idx || parts.len() <= scenario_idx {
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

        let scenario = parts[scenario_idx].trim();
        if scenario.is_empty() {
            bail!("baseline line {line_number} has empty scenario value");
        }

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

        out.insert(
            scenario.to_string(),
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

fn save_baseline(path: &PathBuf, results: &[BenchResult]) -> Result<()> {
    let mut out =
        String::from("scenario,avg_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,stddev_ms,fps\n");
    for result in results {
        out.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3}\n",
            result.scenario,
            result.avg_ms,
            result.p50_ms,
            result.p95_ms,
            result.p99_ms,
            result.min_ms,
            result.max_ms,
            result.stddev_ms,
            result.fps,
        ));
    }

    fs::write(path, out)
        .with_context(|| format!("failed to write baseline file {}", path.display()))
}

fn check_regression(
    baseline: &HashMap<String, BaselineEntry>,
    current: &[BenchResult],
    max_regression_pct: f64,
    metric: RegressionMetric,
) -> Result<()> {
    let mut regressions = Vec::new();

    for result in current {
        let Some(base) = baseline.get(&result.scenario) else {
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
                result.scenario,
                metric.as_str(),
                delta_pct,
                base_value,
                current_value,
                max_regression_pct,
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
        "{:<28} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "scenario", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "stddev", "fps"
    );
    for result in results {
        println!(
            "{:<28} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>10.2}",
            result.scenario,
            result.avg_ms,
            result.p50_ms,
            result.p95_ms,
            result.p99_ms,
            result.min_ms,
            result.max_ms,
            result.stddev_ms,
            result.fps,
        );
    }
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let all_scenarios = scenario_catalog();
    let available_scenarios = all_scenarios
        .iter()
        .map(|scenario| scenario.name)
        .collect::<Vec<_>>()
        .join(",");
    let scenarios: Vec<Scenario> = all_scenarios
        .into_iter()
        .filter(|scenario| {
            config
                .scenario_filter
                .as_ref()
                .is_none_or(|needle| scenario.name.eq_ignore_ascii_case(needle))
        })
        .collect();

    if scenarios.is_empty() {
        bail!(
            "no scenarios matched the requested filter; available scenarios: {}",
            available_scenarios
        );
    }

    println!(
        "Running conversion benchmark: warmup={} iters={} scenarios={} regression_metrics={}",
        config.warmup_iters,
        config.measure_iters,
        scenarios
            .iter()
            .map(|scenario| scenario.name)
            .collect::<Vec<_>>()
            .join(","),
        config
            .regression_metrics
            .iter()
            .map(|metric| metric.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );

    let mut results = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        println!("Benchmarking {}...", scenario.name);
        let result = run_scenario(scenario, config.warmup_iters, config.measure_iters)?;
        results.push(result);
    }

    print_results(&results);

    if let Some(path) = &config.save_baseline_path {
        save_baseline(path, &results)?;
        println!("Saved baseline to {}", path.display());
    }

    if let Some(path) = &config.baseline_path {
        let baseline = load_baseline(path)?;
        for metric in &config.regression_metrics {
            check_regression(&baseline, &results, config.max_regression_pct, *metric)?;
            println!(
                "Regression check passed ({}, max allowed regression: {:.2}%)",
                metric.as_str(),
                config.max_regression_pct,
            );
        }
    }

    Ok(())
}
