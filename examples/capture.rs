use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use snow_capture::{
    CaptureMode, CaptureSession, CaptureTarget, backend::CaptureBackendKind, frame::Frame,
};

fn save_frame_png(frame: &Frame, path: &Path) -> anyhow::Result<()> {
    image::save_buffer(
        path,
        frame.as_rgba_bytes(),
        frame.width(),
        frame.height(),
        image::ColorType::Rgba8,
    )
    .map_err(|e| anyhow::anyhow!("failed to write PNG to {}: {e}", path.display()))
}

fn capture_primary_to_png(kind: CaptureBackendKind, label: &str, output_path: &str) -> Result<()> {
    let target = CaptureTarget::PrimaryMonitor;

    let begin = Instant::now();
    let mut session = CaptureSession::builder()
        .with_backend_kind(kind)
        .capture_mode(CaptureMode::Screenshot)
        .build()
        .with_context(|| format!("failed to initialize {label} capture session"))?;
    let elapsed = begin.elapsed();

    println!(
        "Initialized {label} capture session in {:.3} ms",
        elapsed.as_secs_f64() * 1000.0
    );

    let mut frame = Frame::empty();

    let begin = Instant::now();
    session
        .capture_frame_into(&target, &mut frame)
        .with_context(|| format!("failed to capture frame using {label}"))?;
    let elapsed = begin.elapsed();

    println!(
        "Captured {label} frame: {}x{} in {:.3} ms",
        frame.width(),
        frame.height(),
        elapsed.as_secs_f64() * 1000.0
    );

    save_frame_png(&frame, Path::new(output_path))?;
    println!("Saved {label} capture to {output_path}");
    Ok(())
}

fn main() -> Result<()> {
    capture_primary_to_png(CaptureBackendKind::Gdi, "GDI", "./capture-gdi.png")?;
    capture_primary_to_png(
        CaptureBackendKind::DxgiDuplication,
        "DXGI (HDR Linear)",
        "./capture-dxgi-hdr-linear.png",
    )?;
    capture_primary_to_png(
        CaptureBackendKind::WindowsGraphicsCapture,
        "WGC",
        "./capture-wgc.png",
    )?;

    Ok(())
}
