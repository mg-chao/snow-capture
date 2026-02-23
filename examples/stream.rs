use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use snow_capture::backend::CaptureBackendKind;
use snow_capture::frame::CaptureEvent;
use snow_capture::streaming::StreamConfig;
use snow_capture::{CaptureMode, CaptureSession, CaptureTarget, FrameTimestampAnchor};

fn main() -> Result<()> {
    let target = CaptureTarget::PrimaryMonitor;

    let session = CaptureSession::builder()
        .with_backend_kind(CaptureBackendKind::DxgiDuplication)
        .capture_mode(CaptureMode::ScreenRecording)
        .capture_cursor(true)
        .build()
        .context("failed to build capture session")?;

    let stream = session
        .start_streaming(
            target,
            StreamConfig {
                target_fps: 60,
                buffer_depth: 3,
                adaptive_fps: true,
                ..Default::default()
            },
        )
        .context("failed to start streaming")?;

    let stats = stream.stats().clone();
    let start = Instant::now();
    let run_duration = Duration::from_secs(5);
    let mut frame_count: u64 = 0;
    let mut duplicate_count: u64 = 0;
    let mut anchor: Option<FrameTimestampAnchor> = None;

    println!("Streaming for {run_duration:?} at 60 fps (adaptive)...");

    while start.elapsed() < run_duration {
        match stream.recv_timeout(Duration::from_millis(500)) {
            Ok(CaptureEvent::Frame(frame)) => {
                frame_count += 1;
                if frame.metadata.is_duplicate {
                    duplicate_count += 1;
                }

                let ts_anchor = anchor.get_or_insert_with(|| {
                    let a = FrameTimestampAnchor::from_first_frame(&frame.metadata);
                    if let Some(qpc) = a.origin_qpc_ticks() {
                        println!("  anchor: qpc_origin={qpc}, freq={}", a.qpc_frequency());
                    }
                    a
                });

                if frame_count % 60 == 0 {
                    let stream_ts = ts_anchor.stream_relative(&frame.metadata);
                    let snap = stats.snapshot();
                    let dirty = frame.metadata.dirty_rects.len();
                    let has_cursor = frame.metadata.cursor.is_some();
                    let cursor_visible =
                        frame.metadata.cursor.as_ref().map_or(false, |c| c.visible);
                    let cap_lat = frame
                        .metadata
                        .capture_duration
                        .map_or(0.0, |d| d.as_secs_f64() * 1000.0);
                    println!(
                        "  frame #{frame_count}: {}x{}, {:.1} fps (live), \
                         stream_ts={stream_ts:.3?}, cap_lat={cap_lat:.2}ms, \
                         buf_fill={:.0}%, avg_lat={:.2}ms, \
                         {dirty} dirty rects, cursor={has_cursor} visible={cursor_visible}, \
                         color={:?}, seq={}, dup={duplicate_count}, dropped={}",
                        frame.width(),
                        frame.height(),
                        snap.current_fps,
                        stream.buffer_fill_percent() * 100.0,
                        snap.capture_latency_avg.as_secs_f64() * 1000.0,
                        frame.metadata.color_space,
                        frame.metadata.sequence,
                        snap.frames_dropped,
                    );
                }
            }
            Ok(CaptureEvent::ResolutionChanged {
                old_width,
                old_height,
                new_width,
                new_height,
            }) => {
                println!(
                    "  resolution changed: {old_width}x{old_height} -> {new_width}x{new_height}"
                );
            }
            Ok(CaptureEvent::FrameDropped { sequence }) => {
                println!("  frame dropped: seq={sequence}");
            }
            Ok(CaptureEvent::Paused { at }) => {
                println!("  stream paused at {at:?}");
            }
            Ok(CaptureEvent::Resumed { at, gap }) => {
                println!("  stream resumed at {at:?}, gap={gap:.3?}");
            }
            Ok(CaptureEvent::StreamEnded) => {
                println!("  stream ended cleanly");
                break;
            }
            Ok(CaptureEvent::Error(err)) => {
                println!("  stream error: {err}");
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                println!("  (no frame received in 500ms)");
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                println!("  stream ended unexpectedly");
                break;
            }
        }
    }

    println!("\nPausing stream for 1 second...");
    stream.pause();
    std::thread::sleep(Duration::from_secs(1));
    println!("Resuming...");
    stream.resume();
    std::thread::sleep(Duration::from_secs(1));

    let tail_events = stream.stop_and_drain();
    println!("\nDrained {} tail events after stop", tail_events.len());

    let snap = stats.snapshot();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nDone: {} captured, {} dropped, {} errors recovered, \
         avg capture latency={:.2}ms, \
         {duplicate_count} duplicates in {elapsed:.2}s",
        snap.frames_captured,
        snap.frames_dropped,
        snap.errors_recovered,
        snap.capture_latency_avg.as_secs_f64() * 1000.0,
    );

    Ok(())
}
