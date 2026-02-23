//! Continuous capture streaming with frame pacing, backpressure, and
//! adaptive rate control.
//!
//! The streaming module runs a capture loop on a dedicated thread,
//! delivering `CaptureEvent`s through a bounded channel. The caller
//! consumes events from the receiver at its own pace (e.g. feeding
//! an encoder). When the receiver falls behind, the oldest frames
//! are dropped and a `CaptureEvent::FrameDropped` notification is
//! sent so recorders can insert duplicate frames for A/V sync.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::CaptureTarget;
use crate::capture_session::CaptureSession;
use crate::error::{CaptureError, CaptureResult};
use crate::frame::{CaptureEvent, Frame};

/// Configuration for a continuous capture stream.
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// Target frames per second. The stream thread will pace captures
    /// to approximate this rate. Set to `0` for uncapped (capture as
    /// fast as the backend allows).
    pub target_fps: u32,
    /// Maximum number of frames buffered in the channel before the
    /// stream starts dropping the oldest frames. Higher values add
    /// latency but tolerate encoder stalls better.
    pub buffer_depth: usize,
    /// Maximum number of consecutive transient errors before the
    /// stream thread gives up and exits.
    pub max_consecutive_errors: usize,
    /// Enable adaptive frame rate reduction under sustained
    /// backpressure. When the receiver can't keep up, the capture
    /// rate is temporarily halved (down to `min_fps`), then ramped
    /// back up when the consumer catches up.
    pub adaptive_fps: bool,
    /// Minimum FPS when adaptive rate reduction is active.
    /// Ignored when `adaptive_fps` is `false`.
    pub min_fps: u32,
    /// When `true`, the stream automatically pauses after sending a
    /// `ResolutionChanged` event, giving the consumer time to
    /// reconfigure its encoder before calling `resume()`.
    pub pause_on_resolution_change: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            buffer_depth: 3,
            max_consecutive_errors: 30,
            adaptive_fps: false,
            min_fps: 15,
            pause_on_resolution_change: false,
        }
    }
}

/// Live statistics about the running stream, updated atomically by the
/// capture thread. Read these from any thread via `StreamHandle::stats()`.
#[derive(Debug)]
pub struct StreamStats {
    /// Total frames captured since the stream started.
    pub frames_captured: AtomicU64,
    /// Total frames dropped due to backpressure (receiver too slow).
    pub frames_dropped: AtomicU64,
    /// Total transient errors encountered and recovered from.
    pub errors_recovered: AtomicU64,
    /// Current effective FPS (updated once per second).
    pub current_fps: AtomicU64,
    /// Number of frames currently sitting in the channel buffer.
    /// Stored as a plain u64; compare against `StreamConfig::buffer_depth`
    /// to get a fill percentage.
    pub buffer_fill: AtomicU64,
    /// Exponentially-weighted moving average of per-frame capture
    /// latency in nanoseconds. Useful for detecting GPU readback
    /// bottlenecks. Stored as `f64` bits.
    pub capture_latency_avg_ns: AtomicU64,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            frames_captured: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            errors_recovered: AtomicU64::new(0),
            current_fps: AtomicU64::new(0),
            buffer_fill: AtomicU64::new(0),
            capture_latency_avg_ns: AtomicU64::new(0),
        }
    }
}

impl StreamStats {
    /// Snapshot the current stats into plain values.
    pub fn snapshot(&self) -> StreamStatsSnapshot {
        StreamStatsSnapshot {
            frames_captured: self.frames_captured.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            errors_recovered: self.errors_recovered.load(Ordering::Relaxed),
            current_fps: f64::from_bits(self.current_fps.load(Ordering::Relaxed)),
            buffer_fill: self.buffer_fill.load(Ordering::Relaxed),
            capture_latency_avg: Duration::from_nanos(f64::from_bits(
                self.capture_latency_avg_ns.load(Ordering::Relaxed),
            ) as u64),
        }
    }
}

/// A point-in-time copy of stream statistics.
#[derive(Clone, Debug, Default)]
pub struct StreamStatsSnapshot {
    pub frames_captured: u64,
    pub frames_dropped: u64,
    pub errors_recovered: u64,
    pub current_fps: f64,
    /// Number of frames currently buffered in the channel.
    pub buffer_fill: u64,
    /// Exponentially-weighted moving average of per-frame capture latency.
    pub capture_latency_avg: Duration,
}

/// Handle to a running capture stream. Dropping the handle stops the
/// background capture thread.
pub struct StreamHandle {
    receiver: mpsc::Receiver<CaptureEvent>,
    stop_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
    stats: Arc<StreamStats>,
    join_handle: Option<std::thread::JoinHandle<()>>,
    buffer_depth: usize,
}

impl StreamHandle {
    /// Start the streaming capture loop on a background thread.
    pub(crate) fn start(
        mut session: CaptureSession,
        target: CaptureTarget,
        config: StreamConfig,
    ) -> CaptureResult<Self> {
        let buffer_depth = config.buffer_depth.max(1);
        let (tx, rx) = mpsc::sync_channel::<CaptureEvent>(buffer_depth);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(StreamStats::default());

        let stop = stop_flag.clone();
        let pause = pause_flag.clone();
        let stats_clone = stats.clone();

        let join_handle = std::thread::Builder::new()
            .name("snow-capture-stream".to_string())
            .spawn(move || {
                stream_loop(
                    &mut session,
                    &target,
                    &config,
                    &tx,
                    &stop,
                    &pause,
                    &stats_clone,
                );
            })
            .map_err(|e| {
                CaptureError::Platform(anyhow::anyhow!(
                    "failed to spawn capture stream thread: {e}"
                ))
            })?;

        Ok(Self {
            receiver: rx,
            stop_flag,
            pause_flag,
            stats,
            join_handle: Some(join_handle),
            buffer_depth,
        })
    }

    /// Get the raw receiver end of the capture event channel.
    ///
    /// **Note:** Prefer `recv()`, `try_recv()`, or `recv_timeout()` on
    /// `StreamHandle` directly — those methods keep `buffer_fill` accurate.
    /// Reading from the raw receiver bypasses fill tracking.
    pub fn receiver(&self) -> &mpsc::Receiver<CaptureEvent> {
        &self.receiver
    }

    /// Consume the handle and return the raw receiver. The stream thread
    /// will be signaled to stop and joined.
    ///
    /// **Note:** `buffer_fill` will no longer be updated after this call.
    pub fn into_receiver(mut self) -> mpsc::Receiver<CaptureEvent> {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
        let this = std::mem::ManuallyDrop::new(self);
        unsafe { std::ptr::read(&this.receiver) }
    }

    /// Receive the next capture event, blocking until one is available
    /// or the channel disconnects. Automatically updates `buffer_fill`
    /// when a `Frame` event is consumed.
    pub fn recv(&self) -> Result<CaptureEvent, mpsc::RecvError> {
        let event = self.receiver.recv()?;
        if matches!(&event, CaptureEvent::Frame(_)) {
            self.stats.buffer_fill.fetch_sub(1, Ordering::Release);
        }
        Ok(event)
    }

    /// Try to receive a capture event without blocking. Automatically
    /// updates `buffer_fill` when a `Frame` event is consumed.
    pub fn try_recv(&self) -> Result<CaptureEvent, mpsc::TryRecvError> {
        let event = self.receiver.try_recv()?;
        if matches!(&event, CaptureEvent::Frame(_)) {
            self.stats.buffer_fill.fetch_sub(1, Ordering::Release);
        }
        Ok(event)
    }

    /// Receive a capture event with a timeout. Automatically updates
    /// `buffer_fill` when a `Frame` event is consumed.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<CaptureEvent, mpsc::RecvTimeoutError> {
        let event = self.receiver.recv_timeout(timeout)?;
        if matches!(&event, CaptureEvent::Frame(_)) {
            self.stats.buffer_fill.fetch_sub(1, Ordering::Release);
        }
        Ok(event)
    }

    /// Signal the stream thread to stop. Non-blocking — the thread will
    /// exit on its next loop iteration.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }

    /// Pause the capture stream. The capture thread idles without
    /// releasing the underlying OS capture resources, so resume is
    /// near-instant.
    pub fn pause(&self) {
        self.pause_flag.store(true, Ordering::Release);
    }

    /// Resume a paused capture stream.
    pub fn resume(&self) {
        self.pause_flag.store(false, Ordering::Release);
    }

    /// Whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::Acquire)
    }

    /// Check whether the stream thread is still running.
    pub fn is_running(&self) -> bool {
        self.join_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }

    /// Get a reference to the live stream statistics.
    pub fn stats(&self) -> &Arc<StreamStats> {
        &self.stats
    }

    /// Current buffer fill level as a fraction in `[0.0, 1.0]`.
    /// Useful for proactive quality adjustment before drops occur.
    pub fn buffer_fill_percent(&self) -> f64 {
        if self.buffer_depth == 0 {
            return 0.0;
        }
        let fill = self.stats.buffer_fill.load(Ordering::Relaxed);
        (fill as f64 / self.buffer_depth as f64).min(1.0)
    }

    /// Signal the stream to stop and drain all remaining buffered
    /// events. Returns an iterator over the final events so the
    /// recorder can flush its encoder without losing the tail frames.
    pub fn stop_and_drain(mut self) -> Vec<CaptureEvent> {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
        // Drain everything left in the channel.
        let mut events = Vec::new();
        while let Ok(event) = self.receiver.try_recv() {
            events.push(event);
        }
        // Prevent Drop from joining again.
        let _ = std::mem::ManuallyDrop::new(self);
        events
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn stream_loop(
    session: &mut CaptureSession,
    target: &CaptureTarget,
    config: &StreamConfig,
    tx: &mpsc::SyncSender<CaptureEvent>,
    stop: &AtomicBool,
    pause: &AtomicBool,
    stats: &StreamStats,
) {
    let base_interval = if config.target_fps > 0 {
        Some(Duration::from_secs_f64(1.0 / config.target_fps as f64))
    } else {
        None
    };

    let min_interval = if config.adaptive_fps && config.min_fps > 0 {
        Some(Duration::from_secs_f64(1.0 / config.min_fps as f64))
    } else {
        None
    };

    let mut reuse_frame: Option<Frame> = None;
    let mut consecutive_errors: usize = 0;
    let mut last_width: u32 = 0;
    let mut last_height: u32 = 0;

    // Smooth adaptive pacing state (EWMA-based).
    let mut current_interval = base_interval;
    // Exponential smoothing factor for adaptive pacing.
    // Higher values react faster to backpressure changes.
    const ADAPTIVE_ALPHA: f64 = 0.15;
    // When the drop ratio over the recent window exceeds this,
    // the interval is nudged longer.
    const DROP_RATIO_THRESHOLD: f64 = 0.10;
    // Size of the sliding window for drop ratio calculation.
    const ADAPTIVE_WINDOW: u32 = 30;
    let mut window_drops: u32 = 0;
    let mut window_total: u32 = 0;

    // Capture latency EWMA state.
    let mut latency_avg_ns: f64 = 0.0;
    const LATENCY_ALPHA: f64 = 0.1;

    // Buffer fill tracking — stats.buffer_fill is the shared atomic
    // counter. The producer (this loop) increments on successful send,
    // and the consumer decrements via StreamHandle::recv* methods.

    // FPS measurement.
    let mut fps_counter: u64 = 0;
    let mut fps_epoch = Instant::now();

    // Pause/resume lifecycle tracking.
    let mut was_paused = false;
    let mut pause_started: Option<Instant> = None;

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }

        // Pause handling with lifecycle events.
        if pause.load(Ordering::Acquire) {
            if !was_paused {
                // Entering pause — send Paused event.
                let now = Instant::now();
                pause_started = Some(now);
                let _ = tx.try_send(CaptureEvent::Paused { at: now });
                was_paused = true;
            }
            std::thread::sleep(Duration::from_millis(50));
            // Reset FPS counter across pause boundaries.
            fps_counter = 0;
            fps_epoch = Instant::now();
            continue;
        } else if was_paused {
            // Exiting pause — send Resumed event.
            let now = Instant::now();
            let gap = pause_started
                .map(|s| now.saturating_duration_since(s))
                .unwrap_or(Duration::ZERO);
            let _ = tx.try_send(CaptureEvent::Resumed { at: now, gap });
            was_paused = false;
            pause_started = None;
        }

        let frame_start = Instant::now();

        let capture_result = match reuse_frame.take() {
            Some(f) => session.capture_frame_reuse(target, f),
            None => session.capture_frame(target),
        };

        let capture_elapsed = frame_start.elapsed();

        match capture_result {
            Ok(mut frame) => {
                consecutive_errors = 0;

                // Record capture latency on the frame.
                frame.metadata.capture_duration = Some(capture_elapsed);

                // Update EWMA capture latency.
                let sample_ns = capture_elapsed.as_nanos() as f64;
                latency_avg_ns = LATENCY_ALPHA * sample_ns + (1.0 - LATENCY_ALPHA) * latency_avg_ns;
                stats
                    .capture_latency_avg_ns
                    .store(latency_avg_ns.to_bits(), Ordering::Relaxed);

                let seq = frame.metadata.sequence;

                // Detect resolution changes.
                let (w, h) = frame.dimensions();
                if last_width != 0 && last_height != 0 && (w != last_width || h != last_height) {
                    let event = CaptureEvent::ResolutionChanged {
                        old_width: last_width,
                        old_height: last_height,
                        new_width: w,
                        new_height: h,
                    };
                    let _ = tx.try_send(event);

                    // Auto-pause on resolution change if configured.
                    if config.pause_on_resolution_change {
                        pause.store(true, Ordering::Release);
                    }
                }
                last_width = w;
                last_height = h;

                stats.frames_captured.fetch_add(1, Ordering::Relaxed);

                match tx.try_send(CaptureEvent::Frame(frame)) {
                    Ok(()) => {
                        stats.buffer_fill.fetch_add(1, Ordering::Release);

                        // Adaptive pacing: smooth EWMA-based approach.
                        window_total += 1;
                        if config.adaptive_fps && window_total >= ADAPTIVE_WINDOW {
                            let drop_ratio = window_drops as f64 / window_total as f64;
                            if let (Some(cur), Some(base), Some(max)) =
                                (current_interval, base_interval, min_interval)
                            {
                                let cur_ns = cur.as_nanos() as f64;
                                let target_ns = if drop_ratio > DROP_RATIO_THRESHOLD {
                                    // Nudge slower toward min_fps.
                                    (cur_ns * 1.5).min(max.as_nanos() as f64)
                                } else {
                                    // Nudge faster toward target_fps.
                                    (cur_ns * 0.8).max(base.as_nanos() as f64)
                                };
                                let smoothed =
                                    ADAPTIVE_ALPHA * target_ns + (1.0 - ADAPTIVE_ALPHA) * cur_ns;
                                current_interval = Some(Duration::from_nanos(smoothed as u64));
                            }
                            window_drops = 0;
                            window_total = 0;
                        }
                    }
                    Err(mpsc::TrySendError::Full(CaptureEvent::Frame(dropped))) => {
                        stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                        // Frame never entered the channel, so buffer_fill
                        // is unchanged.
                        // Notify receiver about the drop.
                        let _ = tx.try_send(CaptureEvent::FrameDropped { sequence: seq });
                        reuse_frame = Some(dropped);

                        window_drops += 1;
                        window_total += 1;
                        if config.adaptive_fps && window_total >= ADAPTIVE_WINDOW {
                            let drop_ratio = window_drops as f64 / window_total as f64;
                            if let (Some(cur), Some(base), Some(max)) =
                                (current_interval, base_interval, min_interval)
                            {
                                let cur_ns = cur.as_nanos() as f64;
                                let target_ns = if drop_ratio > DROP_RATIO_THRESHOLD {
                                    (cur_ns * 1.5).min(max.as_nanos() as f64)
                                } else {
                                    (cur_ns * 0.8).max(base.as_nanos() as f64)
                                };
                                let smoothed =
                                    ADAPTIVE_ALPHA * target_ns + (1.0 - ADAPTIVE_ALPHA) * cur_ns;
                                current_interval = Some(Duration::from_nanos(smoothed as u64));
                            }
                            window_drops = 0;
                            window_total = 0;
                        }
                    }
                    Err(mpsc::TrySendError::Full(_)) => {}
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        break;
                    }
                }
            }
            Err(ref e) if e.is_retryable() => {
                consecutive_errors += 1;
                stats.errors_recovered.fetch_add(1, Ordering::Relaxed);
                if consecutive_errors >= config.max_consecutive_errors {
                    // Send fatal error event before exiting.
                    let _ = tx.try_send(CaptureEvent::Error(e.to_sendable()));
                    break;
                }
                std::thread::sleep(Duration::from_millis(16));
                continue;
            }
            Err(e) => {
                // Fatal error — notify receiver and exit.
                let _ = tx.try_send(CaptureEvent::Error(e.to_sendable()));
                break;
            }
        }

        // Update FPS counter.
        fps_counter += 1;
        let fps_elapsed = fps_epoch.elapsed();
        if fps_elapsed >= Duration::from_secs(1) {
            let fps = fps_counter as f64 / fps_elapsed.as_secs_f64();
            stats.current_fps.store(fps.to_bits(), Ordering::Relaxed);
            fps_counter = 0;
            fps_epoch = Instant::now();
        }

        // Frame pacing.
        if let Some(interval) = current_interval {
            let elapsed = frame_start.elapsed();
            if elapsed < interval {
                spin_sleep(interval - elapsed);
            }
        }
    }

    // Send StreamEnded sentinel so the consumer knows no more events
    // will arrive and can flush its encoder.
    let _ = tx.try_send(CaptureEvent::StreamEnded);
}

/// High-precision sleep that uses spin-waiting for the final sub-millisecond
/// portion to avoid Windows timer resolution issues.
fn spin_sleep(duration: Duration) {
    const SPIN_THRESHOLD: Duration = Duration::from_micros(1500);

    if duration > SPIN_THRESHOLD {
        std::thread::sleep(duration - SPIN_THRESHOLD);
    }

    let target = Instant::now() + duration;
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

// ---------------------------------------------------------------------------
// Async (tokio) stream wrapper
// ---------------------------------------------------------------------------

/// Async wrapper around `StreamHandle` for tokio-based recording pipelines.
///
/// Requires the `tokio-stream` feature.
#[cfg(feature = "tokio-stream")]
pub struct AsyncStreamHandle {
    inner: StreamHandle,
    /// Tokio channel for async event delivery.
    async_rx: tokio::sync::mpsc::Receiver<CaptureEvent>,
    /// Bridge thread that moves events from the sync channel to the
    /// tokio channel.
    _bridge_handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "tokio-stream")]
impl AsyncStreamHandle {
    /// Start a capture stream with an async event receiver.
    ///
    /// Internally spawns the same capture thread as `StreamHandle`, plus
    /// a lightweight bridge thread that forwards events into a
    /// `tokio::sync::mpsc` channel.
    pub fn start(
        session: CaptureSession,
        target: CaptureTarget,
        config: StreamConfig,
    ) -> CaptureResult<Self> {
        let handle = StreamHandle::start(session, target, config)?;
        let sync_rx = unsafe {
            // Take the receiver out of the handle so we can bridge it.
            // We'll reconstruct the handle without the receiver.
            std::ptr::read(&handle.receiver)
        };
        // Prevent double-free of receiver in the original handle.
        let handle = std::mem::ManuallyDrop::new(handle);
        let stop_flag = handle.stop_flag.clone();
        let pause_flag = handle.pause_flag.clone();
        let stats = handle.stats.clone();
        let join_handle_inner = unsafe { std::ptr::read(&handle.join_handle) };

        let (async_tx, async_rx) = tokio::sync::mpsc::channel::<CaptureEvent>(32);

        let bridge_stop = stop_flag.clone();
        let bridge_stats = stats.clone();
        let bridge_handle = std::thread::Builder::new()
            .name("snow-capture-async-bridge".to_string())
            .spawn(move || {
                loop {
                    match sync_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(event) => {
                            let is_frame = matches!(&event, CaptureEvent::Frame(_));
                            if async_tx.blocking_send(event).is_err() {
                                break;
                            }
                            // Decrement buffer_fill after successfully
                            // pulling a frame out of the sync channel.
                            if is_frame {
                                bridge_stats.buffer_fill.fetch_sub(1, Ordering::Release);
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if bridge_stop.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .ok();

        // Reconstruct a StreamHandle that owns the thread but not the
        // sync receiver (which the bridge now owns).
        let (dummy_tx, dummy_rx) = mpsc::sync_channel(1);
        drop(dummy_tx);
        let inner = StreamHandle {
            receiver: dummy_rx,
            stop_flag,
            pause_flag,
            stats,
            join_handle: join_handle_inner,
            buffer_depth: 0, // Not used directly; stats are shared.
        };

        Ok(Self {
            inner,
            async_rx,
            _bridge_handle: bridge_handle,
        })
    }

    /// Await the next capture event.
    pub async fn next_event(&mut self) -> Option<CaptureEvent> {
        self.async_rx.recv().await
    }

    /// Signal the stream to stop.
    pub fn stop(&self) {
        self.inner.stop();
    }

    /// Pause the capture stream.
    pub fn pause(&self) {
        self.inner.pause();
    }

    /// Resume a paused capture stream.
    pub fn resume(&self) {
        self.inner.resume();
    }

    /// Whether the stream is currently paused.
    pub fn is_paused(&self) -> bool {
        self.inner.is_paused()
    }

    /// Check whether the stream thread is still running.
    pub fn is_running(&self) -> bool {
        self.inner.is_running()
    }

    /// Get a reference to the live stream statistics.
    pub fn stats(&self) -> &Arc<StreamStats> {
        self.inner.stats()
    }
}
