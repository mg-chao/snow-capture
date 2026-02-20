use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::CaptureTarget;
use crate::backend::{self, CaptureBackend, CaptureMode, CursorCaptureConfig, MonitorCapturer};
use crate::error::{CaptureError, CaptureResult};
use crate::frame::Frame;
use crate::monitor::{MonitorId, MonitorKey};
use crate::region::MonitorLayout;
use crate::streaming::{StreamConfig, StreamHandle};
use crate::window::{WindowId, WindowKey};

#[derive(Clone, Copy, Debug)]
pub struct CaptureSessionConfig {
    pub capture_retry_count: usize,
    /// When `true`, cursor shape and position data will be attached to
    /// each captured frame's metadata.
    pub capture_cursor: bool,
    /// Capture intent used to tune backend behavior for screenshot
    /// vs. continuous recording workloads.
    pub mode: CaptureMode,
}

impl Default for CaptureSessionConfig {
    fn default() -> Self {
        Self {
            capture_retry_count: 1,
            capture_cursor: false,
            mode: CaptureMode::Screenshot,
        }
    }
}

pub struct CaptureSessionBuilder {
    backend_override: Option<Arc<dyn CaptureBackend>>,
    backend_kind: backend::CaptureBackendKind,
    auto_backend_policy: backend::AutoBackendPolicy,
    config: CaptureSessionConfig,
}

impl CaptureSessionBuilder {
    pub fn new() -> Self {
        Self {
            backend_override: None,
            backend_kind: backend::CaptureBackendKind::Auto,
            auto_backend_policy: backend::AutoBackendPolicy::default(),
            config: CaptureSessionConfig::default(),
        }
    }

    pub fn with_backend(mut self, backend: Arc<dyn CaptureBackend>) -> Self {
        self.backend_override = Some(backend);
        self
    }

    pub fn with_backend_kind(mut self, kind: backend::CaptureBackendKind) -> Self {
        self.backend_override = None;
        self.backend_kind = kind;
        self
    }

    pub fn with_auto_backend_policy(
        mut self,
        auto_backend_policy: backend::AutoBackendPolicy,
    ) -> Self {
        self.auto_backend_policy = auto_backend_policy;
        self
    }

    pub fn capture_retry_count(mut self, capture_retry_count: usize) -> Self {
        self.config.capture_retry_count = capture_retry_count;
        self
    }

    /// Enable cursor capture. When enabled, each frame's metadata will
    /// include cursor shape and position data (if supported by the backend).
    pub fn capture_cursor(mut self, enabled: bool) -> Self {
        self.config.capture_cursor = enabled;
        self
    }

    /// Configure the capture intent for backend tuning.
    pub fn capture_mode(mut self, mode: CaptureMode) -> Self {
        self.config.mode = mode;
        self
    }

    pub fn build(self) -> CaptureResult<CaptureSession> {
        let backend = match self.backend_override {
            Some(b) => b,
            None => backend::backend_for_kind_with_auto_policy(
                self.backend_kind,
                self.auto_backend_policy,
            )?,
        };
        // Pre-initialize conversion resources (thread pool, LUT, kernel
        // selection) so the first capture doesn't pay the one-time cost.
        crate::convert::warmup();
        Ok(CaptureSession {
            backend,
            capturers: FxHashMap::default(),
            window_capturers: FxHashMap::default(),
            config: self.config,
            sequence: 0,
            layout: None,
        })
    }
}

impl Default for CaptureSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CaptureSession {
    backend: Arc<dyn CaptureBackend>,
    capturers: FxHashMap<MonitorKey, Box<dyn MonitorCapturer>>,
    window_capturers: FxHashMap<WindowKey, Box<dyn MonitorCapturer>>,
    config: CaptureSessionConfig,
    sequence: u64,
    /// Lazily-initialized monitor layout for region capture.
    layout: Option<MonitorLayout>,
}

impl CaptureSession {
    pub fn builder() -> CaptureSessionBuilder {
        CaptureSessionBuilder::new()
    }

    pub fn new() -> CaptureResult<Self> {
        Self::builder().build()
    }

    pub fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>> {
        self.backend.enumerate_monitors()
    }

    pub fn primary_monitor(&self) -> CaptureResult<MonitorId> {
        self.backend.primary_monitor()
    }

    pub fn capture_frame(&mut self, target: &CaptureTarget) -> CaptureResult<Frame> {
        self.do_capture(target, None)
    }

    pub fn capture_frame_reuse(
        &mut self,
        target: &CaptureTarget,
        frame: Frame,
    ) -> CaptureResult<Frame> {
        self.do_capture(target, Some(frame))
    }

    pub fn capture_frame_into(
        &mut self,
        target: &CaptureTarget,
        frame: &mut Frame,
    ) -> CaptureResult<()> {
        let reused = std::mem::replace(frame, Frame::empty());
        *frame = self.capture_frame_reuse(target, reused)?;
        Ok(())
    }

    /// Start a continuous capture stream on a background thread.
    ///
    /// Returns a `StreamHandle` that provides a `Receiver<CaptureEvent>`
    /// for consuming frames and events. The stream runs until the handle
    /// is dropped or `StreamHandle::stop()` is called.
    ///
    /// The stream thread owns its own `CaptureSession` internally, so
    /// this method consumes `self`.
    pub fn start_streaming(
        self,
        target: CaptureTarget,
        config: StreamConfig,
    ) -> CaptureResult<StreamHandle> {
        let mut session = self;
        // Streaming is always a recording workload; force the mode so
        // backends can keep high-throughput optimizations enabled.
        session.set_capture_mode(CaptureMode::ScreenRecording);
        StreamHandle::start(session, target, config)
    }

    /// Returns the active capture mode.
    pub fn capture_mode(&self) -> CaptureMode {
        self.config.mode
    }

    /// Update capture mode for both future and already-initialized
    /// monitor capturers.
    pub fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.config.mode = mode;
        for capturer in self.capturers.values_mut() {
            capturer.set_capture_mode(mode);
        }
        for capturer in self.window_capturers.values_mut() {
            capturer.set_capture_mode(mode);
        }
    }

    fn resolve_target(&self, target: &CaptureTarget) -> CaptureResult<MonitorId> {
        match target {
            CaptureTarget::PrimaryMonitor => self.backend.primary_monitor(),
            CaptureTarget::Monitor(id) => self
                .backend
                .enumerate_monitors()?
                .into_iter()
                .find(|candidate| candidate.key() == id.key())
                .ok_or_else(|| CaptureError::InvalidTarget(id.stable_id())),
            CaptureTarget::Region(_) => {
                // Region capture doesn't resolve to a single monitor.
                Err(CaptureError::InvalidConfig(
                    "resolve_target called for Region target".into(),
                ))
            }
            CaptureTarget::Window(_) => Err(CaptureError::InvalidConfig(
                "resolve_target called for Window target".into(),
            )),
        }
    }

    fn get_or_create_capturer(
        &mut self,
        monitor: &MonitorId,
    ) -> CaptureResult<&mut Box<dyn MonitorCapturer>> {
        let key = monitor.key();
        if !self.capturers.contains_key(&key) {
            let mut capturer = self.backend.create_monitor_capturer(monitor)?;
            capturer.set_capture_mode(self.config.mode);
            if self.config.capture_cursor {
                capturer.set_cursor_config(CursorCaptureConfig {
                    capture_cursor: true,
                });
            }
            self.capturers.insert(key, capturer);
        }
        Ok(self.capturers.get_mut(&key).unwrap())
    }

    fn ensure_window_capturer(&mut self, window: &WindowId) -> CaptureResult<()> {
        let key = window.key();

        if self.window_capturers.contains_key(&key) {
            return Ok(());
        }

        let mut capturer = self.backend.create_window_capturer(window)?;
        capturer.set_capture_mode(self.config.mode);
        if self.config.capture_cursor {
            capturer.set_cursor_config(CursorCaptureConfig {
                capture_cursor: true,
            });
        }
        self.window_capturers.insert(key, capturer);
        Ok(())
    }

    fn get_window_capturer(&mut self, window: &WindowId) -> &mut Box<dyn MonitorCapturer> {
        self.window_capturers.get_mut(&window.key()).unwrap()
    }

    fn do_capture(&mut self, target: &CaptureTarget, reuse: Option<Frame>) -> CaptureResult<Frame> {
        match target {
            CaptureTarget::Region(region) => return self.do_capture_region(region, reuse),
            CaptureTarget::Window(window) => return self.do_capture_window(window, reuse),
            _ => {}
        }

        let monitor = self.resolve_target(target)?;
        let key = monitor.key();
        let max_retries = self.config.capture_retry_count;

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let cap_start = std::time::Instant::now();
        let first_result = self.get_or_create_capturer(&monitor)?.capture(reuse);
        let cap_dur = cap_start.elapsed();
        match first_result {
            Ok(mut frame) => {
                frame.metadata.sequence = seq;
                if frame.metadata.capture_duration.is_none() {
                    frame.metadata.capture_duration = Some(cap_dur);
                }
                return Ok(frame);
            }
            Err(error) if error.requires_worker_reset() && max_retries > 0 => {
                self.capturers.remove(&key);
            }
            Err(error) => return Err(error),
        }

        // Retry loop after capturer reset.
        for attempt in 0..max_retries {
            let retry_start = std::time::Instant::now();
            let result = self.get_or_create_capturer(&monitor)?.capture(None);
            let retry_dur = retry_start.elapsed();
            match result {
                Ok(mut frame) => {
                    frame.metadata.sequence = seq;
                    if frame.metadata.capture_duration.is_none() {
                        frame.metadata.capture_duration = Some(retry_dur);
                    }
                    return Ok(frame);
                }
                Err(error) if error.requires_worker_reset() && attempt + 1 < max_retries => {
                    self.capturers.remove(&key);
                }
                Err(error) => return Err(error),
            }
        }

        Err(CaptureError::WorkerDead)
    }

    fn do_capture_window(
        &mut self,
        window: &WindowId,
        reuse: Option<Frame>,
    ) -> CaptureResult<Frame> {
        self.ensure_window_capturer(window)?;

        let key = window.key();
        let max_retries = self.config.capture_retry_count;

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let cap_start = std::time::Instant::now();
        let first_result = self.get_window_capturer(window).capture(reuse);
        let cap_dur = cap_start.elapsed();
        match first_result {
            Ok(mut frame) => {
                frame.metadata.sequence = seq;
                if frame.metadata.capture_duration.is_none() {
                    frame.metadata.capture_duration = Some(cap_dur);
                }
                return Ok(frame);
            }
            Err(error) if error.requires_worker_reset() && max_retries > 0 => {
                self.window_capturers.remove(&key);
            }
            Err(error) => return Err(error),
        }

        for attempt in 0..max_retries {
            if !self.window_capturers.contains_key(&key) {
                self.ensure_window_capturer(window)?;
            }
            let retry_start = std::time::Instant::now();
            let result = self.get_window_capturer(window).capture(None);
            let retry_dur = retry_start.elapsed();
            match result {
                Ok(mut frame) => {
                    frame.metadata.sequence = seq;
                    if frame.metadata.capture_duration.is_none() {
                        frame.metadata.capture_duration = Some(retry_dur);
                    }
                    return Ok(frame);
                }
                Err(error) if error.requires_worker_reset() && attempt + 1 < max_retries => {
                    self.window_capturers.remove(&key);
                }
                Err(error) => return Err(error),
            }
        }

        Err(CaptureError::WorkerDead)
    }

    /// Capture a region that may span multiple monitors by compositing
    /// individual monitor captures into a single output frame.
    fn do_capture_region(
        &mut self,
        region: &crate::region::CaptureRegion,
        reuse: Option<Frame>,
    ) -> CaptureResult<Frame> {
        // Lazily snapshot the monitor layout on first region capture.
        if self.layout.is_none() {
            let monitors = self.backend.enumerate_monitors()?;
            self.layout = Some(MonitorLayout::snapshot_from_monitors(monitors)?);
        }
        let layout = self.layout.as_ref().unwrap();

        let overlaps = layout.overlapping_monitors(region);
        if overlaps.is_empty() {
            return Err(CaptureError::InvalidTarget(
                "region does not overlap any monitor".into(),
            ));
        }

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let out_w = region.width;
        let out_h = region.height;

        // Prepare output frame.
        let mut out_frame = reuse.unwrap_or_else(Frame::empty);
        out_frame.ensure_rgba_capacity(out_w, out_h)?;
        // Zero-fill so gaps between monitors are black.
        let out_buf = out_frame.as_mut_rgba_bytes();
        out_buf.iter_mut().for_each(|b| *b = 0);

        let mut latest_capture_time = None;
        let mut latest_present_qpc = None;

        for (mon_geo, intersection) in &overlaps {
            // Capture the full monitor.
            let mon_frame = {
                let capturer = self.get_or_create_capturer(&mon_geo.monitor)?;
                capturer.capture(None)?
            };

            // Track the latest timestamps.
            if let Some(t) = mon_frame.metadata.capture_time {
                latest_capture_time = Some(t);
            }
            if let Some(q) = mon_frame.metadata.present_time_qpc {
                latest_present_qpc = Some(q);
            }

            let src_bytes = mon_frame.as_rgba_bytes();
            let src_w = mon_frame.width() as usize;

            // Source rect within the monitor frame.
            let src_x = (intersection.x - mon_geo.x) as usize;
            let src_y = (intersection.y - mon_geo.y) as usize;
            // Destination rect within the output frame.
            let dst_x = (intersection.x - region.x) as usize;
            let dst_y = (intersection.y - region.y) as usize;
            let copy_w = intersection.width as usize;
            let copy_h = intersection.height as usize;

            let out_buf = out_frame.as_mut_rgba_bytes();
            for row in 0..copy_h {
                let src_off = ((src_y + row) * src_w + src_x) * 4;
                let dst_off = ((dst_y + row) * out_w as usize + dst_x) * 4;
                let len = copy_w * 4;
                if src_off + len <= src_bytes.len() && dst_off + len <= out_buf.len() {
                    out_buf[dst_off..dst_off + len]
                        .copy_from_slice(&src_bytes[src_off..src_off + len]);
                }
            }
        }

        out_frame.metadata.sequence = seq;
        out_frame.metadata.capture_time = latest_capture_time;
        out_frame.metadata.present_time_qpc = latest_present_qpc;
        Ok(out_frame)
    }
}
