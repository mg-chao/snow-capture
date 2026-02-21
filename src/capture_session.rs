use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::CaptureTarget;
use crate::backend::{
    self, CaptureBackend, CaptureBlitRegion, CaptureMode, CaptureSampleMetadata,
    CursorCaptureConfig, MonitorCapturer,
};
use crate::error::{CaptureError, CaptureResult};
use crate::frame::Frame;
use crate::monitor::{MonitorId, MonitorKey};
use crate::region::{CaptureRegion, MonitorLayout};
use crate::streaming::{StreamConfig, StreamHandle};
use crate::window::{WindowId, WindowKey};

#[derive(Clone, Debug)]
struct RegionPlanEntry {
    monitor: MonitorId,
    monitor_key: MonitorKey,
    blit: CaptureBlitRegion,
}

#[derive(Clone, Debug)]
struct PreparedRegionPlan {
    region: CaptureRegion,
    output_width: u32,
    output_height: u32,
    entries: Vec<RegionPlanEntry>,
}

fn copy_region_rgba(src: &Frame, blit: CaptureBlitRegion, dst: &mut Frame) -> CaptureResult<()> {
    if blit.width == 0 || blit.height == 0 {
        return Ok(());
    }

    let src_w = src.width() as usize;
    let src_h = src.height() as usize;
    let dst_w = dst.width() as usize;
    let dst_h = dst.height() as usize;

    let src_x = blit.src_x as usize;
    let src_y = blit.src_y as usize;
    let dst_x = blit.dst_x as usize;
    let dst_y = blit.dst_y as usize;
    let copy_w = blit.width as usize;
    let copy_h = blit.height as usize;

    let src_right = src_x
        .checked_add(copy_w)
        .ok_or(CaptureError::BufferOverflow)?;
    let src_bottom = src_y
        .checked_add(copy_h)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_right = dst_x
        .checked_add(copy_w)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_bottom = dst_y
        .checked_add(copy_h)
        .ok_or(CaptureError::BufferOverflow)?;

    if src_right > src_w || src_bottom > src_h || dst_right > dst_w || dst_bottom > dst_h {
        return Err(CaptureError::BufferOverflow);
    }

    let src_stride = src_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
    let dst_stride = dst_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
    let row_bytes = copy_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;

    let src_bytes = src.as_rgba_bytes();
    let dst_bytes = dst.as_mut_rgba_bytes();
    let src_required_len = src_stride
        .checked_mul(src_h)
        .ok_or(CaptureError::BufferOverflow)?;
    if src_bytes.len() < src_required_len {
        return Err(CaptureError::BufferOverflow);
    }
    let dst_required_len = dst_stride
        .checked_mul(dst_h)
        .ok_or(CaptureError::BufferOverflow)?;
    if dst_bytes.len() < dst_required_len {
        return Err(CaptureError::BufferOverflow);
    }
    let src_start = src_y
        .checked_mul(src_stride)
        .and_then(|off| src_x.checked_mul(4).and_then(|xoff| off.checked_add(xoff)))
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_start = dst_y
        .checked_mul(dst_stride)
        .and_then(|off| dst_x.checked_mul(4).and_then(|xoff| off.checked_add(xoff)))
        .ok_or(CaptureError::BufferOverflow)?;

    let mut src_row_start = src_start;
    let mut dst_row_start = dst_start;
    for _ in 0..copy_h {
        let src_row_end = src_row_start
            .checked_add(row_bytes)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_row_end = dst_row_start
            .checked_add(row_bytes)
            .ok_or(CaptureError::BufferOverflow)?;

        dst_bytes[dst_row_start..dst_row_end]
            .copy_from_slice(&src_bytes[src_row_start..src_row_end]);

        src_row_start = src_row_start
            .checked_add(src_stride)
            .ok_or(CaptureError::BufferOverflow)?;
        dst_row_start = dst_row_start
            .checked_add(dst_stride)
            .ok_or(CaptureError::BufferOverflow)?;
    }
    Ok(())
}

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
            monitor_output_history: FxHashMap::default(),
            window_output_history: FxHashMap::default(),
            config: self.config,
            sequence: 0,
            layout: None,
            prepared_region_plan: None,
            region_fallback_frames: FxHashMap::default(),
            region_output_history_valid: false,
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
    /// Last output sequence per monitor. Reuse is considered to have
    /// valid history only when its metadata sequence matches this value.
    monitor_output_history: FxHashMap<MonitorKey, u64>,
    /// Per-window map mirroring `monitor_output_history`.
    window_output_history: FxHashMap<WindowKey, u64>,
    config: CaptureSessionConfig,
    sequence: u64,
    /// Lazily-initialized monitor layout for region capture.
    layout: Option<MonitorLayout>,
    prepared_region_plan: Option<PreparedRegionPlan>,
    region_fallback_frames: FxHashMap<MonitorKey, Frame>,
    region_output_history_valid: bool,
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
        if self.config.mode != mode {
            // Reuse history is mode-sensitive (for example, some
            // backend caches only advance in recording mode), so reset
            // all reuse provenance on mode transitions.
            self.monitor_output_history.clear();
            self.window_output_history.clear();
            self.region_output_history_valid = false;
            self.region_fallback_frames.clear();
        }
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
            CaptureTarget::Window(window) => {
                self.region_output_history_valid = false;
                return self.do_capture_window(window, reuse);
            }
            _ => {}
        }
        self.region_output_history_valid = false;

        let monitor = self.resolve_target(target)?;
        let key = monitor.key();
        let max_retries = self.config.capture_retry_count;
        let reuse_sequence = reuse.as_ref().map(|frame| frame.metadata.sequence);
        let destination_has_history = matches!(
            (reuse_sequence, self.monitor_output_history.get(&key)),
            (Some(reuse_seq), Some(last_seq)) if reuse_seq == *last_seq
        );

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let cap_start = std::time::Instant::now();
        let first_result = self
            .get_or_create_capturer(&monitor)?
            .capture_with_history_hint(reuse, destination_has_history);
        let cap_dur = cap_start.elapsed();
        match first_result {
            Ok(mut frame) => {
                frame.metadata.sequence = seq;
                self.monitor_output_history.insert(key, seq);
                if frame.metadata.capture_duration.is_none() {
                    frame.metadata.capture_duration = Some(cap_dur);
                }
                return Ok(frame);
            }
            Err(error) if error.requires_worker_reset() && max_retries > 0 => {
                self.capturers.remove(&key);
                self.monitor_output_history.remove(&key);
            }
            Err(error) => return Err(error),
        }

        // Retry loop after capturer reset.
        for attempt in 0..max_retries {
            let retry_start = std::time::Instant::now();
            let result = self
                .get_or_create_capturer(&monitor)?
                .capture_with_history_hint(None, false);
            let retry_dur = retry_start.elapsed();
            match result {
                Ok(mut frame) => {
                    frame.metadata.sequence = seq;
                    self.monitor_output_history.insert(key, seq);
                    if frame.metadata.capture_duration.is_none() {
                        frame.metadata.capture_duration = Some(retry_dur);
                    }
                    return Ok(frame);
                }
                Err(error) if error.requires_worker_reset() && attempt + 1 < max_retries => {
                    self.capturers.remove(&key);
                    self.monitor_output_history.remove(&key);
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
        let reuse_sequence = reuse.as_ref().map(|frame| frame.metadata.sequence);
        let destination_has_history = matches!(
            (reuse_sequence, self.window_output_history.get(&key)),
            (Some(reuse_seq), Some(last_seq)) if reuse_seq == *last_seq
        );

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let cap_start = std::time::Instant::now();
        let first_result = self
            .get_window_capturer(window)
            .capture_with_history_hint(reuse, destination_has_history);
        let cap_dur = cap_start.elapsed();
        match first_result {
            Ok(mut frame) => {
                frame.metadata.sequence = seq;
                self.window_output_history.insert(key, seq);
                if frame.metadata.capture_duration.is_none() {
                    frame.metadata.capture_duration = Some(cap_dur);
                }
                return Ok(frame);
            }
            Err(error) if error.requires_worker_reset() && max_retries > 0 => {
                self.window_capturers.remove(&key);
                self.window_output_history.remove(&key);
            }
            Err(error) => return Err(error),
        }

        for attempt in 0..max_retries {
            if !self.window_capturers.contains_key(&key) {
                self.ensure_window_capturer(window)?;
            }
            let retry_start = std::time::Instant::now();
            let result = self
                .get_window_capturer(window)
                .capture_with_history_hint(None, false);
            let retry_dur = retry_start.elapsed();
            match result {
                Ok(mut frame) => {
                    frame.metadata.sequence = seq;
                    self.window_output_history.insert(key, seq);
                    if frame.metadata.capture_duration.is_none() {
                        frame.metadata.capture_duration = Some(retry_dur);
                    }
                    return Ok(frame);
                }
                Err(error) if error.requires_worker_reset() && attempt + 1 < max_retries => {
                    self.window_capturers.remove(&key);
                    self.window_output_history.remove(&key);
                }
                Err(error) => return Err(error),
            }
        }

        Err(CaptureError::WorkerDead)
    }

    /// Capture a region that may span multiple monitors by compositing
    /// individual monitor captures into a single output frame.
    fn prepare_region_plan(
        &mut self,
        region: &CaptureRegion,
    ) -> CaptureResult<(PreparedRegionPlan, bool)> {
        // Lazily snapshot the monitor layout on first region capture.
        if self.layout.is_none() {
            let monitors = self.backend.enumerate_monitors()?;
            self.layout = Some(MonitorLayout::snapshot_from_monitors(monitors)?);
        }

        let needs_rebuild = self
            .prepared_region_plan
            .as_ref()
            .is_none_or(|cached| cached.region != *region);
        if needs_rebuild {
            let layout = self.layout.as_ref().unwrap();
            let overlaps = layout.overlapping_monitors(region);
            if overlaps.is_empty() {
                return Err(CaptureError::InvalidTarget(
                    "region does not overlap any monitor".into(),
                ));
            }

            let mut entries = Vec::with_capacity(overlaps.len());
            for (mon_geo, intersection) in overlaps {
                entries.push(RegionPlanEntry {
                    monitor_key: mon_geo.monitor.key(),
                    monitor: mon_geo.monitor,
                    blit: CaptureBlitRegion {
                        src_x: (intersection.x - mon_geo.x) as u32,
                        src_y: (intersection.y - mon_geo.y) as u32,
                        width: intersection.width,
                        height: intersection.height,
                        dst_x: (intersection.x - region.x) as u32,
                        dst_y: (intersection.y - region.y) as u32,
                    },
                });
            }

            self.prepared_region_plan = Some(PreparedRegionPlan {
                region: *region,
                output_width: region.width,
                output_height: region.height,
                entries,
            });
        }

        Ok((
            self.prepared_region_plan.as_ref().unwrap().clone(),
            needs_rebuild,
        ))
    }

    fn do_capture_region(
        &mut self,
        region: &CaptureRegion,
        reuse: Option<Frame>,
    ) -> CaptureResult<Frame> {
        let (plan, plan_changed) = self.prepare_region_plan(region)?;

        self.sequence = self.sequence.wrapping_add(1);
        let seq = self.sequence;

        let out_w = plan.output_width;
        let out_h = plan.output_height;

        // Prepare output frame.
        let mut out_frame = reuse.unwrap_or_else(Frame::empty);
        let had_region_history = self.region_output_history_valid;
        self.region_output_history_valid = false;
        let destination_has_history = had_region_history
            && !plan_changed
            && out_frame.width() == out_w
            && out_frame.height() == out_h
            && !out_frame.as_rgba_bytes().is_empty();
        out_frame.ensure_rgba_capacity(out_w, out_h)?;
        out_frame.reset_metadata();
        // Initialize only when first created or when the region target changed.
        if !destination_has_history {
            out_frame.as_mut_rgba_bytes().fill(0);
        }

        let full_region_area = u64::from(out_w) * u64::from(out_h);
        let covered_region_area = plan.entries.iter().fold(0u64, |area, entry| {
            area.saturating_add(u64::from(entry.blit.width) * u64::from(entry.blit.height))
        });
        let region_fully_covered = covered_region_area == full_region_area;

        if region_fully_covered && let Some(first_entry) = plan.entries.first() {
            let direct_sample = {
                let capturer = self.get_or_create_capturer(&first_entry.monitor)?;
                capturer.capture_desktop_region_into(
                    region.x,
                    region.y,
                    out_w,
                    out_h,
                    &mut out_frame,
                    destination_has_history,
                )
            };
            if let Ok(Some(sample)) = direct_sample {
                out_frame.metadata.sequence = seq;
                out_frame.metadata.capture_time = sample.capture_time;
                out_frame.metadata.present_time_qpc = sample.present_time_qpc;
                out_frame.metadata.is_duplicate = destination_has_history && sample.is_duplicate;
                self.region_output_history_valid = true;
                return Ok(out_frame);
            }
        }

        let mut latest_capture_time = None;
        let mut latest_present_qpc = None;
        let mut all_duplicate = destination_has_history;

        for entry in &plan.entries {
            let sample = {
                let capturer = self.get_or_create_capturer(&entry.monitor)?;
                capturer.capture_region_into(entry.blit, &mut out_frame, destination_has_history)?
            };

            let sample = if let Some(sample) = sample {
                sample
            } else {
                let reuse_frame = self.region_fallback_frames.remove(&entry.monitor_key);
                let reuse_has_history = reuse_frame.is_some();
                let monitor_frame = {
                    let capturer = self.get_or_create_capturer(&entry.monitor)?;
                    capturer.capture_with_history_hint(reuse_frame, reuse_has_history)?
                };

                copy_region_rgba(&monitor_frame, entry.blit, &mut out_frame)?;
                let sample = CaptureSampleMetadata {
                    capture_time: monitor_frame.metadata.capture_time,
                    present_time_qpc: monitor_frame.metadata.present_time_qpc,
                    is_duplicate: monitor_frame.metadata.is_duplicate,
                };
                self.region_fallback_frames
                    .insert(entry.monitor_key, monitor_frame);
                sample
            };

            if let Some(t) = sample.capture_time {
                latest_capture_time = Some(
                    latest_capture_time.map_or(t, |current: std::time::Instant| current.max(t)),
                );
            }
            if let Some(q) = sample.present_time_qpc {
                latest_present_qpc =
                    Some(latest_present_qpc.map_or(q, |current: i64| current.max(q)));
            }
            all_duplicate &= sample.is_duplicate;
        }

        out_frame.metadata.sequence = seq;
        out_frame.metadata.capture_time = latest_capture_time;
        out_frame.metadata.present_time_qpc = latest_present_qpc;
        out_frame.metadata.is_duplicate = all_duplicate;
        self.region_output_history_valid = true;
        Ok(out_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    struct MockBackend {
        monitor: MonitorId,
        history_hints: Arc<Mutex<Vec<bool>>>,
    }

    impl MockBackend {
        fn new(history_hints: Arc<Mutex<Vec<bool>>>) -> Self {
            Self {
                monitor: MonitorId::from_parts(1, 2, 0, "mock-monitor", true),
                history_hints,
            }
        }
    }

    struct MockMonitorCapturer {
        history_hints: Arc<Mutex<Vec<bool>>>,
    }

    impl MonitorCapturer for MockMonitorCapturer {
        fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
            self.capture_with_history_hint(reuse, false)
        }

        fn capture_with_history_hint(
            &mut self,
            reuse: Option<Frame>,
            destination_has_history: bool,
        ) -> CaptureResult<Frame> {
            self.history_hints
                .lock()
                .unwrap()
                .push(destination_has_history);
            let mut frame = reuse.unwrap_or_else(Frame::empty);
            frame.ensure_rgba_capacity(4, 4)?;
            frame.reset_metadata();
            Ok(frame)
        }
    }

    impl CaptureBackend for MockBackend {
        fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>> {
            Ok(vec![self.monitor.clone()])
        }

        fn primary_monitor(&self) -> CaptureResult<MonitorId> {
            Ok(self.monitor.clone())
        }

        fn create_monitor_capturer(
            &self,
            _monitor: &MonitorId,
        ) -> CaptureResult<Box<dyn MonitorCapturer>> {
            Ok(Box::new(MockMonitorCapturer {
                history_hints: Arc::clone(&self.history_hints),
            }))
        }
    }

    struct DirectRegionBackend {
        monitor: MonitorId,
        desktop_calls: Arc<Mutex<usize>>,
        region_calls: Arc<Mutex<usize>>,
        supports_desktop_direct: bool,
        desktop_should_error: bool,
    }

    struct DirectRegionCapturer {
        desktop_calls: Arc<Mutex<usize>>,
        region_calls: Arc<Mutex<usize>>,
        supports_desktop_direct: bool,
        desktop_should_error: bool,
    }

    impl MonitorCapturer for DirectRegionCapturer {
        fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
            Ok(reuse.unwrap_or_else(Frame::empty))
        }

        fn capture_region_into(
            &mut self,
            _blit: CaptureBlitRegion,
            _destination: &mut Frame,
            _destination_has_history: bool,
        ) -> CaptureResult<Option<CaptureSampleMetadata>> {
            *self.region_calls.lock().unwrap() += 1;
            Ok(Some(CaptureSampleMetadata {
                capture_time: Some(Instant::now()),
                present_time_qpc: Some(0),
                is_duplicate: false,
            }))
        }

        fn capture_desktop_region_into(
            &mut self,
            _x: i32,
            _y: i32,
            _width: u32,
            _height: u32,
            _destination: &mut Frame,
            _destination_has_history: bool,
        ) -> CaptureResult<Option<CaptureSampleMetadata>> {
            *self.desktop_calls.lock().unwrap() += 1;
            if self.desktop_should_error {
                return Err(CaptureError::Platform(anyhow::anyhow!(
                    "mock desktop direct failure"
                )));
            }
            if !self.supports_desktop_direct {
                return Ok(None);
            }
            Ok(Some(CaptureSampleMetadata {
                capture_time: Some(Instant::now()),
                present_time_qpc: Some(0),
                is_duplicate: true,
            }))
        }
    }

    impl CaptureBackend for DirectRegionBackend {
        fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>> {
            Ok(vec![self.monitor.clone()])
        }

        fn primary_monitor(&self) -> CaptureResult<MonitorId> {
            Ok(self.monitor.clone())
        }

        fn create_monitor_capturer(
            &self,
            _monitor: &MonitorId,
        ) -> CaptureResult<Box<dyn MonitorCapturer>> {
            Ok(Box::new(DirectRegionCapturer {
                desktop_calls: Arc::clone(&self.desktop_calls),
                region_calls: Arc::clone(&self.region_calls),
                supports_desktop_direct: self.supports_desktop_direct,
                desktop_should_error: self.desktop_should_error,
            }))
        }
    }

    fn mock_layout(monitor: &MonitorId, x: i32, y: i32, width: u32, height: u32) -> MonitorLayout {
        MonitorLayout {
            monitors: vec![crate::region::MonitorGeometry {
                monitor: monitor.clone(),
                x,
                y,
                width,
                height,
            }],
            virtual_left: x,
            virtual_top: y,
            virtual_width: width,
            virtual_height: height,
        }
    }

    #[test]
    fn monitor_reuse_history_hint_requires_matching_sequence() -> CaptureResult<()> {
        let history_hints = Arc::new(Mutex::new(Vec::new()));
        let backend: Arc<dyn CaptureBackend> =
            Arc::new(MockBackend::new(Arc::clone(&history_hints)));
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        let target = CaptureTarget::PrimaryMonitor;

        let mut first = session.capture_frame(&target)?;
        first.metadata.sequence = first.metadata.sequence.wrapping_add(1);
        let second = session.capture_frame_reuse(&target, first)?;
        let _third = session.capture_frame_reuse(&target, second)?;

        let recorded = history_hints.lock().unwrap().clone();
        assert_eq!(recorded, vec![false, false, true]);
        Ok(())
    }

    #[test]
    fn changing_capture_mode_clears_reuse_history() -> CaptureResult<()> {
        let history_hints = Arc::new(Mutex::new(Vec::new()));
        let backend: Arc<dyn CaptureBackend> =
            Arc::new(MockBackend::new(Arc::clone(&history_hints)));
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        let target = CaptureTarget::PrimaryMonitor;

        let first = session.capture_frame(&target)?;
        session.set_capture_mode(CaptureMode::ScreenRecording);
        let _second = session.capture_frame_reuse(&target, first)?;

        let recorded = history_hints.lock().unwrap().clone();
        assert_eq!(recorded, vec![false, false]);
        Ok(())
    }

    #[test]
    fn region_capture_uses_desktop_direct_path_when_supported() -> CaptureResult<()> {
        let desktop_calls = Arc::new(Mutex::new(0usize));
        let region_calls = Arc::new(Mutex::new(0usize));
        let monitor = MonitorId::from_parts(1, 2, 0, "mock-monitor", true);
        let backend: Arc<dyn CaptureBackend> = Arc::new(DirectRegionBackend {
            monitor: monitor.clone(),
            desktop_calls: Arc::clone(&desktop_calls),
            region_calls: Arc::clone(&region_calls),
            supports_desktop_direct: true,
            desktop_should_error: false,
        });
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        session.layout = Some(mock_layout(&monitor, 0, 0, 1920, 1080));

        let target = CaptureTarget::Region(CaptureRegion::new(100, 50, 640, 360)?);
        let frame = session.capture_frame(&target)?;
        assert!(!frame.metadata.is_duplicate);
        let next_frame = session.capture_frame_reuse(&target, frame)?;

        assert!(next_frame.metadata.is_duplicate);
        assert_eq!(*desktop_calls.lock().unwrap(), 2);
        assert_eq!(*region_calls.lock().unwrap(), 0);
        Ok(())
    }

    #[test]
    fn region_capture_falls_back_when_desktop_direct_path_is_unavailable() -> CaptureResult<()> {
        let desktop_calls = Arc::new(Mutex::new(0usize));
        let region_calls = Arc::new(Mutex::new(0usize));
        let monitor = MonitorId::from_parts(1, 2, 0, "mock-monitor", true);
        let backend: Arc<dyn CaptureBackend> = Arc::new(DirectRegionBackend {
            monitor: monitor.clone(),
            desktop_calls: Arc::clone(&desktop_calls),
            region_calls: Arc::clone(&region_calls),
            supports_desktop_direct: false,
            desktop_should_error: false,
        });
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        session.layout = Some(mock_layout(&monitor, 0, 0, 1920, 1080));

        let target = CaptureTarget::Region(CaptureRegion::new(100, 50, 640, 360)?);
        let _frame = session.capture_frame(&target)?;

        assert_eq!(*desktop_calls.lock().unwrap(), 1);
        assert_eq!(*region_calls.lock().unwrap(), 1);
        Ok(())
    }

    #[test]
    fn region_capture_falls_back_when_desktop_direct_path_errors() -> CaptureResult<()> {
        let desktop_calls = Arc::new(Mutex::new(0usize));
        let region_calls = Arc::new(Mutex::new(0usize));
        let monitor = MonitorId::from_parts(1, 2, 0, "mock-monitor", true);
        let backend: Arc<dyn CaptureBackend> = Arc::new(DirectRegionBackend {
            monitor: monitor.clone(),
            desktop_calls: Arc::clone(&desktop_calls),
            region_calls: Arc::clone(&region_calls),
            supports_desktop_direct: true,
            desktop_should_error: true,
        });
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        session.layout = Some(mock_layout(&monitor, 0, 0, 1920, 1080));

        let target = CaptureTarget::Region(CaptureRegion::new(100, 50, 640, 360)?);
        let _frame = session.capture_frame(&target)?;

        assert_eq!(*desktop_calls.lock().unwrap(), 1);
        assert_eq!(*region_calls.lock().unwrap(), 1);
        Ok(())
    }

    #[test]
    fn region_capture_skips_desktop_direct_path_for_partial_coverage() -> CaptureResult<()> {
        let desktop_calls = Arc::new(Mutex::new(0usize));
        let region_calls = Arc::new(Mutex::new(0usize));
        let monitor = MonitorId::from_parts(1, 2, 0, "mock-monitor", true);
        let backend: Arc<dyn CaptureBackend> = Arc::new(DirectRegionBackend {
            monitor: monitor.clone(),
            desktop_calls: Arc::clone(&desktop_calls),
            region_calls: Arc::clone(&region_calls),
            supports_desktop_direct: true,
            desktop_should_error: false,
        });
        let mut session = CaptureSession::builder().with_backend(backend).build()?;
        session.layout = Some(mock_layout(&monitor, 0, 0, 128, 128));

        // Region extends beyond monitor bounds, so only a partial overlap is
        // covered by monitor entries and the desktop-direct fast path should
        // not run.
        let target = CaptureTarget::Region(CaptureRegion::new(-16, -16, 128, 128)?);
        let _frame = session.capture_frame(&target)?;

        assert_eq!(*desktop_calls.lock().unwrap(), 0);
        assert_eq!(*region_calls.lock().unwrap(), 1);
        Ok(())
    }
}
