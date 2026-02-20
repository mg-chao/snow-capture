use std::sync::Arc;
use std::time::Instant;

use crate::error::CaptureError;
use crate::error::CaptureResult;
use crate::frame::Frame;
use crate::monitor::MonitorId;
use crate::window::WindowId;

/// Capture intent used to tune backend behavior for latency/throughput.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    /// Favor low-overhead single-shot behavior for snapshots.
    Screenshot,
    /// Favor sustained throughput for continuous recording pipelines.
    ScreenRecording,
}

impl Default for CaptureMode {
    fn default() -> Self {
        Self::Screenshot
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureBackendKind {
    Auto,

    DxgiDuplication,

    WindowsGraphicsCapture,

    Gdi,
}

impl CaptureBackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DxgiDuplication => "dxgi",
            Self::WindowsGraphicsCapture => "wgc",
            Self::Gdi => "gdi",
        }
    }
}

#[derive(Clone, Debug)]
pub struct AutoBackendPolicy {
    pub priority: Vec<CaptureBackendKind>,
}

impl AutoBackendPolicy {
    pub fn normalized_priority(&self) -> Vec<CaptureBackendKind> {
        let mut normalized = Vec::new();
        for kind in &self.priority {
            if *kind == CaptureBackendKind::Auto {
                continue;
            }
            if !normalized.contains(kind) {
                normalized.push(*kind);
            }
        }
        if normalized.is_empty() {
            normalized.extend(DEFAULT_AUTO_BACKEND_PRIORITY);
        }
        normalized
    }
}

impl Default for AutoBackendPolicy {
    fn default() -> Self {
        Self {
            priority: DEFAULT_AUTO_BACKEND_PRIORITY.to_vec(),
        }
    }
}

pub const DEFAULT_AUTO_BACKEND_PRIORITY: [CaptureBackendKind; 3] = [
    CaptureBackendKind::DxgiDuplication,
    CaptureBackendKind::WindowsGraphicsCapture,
    CaptureBackendKind::Gdi,
];

/// Configuration for cursor capture behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorCaptureConfig {
    /// When `true`, the backend will capture cursor shape and position
    /// data and attach it to `Frame::metadata.cursor`.
    pub capture_cursor: bool,
}

/// Source/destination rectangle pair used for partial monitor capture writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureBlitRegion {
    pub src_x: u32,
    pub src_y: u32,
    pub width: u32,
    pub height: u32,
    pub dst_x: u32,
    pub dst_y: u32,
}

/// Timing and duplicate metadata produced by a capture operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureSampleMetadata {
    pub capture_time: Option<Instant>,
    pub present_time_qpc: Option<i64>,
    pub is_duplicate: bool,
}

pub trait MonitorCapturer: Send {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame>;

    /// Optional accelerated path for writing only a source sub-rectangle
    /// into an already-allocated destination frame.
    ///
    /// Returns `Ok(Some(..))` when the backend handled the partial write
    /// directly, or `Ok(None)` to request the caller fall back to full-frame
    /// capture plus CPU blit.
    fn capture_region_into(
        &mut self,
        _blit: CaptureBlitRegion,
        _destination: &mut Frame,
        _destination_has_history: bool,
    ) -> CaptureResult<Option<CaptureSampleMetadata>> {
        Ok(None)
    }

    /// Set capture mode so backends can tune buffering/conversion policy.
    fn set_capture_mode(&mut self, _mode: CaptureMode) {}

    /// Set cursor capture configuration. Backends that don't support
    /// cursor capture may ignore this.
    fn set_cursor_config(&mut self, _config: CursorCaptureConfig) {}
}

pub trait CaptureBackend: Send + Sync {
    fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>>;
    fn primary_monitor(&self) -> CaptureResult<MonitorId>;
    fn create_monitor_capturer(
        &self,
        monitor: &MonitorId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>>;
    fn create_window_capturer(
        &self,
        _window: &WindowId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        Err(CaptureError::BackendUnavailable(
            "window capture is not supported by this backend".into(),
        ))
    }
}

pub fn default_backend() -> CaptureResult<Arc<dyn CaptureBackend>> {
    backend_for_kind_with_auto_policy(CaptureBackendKind::Auto, AutoBackendPolicy::default())
}

pub fn backend_for_kind(kind: CaptureBackendKind) -> CaptureResult<Arc<dyn CaptureBackend>> {
    backend_for_kind_with_auto_policy(kind, AutoBackendPolicy::default())
}

pub fn backend_for_kind_with_auto_policy(
    kind: CaptureBackendKind,
    auto_policy: AutoBackendPolicy,
) -> CaptureResult<Arc<dyn CaptureBackend>> {
    crate::platform::build_backend(kind, auto_policy)
}
