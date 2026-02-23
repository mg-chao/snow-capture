pub(crate) mod com;
pub(crate) mod d3d11;
pub(crate) mod dirty_rect;
pub(crate) mod display_change;
pub(crate) mod duplication;
pub(crate) mod gdi;
pub(crate) mod gpu_tonemap;
pub(crate) mod monitor;
pub(crate) mod region_pipeline;
pub(crate) mod surface;
pub(crate) mod wgc;

use std::sync::Arc;

use crate::backend::{AutoBackendPolicy, CaptureBackend, CaptureBackendKind, MonitorCapturer};
use crate::error::{CaptureError, CaptureErrorClass, CaptureResult};
use crate::monitor::MonitorId;
use crate::window::WindowId;

pub(crate) struct WindowsBackend {
    resolver: Arc<monitor::MonitorResolver>,
    kind: CaptureBackendKind,
    auto_policy: AutoBackendPolicy,
}

impl WindowsBackend {
    pub(crate) fn with_kind_and_policy(
        kind: CaptureBackendKind,
        auto_policy: AutoBackendPolicy,
    ) -> CaptureResult<Self> {
        let display_cache = display_change::DisplayInfoCache::new()?;
        let resolver = Arc::new(monitor::MonitorResolver::with_display_cache(display_cache));
        Ok(Self {
            resolver,
            kind,
            auto_policy,
        })
    }

    fn create_by_kind(
        &self,
        kind: CaptureBackendKind,
        monitor: &MonitorId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        match kind {
            CaptureBackendKind::Auto => Err(CaptureError::InvalidConfig(
                "auto backend selection is handled separately".to_string(),
            )),
            CaptureBackendKind::DxgiDuplication => Ok(Box::new(
                duplication::WindowsMonitorCapturer::new(monitor, self.resolver.clone())?,
            )),
            CaptureBackendKind::WindowsGraphicsCapture => Ok(Box::new(
                wgc::WindowsMonitorCapturer::new(monitor, self.resolver.clone())?,
            )),
            CaptureBackendKind::Gdi => Ok(Box::new(gdi::WindowsMonitorCapturer::new(
                monitor,
                self.resolver.clone(),
            )?)),
        }
    }

    fn create_window_by_kind(
        &self,
        kind: CaptureBackendKind,
        window: &WindowId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        match kind {
            CaptureBackendKind::Auto => Err(CaptureError::InvalidConfig(
                "auto backend selection is handled separately".to_string(),
            )),
            CaptureBackendKind::DxgiDuplication => Ok(Box::new(
                duplication::WindowsDxgiWindowCapturer::new(window, self.resolver.clone())?,
            )),
            CaptureBackendKind::WindowsGraphicsCapture => {
                Ok(Box::new(wgc::WindowsWindowCapturer::new(window)?))
            }
            CaptureBackendKind::Gdi => Ok(Box::new(gdi::WindowsWindowCapturer::new(window)?)),
        }
    }

    fn create_auto_capturer(&self, monitor: &MonitorId) -> CaptureResult<Box<dyn MonitorCapturer>> {
        let mut errors: Vec<(CaptureBackendKind, CaptureError)> = Vec::new();

        for kind in self.auto_policy.normalized_priority() {
            match self.create_by_kind(kind, monitor) {
                Ok(capturer) => return Ok(capturer),
                Err(err) => {
                    if matches!(err, CaptureError::MonitorLost) {
                        return Err(err);
                    }
                    errors.push((kind, err));
                }
            }
        }

        Err(CaptureError::BackendUnavailable(format!(
            "failed to initialize auto backend for {}: {}",
            monitor.name(),
            format_backend_errors(&errors)
        )))
    }

    fn create_auto_window_capturer(
        &self,
        window: &WindowId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        let mut errors: Vec<(CaptureBackendKind, CaptureError)> = Vec::new();

        for kind in self.auto_policy.normalized_priority() {
            match self.create_window_by_kind(kind, window) {
                Ok(capturer) => return Ok(capturer),
                Err(err) if err.class() == CaptureErrorClass::InvalidInput => return Err(err),
                Err(err) => errors.push((kind, err)),
            }
        }

        Err(CaptureError::BackendUnavailable(format!(
            "failed to initialize auto window backend for {}: {}",
            window.stable_id(),
            format_backend_errors(&errors)
        )))
    }
}

fn format_backend_errors(errors: &[(CaptureBackendKind, CaptureError)]) -> String {
    let mut combined = String::new();
    for (index, (kind, error)) in errors.iter().enumerate() {
        if index != 0 {
            combined.push_str("; ");
        }
        combined.push_str(kind.as_str());
        combined.push_str(": ");
        combined.push_str(&error.to_string());
    }
    combined
}

impl CaptureBackend for WindowsBackend {
    fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>> {
        self.resolver.enumerate_monitors()
    }

    fn primary_monitor(&self) -> CaptureResult<MonitorId> {
        self.resolver.primary_monitor()
    }

    fn create_monitor_capturer(
        &self,
        monitor: &MonitorId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        match self.kind {
            CaptureBackendKind::Auto => self.create_auto_capturer(monitor),
            other => self.create_by_kind(other, monitor),
        }
    }

    fn create_window_capturer(&self, window: &WindowId) -> CaptureResult<Box<dyn MonitorCapturer>> {
        match self.kind {
            CaptureBackendKind::Auto => self.create_auto_window_capturer(window),
            other => self.create_window_by_kind(other, window),
        }
    }
}
