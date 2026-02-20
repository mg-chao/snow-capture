use std::sync::Arc;

#[cfg(not(target_os = "windows"))]
use crate::backend::MonitorCapturer;
use crate::backend::{AutoBackendPolicy, CaptureBackend, CaptureBackendKind};
#[cfg(not(target_os = "windows"))]
use crate::error::CaptureError;
#[cfg(not(target_os = "windows"))]
use crate::error::CaptureResult;
#[cfg(not(target_os = "windows"))]
use crate::monitor::MonitorId;
#[cfg(not(target_os = "windows"))]
use crate::window::WindowId;

#[cfg(target_os = "windows")]
pub(crate) mod windows;

#[cfg(not(target_os = "windows"))]
fn unsupported_error() -> CaptureError {
    CaptureError::Platform(anyhow::anyhow!(
        "screen capture is only supported on Windows"
    ))
}

#[cfg(not(target_os = "windows"))]
struct UnsupportedBackend;

#[cfg(not(target_os = "windows"))]
impl CaptureBackend for UnsupportedBackend {
    fn enumerate_monitors(&self) -> CaptureResult<Vec<MonitorId>> {
        Err(unsupported_error())
    }

    fn primary_monitor(&self) -> CaptureResult<MonitorId> {
        Err(unsupported_error())
    }

    fn create_monitor_capturer(
        &self,
        _monitor: &MonitorId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        Err(unsupported_error())
    }

    fn create_window_capturer(
        &self,
        _window: &WindowId,
    ) -> CaptureResult<Box<dyn MonitorCapturer>> {
        Err(unsupported_error())
    }
}

#[cfg(target_os = "windows")]
pub(crate) fn build_backend(
    kind: CaptureBackendKind,
    auto_policy: AutoBackendPolicy,
) -> crate::error::CaptureResult<Arc<dyn CaptureBackend>> {
    Ok(Arc::new(windows::WindowsBackend::with_kind_and_policy(
        kind,
        auto_policy,
    )?))
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn build_backend(
    _kind: CaptureBackendKind,
    _auto_policy: AutoBackendPolicy,
) -> crate::error::CaptureResult<Arc<dyn CaptureBackend>> {
    Ok(Arc::new(UnsupportedBackend))
}
