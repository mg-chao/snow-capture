pub mod backend;
pub mod capture_session;
pub mod convert;
pub(crate) mod env_config;
pub mod error;
pub mod frame;
pub mod monitor;
mod platform;
pub mod region;
pub mod streaming;
pub mod window;

use error::CaptureResult;
use frame::Frame;

#[derive(Clone)]
pub enum CaptureTarget {
    PrimaryMonitor,

    Monitor(monitor::MonitorId),

    /// Capture a top-level window by native window handle.
    ///
    /// On the DXGI duplication backend, the window's monitor is captured
    /// and the result is cropped to the window's desktop bounds.  WGC
    /// captures the window directly. GDI captures the window directly
    /// via native Win32 window rendering.
    Window(window::WindowId),

    /// Capture a rectangular region in virtual desktop coordinates.
    /// The region may span multiple monitors. The layout is snapshotted
    /// once at session creation via [`MonitorLayout`](region::MonitorLayout).
    Region(region::CaptureRegion),
}

pub use backend::CaptureMode;
pub use capture_session::{CaptureSession, CaptureSessionBuilder, CaptureSessionConfig};
pub use frame::{
    CaptureEvent, ColorSpace, CursorData, DirtyRect, FrameMetadata, FrameTimestampAnchor,
};
pub use monitor::MonitorId;
pub use region::{CaptureRegion, MonitorLayout};
pub use streaming::{StreamConfig, StreamHandle, StreamStats, StreamStatsSnapshot};
pub use window::WindowId;

#[cfg(feature = "tokio-stream")]
pub use streaming::AsyncStreamHandle;

pub fn capture_once(target: &CaptureTarget) -> CaptureResult<Frame> {
    let mut session = CaptureSession::new()?;
    session.capture_frame(target)
}
