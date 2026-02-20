use std::fmt;

#[derive(Debug)]
pub enum CaptureError {
    InvalidTarget(String),

    MonitorLost,

    NoPrimaryMonitor,

    AccessLost,

    Timeout,

    UnsupportedFormat(String),

    BufferOverflow,

    InvalidConfig(String),

    WorkerDead,

    BackendUnavailable(String),

    Canceled,

    /// The capture source resolution changed during a streaming session.
    /// Contains (new_width, new_height). The stream will automatically
    /// deliver a `CaptureEvent::ResolutionChanged` event.
    ResolutionChanged(u32, u32),

    Platform(anyhow::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureErrorClass {
    InvalidInput,
    Unsupported,
    Transient,
    Fatal,
}

impl CaptureError {
    pub fn class(&self) -> CaptureErrorClass {
        match self {
            Self::InvalidTarget(_) | Self::NoPrimaryMonitor | Self::InvalidConfig(_) => {
                CaptureErrorClass::InvalidInput
            }
            Self::UnsupportedFormat(_) | Self::BackendUnavailable(_) => {
                CaptureErrorClass::Unsupported
            }
            Self::AccessLost
            | Self::Timeout
            | Self::WorkerDead
            | Self::MonitorLost
            | Self::Canceled
            | Self::ResolutionChanged(_, _) => CaptureErrorClass::Transient,
            Self::BufferOverflow | Self::Platform(_) => CaptureErrorClass::Fatal,
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(self.class(), CaptureErrorClass::Transient)
    }

    pub fn requires_worker_reset(&self) -> bool {
        matches!(
            self,
            Self::MonitorLost | Self::AccessLost | Self::WorkerDead
        )
    }

    /// Create a string-based copy of this error suitable for sending
    /// through channels. The `Platform` variant loses its inner
    /// `anyhow::Error` chain and becomes a formatted string.
    pub fn to_sendable(&self) -> Self {
        match self {
            Self::InvalidTarget(s) => Self::InvalidTarget(s.clone()),
            Self::MonitorLost => Self::MonitorLost,
            Self::NoPrimaryMonitor => Self::NoPrimaryMonitor,
            Self::AccessLost => Self::AccessLost,
            Self::Timeout => Self::Timeout,
            Self::UnsupportedFormat(s) => Self::UnsupportedFormat(s.clone()),
            Self::BufferOverflow => Self::BufferOverflow,
            Self::InvalidConfig(s) => Self::InvalidConfig(s.clone()),
            Self::WorkerDead => Self::WorkerDead,
            Self::BackendUnavailable(s) => Self::BackendUnavailable(s.clone()),
            Self::Canceled => Self::Canceled,
            Self::ResolutionChanged(w, h) => Self::ResolutionChanged(*w, *h),
            Self::Platform(inner) => Self::Platform(anyhow::anyhow!("{inner:#}")),
        }
    }
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTarget(id) => write!(
                f,
                "requested monitor target is not available in this session/backend: {id}"
            ),
            Self::MonitorLost => write!(f, "requested monitor is no longer available"),
            Self::NoPrimaryMonitor => write!(f, "no primary monitor found"),
            Self::AccessLost => write!(f, "capture access lost"),
            Self::Timeout => write!(f, "failed to acquire desktop frame within timeout"),
            Self::UnsupportedFormat(fmt_name) => {
                write!(f, "unsupported desktop texture format: {fmt_name}")
            }
            Self::BufferOverflow => write!(f, "frame buffer size overflow"),
            Self::InvalidConfig(message) => write!(f, "invalid capture configuration: {message}"),
            Self::WorkerDead => write!(f, "capture worker is not running"),
            Self::BackendUnavailable(message) => {
                write!(f, "no available backend implementation: {message}")
            }
            Self::Canceled => write!(f, "capture request was canceled"),
            Self::ResolutionChanged(w, h) => {
                write!(f, "capture source resolution changed to {w}x{h}")
            }
            Self::Platform(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for CaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Platform(inner) => Some(inner.as_ref()),
            _ => None,
        }
    }
}

pub type CaptureResult<T> = Result<T, CaptureError>;
