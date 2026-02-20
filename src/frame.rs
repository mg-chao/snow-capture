use std::time::{Duration, Instant};

use crate::error::{CaptureError, CaptureResult};

/// Color space / transfer function describing the frame's pixel data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorSpace {
    /// Standard sRGB (BT.709 primaries, sRGB transfer function).
    /// This is the default for SDR captures and tonemapped HDR output.
    #[default]
    Srgb,
    /// Scene-referred linear light (BT.709 primaries, linear gamma).
    /// Produced when the DXGI backend captures an HDR surface without
    /// tonemapping.
    LinearSrgb,
    /// HDR10 / PQ (BT.2020 primaries, SMPTE ST 2084 transfer function).
    Hdr10Pq,
    /// Hybrid Log-Gamma (BT.2020 primaries, ARIB STD-B67).
    Hlg,
}

/// Minimum allocation size to attempt large-page backing.
/// 4K RGBA = 3840×2160×4 ≈ 33 MB — well above the 2 MB large page size.
/// We only bother for allocations ≥ 4 MB to avoid overhead on small captures.
const LARGE_PAGE_MIN_BYTES: usize = 4 * 1024 * 1024;

/// A rectangle describing a dirty (changed) region of the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Cursor shape data captured alongside the frame.
#[derive(Clone, Debug)]
pub struct CursorData {
    /// Cursor hotspot X relative to the cursor image.
    pub hotspot_x: u32,
    /// Cursor hotspot Y relative to the cursor image.
    pub hotspot_y: u32,
    /// Cursor position X in desktop coordinates.
    pub position_x: i32,
    /// Cursor position Y in desktop coordinates.
    pub position_y: i32,
    /// Whether the cursor is currently visible.
    pub visible: bool,
    /// Cursor image width in pixels.
    pub shape_width: u32,
    /// Cursor image height in pixels.
    pub shape_height: u32,
    /// RGBA8 pixel data for the cursor shape.
    pub shape_rgba: Vec<u8>,
}

/// Metadata attached to each captured frame for recording pipelines.
#[derive(Clone, Debug, Default)]
pub struct FrameMetadata {
    /// Monotonic timestamp taken immediately after the frame was acquired
    /// from the OS capture API. Use for frame pacing and drop detection.
    pub capture_time: Option<Instant>,
    /// OS presentation timestamp in QPC ticks (100ns units on Windows).
    /// Sourced from `DXGI_OUTDUPL_FRAME_INFO.LastPresentTime` or
    /// `Direct3D11CaptureFrame.SystemRelativeTime`. More accurate than
    /// `capture_time` for A/V sync.
    pub present_time_qpc: Option<i64>,
    /// Wall-clock time spent inside the capture call (GPU readback,
    /// staging copy, pixel conversion). Lets recorders detect when the
    /// capture pipeline itself is the bottleneck vs. the encoder.
    pub capture_duration: Option<Duration>,
    /// Whether this frame's content is identical to the previous frame.
    /// `true` means no new desktop present occurred — a recorder can skip
    /// encoding this frame to save bitrate.
    pub is_duplicate: bool,
    /// Dirty rectangles describing which regions changed since the last
    /// frame. Empty when the backend doesn't support damage tracking or
    /// when the entire frame changed.
    pub dirty_rects: Vec<DirtyRect>,
    /// Cursor shape and position at the time of capture. `None` when
    /// cursor capture is not enabled or not supported by the backend.
    pub cursor: Option<CursorData>,
    /// Monotonic sequence number incremented for each capture call.
    /// Useful for correlating frames across threads.
    pub sequence: u64,
    /// Color space / transfer function of the pixel data. Defaults to
    /// `Srgb` for standard dynamic range captures. HDR pipelines can
    /// check this to decide whether tonemapping or passthrough is needed.
    pub color_space: ColorSpace,
}

/// Anchor for converting raw QPC ticks into stream-relative durations.
///
/// Created from the first frame's `present_time_qpc` (or `capture_time`
/// when QPC is unavailable). All subsequent frames can be mapped to a
/// consistent `Duration` offset from stream start via
/// [`stream_relative`](Self::stream_relative).
#[derive(Clone, Debug)]
pub struct FrameTimestampAnchor {
    /// QPC ticks of the first frame (if available).
    origin_qpc: Option<i64>,
    /// `Instant` of the first frame (fallback when QPC is absent).
    origin_instant: Instant,
    /// QPC frequency (ticks per second), cached at construction.
    qpc_frequency: i64,
}

impl FrameTimestampAnchor {
    /// Build an anchor from the first captured frame's metadata.
    pub fn from_first_frame(meta: &FrameMetadata) -> Self {
        let qpc_frequency = qpc_frequency_cached();
        Self {
            origin_qpc: meta.present_time_qpc,
            origin_instant: meta.capture_time.unwrap_or_else(Instant::now),
            qpc_frequency,
        }
    }

    /// Raw QPC ticks of the stream origin frame. Useful for correlating
    /// with WASAPI audio timestamps which also use QPC.
    pub fn origin_qpc_ticks(&self) -> Option<i64> {
        self.origin_qpc
    }

    /// QPC tick frequency (ticks per second). Returns 0 on non-Windows.
    pub fn qpc_frequency(&self) -> i64 {
        self.qpc_frequency
    }

    /// The `Instant` of the stream origin frame.
    pub fn origin_instant(&self) -> Instant {
        self.origin_instant
    }

    /// Convert a frame's timestamp to a stream-relative `Duration`.
    ///
    /// Uses QPC when both the anchor and the frame have QPC timestamps,
    /// otherwise falls back to `capture_time` difference.
    pub fn stream_relative(&self, meta: &FrameMetadata) -> Duration {
        if let (Some(origin), Some(current)) = (self.origin_qpc, meta.present_time_qpc) {
            let delta_ticks = current.saturating_sub(origin).max(0);
            if self.qpc_frequency > 0 {
                let secs = delta_ticks / self.qpc_frequency;
                let remainder = delta_ticks % self.qpc_frequency;
                let nanos = (remainder as u128 * 1_000_000_000) / self.qpc_frequency as u128;
                return Duration::new(secs as u64, nanos as u32);
            }
        }
        // Fallback: Instant-based delta.
        meta.capture_time
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.origin_instant)
    }
}

/// Cached QPC frequency. Returns 0 if unavailable.
fn qpc_frequency_cached() -> i64 {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Performance::QueryPerformanceFrequency;
        let mut freq = 0i64;
        unsafe {
            let _ = QueryPerformanceFrequency(&mut freq);
        }
        freq
    }
    #[cfg(not(target_os = "windows"))]
    {
        0
    }
}

/// Query the current QPC counter value. Returns `None` on non-Windows
/// or if the call fails.
#[cfg(target_os = "windows")]
pub(crate) fn query_qpc_now() -> Option<i64> {
    use windows::Win32::System::Performance::QueryPerformanceCounter;
    let mut ticks = 0i64;
    let ok = unsafe { QueryPerformanceCounter(&mut ticks) };
    if ok.is_ok() { Some(ticks) } else { None }
}

/// Notification sent through the streaming channel when the capture
/// source changes in a way that affects the output.
#[derive(Debug)]
pub enum CaptureEvent {
    /// A new frame is available.
    Frame(Frame),
    /// The capture source resolution changed. A recorder should
    /// reconfigure its encoder with the new dimensions.
    ResolutionChanged {
        old_width: u32,
        old_height: u32,
        new_width: u32,
        new_height: u32,
    },
    /// One or more frames were dropped due to backpressure (the
    /// receiver couldn't keep up). A recorder should insert duplicate
    /// frames to maintain correct A/V timing.
    FrameDropped {
        /// Sequence number of the dropped frame.
        sequence: u64,
    },
    /// The stream was paused. Contains the `Instant` at which the
    /// pause took effect. Recorders can use this to account for the
    /// gap in their muxer timeline.
    Paused { at: Instant },
    /// The stream was resumed after a pause. Contains the `Instant`
    /// of resumption and the total duration of the pause gap.
    Resumed { at: Instant, gap: Duration },
    /// The stream thread is about to exit cleanly (stop was requested).
    /// Sent after the last frame so the consumer knows no more events
    /// will arrive. Useful for flushing encoder queues.
    StreamEnded,
    /// The stream encountered a fatal error and is about to exit.
    /// After receiving this event the channel will disconnect.
    Error(CaptureError),
}

pub struct Frame {
    data: FrameBuffer,
    width: u32,
    height: u32,
    /// Per-frame metadata for recording pipelines.
    pub metadata: FrameMetadata,
}

/// Frame buffer that tries to use large pages (2 MB) via `VirtualAlloc`
/// to reduce TLB misses during parallel pixel conversion.  Falls back
/// to a regular `Vec<u8>` when large pages aren't available or the
/// allocation is too small to benefit.
enum FrameBuffer {
    Vec(Vec<u8>),
    #[cfg(target_os = "windows")]
    LargePage(LargePageAlloc),
}

#[cfg(target_os = "windows")]
struct LargePageAlloc {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

#[cfg(target_os = "windows")]
unsafe impl Send for LargePageAlloc {}
#[cfg(target_os = "windows")]
unsafe impl Sync for LargePageAlloc {}

#[cfg(target_os = "windows")]
impl Drop for LargePageAlloc {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            use windows::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            unsafe {
                let _ = VirtualFree(self.ptr as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn try_alloc_large_pages(size: usize) -> Option<LargePageAlloc> {
    use windows::Win32::System::Memory::{
        GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
    };

    if size < LARGE_PAGE_MIN_BYTES {
        return None;
    }

    let large_page_size = unsafe { GetLargePageMinimum() };
    if large_page_size == 0 {
        return None;
    }

    // Round up to large page boundary
    let aligned_size = (size + large_page_size - 1) & !(large_page_size - 1);

    let ptr = unsafe {
        VirtualAlloc(
            None,
            aligned_size,
            MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
            PAGE_READWRITE,
        )
    };

    if ptr.is_null() {
        return None;
    }

    Some(LargePageAlloc {
        ptr: ptr as *mut u8,
        len: 0,
        capacity: aligned_size,
    })
}

impl FrameBuffer {
    fn new() -> Self {
        FrameBuffer::Vec(Vec::new())
    }

    fn len(&self) -> usize {
        match self {
            FrameBuffer::Vec(v) => v.len(),
            #[cfg(target_os = "windows")]
            FrameBuffer::LargePage(lp) => lp.len,
        }
    }

    fn capacity(&self) -> usize {
        match self {
            FrameBuffer::Vec(v) => v.capacity(),
            #[cfg(target_os = "windows")]
            FrameBuffer::LargePage(lp) => lp.capacity,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            FrameBuffer::Vec(v) => v.as_mut_ptr(),
            #[cfg(target_os = "windows")]
            FrameBuffer::LargePage(lp) => lp.ptr,
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            FrameBuffer::Vec(v) => v.as_slice(),
            #[cfg(target_os = "windows")]
            FrameBuffer::LargePage(lp) => unsafe { std::slice::from_raw_parts(lp.ptr, lp.len) },
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            FrameBuffer::Vec(v) => v.as_mut_slice(),
            #[cfg(target_os = "windows")]
            FrameBuffer::LargePage(lp) => unsafe { std::slice::from_raw_parts_mut(lp.ptr, lp.len) },
        }
    }

    /// Ensure the buffer has exactly `len` bytes available.
    /// Tries large pages for big allocations, falls back to Vec.
    fn ensure_len(&mut self, len: usize) {
        if self.len() == len {
            return;
        }

        // If current backing can hold the new size, just adjust length.
        if len <= self.capacity() {
            match self {
                FrameBuffer::Vec(v) => unsafe { v.set_len(len) },
                #[cfg(target_os = "windows")]
                FrameBuffer::LargePage(lp) => lp.len = len,
            }
            return;
        }

        // Need a new allocation — try large pages first on Windows.
        #[cfg(target_os = "windows")]
        if len >= LARGE_PAGE_MIN_BYTES {
            if let Some(mut lp) = try_alloc_large_pages(len) {
                lp.len = len;
                *self = FrameBuffer::LargePage(lp);
                return;
            }
        }

        // Fall back to Vec with headroom.
        let headroom = len / 8;
        let mut v = Vec::with_capacity(len + headroom);
        unsafe { v.set_len(len) };
        *self = FrameBuffer::Vec(v);
    }
}

impl Frame {
    pub fn empty() -> Self {
        Self {
            data: FrameBuffer::new(),
            width: 0,
            height: 0,
            metadata: FrameMetadata::default(),
        }
    }

    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> CaptureResult<Self> {
        let expected = rgba_len(width, height)?;
        if data.len() != expected {
            return Err(CaptureError::InvalidConfig(format!(
                "RGBA frame data length mismatch: got {}, expected {} for {}x{}",
                data.len(),
                expected,
                width,
                height
            )));
        }

        Ok(Self {
            data: FrameBuffer::Vec(data),
            width,
            height,
            metadata: FrameMetadata::default(),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn as_rgba_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    pub fn as_mut_rgba_bytes(&mut self) -> &mut [u8] {
        self.data.as_mut_slice()
    }

    pub(crate) fn as_mut_rgba_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    pub(crate) fn ensure_rgba_capacity(&mut self, width: u32, height: u32) -> CaptureResult<()> {
        let len = rgba_len(width, height)?;
        self.data.ensure_len(len);
        self.width = width;
        self.height = height;
        Ok(())
    }

    /// Reset metadata fields to defaults, preserving the pixel buffer.
    /// Called at the start of each capture to avoid stale metadata from
    /// a reused frame leaking into the new result.
    pub(crate) fn reset_metadata(&mut self) {
        self.metadata.capture_time = None;
        self.metadata.present_time_qpc = None;
        self.metadata.capture_duration = None;
        self.metadata.is_duplicate = false;
        self.metadata.dirty_rects.clear();
        self.metadata.cursor = None;
        self.metadata.color_space = ColorSpace::default();
        // sequence is set by the session, not reset here
    }
}

fn rgba_len(width: u32, height: u32) -> CaptureResult<usize> {
    let w = usize::try_from(width).map_err(|_| CaptureError::BufferOverflow)?;
    let h = usize::try_from(height).map_err(|_| CaptureError::BufferOverflow)?;
    w.checked_mul(h)
        .and_then(|px| px.checked_mul(4))
        .ok_or(CaptureError::BufferOverflow)
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data_len", &self.data.len())
            .field("metadata", &self.metadata)
            .finish()
    }
}
