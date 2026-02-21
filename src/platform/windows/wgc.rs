use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Context;
use windows::Foundation::{EventRegistrationToken, TypedEventHandler};
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureDirtyRegionMode,
    GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_QUERY_DESC, D3D11_QUERY_EVENT, D3D11_TEXTURE2D_DESC, ID3D11Device,
    ID3D11DeviceContext, ID3D11Query, ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_ACCESS_LOST, IDXGIDevice};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::core::{IInspectable, Interface};

use crate::backend::{CaptureBlitRegion, CaptureMode, CaptureSampleMetadata, CursorCaptureConfig};
use crate::convert::HdrToSdrParams;
use crate::error::{CaptureError, CaptureResult};
use crate::frame::{DirtyRect, Frame};
use crate::monitor::MonitorId;
use crate::window::WindowId;

use super::com::CoInitGuard;
use super::d3d11;
use super::gpu_tonemap::{GpuF16Converter, GpuTonemapper};
use super::monitor::{HdrMonitorMetadata, MonitorResolver};
use super::surface::{self, StagingSampleDesc};

const WGC_FRAME_TIMEOUT: Duration = Duration::from_millis(250);
const WGC_STALE_FRAME_TIMEOUT_MIN: Duration = Duration::from_micros(400);
const WGC_STALE_FRAME_TIMEOUT_MAX: Duration = Duration::from_millis(2);
const WGC_STALE_FRAME_TIMEOUT_INITIAL: Duration = WGC_STALE_FRAME_TIMEOUT_MAX;
const WGC_STALE_FRAME_TIMEOUT_DECREASE_STEP: Duration = Duration::from_micros(200);
const WGC_STALE_FRAME_TIMEOUT_INCREASE_STEP: Duration = Duration::from_micros(100);
const WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_MAX: Duration = Duration::from_micros(900);
const WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_INITIAL: Duration = WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_MAX;
const WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_DECREASE_STEP: Duration = Duration::from_micros(120);
const WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_INCREASE_STEP: Duration = Duration::from_micros(60);
const WGC_FRAME_POOL_BUFFERS: i32 = 2;
const WGC_STAGING_SLOTS: usize = 2;
const WGC_DIRTY_COPY_MAX_RECTS: usize = 192;
const WGC_DIRTY_COPY_MAX_AREA_PERCENT: u64 = 70;
const WGC_STALE_POLL_SPIN_MIN: u32 = 16;
const WGC_STALE_POLL_SPIN_MAX: u32 = 256;
const WGC_STALE_POLL_SPIN_INITIAL: u32 = 64;
const WGC_STALE_POLL_SPIN_INCREASE_STEP: u32 = 16;
const WGC_QUERY_SPIN_MIN_POLLS: u32 = 2;
const WGC_QUERY_SPIN_MAX_POLLS: u32 = 64;
const WGC_QUERY_SPIN_INITIAL_POLLS: u32 = 4;
const WGC_QUERY_SPIN_INCREASE_STEP: u32 = 4;

#[derive(Clone, Copy)]
struct WgcStaleTimeoutConfig {
    min: Duration,
    max: Duration,
    initial: Duration,
    decrease_step: Duration,
    increase_step: Duration,
}

#[inline]
fn env_var_truthy(var_name: &'static str) -> bool {
    std::env::var(var_name)
        .map(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

#[inline]
fn region_dirty_gpu_copy_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_REGION_DIRTY_GPU_COPY"))
}

#[inline]
fn duplicate_dirty_fastpath_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_DUPLICATE_DIRTY_FASTPATH"))
}

#[inline]
fn immediate_stale_return_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_IMMEDIATE_STALE_RETURN"))
}

#[inline]
fn region_duplicate_short_circuit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_REGION_DUPLICATE_SHORTCIRCUIT"))
}

#[inline]
fn borrowed_source_resource_cast_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_BORROWED_SOURCE_RESOURCE"))
}

#[inline]
fn borrowed_slot_resource_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_BORROWED_SLOT_RESOURCE"))
}

#[inline]
fn aggressive_stale_timeout_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_WGC_DISABLE_AGGRESSIVE_STALE_TIMEOUT"))
}

#[inline(always)]
fn stale_timeout_config(aggressive: bool) -> WgcStaleTimeoutConfig {
    if aggressive {
        WgcStaleTimeoutConfig {
            min: WGC_STALE_FRAME_TIMEOUT_MIN,
            max: WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_MAX,
            initial: WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_INITIAL,
            decrease_step: WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_DECREASE_STEP,
            increase_step: WGC_STALE_FRAME_TIMEOUT_AGGRESSIVE_INCREASE_STEP,
        }
    } else {
        WgcStaleTimeoutConfig {
            min: WGC_STALE_FRAME_TIMEOUT_MIN,
            max: WGC_STALE_FRAME_TIMEOUT_MAX,
            initial: WGC_STALE_FRAME_TIMEOUT_INITIAL,
            decrease_step: WGC_STALE_FRAME_TIMEOUT_DECREASE_STEP,
            increase_step: WGC_STALE_FRAME_TIMEOUT_INCREASE_STEP,
        }
    }
}

#[inline(always)]
fn active_stale_timeout_config() -> WgcStaleTimeoutConfig {
    stale_timeout_config(aggressive_stale_timeout_enabled())
}

#[inline(always)]
fn duration_saturating_add_clamped(base: Duration, delta: Duration, max: Duration) -> Duration {
    base.checked_add(delta).unwrap_or(max).min(max)
}

#[inline(always)]
fn duration_saturating_sub_clamped(base: Duration, delta: Duration, min: Duration) -> Duration {
    base.saturating_sub(delta).max(min)
}

fn hdr_to_sdr_params(hdr: HdrMonitorMetadata) -> Option<HdrToSdrParams> {
    if !hdr.advanced_color_enabled {
        return None;
    }

    if !hdr.hdr_enabled {
        return None;
    }

    let sdr_white_level_nits = hdr.sdr_white_level_nits.unwrap_or(80.0);
    let hdr_paper_white_nits = hdr.hdr_paper_white_nits.unwrap_or(80.0);

    Some(HdrToSdrParams {
        hdr_paper_white_nits,
        hdr_maximum_nits: hdr.hdr_maximum_nits.unwrap_or(1000.0),
        sdr_white_level_nits,
    })
}

fn clamp_dirty_rect(rect: DirtyRect, width: u32, height: u32) -> Option<DirtyRect> {
    let x = rect.x.min(width);
    let y = rect.y.min(height);
    if x >= width || y >= height {
        return None;
    }

    let max_w = width - x;
    let max_h = height - y;
    let clamped_w = rect.width.min(max_w);
    let clamped_h = rect.height.min(max_h);
    if clamped_w == 0 || clamped_h == 0 {
        return None;
    }

    Some(DirtyRect {
        x,
        y,
        width: clamped_w,
        height: clamped_h,
    })
}

fn dirty_region_mode_supported(mode: GraphicsCaptureDirtyRegionMode) -> bool {
    mode == GraphicsCaptureDirtyRegionMode::ReportOnly
        || mode == GraphicsCaptureDirtyRegionMode::ReportAndRender
}

fn extract_dirty_rects(
    frame: &Direct3D11CaptureFrame,
    width: u32,
    height: u32,
    out: &mut Vec<DirtyRect>,
) -> Option<GraphicsCaptureDirtyRegionMode> {
    out.clear();
    let mode = frame.DirtyRegionMode().ok()?;
    if !dirty_region_mode_supported(mode) {
        return None;
    }
    let regions = frame.DirtyRegions().ok()?;
    let count = regions.Size().ok()?;

    for idx in 0..count {
        let Ok(rect) = regions.GetAt(idx) else {
            continue;
        };

        if rect.Width <= 0 || rect.Height <= 0 {
            continue;
        }

        let x = rect.X.max(0) as u32;
        let y = rect.Y.max(0) as u32;
        let rect_width = rect.Width as u32;
        let rect_height = rect.Height as u32;

        if let Some(clamped) = clamp_dirty_rect(
            DirtyRect {
                x,
                y,
                width: rect_width,
                height: rect_height,
            },
            width,
            height,
        ) {
            out.push(clamped);
        }
    }

    normalize_dirty_rects_in_place(out, width, height);
    Some(mode)
}

fn intersect_dirty_rects(a: DirtyRect, b: DirtyRect) -> Option<DirtyRect> {
    let a_right = a.x.saturating_add(a.width);
    let a_bottom = a.y.saturating_add(a.height);
    let b_right = b.x.saturating_add(b.width);
    let b_bottom = b.y.saturating_add(b.height);

    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = a_right.min(b_right);
    let bottom = a_bottom.min(b_bottom);
    if right <= x || bottom <= y {
        return None;
    }

    Some(DirtyRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

#[inline(always)]
fn dirty_rect_bounds(rect: DirtyRect) -> (u32, u32) {
    (
        rect.x.saturating_add(rect.width),
        rect.y.saturating_add(rect.height),
    )
}

#[inline(always)]
fn intervals_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start < b_end && b_start < a_end
}

#[inline(always)]
fn intervals_touch_or_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn dirty_rects_can_merge(a: DirtyRect, b: DirtyRect) -> bool {
    let (a_right, a_bottom) = dirty_rect_bounds(a);
    let (b_right, b_bottom) = dirty_rect_bounds(b);

    let horizontal_overlap = intervals_overlap(a.x, a_right, b.x, b_right);
    let vertical_overlap = intervals_overlap(a.y, a_bottom, b.y, b_bottom);
    let horizontal_touch_or_overlap = intervals_touch_or_overlap(a.x, a_right, b.x, b_right);
    let vertical_touch_or_overlap = intervals_touch_or_overlap(a.y, a_bottom, b.y, b_bottom);

    (horizontal_overlap && vertical_touch_or_overlap)
        || (vertical_overlap && horizontal_touch_or_overlap)
}

fn merge_dirty_rects(a: DirtyRect, b: DirtyRect) -> DirtyRect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let (a_right, a_bottom) = dirty_rect_bounds(a);
    let (b_right, b_bottom) = dirty_rect_bounds(b);
    let right = a_right.max(b_right);
    let bottom = a_bottom.max(b_bottom);
    DirtyRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

fn normalize_dirty_rects_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut write = 0usize;
    for read in 0..rects.len() {
        if let Some(clamped) = clamp_dirty_rect(rects[read], width, height) {
            rects[write] = clamped;
            write += 1;
        }
    }
    rects.truncate(write);
    if rects.len() <= 1 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;

        let mut i = 0usize;
        while i < rects.len() {
            let mut j = i + 1;
            while j < rects.len() {
                if dirty_rects_can_merge(rects[i], rects[j]) {
                    rects[i] = merge_dirty_rects(rects[i], rects[j]);
                    rects.swap_remove(j);
                    changed = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    rects.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));
}

fn extract_region_dirty_rects(
    frame: &Direct3D11CaptureFrame,
    source_width: u32,
    source_height: u32,
    blit: CaptureBlitRegion,
    out: &mut Vec<DirtyRect>,
) -> bool {
    out.clear();
    let Some(mode) = frame.DirtyRegionMode().ok() else {
        return false;
    };
    if !dirty_region_mode_supported(mode) {
        return false;
    }
    let Some(regions) = frame.DirtyRegions().ok() else {
        return false;
    };
    let Some(count) = regions.Size().ok() else {
        return false;
    };

    let Some(region_bounds) = clamp_dirty_rect(
        DirtyRect {
            x: blit.src_x,
            y: blit.src_y,
            width: blit.width,
            height: blit.height,
        },
        source_width,
        source_height,
    ) else {
        return false;
    };

    for idx in 0..count {
        let Ok(rect) = regions.GetAt(idx) else {
            continue;
        };

        if rect.Width <= 0 || rect.Height <= 0 {
            continue;
        }

        let x = rect.X.max(0) as u32;
        let y = rect.Y.max(0) as u32;
        let rect_width = rect.Width as u32;
        let rect_height = rect.Height as u32;
        let Some(clamped) = clamp_dirty_rect(
            DirtyRect {
                x,
                y,
                width: rect_width,
                height: rect_height,
            },
            source_width,
            source_height,
        ) else {
            continue;
        };

        let Some(intersection) = intersect_dirty_rects(clamped, region_bounds) else {
            continue;
        };
        out.push(DirtyRect {
            x: intersection.x.saturating_sub(region_bounds.x),
            y: intersection.y.saturating_sub(region_bounds.y),
            width: intersection.width,
            height: intersection.height,
        });
    }

    normalize_dirty_rects_in_place(out, region_bounds.width, region_bounds.height);
    true
}

fn should_use_dirty_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
    if rects.is_empty() || rects.len() > WGC_DIRTY_COPY_MAX_RECTS {
        return false;
    }

    let total_pixels = (width as u64).saturating_mul(height as u64);
    if total_pixels == 0 {
        return false;
    }

    let mut dirty_pixels = 0u64;
    for rect in rects {
        dirty_pixels =
            dirty_pixels.saturating_add((rect.width as u64).saturating_mul(rect.height as u64));
        if dirty_pixels.saturating_mul(100)
            > total_pixels.saturating_mul(WGC_DIRTY_COPY_MAX_AREA_PERCENT)
        {
            return false;
        }
    }

    true
}

#[inline(always)]
fn should_short_circuit_region_duplicate(
    optimization_enabled: bool,
    capture_mode: CaptureMode,
    destination_has_history: bool,
    source_is_duplicate: bool,
    pending_slot_available: bool,
) -> bool {
    optimization_enabled
        && capture_mode == CaptureMode::ScreenRecording
        && destination_has_history
        && source_is_duplicate
        && pending_slot_available
}

#[derive(Default)]
struct WgcStagingSlot {
    staging: Option<ID3D11Texture2D>,
    staging_resource: Option<ID3D11Resource>,
    query: Option<ID3D11Query>,
    source_desc: Option<D3D11_TEXTURE2D_DESC>,
    hdr_to_sdr: Option<HdrToSdrParams>,
    capture_time: Option<Instant>,
    present_time_ticks: i64,
    is_duplicate: bool,
    dirty_mode_available: bool,
    dirty_copy_preferred: bool,
    dirty_rects: Vec<DirtyRect>,
    populated: bool,
}

impl WgcStagingSlot {
    fn invalidate(&mut self) {
        self.staging = None;
        self.staging_resource = None;
        self.query = None;
        self.source_desc = None;
        self.hdr_to_sdr = None;
        self.capture_time = None;
        self.present_time_ticks = 0;
        self.is_duplicate = false;
        self.dirty_mode_available = false;
        self.dirty_copy_preferred = false;
        self.dirty_rects.clear();
        self.populated = false;
    }
}

/// WGC FrameArrived stores the system-relative time alongside the frame.
#[derive(Default)]
struct FrameState {
    sequence: u64,
    latest: Option<Direct3D11CaptureFrame>,
    /// System-relative time from the WGC frame (100ns ticks).
    latest_time_ticks: i64,
    closed: bool,
}

#[derive(Default)]
struct FrameSignal {
    state: Mutex<FrameState>,
    cv: Condvar,
    sequence_hint: AtomicU64,
    closed_hint: AtomicBool,
}

fn poisoned_lock_error() -> CaptureError {
    CaptureError::Platform(anyhow::anyhow!(
        "wgc frame synchronization mutex was poisoned"
    ))
}

fn map_platform_error(error: windows::core::Error, context: &str) -> CaptureError {
    if error.code() == DXGI_ERROR_ACCESS_LOST {
        return CaptureError::AccessLost;
    }
    CaptureError::Platform(anyhow::Error::from(error).context(context.to_string()))
}

#[inline(always)]
fn with_texture_resource<T>(
    texture: &ID3D11Texture2D,
    cast_context: &'static str,
    f: impl FnOnce(&ID3D11Resource) -> CaptureResult<T>,
) -> CaptureResult<T> {
    if borrowed_source_resource_cast_enabled() {
        let raw = texture.as_raw();
        // SAFETY: ID3D11Texture2D inherits from ID3D11Resource, so the raw
        // COM pointer is valid when viewed through the base interface.
        if let Some(resource) = unsafe { ID3D11Resource::from_raw_borrowed(&raw) } {
            return f(resource);
        }
    }

    let owned_resource: ID3D11Resource = texture
        .cast()
        .context(cast_context)
        .map_err(CaptureError::Platform)?;
    f(&owned_resource)
}

fn create_winrt_device(device: &ID3D11Device) -> CaptureResult<IDirect3DDevice> {
    let dxgi_device: IDXGIDevice = device
        .cast()
        .context("failed to cast ID3D11Device to IDXGIDevice")
        .map_err(CaptureError::Platform)?;
    let inspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device) }
        .context("CreateDirect3D11DeviceFromDXGIDevice failed")
        .map_err(CaptureError::Platform)?;
    inspectable
        .cast()
        .context("failed to cast IInspectable to IDirect3DDevice")
        .map_err(CaptureError::Platform)
}

fn create_monitor_capture_item(
    monitor: windows::Win32::Graphics::Gdi::HMONITOR,
) -> CaptureResult<GraphicsCaptureItem> {
    let interop = windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
        .context("failed to get IGraphicsCaptureItemInterop factory")
        .map_err(CaptureError::Platform)?;
    unsafe { interop.CreateForMonitor(monitor) }
        .context("IGraphicsCaptureItemInterop::CreateForMonitor failed")
        .map_err(CaptureError::Platform)
}

fn create_window_capture_item(window: HWND) -> CaptureResult<GraphicsCaptureItem> {
    let interop = windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
        .context("failed to get IGraphicsCaptureItemInterop factory")
        .map_err(CaptureError::Platform)?;
    unsafe { interop.CreateForWindow(window) }
        .context("IGraphicsCaptureItemInterop::CreateForWindow failed")
        .map_err(CaptureError::Platform)
}

pub(crate) fn validate_support() -> CaptureResult<()> {
    let supported = GraphicsCaptureSession::IsSupported()
        .context("GraphicsCaptureSession::IsSupported failed")
        .map_err(CaptureError::Platform)?;
    if supported {
        Ok(())
    } else {
        Err(CaptureError::BackendUnavailable(
            "Windows Graphics Capture is not supported on this system".into(),
        ))
    }
}

struct WindowsGraphicsCaptureCapturer {
    _com: CoInitGuard,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    winrt_device: IDirect3DDevice,
    item: GraphicsCaptureItem,
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame_arrived_token: EventRegistrationToken,
    closed_token: EventRegistrationToken,
    signal: Arc<FrameSignal>,
    last_sequence: u64,
    pool_size: SizeInt32,
    pixel_format: DirectXPixelFormat,
    staging_slots: [WgcStagingSlot; WGC_STAGING_SLOTS],
    next_write_slot: usize,
    /// Most recent submitted slot. In recording mode this is read on the
    /// following capture call so GPU copy and CPU conversion overlap.
    pending_slot: Option<usize>,
    region_slots: [WgcStagingSlot; WGC_STAGING_SLOTS],
    region_next_write_slot: usize,
    region_pending_slot: Option<usize>,
    cached_src_desc: Option<D3D11_TEXTURE2D_DESC>,
    /// Last system-relative time, used for duplicate detection.
    last_present_time: i64,
    /// Last present timestamp emitted to callers.
    last_emitted_present_time: i64,
    stale_poll_spins: u32,
    stale_timeout_config: WgcStaleTimeoutConfig,
    stale_frame_timeout: Duration,
    adaptive_spin_polls: u32,
    region_adaptive_spin_polls: u32,
    capture_mode: CaptureMode,
    /// HDR-to-SDR tonemap parameters, `Some` when the monitor has HDR enabled.
    hdr_to_sdr: Option<HdrToSdrParams>,
    /// GPU compute-shader tonemapper (HDR F16 -> sRGB RGBA8).
    gpu_tonemapper: Option<GpuTonemapper>,
    /// GPU F16->sRGB converter for when source is F16 but no HDR tonemap needed.
    gpu_f16_converter: Option<GpuF16Converter>,
    region_blit: Option<CaptureBlitRegion>,
    cursor_config: CursorCaptureConfig,
    has_frame_history: bool,
    source_dirty_rects_scratch: Vec<DirtyRect>,
    region_dirty_rects_scratch: Vec<DirtyRect>,
}

// SAFETY: WGC COM objects are not Send, but we only access them from
// the owning capture thread. The FrameArrived callback runs on a
// separate thread but only touches the Arc<FrameSignal> which is
// Send+Sync. The streaming module ensures single-threaded access to
// the capturer itself.
unsafe impl Send for WindowsGraphicsCaptureCapturer {}

impl WindowsGraphicsCaptureCapturer {
    fn new(
        com: CoInitGuard,
        device: ID3D11Device,
        context: ID3D11DeviceContext,
        item: GraphicsCaptureItem,
        hdr_metadata: Option<HdrMonitorMetadata>,
    ) -> CaptureResult<Self> {
        let stale_timeout_config = active_stale_timeout_config();
        let winrt_device = create_winrt_device(&device)?;
        let pool_size = item
            .Size()
            .context("GraphicsCaptureItem::Size failed")
            .map_err(CaptureError::Platform)?;

        let hdr_to_sdr = hdr_metadata.and_then(hdr_to_sdr_params);
        let is_hdr = hdr_to_sdr.is_some();

        // Use F16 pixel format when HDR is active so we capture the full
        // dynamic range; fall back to BGRA8 for SDR.
        let pixel_format = if is_hdr {
            DirectXPixelFormat::R16G16B16A16Float
        } else {
            DirectXPixelFormat::B8G8R8A8UIntNormalized
        };

        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &winrt_device,
            pixel_format,
            WGC_FRAME_POOL_BUFFERS,
            pool_size,
        )
        .context("Direct3D11CaptureFramePool::CreateFreeThreaded failed")
        .map_err(CaptureError::Platform)?;
        let session = frame_pool
            .CreateCaptureSession(&item)
            .context("Direct3D11CaptureFramePool::CreateCaptureSession failed")
            .map_err(CaptureError::Platform)?;
        let cursor_config = CursorCaptureConfig::default();
        // Best-effort session tuning:
        // - Disable cursor composition unless explicitly requested.
        // - Disable the capture border where allowed by the OS.
        let _ = session.SetIsCursorCaptureEnabled(cursor_config.capture_cursor);
        let _ = session.SetIsBorderRequired(false);
        let _ = session.SetDirtyRegionMode(GraphicsCaptureDirtyRegionMode::ReportAndRender);

        let signal = Arc::new(FrameSignal::default());
        let signal_for_frames = signal.clone();
        let frame_arrived_token = frame_pool
            .FrameArrived(
                &TypedEventHandler::<Direct3D11CaptureFramePool, IInspectable>::new(
                    move |sender, _| {
                        if let Some(pool) = sender {
                            let mut newest: Option<Direct3D11CaptureFrame> = None;
                            while let Ok(frame) = pool.TryGetNextFrame() {
                                if let Some(previous) = newest.replace(frame) {
                                    let _ = previous.Close();
                                }
                            }
                            if let Some(next_frame) = newest
                                && let Ok(mut state) = signal_for_frames.state.lock()
                            {
                                if let Some(previous) = state.latest.take() {
                                    let _ = previous.Close();
                                }
                                let time_ticks = next_frame
                                    .SystemRelativeTime()
                                    .map(|t| t.Duration)
                                    .unwrap_or(0);
                                state.latest_time_ticks = time_ticks;
                                state.latest = Some(next_frame);
                                state.sequence = state.sequence.wrapping_add(1);
                                signal_for_frames
                                    .sequence_hint
                                    .store(state.sequence, Ordering::Release);
                                signal_for_frames.cv.notify_one();
                            }
                        }
                        Ok(())
                    },
                ),
            )
            .context("Direct3D11CaptureFramePool::FrameArrived registration failed")
            .map_err(CaptureError::Platform)?;

        let signal_for_closed = signal.clone();
        let closed_token = item
            .Closed(
                &TypedEventHandler::<GraphicsCaptureItem, IInspectable>::new(move |_, _| {
                    if let Ok(mut state) = signal_for_closed.state.lock() {
                        state.closed = true;
                        signal_for_closed.closed_hint.store(true, Ordering::Release);
                        signal_for_closed.cv.notify_one();
                    }
                    Ok(())
                }),
            )
            .context("GraphicsCaptureItem::Closed registration failed")
            .map_err(CaptureError::Platform)?;

        session
            .StartCapture()
            .context("GraphicsCaptureSession::StartCapture failed")
            .map_err(CaptureError::Platform)?;

        let gpu_tonemapper = if is_hdr {
            Some(GpuTonemapper::new(&device)?)
        } else {
            None
        };
        // Create the F16 converter for non-HDR F16 sources.
        // Non-fatal if it fails -- we fall back to CPU conversion.
        let gpu_f16_converter = if pixel_format == DirectXPixelFormat::R16G16B16A16Float {
            GpuF16Converter::new(&device).ok()
        } else {
            None
        };

        Ok(Self {
            _com: com,
            device,
            context,
            winrt_device,
            item,
            frame_pool,
            session,
            frame_arrived_token,
            closed_token,
            signal,
            last_sequence: 0,
            pool_size,
            pixel_format,
            staging_slots: std::array::from_fn(|_| WgcStagingSlot::default()),
            next_write_slot: 0,
            pending_slot: None,
            region_slots: std::array::from_fn(|_| WgcStagingSlot::default()),
            region_next_write_slot: 0,
            region_pending_slot: None,
            cached_src_desc: None,
            last_present_time: 0,
            last_emitted_present_time: 0,
            stale_poll_spins: WGC_STALE_POLL_SPIN_INITIAL,
            stale_timeout_config,
            stale_frame_timeout: stale_timeout_config.initial,
            adaptive_spin_polls: WGC_QUERY_SPIN_INITIAL_POLLS,
            region_adaptive_spin_polls: WGC_QUERY_SPIN_INITIAL_POLLS,
            capture_mode: CaptureMode::Screenshot,
            hdr_to_sdr,
            gpu_tonemapper,
            gpu_f16_converter,
            region_blit: None,
            cursor_config,
            has_frame_history: false,
            source_dirty_rects_scratch: Vec::new(),
            region_dirty_rects_scratch: Vec::new(),
        })
    }

    fn try_take_latest_frame(&mut self) -> CaptureResult<Option<(Direct3D11CaptureFrame, i64)>> {
        if self.signal.closed_hint.load(Ordering::Acquire) {
            return Err(CaptureError::MonitorLost);
        }
        let observed_sequence = self.signal.sequence_hint.load(Ordering::Acquire);
        if observed_sequence == self.last_sequence {
            return Ok(None);
        }

        let mut state = self
            .signal
            .state
            .lock()
            .map_err(|_| poisoned_lock_error())?;
        if state.closed {
            self.signal.closed_hint.store(true, Ordering::Release);
            return Err(CaptureError::MonitorLost);
        }
        if state.sequence != self.last_sequence {
            self.last_sequence = state.sequence;
            if let Some(frame) = state.latest.take() {
                let time_ticks = state.latest_time_ticks;
                return Ok(Some((frame, time_ticks)));
            }
        }
        Ok(None)
    }

    fn poll_for_next_frame_stale(
        &mut self,
        timeout: Duration,
    ) -> CaptureResult<Option<(Direct3D11CaptureFrame, i64)>> {
        if let Some(frame) = self.try_take_latest_frame()? {
            return Ok(Some(frame));
        }
        if timeout.is_zero() {
            return Ok(None);
        }

        let deadline = Instant::now() + timeout;
        for _ in 0..self.stale_poll_spins {
            if let Some(frame) = self.try_take_latest_frame()? {
                self.stale_poll_spins = self
                    .stale_poll_spins
                    .saturating_sub(1)
                    .max(WGC_STALE_POLL_SPIN_MIN);
                return Ok(Some(frame));
            }
            if Instant::now() >= deadline {
                break;
            }
            std::hint::spin_loop();
        }

        while Instant::now() < deadline {
            std::thread::yield_now();
            if let Some(frame) = self.try_take_latest_frame()? {
                self.stale_poll_spins = self
                    .stale_poll_spins
                    .saturating_add(1)
                    .min(WGC_STALE_POLL_SPIN_MAX);
                return Ok(Some(frame));
            }
        }

        self.stale_poll_spins = self
            .stale_poll_spins
            .saturating_add(WGC_STALE_POLL_SPIN_INCREASE_STEP)
            .min(WGC_STALE_POLL_SPIN_MAX);
        Ok(None)
    }

    #[inline(always)]
    fn tighten_stale_timeout(&mut self) {
        self.stale_frame_timeout = duration_saturating_sub_clamped(
            self.stale_frame_timeout,
            self.stale_timeout_config.decrease_step,
            self.stale_timeout_config.min,
        );
    }

    #[inline(always)]
    fn relax_stale_timeout(&mut self) {
        self.stale_frame_timeout = duration_saturating_add_clamped(
            self.stale_frame_timeout,
            self.stale_timeout_config.increase_step,
            self.stale_timeout_config.max,
        );
    }

    #[inline(always)]
    fn reset_stale_timeout_window(&mut self) {
        self.stale_frame_timeout = self.stale_timeout_config.initial;
    }

    fn acquire_next_frame_or_stale(
        &mut self,
        allow_stale_return: bool,
    ) -> CaptureResult<Option<(Direct3D11CaptureFrame, i64)>> {
        if allow_stale_return {
            // Low-latency polling path: when we already have a staged frame to
            // reuse, avoid spending the stale timeout budget on every capture
            // call. A single atomic/mutex check is enough; if a new frame lands
            // right after this check we'll pick it up on the next call.
            if immediate_stale_return_enabled() && self.pending_slot.is_some() {
                if let Some(fresh) = self.try_take_latest_frame()? {
                    self.relax_stale_timeout();
                    return Ok(Some(fresh));
                }
                self.tighten_stale_timeout();
                return Ok(None);
            }

            if let Some(fresh) = self.poll_for_next_frame_stale(self.stale_frame_timeout)? {
                self.relax_stale_timeout();
                return Ok(Some(fresh));
            }

            // In recording mode, don't sit on a full frame interval when we
            // already have a previously converted slot. Wait briefly for a
            // just-about-to-arrive frame, then fall back to stale reuse.
            if self.pending_slot.is_some() {
                self.tighten_stale_timeout();
                return Ok(None);
            }
        }
        self.reset_stale_timeout_window();
        self.wait_for_next_frame(WGC_FRAME_TIMEOUT).map(Some)
    }

    fn acquire_next_frame_or_stale_region(
        &mut self,
        allow_stale_return: bool,
    ) -> CaptureResult<Option<(Direct3D11CaptureFrame, i64)>> {
        if allow_stale_return {
            // Mirror the full-frame low-latency path for region capture.
            if immediate_stale_return_enabled() && self.region_pending_slot.is_some() {
                if let Some(fresh) = self.try_take_latest_frame()? {
                    self.relax_stale_timeout();
                    return Ok(Some(fresh));
                }
                self.tighten_stale_timeout();
                return Ok(None);
            }

            let polled = self.poll_for_next_frame_stale(self.stale_frame_timeout)?;
            if polled.is_some() {
                self.relax_stale_timeout();
            } else {
                self.tighten_stale_timeout();
            }
            return Ok(polled);
        }

        self.reset_stale_timeout_window();
        self.wait_for_next_frame(WGC_FRAME_TIMEOUT).map(Some)
    }

    fn wait_for_next_frame(
        &mut self,
        timeout: Duration,
    ) -> CaptureResult<(Direct3D11CaptureFrame, i64)> {
        let deadline = Instant::now() + timeout;
        let mut state = self
            .signal
            .state
            .lock()
            .map_err(|_| poisoned_lock_error())?;

        loop {
            if state.closed {
                self.signal.closed_hint.store(true, Ordering::Release);
                return Err(CaptureError::MonitorLost);
            }

            if state.sequence != self.last_sequence {
                self.last_sequence = state.sequence;
                if let Some(frame) = state.latest.take() {
                    let time_ticks = state.latest_time_ticks;
                    return Ok((frame, time_ticks));
                }
            }

            let now = Instant::now();
            if now >= deadline {
                return Err(CaptureError::Timeout);
            }

            let wait_for = deadline.duration_since(now);
            let (new_state, _timeout) = self
                .signal
                .cv
                .wait_timeout(state, wait_for)
                .map_err(|_| poisoned_lock_error())?;
            state = new_state;
        }
    }

    fn recreate_pool_if_needed(&mut self, frame: &Direct3D11CaptureFrame) -> CaptureResult<()> {
        let content_size = frame.ContentSize().map_err(|error| {
            map_platform_error(error, "Direct3D11CaptureFrame::ContentSize failed")
        })?;
        if content_size.Width <= 0 || content_size.Height <= 0 {
            return Err(CaptureError::Timeout);
        }

        if content_size.Width != self.pool_size.Width
            || content_size.Height != self.pool_size.Height
        {
            self.frame_pool
                .Recreate(
                    &self.winrt_device,
                    self.pixel_format,
                    WGC_FRAME_POOL_BUFFERS,
                    content_size,
                )
                .map_err(|error| {
                    map_platform_error(error, "Direct3D11CaptureFramePool::Recreate failed")
                })?;
            self.pool_size = content_size;
            self.reset_staging_pipeline();
            self.cached_src_desc = None;
        }
        Ok(())
    }

    fn reset_staging_pipeline(&mut self) {
        for slot in &mut self.staging_slots {
            slot.invalidate();
        }
        self.pending_slot = None;
        self.next_write_slot = 0;
        self.reset_region_pipeline();
        self.has_frame_history = false;
        self.last_emitted_present_time = 0;
        self.stale_poll_spins = WGC_STALE_POLL_SPIN_INITIAL;
        self.stale_frame_timeout = self.stale_timeout_config.initial;
        self.adaptive_spin_polls = WGC_QUERY_SPIN_INITIAL_POLLS;
        self.source_dirty_rects_scratch.clear();
    }

    fn reset_region_pipeline(&mut self) {
        for slot in &mut self.region_slots {
            slot.invalidate();
        }
        self.region_pending_slot = None;
        self.region_next_write_slot = 0;
        self.region_adaptive_spin_polls = WGC_QUERY_SPIN_INITIAL_POLLS;
        self.region_blit = None;
        self.region_dirty_rects_scratch.clear();
    }

    fn ensure_region_pipeline_for_blit(&mut self, blit: CaptureBlitRegion) {
        if self.region_blit == Some(blit) {
            return;
        }
        self.reset_region_pipeline();
        self.region_blit = Some(blit);
    }

    fn ensure_staging_slot(
        &mut self,
        slot_idx: usize,
        desc: &D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<()> {
        let slot = &mut self.staging_slots[slot_idx];
        let needs_recreate = slot.source_desc.is_none_or(|cached| {
            cached.Width != desc.Width
                || cached.Height != desc.Height
                || cached.Format != desc.Format
                || cached.SampleDesc.Count != desc.SampleDesc.Count
                || cached.SampleDesc.Quality != desc.SampleDesc.Quality
        });

        if needs_recreate {
            slot.staging = None;
            let staging = surface::ensure_staging_texture(
                &self.device,
                &mut slot.staging,
                desc,
                StagingSampleDesc::Source,
                "failed to create WGC staging texture",
            )?;
            slot.staging_resource = Some(
                staging
                    .cast()
                    .context("failed to cast WGC staging texture to ID3D11Resource")
                    .map_err(CaptureError::Platform)?,
            );
            slot.source_desc = Some(*desc);
            slot.populated = false;
        }

        if slot.query.is_none() {
            let query_desc = D3D11_QUERY_DESC {
                Query: D3D11_QUERY_EVENT,
                ..Default::default()
            };
            let mut query: Option<ID3D11Query> = None;
            unsafe { self.device.CreateQuery(&query_desc, Some(&mut query)) }
                .context("CreateQuery for WGC staging slot failed")
                .map_err(CaptureError::Platform)?;
            slot.query = query;
        }
        Ok(())
    }

    fn region_desc_for_blit(
        source_desc: &D3D11_TEXTURE2D_DESC,
        blit: CaptureBlitRegion,
    ) -> D3D11_TEXTURE2D_DESC {
        let mut region_desc = *source_desc;
        region_desc.Width = blit.width;
        region_desc.Height = blit.height;
        region_desc.MipLevels = 1;
        region_desc.ArraySize = 1;
        region_desc.SampleDesc.Count = 1;
        region_desc.SampleDesc.Quality = 0;
        region_desc
    }

    fn ensure_region_slot(
        &mut self,
        slot_idx: usize,
        desc: &D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<()> {
        let slot = &mut self.region_slots[slot_idx];
        let needs_recreate = slot.source_desc.is_none_or(|cached| {
            cached.Width != desc.Width
                || cached.Height != desc.Height
                || cached.Format != desc.Format
                || cached.SampleDesc.Count != desc.SampleDesc.Count
                || cached.SampleDesc.Quality != desc.SampleDesc.Quality
        });

        if needs_recreate {
            slot.staging = None;
            let staging = surface::ensure_staging_texture(
                &self.device,
                &mut slot.staging,
                desc,
                StagingSampleDesc::SingleSample,
                "failed to create WGC region staging texture",
            )?;
            slot.staging_resource = Some(
                staging
                    .cast::<ID3D11Resource>()
                    .context("failed to cast WGC region staging texture to ID3D11Resource")
                    .map_err(CaptureError::Platform)?,
            );
            slot.source_desc = Some(*desc);
            slot.populated = false;
        }

        if slot.query.is_none() {
            let query_desc = D3D11_QUERY_DESC {
                Query: D3D11_QUERY_EVENT,
                ..Default::default()
            };
            let mut query: Option<ID3D11Query> = None;
            unsafe { self.device.CreateQuery(&query_desc, Some(&mut query)) }
                .context("CreateQuery for WGC region staging slot failed")
                .map_err(CaptureError::Platform)?;
            slot.query = query;
        }
        Ok(())
    }

    fn copy_region_source_to_slot(
        &mut self,
        slot_idx: usize,
        source_resource: &ID3D11Resource,
        blit: CaptureBlitRegion,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        if !borrowed_slot_resource_enabled() {
            let staging_resource = self.region_slots[slot_idx]
                .staging_resource
                .as_ref()
                .ok_or_else(|| {
                    CaptureError::Platform(anyhow::anyhow!(
                        "failed to resolve WGC region staging resource for slot {}",
                        slot_idx
                    ))
                })?
                .clone();
            let query = self.region_slots[slot_idx].query.clone();

            let mut used_dirty_copy = false;
            if can_use_dirty_gpu_copy && region_dirty_gpu_copy_enabled() {
                let use_dirty_copy = {
                    let slot = &self.region_slots[slot_idx];
                    slot.dirty_copy_preferred
                };

                if use_dirty_copy {
                    let slot = &self.region_slots[slot_idx];
                    for rect in &slot.dirty_rects {
                        let source_left = blit
                            .src_x
                            .checked_add(rect.x)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_top = blit
                            .src_y
                            .checked_add(rect.y)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_right = source_left
                            .checked_add(rect.width)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_bottom = source_top
                            .checked_add(rect.height)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_box = D3D11_BOX {
                            left: source_left,
                            top: source_top,
                            front: 0,
                            right: source_right,
                            bottom: source_bottom,
                            back: 1,
                        };
                        unsafe {
                            self.context.CopySubresourceRegion(
                                &staging_resource,
                                0,
                                rect.x,
                                rect.y,
                                0,
                                source_resource,
                                0,
                                Some(&source_box),
                            );
                        }
                    }
                    used_dirty_copy = true;
                }
            }

            if !used_dirty_copy {
                let src_right = blit
                    .src_x
                    .checked_add(blit.width)
                    .ok_or(CaptureError::BufferOverflow)?;
                let src_bottom = blit
                    .src_y
                    .checked_add(blit.height)
                    .ok_or(CaptureError::BufferOverflow)?;
                let source_box = D3D11_BOX {
                    left: blit.src_x,
                    top: blit.src_y,
                    front: 0,
                    right: src_right,
                    bottom: src_bottom,
                    back: 1,
                };

                unsafe {
                    self.context.CopySubresourceRegion(
                        &staging_resource,
                        0,
                        0,
                        0,
                        0,
                        source_resource,
                        0,
                        Some(&source_box),
                    );
                }
            }

            if let Some(query) = query.as_ref() {
                unsafe {
                    self.context.End(query);
                }
            }
            self.region_slots[slot_idx].populated = true;
            return Ok(());
        }

        let mut used_dirty_copy = false;
        {
            let slot = &self.region_slots[slot_idx];
            let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "failed to resolve WGC region staging resource for slot {}",
                    slot_idx
                ))
            })?;

            if can_use_dirty_gpu_copy
                && region_dirty_gpu_copy_enabled()
                && slot.dirty_copy_preferred
            {
                for rect in &slot.dirty_rects {
                    let source_left = blit
                        .src_x
                        .checked_add(rect.x)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let source_top = blit
                        .src_y
                        .checked_add(rect.y)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let source_right = source_left
                        .checked_add(rect.width)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let source_bottom = source_top
                        .checked_add(rect.height)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let source_box = D3D11_BOX {
                        left: source_left,
                        top: source_top,
                        front: 0,
                        right: source_right,
                        bottom: source_bottom,
                        back: 1,
                    };
                    unsafe {
                        self.context.CopySubresourceRegion(
                            staging_resource,
                            0,
                            rect.x,
                            rect.y,
                            0,
                            source_resource,
                            0,
                            Some(&source_box),
                        );
                    }
                }
                used_dirty_copy = true;
            }

            if !used_dirty_copy {
                let src_right = blit
                    .src_x
                    .checked_add(blit.width)
                    .ok_or(CaptureError::BufferOverflow)?;
                let src_bottom = blit
                    .src_y
                    .checked_add(blit.height)
                    .ok_or(CaptureError::BufferOverflow)?;
                let source_box = D3D11_BOX {
                    left: blit.src_x,
                    top: blit.src_y,
                    front: 0,
                    right: src_right,
                    bottom: src_bottom,
                    back: 1,
                };

                unsafe {
                    self.context.CopySubresourceRegion(
                        staging_resource,
                        0,
                        0,
                        0,
                        0,
                        source_resource,
                        0,
                        Some(&source_box),
                    );
                }
            }

            if let Some(query) = slot.query.as_ref() {
                unsafe {
                    self.context.End(query);
                }
            }
        }
        self.region_slots[slot_idx].populated = true;
        Ok(())
    }

    fn copy_source_to_slot(
        &mut self,
        slot_idx: usize,
        source_resource: &ID3D11Resource,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        if !borrowed_slot_resource_enabled() {
            let staging_resource = self.staging_slots[slot_idx]
                .staging_resource
                .as_ref()
                .ok_or_else(|| {
                    CaptureError::Platform(anyhow::anyhow!(
                        "failed to resolve WGC staging resource for slot {}",
                        slot_idx
                    ))
                })?
                .clone();
            let query = self.staging_slots[slot_idx].query.clone();

            let mut used_dirty_copy = false;
            if can_use_dirty_gpu_copy {
                let use_dirty_copy = {
                    let slot = &self.staging_slots[slot_idx];
                    slot.dirty_copy_preferred
                };

                if use_dirty_copy {
                    let slot = &self.staging_slots[slot_idx];
                    for rect in &slot.dirty_rects {
                        let right = rect
                            .x
                            .checked_add(rect.width)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let bottom = rect
                            .y
                            .checked_add(rect.height)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_box = D3D11_BOX {
                            left: rect.x,
                            top: rect.y,
                            front: 0,
                            right,
                            bottom,
                            back: 1,
                        };
                        unsafe {
                            self.context.CopySubresourceRegion(
                                &staging_resource,
                                0,
                                rect.x,
                                rect.y,
                                0,
                                source_resource,
                                0,
                                Some(&source_box),
                            );
                        }
                    }
                    used_dirty_copy = true;
                }
            }

            if !used_dirty_copy {
                unsafe {
                    self.context
                        .CopyResource(&staging_resource, source_resource);
                }
            }

            if let Some(query) = query.as_ref() {
                unsafe {
                    self.context.End(query);
                }
            }
            self.staging_slots[slot_idx].populated = true;
            return Ok(());
        }

        let mut used_dirty_copy = false;
        {
            let slot = &self.staging_slots[slot_idx];
            let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "failed to resolve WGC staging resource for slot {}",
                    slot_idx
                ))
            })?;

            if can_use_dirty_gpu_copy && slot.dirty_copy_preferred {
                for rect in &slot.dirty_rects {
                    let right = rect
                        .x
                        .checked_add(rect.width)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let bottom = rect
                        .y
                        .checked_add(rect.height)
                        .ok_or(CaptureError::BufferOverflow)?;
                    let source_box = D3D11_BOX {
                        left: rect.x,
                        top: rect.y,
                        front: 0,
                        right,
                        bottom,
                        back: 1,
                    };
                    unsafe {
                        self.context.CopySubresourceRegion(
                            staging_resource,
                            0,
                            rect.x,
                            rect.y,
                            0,
                            source_resource,
                            0,
                            Some(&source_box),
                        );
                    }
                }
                used_dirty_copy = true;
            }

            if !used_dirty_copy {
                unsafe {
                    self.context.CopyResource(staging_resource, source_resource);
                }
            }

            if let Some(query) = slot.query.as_ref() {
                unsafe {
                    self.context.End(query);
                }
            }
        }
        self.staging_slots[slot_idx].populated = true;
        Ok(())
    }

    fn query_signaled(&self, query: &ID3D11Query, flags: u32) -> bool {
        let mut data = 0u32;
        let status = unsafe {
            self.context.GetData(
                query,
                Some(&mut data as *mut u32 as *mut _),
                std::mem::size_of::<u32>() as u32,
                flags,
            )
        };
        status.is_ok() && data != 0
    }

    fn slot_query_completed(&self, slot_idx: usize) -> bool {
        const DO_NOT_FLUSH: u32 = 0x1;
        let Some(query) = self.staging_slots[slot_idx].query.as_ref() else {
            return false;
        };
        self.query_signaled(query, DO_NOT_FLUSH)
    }

    fn maybe_flush_after_submit(&self, write_slot: usize, read_slot: usize) {
        if write_slot == read_slot || !self.slot_query_completed(read_slot) {
            unsafe {
                self.context.Flush();
            }
        }
    }

    fn wait_for_slot_copy(&mut self, slot_idx: usize) {
        const DO_NOT_FLUSH: u32 = 0x1;
        let Some(query) = self.staging_slots[slot_idx].query.as_ref() else {
            return;
        };

        let mut completed_in_spin = false;
        for _ in 0..self.adaptive_spin_polls {
            if self.query_signaled(query, DO_NOT_FLUSH) {
                completed_in_spin = true;
                break;
            }
            std::hint::spin_loop();
        }

        if completed_in_spin {
            self.adaptive_spin_polls = self
                .adaptive_spin_polls
                .saturating_sub(1)
                .max(WGC_QUERY_SPIN_MIN_POLLS);
        } else {
            self.adaptive_spin_polls = self
                .adaptive_spin_polls
                .saturating_add(WGC_QUERY_SPIN_INCREASE_STEP)
                .min(WGC_QUERY_SPIN_MAX_POLLS);
        }
    }

    fn region_slot_query_completed(&self, slot_idx: usize) -> bool {
        const DO_NOT_FLUSH: u32 = 0x1;
        let Some(query) = self.region_slots[slot_idx].query.as_ref() else {
            return false;
        };
        self.query_signaled(query, DO_NOT_FLUSH)
    }

    fn maybe_flush_region_after_submit(&self, write_slot: usize, read_slot: usize) {
        if write_slot == read_slot || !self.region_slot_query_completed(read_slot) {
            unsafe {
                self.context.Flush();
            }
        }
    }

    fn wait_for_region_slot_copy(&mut self, slot_idx: usize) {
        const DO_NOT_FLUSH: u32 = 0x1;
        let Some(query) = self.region_slots[slot_idx].query.as_ref() else {
            return;
        };

        let mut completed_in_spin = false;
        for _ in 0..self.region_adaptive_spin_polls {
            if self.query_signaled(query, DO_NOT_FLUSH) {
                completed_in_spin = true;
                break;
            }
            std::hint::spin_loop();
        }

        if completed_in_spin {
            self.region_adaptive_spin_polls = self
                .region_adaptive_spin_polls
                .saturating_sub(1)
                .max(WGC_QUERY_SPIN_MIN_POLLS);
        } else {
            self.region_adaptive_spin_polls = self
                .region_adaptive_spin_polls
                .saturating_add(WGC_QUERY_SPIN_INCREASE_STEP)
                .min(WGC_QUERY_SPIN_MAX_POLLS);
        }
    }

    fn read_region_slot_into_output(
        &mut self,
        slot_idx: usize,
        out: &mut Frame,
        destination_has_history: bool,
        blit: CaptureBlitRegion,
    ) -> CaptureResult<CaptureSampleMetadata> {
        if !self.region_slots[slot_idx].populated {
            return Err(CaptureError::Timeout);
        }

        let slot = &self.region_slots[slot_idx];
        let source_desc = slot.source_desc.ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "WGC region slot is populated but missing source descriptor"
            ))
        })?;
        let capture_time = slot.capture_time.unwrap_or_else(Instant::now);
        let present_time_ticks = slot.present_time_ticks;
        let source_duplicate = slot.is_duplicate;

        let emitted_duplicate =
            present_time_ticks != 0 && present_time_ticks == self.last_emitted_present_time;
        let sample = CaptureSampleMetadata {
            capture_time: Some(capture_time),
            present_time_qpc: if present_time_ticks != 0 {
                Some(present_time_ticks)
            } else {
                None
            },
            is_duplicate: source_duplicate || emitted_duplicate,
        };

        if destination_has_history && sample.is_duplicate {
            if present_time_ticks != 0 {
                self.last_emitted_present_time = present_time_ticks;
            }
            return Ok(sample);
        }

        self.wait_for_region_slot_copy(slot_idx);

        let map_result = (|| -> CaptureResult<()> {
            let slot = &self.region_slots[slot_idx];
            let staging = slot.staging.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "WGC region slot is populated but missing staging texture"
                ))
            })?;
            let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "WGC region slot is populated but missing staging resource"
                ))
            })?;

            let use_dirty_copy = destination_has_history
                && slot.dirty_copy_preferred
                && !slot.dirty_rects.is_empty();

            if use_dirty_copy {
                match surface::map_staging_dirty_rects_to_frame_with_offset(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    &slot.dirty_rects,
                    blit.dst_x,
                    blit.dst_y,
                    true,
                    slot.hdr_to_sdr,
                    "failed to map WGC region staging texture (dirty regions)",
                ) {
                    Ok(converted) if converted > 0 => Ok(()),
                    Ok(_) | Err(_) => surface::map_staging_rect_to_frame(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &source_desc,
                        out,
                        CaptureBlitRegion {
                            src_x: 0,
                            src_y: 0,
                            width: source_desc.Width,
                            height: source_desc.Height,
                            dst_x: blit.dst_x,
                            dst_y: blit.dst_y,
                        },
                        slot.hdr_to_sdr,
                        "failed to map WGC region staging texture",
                    ),
                }
            } else {
                surface::map_staging_rect_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    CaptureBlitRegion {
                        src_x: 0,
                        src_y: 0,
                        width: source_desc.Width,
                        height: source_desc.Height,
                        dst_x: blit.dst_x,
                        dst_y: blit.dst_y,
                    },
                    slot.hdr_to_sdr,
                    "failed to map WGC region staging texture",
                )
            }
        })();

        map_result?;

        if present_time_ticks != 0 {
            self.last_emitted_present_time = present_time_ticks;
        }
        Ok(sample)
    }

    fn try_short_circuit_region_duplicate(
        &mut self,
        capture_time: Instant,
        present_time_ticks: i64,
    ) -> Option<CaptureSampleMetadata> {
        let Some(slot_idx) = self.region_pending_slot else {
            return None;
        };
        if !self.region_slots[slot_idx].populated {
            return None;
        }

        if present_time_ticks != 0 {
            self.last_emitted_present_time = present_time_ticks;
        }

        Some(CaptureSampleMetadata {
            capture_time: Some(capture_time),
            present_time_qpc: if present_time_ticks != 0 {
                Some(present_time_ticks)
            } else {
                None
            },
            is_duplicate: true,
        })
    }

    fn read_slot_into_output(
        &mut self,
        slot_idx: usize,
        out: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<()> {
        if !self.staging_slots[slot_idx].populated {
            return Err(CaptureError::Timeout);
        }

        let slot = &self.staging_slots[slot_idx];
        let source_desc = slot.source_desc.ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "WGC staging slot is populated but missing source descriptor"
            ))
        })?;
        let capture_time = slot.capture_time.unwrap_or_else(Instant::now);
        let present_time_ticks = slot.present_time_ticks;
        let source_duplicate = slot.is_duplicate;
        let dirty_mode_available = slot.dirty_mode_available;

        let emitted_duplicate =
            present_time_ticks != 0 && present_time_ticks == self.last_emitted_present_time;
        let is_duplicate = source_duplicate || emitted_duplicate;

        out.metadata.capture_time = Some(capture_time);
        out.metadata.present_time_qpc = if present_time_ticks != 0 {
            Some(present_time_ticks)
        } else {
            None
        };
        out.metadata.is_duplicate = is_duplicate;

        let out_matches_source =
            out.width() == source_desc.Width && out.height() == source_desc.Height;
        if self.has_frame_history && destination_has_history && is_duplicate && out_matches_source {
            out.metadata.dirty_rects.clear();
            if present_time_ticks != 0 {
                self.last_emitted_present_time = present_time_ticks;
            }
            return Ok(());
        }

        self.wait_for_slot_copy(slot_idx);

        let map_result = (|| -> CaptureResult<()> {
            let slot = &self.staging_slots[slot_idx];
            let staging = slot.staging.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "WGC staging slot is populated but missing staging texture"
                ))
            })?;
            let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "WGC staging slot is populated but missing staging resource"
                ))
            })?;
            let use_dirty_copy = self.has_frame_history
                && out_matches_source
                && slot.dirty_copy_preferred
                && !slot.dirty_rects.is_empty();

            if use_dirty_copy {
                match surface::map_staging_dirty_rects_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    &slot.dirty_rects,
                    true,
                    slot.hdr_to_sdr,
                    "failed to map WGC staging texture (dirty regions)",
                ) {
                    Ok(converted) if converted > 0 => Ok(()),
                    Ok(_) | Err(_) => surface::map_staging_to_frame(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &source_desc,
                        out,
                        slot.hdr_to_sdr,
                        "failed to map WGC staging texture",
                    ),
                }
            } else {
                surface::map_staging_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    slot.hdr_to_sdr,
                    "failed to map WGC staging texture",
                )
            }
        })();

        map_result?;

        if dirty_mode_available {
            out.metadata.dirty_rects.clear();
            out.metadata
                .dirty_rects
                .extend_from_slice(&self.staging_slots[slot_idx].dirty_rects);
        } else {
            out.metadata.dirty_rects.clear();
        }

        if present_time_ticks != 0 {
            self.last_emitted_present_time = present_time_ticks;
        }
        Ok(())
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        out: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<CaptureSampleMetadata> {
        if blit.width == 0 || blit.height == 0 {
            return Err(CaptureError::InvalidConfig(
                "capture region dimensions must be non-zero".into(),
            ));
        }

        // Region capture uses a dedicated sub-rect readback path. Reset the
        // full-frame staging pipeline state so switching between full capture
        // and region capture cannot consume stale pending slots.
        self.pending_slot = None;
        self.next_write_slot = 0;
        self.ensure_region_pipeline_for_blit(blit);

        let allow_stale_return =
            self.capture_mode == CaptureMode::ScreenRecording && destination_has_history;
        let capture_time = Instant::now();

        let maybe_capture = self.acquire_next_frame_or_stale_region(allow_stale_return)?;
        let (capture_frame, time_ticks) = if let Some(capture) = maybe_capture {
            capture
        } else if let Some(slot_idx) = self.region_pending_slot {
            match self.read_region_slot_into_output(slot_idx, out, destination_has_history, blit) {
                Ok(mut sample) => {
                    sample.capture_time = Some(capture_time);
                    sample.is_duplicate = true;
                    self.has_frame_history = true;
                    return Ok(sample);
                }
                Err(_) => {
                    self.reset_region_pipeline();
                    self.has_frame_history = false;
                    return Ok(CaptureSampleMetadata {
                        capture_time: Some(capture_time),
                        present_time_qpc: None,
                        is_duplicate: true,
                    });
                }
            }
        } else {
            return Ok(CaptureSampleMetadata {
                capture_time: Some(capture_time),
                present_time_qpc: None,
                is_duplicate: true,
            });
        };

        let previous_present_time = self.last_present_time;
        let source_is_duplicate = time_ticks != 0 && time_ticks == previous_present_time;
        if time_ticks != 0 {
            self.last_present_time = time_ticks;
        }

        if should_short_circuit_region_duplicate(
            region_duplicate_short_circuit_enabled(),
            self.capture_mode,
            destination_has_history,
            source_is_duplicate,
            self.region_pending_slot.is_some(),
        ) {
            if let Some(sample) = self.try_short_circuit_region_duplicate(capture_time, time_ticks)
            {
                let _ = capture_frame.Close();
                self.has_frame_history = true;
                return Ok(sample);
            }
        }

        let mut region_dirty_rects = std::mem::take(&mut self.region_dirty_rects_scratch);
        let capture_result = (|| -> CaptureResult<CaptureSampleMetadata> {
            self.recreate_pool_if_needed(&capture_frame)?;

            let frame_surface = capture_frame.Surface().map_err(|error| {
                map_platform_error(error, "Direct3D11CaptureFrame::Surface failed")
            })?;
            let frame_dxgi_interface: IDirect3DDxgiInterfaceAccess = frame_surface
                .cast()
                .context("failed to cast frame surface to IDirect3DDxgiInterfaceAccess")
                .map_err(CaptureError::Platform)?;
            let frame_texture: ID3D11Texture2D = unsafe { frame_dxgi_interface.GetInterface() }
                .map_err(|error| {
                    map_platform_error(error, "IDirect3DDxgiInterfaceAccess::GetInterface failed")
                })?;

            let src_desc = match self.cached_src_desc {
                Some(desc) => desc,
                None => {
                    let mut desc = D3D11_TEXTURE2D_DESC::default();
                    unsafe { frame_texture.GetDesc(&mut desc) };
                    self.cached_src_desc = Some(desc);
                    desc
                }
            };

            let (effective_source, effective_desc, effective_hdr) =
                if src_desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT {
                    if let (Some(params), Some(tonemapper)) =
                        (self.hdr_to_sdr, self.gpu_tonemapper.as_mut())
                    {
                        let output = tonemapper.tonemap(
                            &self.device,
                            &self.context,
                            &frame_texture,
                            &src_desc,
                            params.sanitized(),
                        )?;
                        let mut out_desc = D3D11_TEXTURE2D_DESC::default();
                        unsafe { output.GetDesc(&mut out_desc) };
                        (output.clone(), out_desc, None)
                    } else if let Some(converter) = self.gpu_f16_converter.as_mut() {
                        let output = converter.convert(
                            &self.device,
                            &self.context,
                            &frame_texture,
                            &src_desc,
                        )?;
                        let mut out_desc = D3D11_TEXTURE2D_DESC::default();
                        unsafe { output.GetDesc(&mut out_desc) };
                        (output.clone(), out_desc, None)
                    } else {
                        (frame_texture.clone(), src_desc, self.hdr_to_sdr)
                    }
                } else {
                    (frame_texture.clone(), src_desc, self.hdr_to_sdr)
                };

            let src_right = blit
                .src_x
                .checked_add(blit.width)
                .ok_or(CaptureError::BufferOverflow)?;
            let src_bottom = blit
                .src_y
                .checked_add(blit.height)
                .ok_or(CaptureError::BufferOverflow)?;
            if src_right > effective_desc.Width || src_bottom > effective_desc.Height {
                return Err(CaptureError::BufferOverflow);
            }

            let region_desc = Self::region_desc_for_blit(&effective_desc, blit);
            let (region_dirty_available, region_unchanged) =
                if source_is_duplicate && duplicate_dirty_fastpath_enabled() {
                    region_dirty_rects.clear();
                    (true, true)
                } else {
                    let available = extract_region_dirty_rects(
                        &capture_frame,
                        effective_desc.Width,
                        effective_desc.Height,
                        blit,
                        &mut region_dirty_rects,
                    );
                    if !available {
                        region_dirty_rects.clear();
                    }
                    (available, available && region_dirty_rects.is_empty())
                };

            if self.capture_mode != CaptureMode::ScreenRecording
                && self.has_frame_history
                && destination_has_history
                && (source_is_duplicate || region_unchanged)
            {
                return Ok(CaptureSampleMetadata {
                    capture_time: Some(capture_time),
                    present_time_qpc: if time_ticks != 0 {
                        Some(time_ticks)
                    } else {
                        None
                    },
                    is_duplicate: true,
                });
            }

            let recording_mode = self.capture_mode == CaptureMode::ScreenRecording;
            let write_slot = if recording_mode {
                self.region_next_write_slot % WGC_STAGING_SLOTS
            } else {
                0
            };
            let read_slot = if recording_mode {
                self.region_pending_slot.unwrap_or(write_slot)
            } else {
                write_slot
            };

            let skip_submit_copy = recording_mode
                && self.region_pending_slot.is_some()
                && (source_is_duplicate || region_unchanged);

            let read_slot = if skip_submit_copy {
                let slot_idx = self.region_pending_slot.unwrap_or(read_slot);
                let slot = &mut self.region_slots[slot_idx];
                slot.capture_time = Some(capture_time);
                slot.present_time_ticks = time_ticks;
                slot.is_duplicate = true;
                slot.dirty_mode_available = region_dirty_available;
                slot.dirty_copy_preferred = false;
                slot.dirty_rects.clear();
                slot.populated = true;
                slot_idx
            } else {
                self.ensure_region_slot(write_slot, &region_desc)?;
                let can_use_dirty_gpu_copy = {
                    let slot = &self.region_slots[write_slot];
                    slot.populated
                        && slot.present_time_ticks != 0
                        && slot.present_time_ticks == previous_present_time
                };
                {
                    let slot = &mut self.region_slots[write_slot];
                    slot.capture_time = Some(capture_time);
                    slot.present_time_ticks = time_ticks;
                    slot.is_duplicate = source_is_duplicate || region_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(region_desc);
                    slot.dirty_mode_available = region_dirty_available;
                    slot.dirty_rects.clear();
                    slot.dirty_rects.extend_from_slice(&region_dirty_rects);
                    slot.dirty_copy_preferred = region_dirty_available
                        && should_use_dirty_copy(
                            &slot.dirty_rects,
                            region_desc.Width,
                            region_desc.Height,
                        );
                    slot.populated = true;
                }

                with_texture_resource(
                    &effective_source,
                    "failed to cast WGC region source texture to ID3D11Resource",
                    |source_resource| {
                        self.copy_region_source_to_slot(
                            write_slot,
                            source_resource,
                            blit,
                            can_use_dirty_gpu_copy,
                        )
                    },
                )?;
                self.maybe_flush_region_after_submit(write_slot, read_slot);
                read_slot
            };

            let sample =
                self.read_region_slot_into_output(read_slot, out, destination_has_history, blit)?;

            if recording_mode {
                if !skip_submit_copy {
                    self.region_pending_slot = Some(write_slot);
                    self.region_next_write_slot = (write_slot + 1) % WGC_STAGING_SLOTS;
                }
            } else {
                self.region_pending_slot = None;
                self.region_next_write_slot = 0;
            }

            Ok(sample)
        })();

        region_dirty_rects.clear();
        self.region_dirty_rects_scratch = region_dirty_rects;

        let _ = capture_frame.Close();
        match capture_result {
            Ok(sample) => {
                self.has_frame_history = true;
                Ok(sample)
            }
            Err(err) => {
                self.reset_region_pipeline();
                self.has_frame_history = false;
                Err(err)
            }
        }
    }

    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.region_pending_slot = None;
        self.region_next_write_slot = 0;
        self.region_blit = None;

        let mut out = reuse.unwrap_or_else(Frame::empty);
        let destination_has_history =
            out.metadata.capture_time.is_some() && !out.as_rgba_bytes().is_empty();
        out.reset_metadata();

        let allow_stale_return = self.capture_mode == CaptureMode::ScreenRecording
            && out.width() > 0
            && out.height() > 0;
        let capture_time = Instant::now();

        let maybe_capture = self.acquire_next_frame_or_stale(allow_stale_return)?;
        let (capture_frame, time_ticks) = if let Some(capture) = maybe_capture {
            capture
        } else if let Some(slot_idx) = self.pending_slot {
            if self
                .read_slot_into_output(slot_idx, &mut out, destination_has_history)
                .is_ok()
            {
                out.metadata.capture_time = Some(capture_time);
                out.metadata.is_duplicate = true;
                self.has_frame_history = true;
                return Ok(out);
            }
            self.reset_staging_pipeline();
            out.metadata.capture_time = Some(capture_time);
            out.metadata.is_duplicate = true;
            return Ok(out);
        } else {
            out.metadata.capture_time = Some(capture_time);
            out.metadata.is_duplicate = true;
            return Ok(out);
        };

        let previous_present_time = self.last_present_time;
        let source_is_duplicate = time_ticks != 0 && time_ticks == previous_present_time;
        if time_ticks != 0 {
            self.last_present_time = time_ticks;
        }

        let mut source_dirty_rects = std::mem::take(&mut self.source_dirty_rects_scratch);
        let capture_result = (|| -> CaptureResult<()> {
            self.recreate_pool_if_needed(&capture_frame)?;

            let frame_surface = capture_frame.Surface().map_err(|error| {
                map_platform_error(error, "Direct3D11CaptureFrame::Surface failed")
            })?;
            let frame_dxgi_interface: IDirect3DDxgiInterfaceAccess = frame_surface
                .cast()
                .context("failed to cast frame surface to IDirect3DDxgiInterfaceAccess")
                .map_err(CaptureError::Platform)?;
            let frame_texture: ID3D11Texture2D = unsafe { frame_dxgi_interface.GetInterface() }
                .map_err(|error| {
                    map_platform_error(error, "IDirect3DDxgiInterfaceAccess::GetInterface failed")
                })?;

            let src_desc = match self.cached_src_desc {
                Some(desc) => desc,
                None => {
                    let mut desc = D3D11_TEXTURE2D_DESC::default();
                    unsafe { frame_texture.GetDesc(&mut desc) };
                    self.cached_src_desc = Some(desc);
                    desc
                }
            };

            // GPU tonemap path: if HDR and source is F16, run compute shader on GPU.
            let (effective_source, effective_desc, effective_hdr) =
                if src_desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT {
                    if let (Some(params), Some(tonemapper)) =
                        (self.hdr_to_sdr, self.gpu_tonemapper.as_mut())
                    {
                        let output = tonemapper.tonemap(
                            &self.device,
                            &self.context,
                            &frame_texture,
                            &src_desc,
                            params.sanitized(),
                        )?;
                        let mut out_desc = D3D11_TEXTURE2D_DESC::default();
                        unsafe { output.GetDesc(&mut out_desc) };
                        (output.clone(), out_desc, None)
                    } else if let Some(converter) = self.gpu_f16_converter.as_mut() {
                        // No HDR tonemap needed, but source is F16 -- convert
                        // to RGBA8 sRGB on the GPU to avoid the expensive
                        // CPU-side F16->sRGB SIMD path.
                        let output = converter.convert(
                            &self.device,
                            &self.context,
                            &frame_texture,
                            &src_desc,
                        )?;
                        let mut out_desc = D3D11_TEXTURE2D_DESC::default();
                        unsafe { output.GetDesc(&mut out_desc) };
                        (output.clone(), out_desc, None)
                    } else {
                        (frame_texture.clone(), src_desc, self.hdr_to_sdr)
                    }
                } else {
                    (frame_texture.clone(), src_desc, self.hdr_to_sdr)
                };

            let (source_dirty_available, source_unchanged) =
                if source_is_duplicate && duplicate_dirty_fastpath_enabled() {
                    source_dirty_rects.clear();
                    (true, true)
                } else {
                    let source_dirty_mode = extract_dirty_rects(
                        &capture_frame,
                        effective_desc.Width,
                        effective_desc.Height,
                        &mut source_dirty_rects,
                    );
                    let source_dirty_available = source_dirty_mode.is_some();
                    if !source_dirty_available {
                        source_dirty_rects.clear();
                    }
                    (
                        source_dirty_available,
                        source_dirty_available && source_dirty_rects.is_empty(),
                    )
                };

            let emitted_duplicate = time_ticks != 0 && time_ticks == self.last_emitted_present_time;
            if self.capture_mode != CaptureMode::ScreenRecording
                && (source_is_duplicate || source_unchanged)
                && self.has_frame_history
                && destination_has_history
                && out.width() == effective_desc.Width
                && out.height() == effective_desc.Height
            {
                out.metadata.capture_time = Some(capture_time);
                out.metadata.present_time_qpc = if time_ticks != 0 {
                    Some(time_ticks)
                } else {
                    None
                };
                out.metadata.is_duplicate = source_is_duplicate || emitted_duplicate;
                out.metadata.dirty_rects.clear();
                if time_ticks != 0 {
                    self.last_emitted_present_time = time_ticks;
                }
                return Ok(());
            }

            let write_slot = if self.capture_mode == CaptureMode::ScreenRecording {
                self.next_write_slot % WGC_STAGING_SLOTS
            } else {
                0
            };
            let read_slot = if self.capture_mode == CaptureMode::ScreenRecording {
                self.pending_slot.unwrap_or(write_slot)
            } else {
                write_slot
            };

            let skip_submit_copy = self.capture_mode == CaptureMode::ScreenRecording
                && self.pending_slot.is_some()
                && (source_is_duplicate || source_unchanged);

            let read_slot = if skip_submit_copy {
                let slot_idx = self.pending_slot.unwrap_or(read_slot);
                let slot = &mut self.staging_slots[slot_idx];
                slot.capture_time = Some(capture_time);
                slot.present_time_ticks = time_ticks;
                slot.is_duplicate = true;
                slot.dirty_mode_available = source_dirty_available;
                slot.dirty_copy_preferred = false;
                slot.dirty_rects.clear();
                slot.populated = true;
                slot_idx
            } else {
                self.ensure_staging_slot(write_slot, &effective_desc)?;
                let can_use_dirty_gpu_copy = {
                    let slot = &self.staging_slots[write_slot];
                    slot.populated
                        && slot.present_time_ticks != 0
                        && slot.present_time_ticks == previous_present_time
                };
                {
                    let slot = &mut self.staging_slots[write_slot];
                    slot.dirty_mode_available = source_dirty_available;
                    slot.dirty_rects.clear();
                    slot.dirty_rects.extend_from_slice(&source_dirty_rects);
                    slot.dirty_copy_preferred = source_dirty_available
                        && should_use_dirty_copy(
                            &slot.dirty_rects,
                            effective_desc.Width,
                            effective_desc.Height,
                        );
                    slot.capture_time = Some(capture_time);
                    slot.present_time_ticks = time_ticks;
                    slot.is_duplicate = source_is_duplicate || source_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(effective_desc);
                    slot.populated = true;
                }

                with_texture_resource(
                    &effective_source,
                    "failed to cast WGC frame texture to ID3D11Resource",
                    |source_resource| {
                        self.copy_source_to_slot(
                            write_slot,
                            source_resource,
                            can_use_dirty_gpu_copy,
                        )
                    },
                )?;
                self.maybe_flush_after_submit(write_slot, read_slot);
                read_slot
            };

            self.read_slot_into_output(read_slot, &mut out, destination_has_history)?;

            if self.capture_mode == CaptureMode::ScreenRecording {
                if !skip_submit_copy {
                    self.pending_slot = Some(write_slot);
                    self.next_write_slot = (write_slot + 1) % WGC_STAGING_SLOTS;
                }
            } else {
                self.pending_slot = None;
                self.next_write_slot = 0;
            }

            Ok(())
        })();

        source_dirty_rects.clear();
        self.source_dirty_rects_scratch = source_dirty_rects;

        let _ = capture_frame.Close();
        if let Err(err) = capture_result {
            self.reset_staging_pipeline();
            self.has_frame_history = false;
            return Err(err);
        }
        self.has_frame_history = true;
        Ok(out)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        if self.capture_mode == mode {
            return;
        }
        self.capture_mode = mode;
        self.reset_staging_pipeline();
    }

    fn set_cursor_config(&mut self, config: CursorCaptureConfig) {
        self.cursor_config = config;
        let _ = self
            .session
            .SetIsCursorCaptureEnabled(self.cursor_config.capture_cursor);
    }
}

impl Drop for WindowsGraphicsCaptureCapturer {
    fn drop(&mut self) {
        let _ = self.frame_pool.RemoveFrameArrived(self.frame_arrived_token);
        let _ = self.item.RemoveClosed(self.closed_token);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
    }
}

pub(crate) struct WindowsMonitorCapturer {
    inner: WindowsGraphicsCaptureCapturer,
}

impl WindowsMonitorCapturer {
    pub(crate) fn new(monitor: &MonitorId, resolver: Arc<MonitorResolver>) -> CaptureResult<Self> {
        validate_support()?;
        let com = CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
        let resolved = resolver.resolve_monitor(monitor)?;
        let (device, context) = d3d11::create_d3d11_device_for_adapter(&resolved.adapter, false)
            .map_err(CaptureError::Platform)?;
        let item = create_monitor_capture_item(resolved.handle)?;
        let inner = WindowsGraphicsCaptureCapturer::new(
            com,
            device,
            context,
            item,
            Some(resolved.hdr_metadata),
        )?;
        Ok(Self { inner })
    }
}

impl crate::backend::MonitorCapturer for WindowsMonitorCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.inner.capture(reuse)
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<Option<CaptureSampleMetadata>> {
        self.inner
            .capture_region_into(blit, destination, destination_has_history)
            .map(Some)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.inner.set_capture_mode(mode);
    }

    fn set_cursor_config(&mut self, config: CursorCaptureConfig) {
        self.inner.set_cursor_config(config);
    }
}

pub(crate) struct WindowsWindowCapturer {
    inner: WindowsGraphicsCaptureCapturer,
}

impl WindowsWindowCapturer {
    pub(crate) fn new(window: &WindowId) -> CaptureResult<Self> {
        validate_support()?;
        let com = CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
        let hwnd = HWND(window.raw_handle() as *mut std::ffi::c_void);
        if hwnd.0.is_null() {
            return Err(CaptureError::InvalidTarget(format!(
                "window handle is null: {}",
                window.stable_id()
            )));
        }
        let (device, context) =
            d3d11::create_d3d11_device_default(false).map_err(CaptureError::Platform)?;
        let item = create_window_capture_item(hwnd)?;
        let inner = WindowsGraphicsCaptureCapturer::new(com, device, context, item, None)?;
        Ok(Self { inner })
    }
}

impl crate::backend::MonitorCapturer for WindowsWindowCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.inner.capture(reuse)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.inner.set_capture_mode(mode);
    }

    fn set_cursor_config(&mut self, config: CursorCaptureConfig) {
        self.inner.set_cursor_config(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_copy_heuristic_accepts_small_sparse_updates() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 320,
                height: 180,
            },
            DirtyRect {
                x: 1280,
                y: 720,
                width: 200,
                height: 120,
            },
        ];
        assert!(should_use_dirty_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_copy_heuristic_rejects_large_dirty_area() {
        let rects = vec![DirtyRect {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        }];
        assert!(!should_use_dirty_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_copy_heuristic_rejects_excessive_rect_count() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            };
            WGC_DIRTY_COPY_MAX_RECTS + 1
        ];
        assert!(!should_use_dirty_copy(&rects, 1920, 1080));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_recording_mode() {
        assert!(!should_short_circuit_region_duplicate(
            true,
            CaptureMode::Screenshot,
            true,
            true,
            true,
        ));
        assert!(should_short_circuit_region_duplicate(
            true,
            CaptureMode::ScreenRecording,
            true,
            true,
            true,
        ));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_duplicate_with_history() {
        assert!(!should_short_circuit_region_duplicate(
            true,
            CaptureMode::ScreenRecording,
            false,
            true,
            true,
        ));
        assert!(!should_short_circuit_region_duplicate(
            true,
            CaptureMode::ScreenRecording,
            true,
            false,
            true,
        ));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_enabled_and_pending_slot() {
        assert!(!should_short_circuit_region_duplicate(
            false,
            CaptureMode::ScreenRecording,
            true,
            true,
            true,
        ));
        assert!(!should_short_circuit_region_duplicate(
            true,
            CaptureMode::ScreenRecording,
            true,
            true,
            false,
        ));
    }

    #[test]
    fn normalize_dirty_rects_merges_touching_spans() {
        let mut rects = vec![
            DirtyRect {
                x: 100,
                y: 100,
                width: 40,
                height: 20,
            },
            DirtyRect {
                x: 140,
                y: 100,
                width: 30,
                height: 20,
            },
            DirtyRect {
                x: 150,
                y: 115,
                width: 20,
                height: 25,
            },
        ];

        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);

        assert_eq!(
            rects,
            vec![DirtyRect {
                x: 100,
                y: 100,
                width: 70,
                height: 40
            }]
        );
    }

    #[test]
    fn normalize_dirty_rects_clamps_and_discards_out_of_bounds() {
        let mut rects = vec![
            DirtyRect {
                x: 1910,
                y: 1070,
                width: 40,
                height: 20,
            },
            DirtyRect {
                x: 3000,
                y: 40,
                width: 10,
                height: 10,
            },
            DirtyRect {
                x: 10,
                y: 20,
                width: 0,
                height: 50,
            },
        ];

        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);

        assert_eq!(
            rects,
            vec![DirtyRect {
                x: 1910,
                y: 1070,
                width: 10,
                height: 10
            }]
        );
    }

    #[test]
    fn normalize_dirty_rects_keeps_corner_contact_separate() {
        let mut rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            DirtyRect {
                x: 16,
                y: 16,
                width: 8,
                height: 8,
            },
        ];

        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);

        assert_eq!(
            rects,
            vec![
                DirtyRect {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16
                },
                DirtyRect {
                    x: 16,
                    y: 16,
                    width: 8,
                    height: 8
                },
            ]
        );
    }

    #[test]
    fn normalize_dirty_rects_merges_after_late_expansion() {
        let mut rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            DirtyRect {
                x: 0,
                y: 0,
                width: 1,
                height: 2,
            },
            DirtyRect {
                x: 0,
                y: 0,
                width: 1,
                height: 3,
            },
            DirtyRect {
                x: 0,
                y: 3,
                width: 1,
                height: 1,
            },
        ];

        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);

        assert_eq!(
            rects,
            vec![DirtyRect {
                x: 0,
                y: 0,
                width: 1,
                height: 4
            }]
        );
    }

    #[test]
    fn intersect_dirty_rects_returns_overlap() {
        let a = DirtyRect {
            x: 100,
            y: 50,
            width: 200,
            height: 120,
        };
        let b = DirtyRect {
            x: 220,
            y: 100,
            width: 160,
            height: 120,
        };
        let overlap = intersect_dirty_rects(a, b).expect("expected overlap");
        assert_eq!(
            overlap,
            DirtyRect {
                x: 220,
                y: 100,
                width: 80,
                height: 70,
            }
        );
    }

    #[test]
    fn intersect_dirty_rects_returns_none_for_disjoint_rects() {
        let a = DirtyRect {
            x: 0,
            y: 0,
            width: 100,
            height: 100,
        };
        let b = DirtyRect {
            x: 200,
            y: 200,
            width: 50,
            height: 50,
        };
        assert!(intersect_dirty_rects(a, b).is_none());
    }

    #[test]
    fn dirty_region_mode_guard_rejects_unknown_mode_values() {
        let unknown = GraphicsCaptureDirtyRegionMode(-1);
        assert!(!dirty_region_mode_supported(unknown));
        assert!(dirty_region_mode_supported(
            GraphicsCaptureDirtyRegionMode::ReportOnly
        ));
        assert!(dirty_region_mode_supported(
            GraphicsCaptureDirtyRegionMode::ReportAndRender
        ));
    }

    #[test]
    fn aggressive_stale_timeout_profile_is_tighter_than_legacy_profile() {
        let legacy = stale_timeout_config(false);
        let aggressive = stale_timeout_config(true);
        assert!(aggressive.initial < legacy.initial);
        assert!(aggressive.max < legacy.max);
        assert!(aggressive.increase_step <= legacy.increase_step);
        assert!(aggressive.decrease_step <= legacy.decrease_step);
        assert_eq!(aggressive.min, legacy.min);
    }

    #[test]
    fn stale_timeout_profiles_respect_bounds() {
        for config in [stale_timeout_config(false), stale_timeout_config(true)] {
            assert!(config.min <= config.initial);
            assert!(config.initial <= config.max);
            assert!(!config.decrease_step.is_zero());
            assert!(!config.increase_step.is_zero());
        }
    }

    #[test]
    fn stale_timeout_add_clamp_respects_upper_bound() {
        let value = duration_saturating_add_clamped(
            Duration::from_micros(1900),
            Duration::from_micros(300),
            WGC_STALE_FRAME_TIMEOUT_MAX,
        );
        assert_eq!(value, WGC_STALE_FRAME_TIMEOUT_MAX);
    }

    #[test]
    fn stale_timeout_sub_clamp_respects_lower_bound() {
        let value = duration_saturating_sub_clamped(
            Duration::from_micros(500),
            Duration::from_micros(200),
            WGC_STALE_FRAME_TIMEOUT_MIN,
        );
        assert_eq!(value, WGC_STALE_FRAME_TIMEOUT_MIN);
    }
}
