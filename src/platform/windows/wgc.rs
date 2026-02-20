use std::sync::{Arc, Condvar, Mutex};
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
    D3D11_BOX, D3D11_TEXTURE2D_DESC, ID3D11Device, ID3D11DeviceContext, ID3D11Resource,
    ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT;
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_ACCESS_LOST, IDXGIDevice};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::core::{IInspectable, Interface};

use crate::backend::{CaptureMode, CursorCaptureConfig};
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
const WGC_STALE_FRAME_TIMEOUT: Duration = Duration::from_millis(8);
const WGC_FRAME_POOL_BUFFERS: i32 = 2;
const WGC_DIRTY_COPY_MAX_RECTS: usize = 192;
const WGC_DIRTY_COPY_MAX_AREA_PERCENT: u64 = 70;

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

fn extract_dirty_rects(
    frame: &Direct3D11CaptureFrame,
    width: u32,
    height: u32,
    out: &mut Vec<DirtyRect>,
) -> Option<GraphicsCaptureDirtyRegionMode> {
    out.clear();
    let mode = frame.DirtyRegionMode().ok()?;
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

    Some(mode)
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

fn copy_dirty_rects_to_staging(
    context: &ID3D11DeviceContext,
    source_resource: &ID3D11Resource,
    staging_resource: &ID3D11Resource,
    rects: &[DirtyRect],
) {
    for rect in rects {
        if rect.width == 0 || rect.height == 0 {
            continue;
        }
        let right = rect.x.saturating_add(rect.width);
        let bottom = rect.y.saturating_add(rect.height);
        if right <= rect.x || bottom <= rect.y {
            continue;
        }
        let src_box = D3D11_BOX {
            left: rect.x,
            top: rect.y,
            front: 0,
            right,
            bottom,
            back: 1,
        };
        unsafe {
            context.CopySubresourceRegion(
                staging_resource,
                0,
                rect.x,
                rect.y,
                0,
                source_resource,
                0,
                Some(&src_box),
            );
        }
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
    staging: Option<ID3D11Texture2D>,
    staging_resource: Option<ID3D11Resource>,
    cached_src_desc: Option<D3D11_TEXTURE2D_DESC>,
    /// Last system-relative time, used for duplicate detection.
    last_present_time: i64,
    capture_mode: CaptureMode,
    /// HDR-to-SDR tonemap parameters, `Some` when the monitor has HDR enabled.
    hdr_to_sdr: Option<HdrToSdrParams>,
    /// GPU compute-shader tonemapper (HDR F16 → sRGB RGBA8).
    gpu_tonemapper: Option<GpuTonemapper>,
    /// GPU F16→sRGB converter for when source is F16 but no HDR tonemap needed.
    gpu_f16_converter: Option<GpuF16Converter>,
    cursor_config: CursorCaptureConfig,
    has_frame_history: bool,
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
                                signal_for_frames.cv.notify_all();
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
                        signal_for_closed.cv.notify_all();
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
            staging: None,
            staging_resource: None,
            cached_src_desc: None,
            last_present_time: 0,
            capture_mode: CaptureMode::Screenshot,
            hdr_to_sdr,
            gpu_tonemapper,
            gpu_f16_converter,
            cursor_config,
            has_frame_history: false,
        })
    }

    fn wait_for_next_frame(
        &mut self,
        allow_stale_return: bool,
    ) -> CaptureResult<(Direct3D11CaptureFrame, i64)> {
        let timeout = if allow_stale_return {
            WGC_STALE_FRAME_TIMEOUT
        } else {
            WGC_FRAME_TIMEOUT
        };
        let deadline = Instant::now() + timeout;
        let mut state = self
            .signal
            .state
            .lock()
            .map_err(|_| poisoned_lock_error())?;

        loop {
            if state.closed {
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
            self.staging = None;
            self.staging_resource = None;
            self.cached_src_desc = None;
            self.has_frame_history = false;
        }
        Ok(())
    }

    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        let mut out = reuse.unwrap_or_else(Frame::empty);
        out.reset_metadata();

        let allow_stale_return = self.capture_mode == CaptureMode::ScreenRecording
            && out.width() > 0
            && out.height() > 0;
        let capture_time = Instant::now();

        let (capture_frame, time_ticks) = match self.wait_for_next_frame(allow_stale_return) {
            Ok(result) => result,
            Err(CaptureError::Timeout) => {
                if allow_stale_return {
                    out.metadata.capture_time = Some(capture_time);
                    out.metadata.is_duplicate = true;
                    return Ok(out);
                }
                return Err(CaptureError::Timeout);
            }
            Err(error) => return Err(error),
        };

        out.metadata.capture_time = Some(capture_time);
        out.metadata.present_time_qpc = if time_ticks != 0 {
            Some(time_ticks)
        } else {
            None
        };
        out.metadata.is_duplicate = time_ticks != 0 && time_ticks == self.last_present_time;
        if time_ticks != 0 {
            self.last_present_time = time_ticks;
        }

        self.recreate_pool_if_needed(&capture_frame)?;

        let frame_surface = capture_frame
            .Surface()
            .map_err(|error| map_platform_error(error, "Direct3D11CaptureFrame::Surface failed"))?;
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
        let (effective_source, effective_desc, effective_hdr) = if src_desc.Format
            == DXGI_FORMAT_R16G16B16A16_FLOAT
        {
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
                let output =
                    converter.convert(&self.device, &self.context, &frame_texture, &src_desc)?;
                let mut out_desc = D3D11_TEXTURE2D_DESC::default();
                unsafe { output.GetDesc(&mut out_desc) };
                (output.clone(), out_desc, None)
            } else {
                (frame_texture.clone(), src_desc, self.hdr_to_sdr)
            }
        } else {
            (frame_texture.clone(), src_desc, self.hdr_to_sdr)
        };

        let out_matches_source =
            out.width() == effective_desc.Width && out.height() == effective_desc.Height;
        if self.has_frame_history && out.metadata.is_duplicate && out_matches_source {
            out.metadata.dirty_rects.clear();
            let _ = capture_frame.Close();
            return Ok(out);
        }

        let dirty_mode = extract_dirty_rects(
            &capture_frame,
            effective_desc.Width,
            effective_desc.Height,
            &mut out.metadata.dirty_rects,
        );
        let had_compatible_staging = self.staging.as_ref().is_some_and(|existing| {
            let mut existing_desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { existing.GetDesc(&mut existing_desc) };
            existing_desc.Width == effective_desc.Width
                && existing_desc.Height == effective_desc.Height
                && existing_desc.Format == effective_desc.Format
        });

        let staging = surface::ensure_staging_texture(
            &self.device,
            &mut self.staging,
            &effective_desc,
            StagingSampleDesc::Source,
            "failed to create WGC staging texture",
        )?;
        if self.staging_resource.is_none()
            || self
                .cached_src_desc
                .map_or(true, |d| d.Format != effective_desc.Format)
        {
            self.staging_resource = Some(
                staging
                    .cast()
                    .context("failed to cast WGC staging texture to ID3D11Resource")
                    .map_err(CaptureError::Platform)?,
            );
        }
        let staging_resource = self.staging_resource.as_ref().unwrap();
        let source_resource: ID3D11Resource = effective_source
            .cast()
            .context("failed to cast WGC frame texture to ID3D11Resource")
            .map_err(CaptureError::Platform)?;
        let use_dirty_copy = self.has_frame_history
            && had_compatible_staging
            && out_matches_source
            && dirty_mode.is_some()
            && !out.metadata.dirty_rects.is_empty()
            && should_use_dirty_copy(
                &out.metadata.dirty_rects,
                effective_desc.Width,
                effective_desc.Height,
            );
        let dirty_rects_for_copy = if use_dirty_copy {
            Some(out.metadata.dirty_rects.clone())
        } else {
            None
        };

        if use_dirty_copy {
            copy_dirty_rects_to_staging(
                &self.context,
                &source_resource,
                staging_resource,
                dirty_rects_for_copy.as_ref().unwrap(),
            );
        }

        let copy_result = if use_dirty_copy {
            match surface::map_staging_dirty_rects_to_frame(
                &self.context,
                staging,
                Some(staging_resource),
                &effective_desc,
                &mut out,
                dirty_rects_for_copy.as_ref().unwrap(),
                effective_hdr,
                "failed to map WGC staging texture (dirty regions)",
            ) {
                Ok(converted) if converted > 0 => Ok(()),
                Ok(_) | Err(_) => {
                    unsafe {
                        self.context
                            .CopyResource(staging_resource, &source_resource);
                    }
                    surface::map_staging_to_frame(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &effective_desc,
                        &mut out,
                        effective_hdr,
                        "failed to map WGC staging texture",
                    )
                }
            }
        } else {
            unsafe {
                self.context
                    .CopyResource(staging_resource, &source_resource);
            }
            surface::map_staging_to_frame(
                &self.context,
                staging,
                Some(staging_resource),
                &effective_desc,
                &mut out,
                effective_hdr,
                "failed to map WGC staging texture",
            )
        };

        if dirty_mode.is_none() {
            out.metadata.dirty_rects.clear();
        }

        let _ = capture_frame.Close();
        if let Err(err) = copy_result {
            self.has_frame_history = false;
            return Err(err);
        }
        self.has_frame_history = true;
        Ok(out)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
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
