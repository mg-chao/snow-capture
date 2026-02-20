use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use windows::Win32::Foundation::{HANDLE, HWND};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CreateCompatibleDC, CreateDIBSection,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetMonitorInfoW, GetWindowDC, HBITMAP, HDC,
    HGDIOBJ, HMONITOR, MONITORINFO, MONITORINFOEXW, ReleaseDC, SRCCOPY, SelectObject,
};

use crate::backend::CaptureMode;
use crate::convert;
use crate::error::{CaptureError, CaptureResult};
use crate::frame::Frame;
use crate::monitor::MonitorId;

use super::com::CoInitGuard;
use super::monitor::MonitorResolver;

#[derive(Clone, Copy)]
struct MonitorGeometry {
    handle: HMONITOR,
    left: i32,
    top: i32,
    width: i32,
    height: i32,
}

fn geometry_from_handle(handle: HMONITOR) -> CaptureResult<MonitorGeometry> {
    let mut info = MONITORINFOEXW {
        monitorInfo: MONITORINFO {
            cbSize: size_of::<MONITORINFOEXW>() as u32,
            ..Default::default()
        },
        ..Default::default()
    };

    if !unsafe { GetMonitorInfoW(handle, (&mut info as *mut MONITORINFOEXW).cast()) }.as_bool() {
        return Err(CaptureError::MonitorLost);
    }

    let rect = info.monitorInfo.rcMonitor;
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    if width <= 0 || height <= 0 {
        return Err(CaptureError::Platform(anyhow::anyhow!(
            "monitor geometry is invalid ({width}x{height})"
        )));
    }

    Ok(MonitorGeometry {
        handle,
        left: rect.left,
        top: rect.top,
        width,
        height,
    })
}

fn resolve_geometry(resolver: &MonitorResolver, id: &MonitorId) -> CaptureResult<MonitorGeometry> {
    let resolved = resolver.resolve_monitor(id)?;
    geometry_from_handle(resolved.handle)
}

struct GdiResources {
    screen_dc: HDC,
    mem_dc: HDC,
    bitmap: Option<HBITMAP>,
    old_bitmap: Option<HGDIOBJ>,
    bits: *mut u8,
    width: i32,
    height: i32,
    stride: usize,
}

impl GdiResources {
    fn new() -> CaptureResult<Self> {
        let screen_dc = unsafe { GetDC(HWND(null_mut())) };
        if screen_dc.0.is_null() {
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "GetDC(NULL) returned null"
            )));
        }

        let mem_dc = unsafe { CreateCompatibleDC(screen_dc) };
        if mem_dc.0.is_null() {
            unsafe {
                let _ = ReleaseDC(HWND(null_mut()), screen_dc);
            }
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "CreateCompatibleDC failed"
            )));
        }

        Ok(Self {
            screen_dc,
            mem_dc,
            bitmap: None,
            old_bitmap: None,
            bits: null_mut(),
            width: 0,
            height: 0,
            stride: 0,
        })
    }

    /// Re-acquire the desktop screen DC.  Called when the display
    /// configuration has changed so the old DC may be stale.
    fn refresh_screen_dc(&mut self) -> CaptureResult<()> {
        // Release the old DC first.
        if !self.screen_dc.0.is_null() {
            unsafe {
                let _ = ReleaseDC(HWND(null_mut()), self.screen_dc);
            }
        }

        let new_dc = unsafe { GetDC(HWND(null_mut())) };
        if new_dc.0.is_null() {
            self.screen_dc = HDC(null_mut());
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "GetDC(NULL) returned null during refresh"
            )));
        }
        self.screen_dc = new_dc;

        // The memory DC was created from the old screen DC.  Recreate it
        // so it is compatible with the new one.
        if !self.mem_dc.0.is_null() {
            // Deselect the bitmap first so we can delete the old mem DC.
            self.release_bitmap();
            unsafe {
                let _ = DeleteDC(self.mem_dc);
            }
        }

        let new_mem_dc = unsafe { CreateCompatibleDC(self.screen_dc) };
        if new_mem_dc.0.is_null() {
            self.mem_dc = HDC(null_mut());
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "CreateCompatibleDC failed during refresh"
            )));
        }
        self.mem_dc = new_mem_dc;
        Ok(())
    }

    fn ensure_surface(&mut self, width: i32, height: i32) -> CaptureResult<()> {
        if width <= 0 || height <= 0 {
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "invalid gdi surface size {width}x{height}"
            )));
        }

        if self.bitmap.is_some() && self.width == width && self.height == height {
            return Ok(());
        }

        self.release_bitmap();

        let mut info = BITMAPINFO::default();
        info.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
        info.bmiHeader.biWidth = width;

        info.bmiHeader.biHeight = -height;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB.0;

        let mut bits: *mut c_void = null_mut();
        let bitmap = unsafe {
            CreateDIBSection(
                self.mem_dc,
                &info,
                DIB_RGB_COLORS,
                &mut bits,
                HANDLE::default(),
                0,
            )
        }
        .context("CreateDIBSection failed")
        .map_err(CaptureError::Platform)?;

        let selected = unsafe { SelectObject(self.mem_dc, bitmap) };
        if selected.0.is_null() {
            unsafe {
                let _ = DeleteObject(bitmap);
            }
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "SelectObject failed for gdi capture bitmap"
            )));
        }

        self.bitmap = Some(bitmap);
        self.old_bitmap = Some(selected);
        self.bits = bits.cast();
        self.width = width;
        self.height = height;
        self.stride = usize::try_from(width)
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or(CaptureError::BufferOverflow)?;
        Ok(())
    }

    /// Capture the monitor region and return an RGBA frame.
    ///
    /// The strategy is to BitBlt into the DIB section, then perform an
    /// in-place BGRA→RGBA swizzle directly in that buffer, and finally
    /// bulk-copy the result into the `Frame`.  When `src == dst` the
    /// SIMD kernels read and write the same cache lines, cutting memory
    /// bandwidth roughly in half compared to a separate src→dst copy.
    fn read_surface_to_rgba(
        &self,
        width: i32,
        height: i32,
        reuse: Option<Frame>,
        mode: CaptureMode,
    ) -> CaptureResult<Frame> {
        let width_u32 = u32::try_from(width).map_err(|_| CaptureError::BufferOverflow)?;
        let height_u32 = u32::try_from(height).map_err(|_| CaptureError::BufferOverflow)?;
        let width = usize::try_from(width_u32).map_err(|_| CaptureError::BufferOverflow)?;
        let height = usize::try_from(height_u32).map_err(|_| CaptureError::BufferOverflow)?;
        let pixel_count = width
            .checked_mul(height)
            .ok_or(CaptureError::BufferOverflow)?;

        let mut frame = reuse.unwrap_or_else(Frame::empty);
        frame.reset_metadata();
        frame.ensure_rgba_capacity(width_u32, height_u32)?;

        // Single-pass: read from DIB section, swizzle BGRA→RGBA, and
        // write directly into the Frame to avoid an extra memcpy.
        unsafe {
            match mode {
                CaptureMode::ScreenRecording => convert::convert_bgra_to_rgba_nt_unchecked(
                    self.bits.cast_const(),
                    frame.as_mut_rgba_ptr(),
                    pixel_count,
                ),
                CaptureMode::Screenshot => convert::convert_bgra_to_rgba_unchecked(
                    self.bits.cast_const(),
                    frame.as_mut_rgba_ptr(),
                    pixel_count,
                ),
            }
        }

        Ok(frame)
    }

    fn capture_to_rgba(
        &mut self,
        geometry: MonitorGeometry,
        reuse: Option<Frame>,
        mode: CaptureMode,
    ) -> CaptureResult<Frame> {
        self.ensure_surface(geometry.width, geometry.height)?;

        unsafe {
            BitBlt(
                self.mem_dc,
                0,
                0,
                geometry.width,
                geometry.height,
                self.screen_dc,
                geometry.left,
                geometry.top,
                SRCCOPY,
            )
        }
        .context("BitBlt failed during GDI monitor capture")
        .map_err(CaptureError::Platform)?;
        self.read_surface_to_rgba(geometry.width, geometry.height, reuse, mode)
    }

    /// Capture a window directly into the backing DIB.
    fn capture_window_to_rgba(
        &mut self,
        hwnd: HWND,
        width: i32,
        height: i32,
        reuse: Option<Frame>,
        mode: CaptureMode,
    ) -> CaptureResult<Frame> {
        self.ensure_surface(width, height)?;

        let mut rendered =
            unsafe { PrintWindow(hwnd, self.mem_dc, PRINT_WINDOW_FLAGS(2)) }.as_bool();
        if !rendered {
            rendered = unsafe { PrintWindow(hwnd, self.mem_dc, PRINT_WINDOW_FLAGS(0)) }.as_bool();
        }
        if !rendered {
            rendered = unsafe { PrintWindow(hwnd, self.mem_dc, PRINT_WINDOW_FLAGS(4)) }.as_bool();
        }
        if !rendered {
            let window_dc = unsafe { GetWindowDC(hwnd) };
            if window_dc.0.is_null() {
                return Err(CaptureError::Platform(anyhow::anyhow!(
                    "PrintWindow failed and GetWindowDC returned null during GDI window capture"
                )));
            }

            let blit_result = unsafe {
                BitBlt(
                    self.mem_dc,
                    0,
                    0,
                    width,
                    height,
                    window_dc,
                    0,
                    0,
                    SRCCOPY,
                )
            };

            unsafe {
                let _ = ReleaseDC(hwnd, window_dc);
            }

            blit_result
                .context("PrintWindow failed and BitBlt fallback failed during GDI window capture")
                .map_err(CaptureError::Platform)?;
        }

        self.read_surface_to_rgba(width, height, reuse, mode)
    }

    fn release_bitmap(&mut self) {
        if let Some(old_bitmap) = self.old_bitmap.take() {
            unsafe {
                let _ = SelectObject(self.mem_dc, old_bitmap);
            }
        }
        if let Some(bitmap) = self.bitmap.take() {
            unsafe {
                let _ = DeleteObject(bitmap);
            }
        }
        self.bits = null_mut();
        self.width = 0;
        self.height = 0;
        self.stride = 0;
    }
}

impl Drop for GdiResources {
    fn drop(&mut self) {
        self.release_bitmap();

        if !self.mem_dc.0.is_null() {
            unsafe {
                let _ = DeleteDC(self.mem_dc);
            }
        }
        if !self.screen_dc.0.is_null() {
            unsafe {
                let _ = ReleaseDC(HWND(null_mut()), self.screen_dc);
            }
        }
    }
}

pub(crate) struct WindowsMonitorCapturer {
    monitor: MonitorId,
    resolver: Arc<MonitorResolver>,
    _com: CoInitGuard,
    resources: GdiResources,
    geometry: MonitorGeometry,
    capture_mode: CaptureMode,
    /// Tracks the `DisplayInfoCache` generation so we only re-query
    /// monitor geometry when `WM_DISPLAYCHANGE` has actually fired.
    last_display_generation: Option<u64>,
}

// SAFETY: GdiResources contains raw HDC/HBITMAP handles which are not
// Send, but we only access them from the owning capture thread.
// The streaming module ensures single-threaded access.
unsafe impl Send for WindowsMonitorCapturer {}

impl WindowsMonitorCapturer {
    pub(crate) fn new(monitor: &MonitorId, resolver: Arc<MonitorResolver>) -> CaptureResult<Self> {
        let com = CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
        let geometry = resolve_geometry(&resolver, monitor)?;
        let mut resources = GdiResources::new()?;
        resources.ensure_surface(geometry.width, geometry.height)?;

        let last_display_generation = resolver.display_generation();

        Ok(Self {
            monitor: monitor.clone(),
            resolver,
            _com: com,
            resources,
            geometry,
            capture_mode: CaptureMode::Screenshot,
            last_display_generation,
        })
    }

    fn refresh_geometry(&mut self) -> CaptureResult<()> {
        let current_gen = self.resolver.display_generation();

        // When backed by the event-driven DisplayInfoCache, skip the
        // refresh entirely if the generation hasn't changed — no
        // WM_DISPLAYCHANGE has fired since our last check.
        if let (Some(current), Some(last)) = (current_gen, self.last_display_generation) {
            if current == last {
                return Ok(());
            }
        }

        self.last_display_generation = current_gen;

        // Display config changed — refresh the screen DC so we don't
        // capture from a stale device context after resolution /
        // composition changes.
        self.resources.refresh_screen_dc()?;

        match geometry_from_handle(self.geometry.handle) {
            Ok(geometry) => {
                self.geometry = geometry;
                Ok(())
            }
            Err(_) => {
                self.geometry = resolve_geometry(&self.resolver, &self.monitor)?;
                Ok(())
            }
        }
    }
}

impl crate::backend::MonitorCapturer for WindowsMonitorCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.refresh_geometry()?;
        let capture_time = Instant::now();
        let mut frame = self
            .resources
            .capture_to_rgba(self.geometry, reuse, self.capture_mode)?;
        frame.metadata.capture_time = Some(capture_time);
        // GDI doesn't provide native presentation timestamps, so we
        // synthesize a QPC value at capture time for consistent timing
        // across backends.
        frame.metadata.present_time_qpc = crate::frame::query_qpc_now();
        Ok(frame)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
    }
}

use crate::backend::MonitorCapturer;
use crate::window::WindowId;
use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, IsIconic, IsWindow, IsWindowVisible};
use windows::Win32::Foundation::RECT;

pub(crate) struct WindowsWindowCapturer {
    _com: CoInitGuard,
    resources: GdiResources,
    hwnd: HWND,
    capture_mode: CaptureMode,
}

// GdiResources contains raw pointers (HDC, HBITMAP, *mut u8) that are
// not inherently Send, but we only access them from the owning capture
// thread.  The streaming module ensures single-threaded access.
unsafe impl Send for WindowsWindowCapturer {}

impl WindowsWindowCapturer {
    pub(crate) fn new(window: &WindowId) -> CaptureResult<Self> {
        let com = CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
        let hwnd = HWND(window.raw_handle() as *mut std::ffi::c_void);
        if hwnd.0.is_null() {
            return Err(CaptureError::InvalidTarget(format!(
                "window handle is null: {}",
                window.stable_id()
            )));
        }
        if !unsafe { IsWindow(hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(format!(
                "window handle is not valid: {}",
                window.stable_id()
            )));
        }
        let resources = GdiResources::new()?;
        Ok(Self {
            _com: com,
            resources,
            hwnd,
            capture_mode: CaptureMode::Screenshot,
        })
    }
}

impl MonitorCapturer for WindowsWindowCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        if !unsafe { IsWindow(self.hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window no longer exists".into(),
            ));
        }
        if unsafe { IsIconic(self.hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window is minimized".into(),
            ));
        }
        if !unsafe { IsWindowVisible(self.hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window is not visible".into(),
            ));
        }

        let mut rect = RECT::default();
        unsafe { GetWindowRect(self.hwnd, &mut rect) }
            .ok()
            .context("GetWindowRect failed")
            .map_err(CaptureError::Platform)?;

        let width = rect.right.saturating_sub(rect.left);
        let height = rect.bottom.saturating_sub(rect.top);
        if width <= 0 || height <= 0 {
            return Err(CaptureError::InvalidTarget(
                "window has empty bounds".into(),
            ));
        }

        let capture_time = Instant::now();
        let mut frame = self.resources.capture_window_to_rgba(
            self.hwnd,
            width,
            height,
            reuse,
            self.capture_mode,
        )?;
        frame.metadata.capture_time = Some(capture_time);
        frame.metadata.present_time_qpc = crate::frame::query_qpc_now();
        Ok(frame)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
    }
}
