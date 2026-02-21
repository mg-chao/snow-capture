use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use windows::Win32::Foundation::{HANDLE, HWND, RECT};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CreateCompatibleDC, CreateDIBSection,
    DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetMonitorInfoW, GetWindowDC, HBITMAP, HDC,
    HGDIOBJ, HMONITOR, MONITORINFO, MONITORINFOEXW, ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, IsIconic, IsWindow, IsWindowVisible};

use crate::backend::{CaptureBlitRegion, CaptureMode, CaptureSampleMetadata};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowCapturePath {
    WindowDcBitBlt,
    PrintWindow(PRINT_WINDOW_FLAGS),
}

const PRINT_WINDOW_RENDER_FULL: PRINT_WINDOW_FLAGS = PRINT_WINDOW_FLAGS(2);
const PRINT_WINDOW_DEFAULT: PRINT_WINDOW_FLAGS = PRINT_WINDOW_FLAGS(0);
const PRINT_WINDOW_EXPERIMENTAL: PRINT_WINDOW_FLAGS = PRINT_WINDOW_FLAGS(4);

const WINDOW_CAPTURE_ORDER_SCREENSHOT: [WindowCapturePath; 4] = [
    WindowCapturePath::PrintWindow(PRINT_WINDOW_RENDER_FULL),
    WindowCapturePath::PrintWindow(PRINT_WINDOW_DEFAULT),
    WindowCapturePath::PrintWindow(PRINT_WINDOW_EXPERIMENTAL),
    WindowCapturePath::WindowDcBitBlt,
];

const WINDOW_CAPTURE_ORDER_RECORDING: [WindowCapturePath; 4] = [
    WindowCapturePath::WindowDcBitBlt,
    WindowCapturePath::PrintWindow(PRINT_WINDOW_RENDER_FULL),
    WindowCapturePath::PrintWindow(PRINT_WINDOW_DEFAULT),
    WindowCapturePath::PrintWindow(PRINT_WINDOW_EXPERIMENTAL),
];

#[inline]
fn gdi_direct_region_capture_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_DIRECT_REGION")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

#[inline]
fn gdi_desktop_direct_region_capture_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_DESKTOP_REGION_DIRECT")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

#[inline]
fn gdi_window_state_cache_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_WINDOW_STATE_CACHE")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

#[inline]
fn gdi_window_state_refresh_interval_frames() -> u32 {
    use std::sync::OnceLock;
    static INTERVAL: OnceLock<u32> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_GDI_WINDOW_STATE_REFRESH_FRAMES")
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .map(|value| value.max(1))
            .unwrap_or(8)
    })
}

#[inline]
fn gdi_incremental_convert_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_INCREMENTAL_CONVERT")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

#[inline]
fn gdi_swap_history_surfaces_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_HISTORY_SURFACE_SWAP")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

#[inline]
fn gdi_row_compare_unroll_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_GDI_ROW_COMPARE_UNROLL")
            .map(|raw| {
                let normalized = raw.trim().to_ascii_lowercase();
                !(normalized == "1"
                    || normalized == "true"
                    || normalized == "yes"
                    || normalized == "on")
            })
            .unwrap_or(true)
    })
}

const GDI_INCREMENTAL_MIN_PIXELS: usize = 160_000;
const GDI_INCREMENTAL_MAX_DIRTY_ROW_NUMERATOR: usize = 3;
const GDI_INCREMENTAL_MAX_DIRTY_ROW_DENOMINATOR: usize = 4;

#[derive(Clone, Copy, Debug)]
struct DirtyRowRun {
    start_row: usize,
    row_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncrementalConvertStatus {
    /// No usable history available yet (first frame or surface resize).
    NotAvailable,
    /// Incremental scan completed and found no changed rows.
    Duplicate,
    /// Incremental conversion completed with `dirty_rows` updated rows.
    Updated(usize),
    /// Incremental scan exceeded the dirty threshold; caller should fall
    /// back to a full-frame conversion.
    TooDirty,
}

type RowCompareKernel = unsafe fn(*const u8, *const u8, usize) -> bool;

#[inline(always)]
fn row_compare_kernel() -> RowCompareKernel {
    use std::sync::OnceLock;
    static KERNEL: OnceLock<RowCompareKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_row_compare_kernel)
}

fn select_row_compare_kernel() -> RowCompareKernel {
    #[cfg(target_arch = "x86_64")]
    {
        let use_unrolled = gdi_row_compare_unroll_enabled();
        if std::arch::is_x86_feature_detected!("avx2") {
            return if use_unrolled {
                row_equal_avx2_unrolled
            } else {
                row_equal_avx2_legacy
            };
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return if use_unrolled {
                row_equal_sse2_unrolled
            } else {
                row_equal_sse2_legacy
            };
        }
    }
    row_equal_scalar
}

#[inline(always)]
fn bgra_row_converter() -> convert::SurfaceRowConverter {
    use std::sync::OnceLock;
    static CONVERTER: OnceLock<convert::SurfaceRowConverter> = OnceLock::new();
    *CONVERTER.get_or_init(|| {
        convert::SurfaceRowConverter::new(
            convert::SurfacePixelFormat::Bgra8,
            convert::SurfaceConversionOptions::default(),
        )
    })
}

unsafe fn row_equal_scalar(lhs: *const u8, rhs: *const u8, len: usize) -> bool {
    let mut offset = 0usize;
    while offset + size_of::<u64>() <= len {
        let left = unsafe { std::ptr::read_unaligned(lhs.add(offset).cast::<u64>()) };
        let right = unsafe { std::ptr::read_unaligned(rhs.add(offset).cast::<u64>()) };
        if left != right {
            return false;
        }
        offset += size_of::<u64>();
    }

    while offset < len {
        if unsafe { *lhs.add(offset) } != unsafe { *rhs.add(offset) } {
            return false;
        }
        offset += 1;
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn row_equal_sse2_legacy(lhs: *const u8, rhs: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};

    let mut offset = 0usize;
    while offset + 16 <= len {
        let left = unsafe { _mm_loadu_si128(lhs.add(offset).cast::<__m128i>()) };
        let right = unsafe { _mm_loadu_si128(rhs.add(offset).cast::<__m128i>()) };
        let equals = _mm_cmpeq_epi8(left, right);
        if _mm_movemask_epi8(equals) != 0xFFFF_i32 {
            return false;
        }
        offset += 16;
    }

    unsafe { row_equal_scalar(lhs.add(offset), rhs.add(offset), len - offset) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn row_equal_sse2_unrolled(lhs: *const u8, rhs: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
    };

    let mut offset = 0usize;
    while offset + 64 <= len {
        let left0 = unsafe { _mm_loadu_si128(lhs.add(offset).cast::<__m128i>()) };
        let right0 = unsafe { _mm_loadu_si128(rhs.add(offset).cast::<__m128i>()) };
        let eq0 = _mm_cmpeq_epi8(left0, right0);
        if _mm_movemask_epi8(eq0) != 0xFFFF_i32 {
            return false;
        }

        let left1 = unsafe { _mm_loadu_si128(lhs.add(offset + 16).cast::<__m128i>()) };
        let right1 = unsafe { _mm_loadu_si128(rhs.add(offset + 16).cast::<__m128i>()) };
        let left2 = unsafe { _mm_loadu_si128(lhs.add(offset + 32).cast::<__m128i>()) };
        let right2 = unsafe { _mm_loadu_si128(rhs.add(offset + 32).cast::<__m128i>()) };
        let left3 = unsafe { _mm_loadu_si128(lhs.add(offset + 48).cast::<__m128i>()) };
        let right3 = unsafe { _mm_loadu_si128(rhs.add(offset + 48).cast::<__m128i>()) };

        let eq1 = _mm_cmpeq_epi8(left1, right1);
        let eq2 = _mm_cmpeq_epi8(left2, right2);
        let eq3 = _mm_cmpeq_epi8(left3, right3);
        let eq12 = _mm_and_si128(eq1, eq2);
        let eq123 = _mm_and_si128(eq12, eq3);
        if _mm_movemask_epi8(eq123) != 0xFFFF_i32 {
            return false;
        }
        offset += 64;
    }

    while offset + 16 <= len {
        let left = unsafe { _mm_loadu_si128(lhs.add(offset).cast::<__m128i>()) };
        let right = unsafe { _mm_loadu_si128(rhs.add(offset).cast::<__m128i>()) };
        let equals = _mm_cmpeq_epi8(left, right);
        if _mm_movemask_epi8(equals) != 0xFFFF_i32 {
            return false;
        }
        offset += 16;
    }

    unsafe { row_equal_scalar(lhs.add(offset), rhs.add(offset), len - offset) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_equal_avx2_legacy(lhs: *const u8, rhs: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    let mut offset = 0usize;
    while offset + 32 <= len {
        let left = unsafe { _mm256_loadu_si256(lhs.add(offset).cast::<__m256i>()) };
        let right = unsafe { _mm256_loadu_si256(rhs.add(offset).cast::<__m256i>()) };
        let equals = _mm256_cmpeq_epi8(left, right);
        if _mm256_movemask_epi8(equals) != -1 {
            return false;
        }
        offset += 32;
    }

    unsafe { row_equal_scalar(lhs.add(offset), rhs.add(offset), len - offset) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_equal_avx2_unrolled(lhs: *const u8, rhs: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_testz_si256, _mm256_xor_si256,
    };

    let mut offset = 0usize;
    while offset + 128 <= len {
        let left0 = unsafe { _mm256_loadu_si256(lhs.add(offset).cast::<__m256i>()) };
        let right0 = unsafe { _mm256_loadu_si256(rhs.add(offset).cast::<__m256i>()) };
        let diff0 = _mm256_xor_si256(left0, right0);
        if _mm256_testz_si256(diff0, diff0) == 0 {
            return false;
        }

        let left1 = unsafe { _mm256_loadu_si256(lhs.add(offset + 32).cast::<__m256i>()) };
        let right1 = unsafe { _mm256_loadu_si256(rhs.add(offset + 32).cast::<__m256i>()) };
        let left2 = unsafe { _mm256_loadu_si256(lhs.add(offset + 64).cast::<__m256i>()) };
        let right2 = unsafe { _mm256_loadu_si256(rhs.add(offset + 64).cast::<__m256i>()) };
        let left3 = unsafe { _mm256_loadu_si256(lhs.add(offset + 96).cast::<__m256i>()) };
        let right3 = unsafe { _mm256_loadu_si256(rhs.add(offset + 96).cast::<__m256i>()) };

        let diff1 = _mm256_xor_si256(left1, right1);
        let diff2 = _mm256_xor_si256(left2, right2);
        let diff3 = _mm256_xor_si256(left3, right3);
        let diff23 = _mm256_or_si256(diff2, diff3);
        let diff_all = _mm256_or_si256(diff1, diff23);
        if _mm256_testz_si256(diff_all, diff_all) == 0 {
            return false;
        }
        offset += 128;
    }

    while offset + 32 <= len {
        let left = unsafe { _mm256_loadu_si256(lhs.add(offset).cast::<__m256i>()) };
        let right = unsafe { _mm256_loadu_si256(rhs.add(offset).cast::<__m256i>()) };
        let diff = _mm256_xor_si256(left, right);
        if _mm256_testz_si256(diff, diff) == 0 {
            return false;
        }
        offset += 32;
    }

    unsafe { row_equal_scalar(lhs.add(offset), rhs.add(offset), len - offset) }
}

#[inline(always)]
fn window_capture_order(mode: CaptureMode) -> &'static [WindowCapturePath; 4] {
    match mode {
        CaptureMode::Screenshot => &WINDOW_CAPTURE_ORDER_SCREENSHOT,
        CaptureMode::ScreenRecording => &WINDOW_CAPTURE_ORDER_RECORDING,
    }
}

fn build_window_capture_attempts(
    mode: CaptureMode,
    preferred_path: Option<WindowCapturePath>,
) -> ([WindowCapturePath; 4], usize) {
    let default_order = window_capture_order(mode);
    let mut attempts = [default_order[0]; 4];
    let mut attempt_count = 0usize;

    let preferred_path = match mode {
        CaptureMode::ScreenRecording => preferred_path,
        CaptureMode::Screenshot => None,
    };

    if let Some(preferred) = preferred_path {
        attempts[attempt_count] = preferred;
        attempt_count += 1;
    }

    for &candidate in default_order {
        if attempts[..attempt_count].contains(&candidate) {
            continue;
        }
        attempts[attempt_count] = candidate;
        attempt_count += 1;
    }

    (attempts, attempt_count)
}

struct GdiResources {
    screen_dc: HDC,
    mem_dc: HDC,
    capture_bitmap: Option<HBITMAP>,
    history_bitmap: Option<HBITMAP>,
    original_bitmap: Option<HGDIOBJ>,
    window_dc: Option<HDC>,
    window_dc_owner: Option<HWND>,
    bits: *mut u8,
    history_bits: *mut u8,
    history_surface_valid: bool,
    width: i32,
    height: i32,
    stride: usize,
    bgra_history: Vec<u8>,
    dirty_row_runs: Vec<DirtyRowRun>,
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
            capture_bitmap: None,
            history_bitmap: None,
            original_bitmap: None,
            window_dc: None,
            window_dc_owner: None,
            bits: null_mut(),
            history_bits: null_mut(),
            history_surface_valid: false,
            width: 0,
            height: 0,
            stride: 0,
            bgra_history: Vec::new(),
            dirty_row_runs: Vec::new(),
        })
    }

    /// Re-acquire the desktop screen DC.  Called when the display
    /// configuration has changed so the old DC may be stale.
    fn refresh_screen_dc(&mut self) -> CaptureResult<()> {
        self.release_window_dc();

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

    fn release_window_dc(&mut self) {
        if let (Some(owner), Some(window_dc)) = (self.window_dc_owner.take(), self.window_dc.take())
            && !window_dc.0.is_null()
        {
            unsafe {
                let _ = ReleaseDC(owner, window_dc);
            }
        }
    }

    fn acquire_window_dc(&mut self, hwnd: HWND) -> CaptureResult<HDC> {
        if let (Some(owner), Some(window_dc)) = (self.window_dc_owner, self.window_dc)
            && owner == hwnd
            && !window_dc.0.is_null()
        {
            return Ok(window_dc);
        }

        self.release_window_dc();

        let window_dc = unsafe { GetWindowDC(hwnd) };
        if window_dc.0.is_null() {
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "GetWindowDC returned null during GDI window capture"
            )));
        }

        self.window_dc_owner = Some(hwnd);
        self.window_dc = Some(window_dc);
        Ok(window_dc)
    }

    fn create_dib_bitmap(&self, width: i32, height: i32) -> CaptureResult<(HBITMAP, *mut u8)> {
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
        if bits.is_null() {
            unsafe {
                let _ = DeleteObject(bitmap);
            }
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "CreateDIBSection returned a null pixel buffer"
            )));
        }
        Ok((bitmap, bits.cast()))
    }

    fn ensure_history_surface(&mut self) -> CaptureResult<()> {
        if !gdi_swap_history_surfaces_enabled() || self.width <= 0 || self.height <= 0 {
            return Ok(());
        }
        if self.history_bitmap.is_some() && !self.history_bits.is_null() {
            return Ok(());
        }

        let (history_bitmap, history_bits) = self.create_dib_bitmap(self.width, self.height)?;
        self.history_bitmap = Some(history_bitmap);
        self.history_bits = history_bits;
        self.history_surface_valid = false;
        Ok(())
    }

    fn swap_capture_and_history_surfaces(&mut self) -> CaptureResult<()> {
        if !gdi_swap_history_surfaces_enabled() {
            return Ok(());
        }
        self.ensure_history_surface()?;

        let Some(current_bitmap) = self.capture_bitmap else {
            return Ok(());
        };
        let Some(next_bitmap) = self.history_bitmap else {
            return Ok(());
        };
        if current_bitmap == next_bitmap {
            return Ok(());
        }

        let selected = unsafe { SelectObject(self.mem_dc, next_bitmap) };
        if selected.0.is_null() {
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "SelectObject failed during gdi history surface swap"
            )));
        }
        if selected.0 != current_bitmap.0 {
            unsafe {
                let _ = SelectObject(self.mem_dc, current_bitmap);
            }
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "unexpected object selected during gdi history surface swap"
            )));
        }

        self.capture_bitmap = Some(next_bitmap);
        self.history_bitmap = Some(current_bitmap);
        std::mem::swap(&mut self.bits, &mut self.history_bits);
        self.history_surface_valid = true;
        Ok(())
    }

    fn ensure_surface(&mut self, width: i32, height: i32) -> CaptureResult<()> {
        if width <= 0 || height <= 0 {
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "invalid gdi surface size {width}x{height}"
            )));
        }

        if self.capture_bitmap.is_some() && self.width == width && self.height == height {
            return Ok(());
        }

        self.release_bitmap();

        let (bitmap, bits) = self.create_dib_bitmap(width, height)?;

        let selected = unsafe { SelectObject(self.mem_dc, bitmap) };
        if selected.0.is_null() {
            unsafe {
                let _ = DeleteObject(bitmap);
            }
            return Err(CaptureError::Platform(anyhow::anyhow!(
                "SelectObject failed for gdi capture bitmap"
            )));
        }

        self.capture_bitmap = Some(bitmap);
        self.original_bitmap = Some(selected);
        self.bits = bits;
        self.history_surface_valid = false;
        self.width = width;
        self.height = height;
        self.stride = usize::try_from(width)
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or(CaptureError::BufferOverflow)?;
        Ok(())
    }

    fn ensure_bgra_history(&mut self, byte_len: usize) -> bool {
        if byte_len == 0 {
            self.bgra_history.clear();
            return false;
        }

        let had_history = self.bgra_history.len() == byte_len;
        if !had_history {
            self.bgra_history.resize(byte_len, 0);
        }
        had_history
    }

    unsafe fn copy_surface_to_history_unchecked(&mut self, byte_len: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bits.cast_const(),
                self.bgra_history.as_mut_ptr(),
                byte_len,
            );
        }
    }

    fn scan_dirty_row_runs<F>(
        &mut self,
        row_bytes: usize,
        height: usize,
        history_base: *mut u8,
        history_stride: usize,
        mut on_changed_row: F,
    ) -> CaptureResult<Option<usize>>
    where
        F: FnMut(*const u8, *mut u8, usize),
    {
        if history_base.is_null() {
            return Ok(None);
        }
        if history_stride < row_bytes {
            return Err(CaptureError::BufferOverflow);
        }

        let max_dirty_rows = height.saturating_mul(GDI_INCREMENTAL_MAX_DIRTY_ROW_NUMERATOR)
            / GDI_INCREMENTAL_MAX_DIRTY_ROW_DENOMINATOR;
        let compare_row = row_compare_kernel();
        self.dirty_row_runs.clear();

        let mut dirty_rows = 0usize;
        let mut run_start = None::<usize>;
        let mut src_row = self.bits.cast_const();
        let mut history_row = history_base;

        for row in 0..height {
            let changed = unsafe { !compare_row(src_row, history_row.cast_const(), row_bytes) };
            if !changed {
                if let Some(start) = run_start.take() {
                    self.dirty_row_runs.push(DirtyRowRun {
                        start_row: start,
                        row_count: row - start,
                    });
                }
            } else {
                dirty_rows = dirty_rows.saturating_add(1);
                on_changed_row(src_row, history_row, row_bytes);
                if run_start.is_none() {
                    run_start = Some(row);
                }

                if dirty_rows > max_dirty_rows {
                    return Ok(None);
                }
            }

            unsafe {
                src_row = src_row.add(self.stride);
                history_row = history_row.add(history_stride);
            }
        }

        if let Some(start) = run_start {
            self.dirty_row_runs.push(DirtyRowRun {
                start_row: start,
                row_count: height - start,
            });
        }

        Ok(Some(dirty_rows))
    }

    fn convert_dirty_row_runs_into(
        &mut self,
        dst_ptr: *mut u8,
        dst_pitch: usize,
        width: usize,
    ) -> CaptureResult<()> {
        let converter = bgra_row_converter();
        let src_base = self.bits.cast_const();
        let dst_addr = dst_ptr as usize;

        for run in &self.dirty_row_runs {
            let src_offset = run
                .start_row
                .checked_mul(self.stride)
                .ok_or(CaptureError::BufferOverflow)?;
            let dst_offset = run
                .start_row
                .checked_mul(dst_pitch)
                .ok_or(CaptureError::BufferOverflow)?;
            let src_ptr = unsafe { src_base.add(src_offset) };
            let dst_ptr = dst_addr
                .checked_add(dst_offset)
                .ok_or(CaptureError::BufferOverflow)? as *mut u8;
            unsafe {
                converter.convert_rows_maybe_parallel_unchecked(
                    src_ptr,
                    self.stride,
                    dst_ptr,
                    dst_pitch,
                    width,
                    run.row_count,
                );
            }
        }
        Ok(())
    }

    fn try_convert_incremental_rows_with_vec_history_into(
        &mut self,
        dst_ptr: *mut u8,
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) -> CaptureResult<IncrementalConvertStatus> {
        let row_bytes = width.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
        if dst_pitch < row_bytes {
            return Err(CaptureError::BufferOverflow);
        }
        let total_bytes = row_bytes
            .checked_mul(height)
            .ok_or(CaptureError::BufferOverflow)?;
        if !self.ensure_bgra_history(total_bytes) {
            return Ok(IncrementalConvertStatus::NotAvailable);
        }

        let history_ptr = self.bgra_history.as_mut_ptr();
        let dirty_rows = self.scan_dirty_row_runs(
            row_bytes,
            height,
            history_ptr,
            row_bytes,
            |src_row, history_row, row_len| unsafe {
                std::ptr::copy_nonoverlapping(src_row, history_row, row_len);
            },
        )?;
        let Some(dirty_rows) = dirty_rows else {
            return Ok(IncrementalConvertStatus::TooDirty);
        };
        if dirty_rows == 0 {
            return Ok(IncrementalConvertStatus::Duplicate);
        }

        self.convert_dirty_row_runs_into(dst_ptr, dst_pitch, width)?;
        Ok(IncrementalConvertStatus::Updated(dirty_rows))
    }

    fn try_convert_incremental_rows_with_surface_history_into(
        &mut self,
        dst_ptr: *mut u8,
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) -> CaptureResult<IncrementalConvertStatus> {
        let row_bytes = width.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
        if dst_pitch < row_bytes {
            return Err(CaptureError::BufferOverflow);
        }
        if !self.history_surface_valid || self.history_bits.is_null() {
            return Ok(IncrementalConvertStatus::NotAvailable);
        }

        let dirty_rows = self.scan_dirty_row_runs(
            row_bytes,
            height,
            self.history_bits,
            self.stride,
            |_src_row, _history_row, _row_len| {},
        )?;
        let Some(dirty_rows) = dirty_rows else {
            return Ok(IncrementalConvertStatus::TooDirty);
        };
        if dirty_rows == 0 {
            return Ok(IncrementalConvertStatus::Duplicate);
        }

        self.convert_dirty_row_runs_into(dst_ptr, dst_pitch, width)?;
        Ok(IncrementalConvertStatus::Updated(dirty_rows))
    }

    fn commit_incremental_history(&mut self, total_bytes: usize) -> CaptureResult<()> {
        if gdi_swap_history_surfaces_enabled() {
            self.swap_capture_and_history_surfaces()?;
        } else {
            self.ensure_bgra_history(total_bytes);
            unsafe {
                self.copy_surface_to_history_unchecked(total_bytes);
            }
        }
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
        &mut self,
        width: i32,
        height: i32,
        reuse: Option<Frame>,
        mode: CaptureMode,
        destination_has_history: bool,
    ) -> CaptureResult<Frame> {
        let width_u32 = u32::try_from(width).map_err(|_| CaptureError::BufferOverflow)?;
        let height_u32 = u32::try_from(height).map_err(|_| CaptureError::BufferOverflow)?;
        let width = usize::try_from(width_u32).map_err(|_| CaptureError::BufferOverflow)?;
        let height = usize::try_from(height_u32).map_err(|_| CaptureError::BufferOverflow)?;
        let pixel_count = width
            .checked_mul(height)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_pitch = width.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;

        let mut frame = reuse.unwrap_or_else(Frame::empty);
        frame.reset_metadata();
        frame.ensure_rgba_capacity(width_u32, height_u32)?;
        let track_incremental_history = mode == CaptureMode::ScreenRecording
            && gdi_incremental_convert_enabled()
            && pixel_count >= GDI_INCREMENTAL_MIN_PIXELS;
        let total_bytes = if track_incremental_history {
            Some(
                width
                    .checked_mul(height)
                    .and_then(|pixels| pixels.checked_mul(4))
                    .ok_or(CaptureError::BufferOverflow)?,
            )
        } else {
            None
        };
        let incremental_enabled = track_incremental_history && destination_has_history;
        let use_surface_history = track_incremental_history && gdi_swap_history_surfaces_enabled();
        if incremental_enabled {
            let incremental_status = if use_surface_history {
                self.try_convert_incremental_rows_with_surface_history_into(
                    frame.as_mut_rgba_ptr(),
                    dst_pitch,
                    width,
                    height,
                )
            } else {
                self.try_convert_incremental_rows_with_vec_history_into(
                    frame.as_mut_rgba_ptr(),
                    dst_pitch,
                    width,
                    height,
                )
            }?;

            match incremental_status {
                IncrementalConvertStatus::Duplicate => {
                    frame.metadata.is_duplicate = true;
                    return Ok(frame);
                }
                IncrementalConvertStatus::Updated(_) => {
                    if use_surface_history && let Some(total_bytes) = total_bytes {
                        self.commit_incremental_history(total_bytes)?;
                    }
                    return Ok(frame);
                }
                IncrementalConvertStatus::NotAvailable | IncrementalConvertStatus::TooDirty => {}
            }
        }

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

        if let Some(total_bytes) = total_bytes {
            self.commit_incremental_history(total_bytes)?;
        }

        Ok(frame)
    }

    fn capture_to_rgba(
        &mut self,
        geometry: MonitorGeometry,
        reuse: Option<Frame>,
        mode: CaptureMode,
        destination_has_history: bool,
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
        self.read_surface_to_rgba(
            geometry.width,
            geometry.height,
            reuse,
            mode,
            destination_has_history,
        )
    }

    /// Capture a desktop-space rectangle and write it directly into an
    /// already allocated destination frame at `dst_x/dst_y`.
    fn capture_rect_into_rgba(
        &mut self,
        source_x: i32,
        source_y: i32,
        copy_width: u32,
        copy_height: u32,
        destination: &mut Frame,
        dst_x: u32,
        dst_y: u32,
        mode: CaptureMode,
        destination_has_history: bool,
    ) -> CaptureResult<bool> {
        if copy_width == 0 || copy_height == 0 {
            return Ok(true);
        }

        let dst_width = destination.width();
        let dst_height = destination.height();
        let dst_right = dst_x
            .checked_add(copy_width)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_bottom = dst_y
            .checked_add(copy_height)
            .ok_or(CaptureError::BufferOverflow)?;
        if dst_right > dst_width || dst_bottom > dst_height {
            return Err(CaptureError::BufferOverflow);
        }

        let copy_w_i32 = i32::try_from(copy_width).map_err(|_| CaptureError::BufferOverflow)?;
        let copy_h_i32 = i32::try_from(copy_height).map_err(|_| CaptureError::BufferOverflow)?;
        self.ensure_surface(copy_w_i32, copy_h_i32)?;

        unsafe {
            BitBlt(
                self.mem_dc,
                0,
                0,
                copy_w_i32,
                copy_h_i32,
                self.screen_dc,
                source_x,
                source_y,
                SRCCOPY,
            )
        }
        .context("BitBlt failed during GDI region capture")
        .map_err(CaptureError::Platform)?;

        let copy_w = usize::try_from(copy_width).map_err(|_| CaptureError::BufferOverflow)?;
        let copy_h = usize::try_from(copy_height).map_err(|_| CaptureError::BufferOverflow)?;
        let dst_pitch = usize::try_from(dst_width)
            .map_err(|_| CaptureError::BufferOverflow)?
            .checked_mul(4)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_height_usize =
            usize::try_from(dst_height).map_err(|_| CaptureError::BufferOverflow)?;
        let dst_required_len = dst_pitch
            .checked_mul(dst_height_usize)
            .ok_or(CaptureError::BufferOverflow)?;
        if destination.as_rgba_bytes().len() < dst_required_len {
            return Err(CaptureError::BufferOverflow);
        }
        let dst_x_usize = usize::try_from(dst_x).map_err(|_| CaptureError::BufferOverflow)?;
        let dst_y_usize = usize::try_from(dst_y).map_err(|_| CaptureError::BufferOverflow)?;
        let dst_offset = dst_y_usize
            .checked_mul(dst_pitch)
            .and_then(|base| {
                dst_x_usize
                    .checked_mul(4)
                    .and_then(|xoff| base.checked_add(xoff))
            })
            .ok_or(CaptureError::BufferOverflow)?;

        let dst_ptr = unsafe { destination.as_mut_rgba_ptr().add(dst_offset) };
        let row_bytes = copy_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
        let pixel_count = copy_w
            .checked_mul(copy_h)
            .ok_or(CaptureError::BufferOverflow)?;
        let track_incremental_history = mode == CaptureMode::ScreenRecording
            && gdi_incremental_convert_enabled()
            && pixel_count >= GDI_INCREMENTAL_MIN_PIXELS;
        let total_bytes = if track_incremental_history {
            Some(
                pixel_count
                    .checked_mul(4)
                    .ok_or(CaptureError::BufferOverflow)?,
            )
        } else {
            None
        };
        let incremental_enabled = track_incremental_history && destination_has_history;
        let use_surface_history = track_incremental_history && gdi_swap_history_surfaces_enabled();
        if incremental_enabled {
            let incremental_status = if use_surface_history {
                self.try_convert_incremental_rows_with_surface_history_into(
                    dst_ptr, dst_pitch, copy_w, copy_h,
                )
            } else {
                self.try_convert_incremental_rows_with_vec_history_into(
                    dst_ptr, dst_pitch, copy_w, copy_h,
                )
            }?;

            match incremental_status {
                IncrementalConvertStatus::Duplicate => return Ok(true),
                IncrementalConvertStatus::Updated(_) => {
                    if use_surface_history && let Some(total_bytes) = total_bytes {
                        self.commit_incremental_history(total_bytes)?;
                    }
                    return Ok(false);
                }
                IncrementalConvertStatus::NotAvailable | IncrementalConvertStatus::TooDirty => {}
            }
        }

        // Fast path: contiguous destination rows (single-monitor region output)
        // can use the dedicated contiguous kernels directly.
        if dst_pitch == row_bytes {
            unsafe {
                match mode {
                    CaptureMode::ScreenRecording => convert::convert_bgra_to_rgba_nt_unchecked(
                        self.bits.cast_const(),
                        dst_ptr,
                        pixel_count,
                    ),
                    CaptureMode::Screenshot => convert::convert_bgra_to_rgba_unchecked(
                        self.bits.cast_const(),
                        dst_ptr,
                        pixel_count,
                    ),
                }
            }
        } else {
            // General path for multi-monitor composites where destination rows
            // include padding to the full region width.
            unsafe {
                convert::convert_surface_to_rgba_unchecked(
                    convert::SurfacePixelFormat::Bgra8,
                    convert::SurfaceLayout::new(
                        self.bits.cast_const(),
                        self.stride,
                        dst_ptr,
                        dst_pitch,
                        copy_w,
                        copy_h,
                    ),
                    convert::SurfaceConversionOptions::default(),
                );
            }
        }

        if let Some(total_bytes) = total_bytes {
            self.commit_incremental_history(total_bytes)?;
        }

        Ok(false)
    }

    /// Capture a monitor sub-rectangle and write it directly into an
    /// already allocated destination frame at `blit.dst_x/dst_y`.
    ///
    /// This avoids the legacy region fallback path of:
    /// 1) full monitor BitBlt + full monitor BGRA->RGBA conversion
    /// 2) CPU crop/copy into the region output frame
    ///
    /// Instead, we only BitBlt the requested source rectangle and convert
    /// those pixels straight into the destination slice.
    fn capture_region_into_rgba(
        &mut self,
        geometry: MonitorGeometry,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        mode: CaptureMode,
        destination_has_history: bool,
    ) -> CaptureResult<bool> {
        if blit.width == 0 || blit.height == 0 {
            return Ok(true);
        }

        let src_width = u32::try_from(geometry.width).map_err(|_| CaptureError::BufferOverflow)?;
        let src_height =
            u32::try_from(geometry.height).map_err(|_| CaptureError::BufferOverflow)?;
        let src_right = blit
            .src_x
            .checked_add(blit.width)
            .ok_or(CaptureError::BufferOverflow)?;
        let src_bottom = blit
            .src_y
            .checked_add(blit.height)
            .ok_or(CaptureError::BufferOverflow)?;
        if src_right > src_width || src_bottom > src_height {
            return Err(CaptureError::BufferOverflow);
        }

        let source_x = i32::try_from(i64::from(geometry.left) + i64::from(blit.src_x))
            .map_err(|_| CaptureError::BufferOverflow)?;
        let source_y = i32::try_from(i64::from(geometry.top) + i64::from(blit.src_y))
            .map_err(|_| CaptureError::BufferOverflow)?;

        self.capture_rect_into_rgba(
            source_x,
            source_y,
            blit.width,
            blit.height,
            destination,
            blit.dst_x,
            blit.dst_y,
            mode,
            destination_has_history,
        )
    }

    fn capture_desktop_region_into_rgba(
        &mut self,
        source_x: i32,
        source_y: i32,
        width: u32,
        height: u32,
        destination: &mut Frame,
        mode: CaptureMode,
        destination_has_history: bool,
    ) -> CaptureResult<bool> {
        self.capture_rect_into_rgba(
            source_x,
            source_y,
            width,
            height,
            destination,
            0,
            0,
            mode,
            destination_has_history,
        )
    }

    /// Capture a window directly into the backing DIB.
    fn capture_window_to_rgba(
        &mut self,
        hwnd: HWND,
        width: i32,
        height: i32,
        reuse: Option<Frame>,
        mode: CaptureMode,
        preferred_path: Option<WindowCapturePath>,
        destination_has_history: bool,
    ) -> CaptureResult<(Frame, WindowCapturePath)> {
        self.ensure_surface(width, height)?;

        let (attempts, attempt_count) = build_window_capture_attempts(mode, preferred_path);

        let mut last_error: Option<CaptureError> = None;
        for &path in &attempts[..attempt_count] {
            match self.try_capture_window_path(hwnd, width, height, path) {
                Ok(()) => {
                    let frame = self.read_surface_to_rgba(
                        width,
                        height,
                        reuse,
                        mode,
                        destination_has_history,
                    )?;
                    return Ok((frame, path));
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!("all GDI window capture strategies failed"))
        }))
    }

    fn try_capture_window_path(
        &mut self,
        hwnd: HWND,
        width: i32,
        height: i32,
        path: WindowCapturePath,
    ) -> CaptureResult<()> {
        match path {
            WindowCapturePath::WindowDcBitBlt => {
                let window_dc = self.acquire_window_dc(hwnd)?;
                unsafe { BitBlt(self.mem_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY) }
                    .context("BitBlt failed during GDI window capture")
                    .map_err(CaptureError::Platform)?;
                Ok(())
            }
            WindowCapturePath::PrintWindow(flags) => {
                self.release_window_dc();
                if unsafe { PrintWindow(hwnd, self.mem_dc, flags) }.as_bool() {
                    return Ok(());
                }

                Err(CaptureError::Platform(anyhow::anyhow!(
                    "PrintWindow failed for flags {:#x}",
                    flags.0
                )))
            }
        }
    }

    fn release_bitmap(&mut self) {
        if let Some(original_bitmap) = self.original_bitmap.take() {
            unsafe {
                let _ = SelectObject(self.mem_dc, original_bitmap);
            }
        }
        if let Some(capture_bitmap) = self.capture_bitmap.take() {
            unsafe {
                let _ = DeleteObject(capture_bitmap);
            }
        }
        if let Some(history_bitmap) = self.history_bitmap.take() {
            unsafe {
                let _ = DeleteObject(history_bitmap);
            }
        }
        self.bits = null_mut();
        self.history_bits = null_mut();
        self.history_surface_valid = false;
        self.width = 0;
        self.height = 0;
        self.stride = 0;
        self.bgra_history.clear();
        self.dirty_row_runs.clear();
    }
}

impl Drop for GdiResources {
    fn drop(&mut self) {
        self.release_window_dc();
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
        if let (Some(current), Some(last)) = (current_gen, self.last_display_generation)
            && current == last
        {
            return Ok(());
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
        self.capture_with_history_hint(reuse, false)
    }

    fn capture_with_history_hint(
        &mut self,
        reuse: Option<Frame>,
        destination_has_history: bool,
    ) -> CaptureResult<Frame> {
        self.refresh_geometry()?;
        let capture_time = Instant::now();
        let mut frame = self.resources.capture_to_rgba(
            self.geometry,
            reuse,
            self.capture_mode,
            destination_has_history,
        )?;
        frame.metadata.capture_time = Some(capture_time);
        // GDI doesn't provide native presentation timestamps, so we
        // synthesize a QPC value at capture time for consistent timing
        // across backends.
        frame.metadata.present_time_qpc = crate::frame::query_qpc_now();
        Ok(frame)
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<Option<CaptureSampleMetadata>> {
        if !gdi_direct_region_capture_enabled() {
            return Ok(None);
        }

        self.refresh_geometry()?;
        let capture_time = Instant::now();
        let is_duplicate = self.resources.capture_region_into_rgba(
            self.geometry,
            blit,
            destination,
            self.capture_mode,
            destination_has_history,
        )?;
        Ok(Some(CaptureSampleMetadata {
            capture_time: Some(capture_time),
            present_time_qpc: crate::frame::query_qpc_now(),
            is_duplicate,
        }))
    }

    fn capture_desktop_region_into(
        &mut self,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        destination: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<Option<CaptureSampleMetadata>> {
        if !gdi_desktop_direct_region_capture_enabled() {
            return Ok(None);
        }
        if width == 0 || height == 0 {
            return Ok(Some(CaptureSampleMetadata {
                capture_time: Some(Instant::now()),
                present_time_qpc: crate::frame::query_qpc_now(),
                is_duplicate: true,
            }));
        }

        self.refresh_geometry()?;
        let capture_time = Instant::now();
        let is_duplicate = self.resources.capture_desktop_region_into_rgba(
            x,
            y,
            width,
            height,
            destination,
            self.capture_mode,
            destination_has_history,
        )?;
        Ok(Some(CaptureSampleMetadata {
            capture_time: Some(capture_time),
            present_time_qpc: crate::frame::query_qpc_now(),
            is_duplicate,
        }))
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
    }
}

use crate::backend::MonitorCapturer;
use crate::window::WindowId;

pub(crate) struct WindowsWindowCapturer {
    _com: CoInitGuard,
    resources: GdiResources,
    hwnd: HWND,
    capture_mode: CaptureMode,
    preferred_path: Option<WindowCapturePath>,
    cached_rect: Option<RECT>,
    frames_until_state_refresh: u32,
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
            preferred_path: None,
            cached_rect: None,
            frames_until_state_refresh: 0,
        })
    }

    fn refresh_window_rect(&mut self, require_visible: bool) -> CaptureResult<RECT> {
        if require_visible && !unsafe { IsWindowVisible(self.hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget("window is not visible".into()));
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

        self.cached_rect = Some(rect);
        self.frames_until_state_refresh =
            gdi_window_state_refresh_interval_frames().saturating_sub(1);
        Ok(rect)
    }

    fn resolve_window_rect(&mut self) -> CaptureResult<(RECT, bool)> {
        if self.capture_mode != CaptureMode::ScreenRecording || !gdi_window_state_cache_enabled() {
            return self.refresh_window_rect(true).map(|rect| (rect, false));
        }

        if self.frames_until_state_refresh > 0 {
            if let Some(rect) = self.cached_rect {
                self.frames_until_state_refresh -= 1;
                return Ok((rect, true));
            }
            self.frames_until_state_refresh = 0;
        }

        self.refresh_window_rect(true).map(|rect| (rect, false))
    }

    fn invalidate_window_state_cache(&mut self) {
        self.cached_rect = None;
        self.frames_until_state_refresh = 0;
    }
}

impl MonitorCapturer for WindowsWindowCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.capture_with_history_hint(reuse, false)
    }

    fn capture_with_history_hint(
        &mut self,
        reuse: Option<Frame>,
        destination_has_history: bool,
    ) -> CaptureResult<Frame> {
        if !unsafe { IsWindow(self.hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window no longer exists".into(),
            ));
        }
        if unsafe { IsIconic(self.hwnd) }.as_bool() {
            self.invalidate_window_state_cache();
            return Err(CaptureError::InvalidTarget("window is minimized".into()));
        }
        if !unsafe { IsWindowVisible(self.hwnd) }.as_bool() {
            self.invalidate_window_state_cache();
            return Err(CaptureError::InvalidTarget("window is not visible".into()));
        }
        let (mut rect, used_cached_rect) = self.resolve_window_rect()?;
        let width = rect.right.saturating_sub(rect.left);
        let height = rect.bottom.saturating_sub(rect.top);
        if width <= 0 || height <= 0 {
            self.invalidate_window_state_cache();
            return Err(CaptureError::InvalidTarget(
                "window has empty bounds".into(),
            ));
        }

        let capture_time = Instant::now();
        let first_attempt = self.resources.capture_window_to_rgba(
            self.hwnd,
            width,
            height,
            reuse,
            self.capture_mode,
            self.preferred_path,
            destination_has_history,
        );
        let (mut frame, used_path) = match first_attempt {
            Ok(result) => result,
            Err(_) if used_cached_rect => {
                rect = self.refresh_window_rect(true)?;
                let retry_width = rect.right.saturating_sub(rect.left);
                let retry_height = rect.bottom.saturating_sub(rect.top);
                if retry_width <= 0 || retry_height <= 0 {
                    self.invalidate_window_state_cache();
                    return Err(CaptureError::InvalidTarget(
                        "window has empty bounds".into(),
                    ));
                }
                self.resources.capture_window_to_rgba(
                    self.hwnd,
                    retry_width,
                    retry_height,
                    None,
                    self.capture_mode,
                    self.preferred_path,
                    false,
                )?
            }
            Err(error) => return Err(error),
        };
        self.preferred_path = match self.capture_mode {
            CaptureMode::ScreenRecording => Some(used_path),
            CaptureMode::Screenshot => None,
        };
        frame.metadata.capture_time = Some(capture_time);
        frame.metadata.present_time_qpc = crate::frame::query_qpc_now();
        Ok(frame)
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        if self.capture_mode != mode {
            self.preferred_path = None;
            self.invalidate_window_state_cache();
        }
        self.capture_mode = mode;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshot_mode_prefers_printwindow_paths() {
        let (attempts, count) = build_window_capture_attempts(CaptureMode::Screenshot, None);
        assert_eq!(count, 4);
        assert_eq!(
            attempts[0],
            WindowCapturePath::PrintWindow(PRINT_WINDOW_RENDER_FULL)
        );
        assert_eq!(
            attempts[1],
            WindowCapturePath::PrintWindow(PRINT_WINDOW_DEFAULT)
        );
        assert_eq!(
            attempts[2],
            WindowCapturePath::PrintWindow(PRINT_WINDOW_EXPERIMENTAL)
        );
        assert_eq!(attempts[3], WindowCapturePath::WindowDcBitBlt);
    }

    #[test]
    fn recording_mode_prefers_cached_window_dc() {
        let (attempts, count) = build_window_capture_attempts(CaptureMode::ScreenRecording, None);
        assert_eq!(count, 4);
        assert_eq!(attempts[0], WindowCapturePath::WindowDcBitBlt);
        assert_eq!(
            attempts[1],
            WindowCapturePath::PrintWindow(PRINT_WINDOW_RENDER_FULL)
        );
    }

    #[test]
    fn preferred_path_is_front_loaded_without_duplicates() {
        let preferred = WindowCapturePath::PrintWindow(PRINT_WINDOW_DEFAULT);
        let (attempts, count) =
            build_window_capture_attempts(CaptureMode::ScreenRecording, Some(preferred));
        assert_eq!(count, 4);
        assert_eq!(attempts[0], preferred);
        for idx in 0..count {
            assert!(!attempts[..idx].contains(&attempts[idx]));
        }
    }

    #[test]
    fn screenshot_mode_ignores_cached_preferred_path() {
        let (attempts, count) = build_window_capture_attempts(
            CaptureMode::Screenshot,
            Some(WindowCapturePath::WindowDcBitBlt),
        );
        assert_eq!(count, 4);
        assert_eq!(
            attempts[0],
            WindowCapturePath::PrintWindow(PRINT_WINDOW_RENDER_FULL)
        );
        assert_eq!(attempts[3], WindowCapturePath::WindowDcBitBlt);
    }

    #[test]
    fn row_compare_scalar_detects_changes() {
        let mut lhs = vec![0u8; 257];
        for (idx, value) in lhs.iter_mut().enumerate() {
            *value = (idx as u8).wrapping_mul(3);
        }
        let mut rhs = lhs.clone();

        assert!(unsafe { row_equal_scalar(lhs.as_ptr(), rhs.as_ptr(), lhs.len()) });

        rhs[128] ^= 0x7F;
        assert!(!unsafe { row_equal_scalar(lhs.as_ptr(), rhs.as_ptr(), lhs.len()) });
    }

    #[test]
    fn selected_row_compare_kernel_matches_scalar() {
        let mut lhs = vec![0u8; 4096];
        for (idx, value) in lhs.iter_mut().enumerate() {
            *value = (idx as u8).wrapping_mul(11);
        }
        let mut rhs = lhs.clone();
        rhs[2048] ^= 0x55;

        let kernel = row_compare_kernel();
        let scalar_equal = unsafe { row_equal_scalar(lhs.as_ptr(), lhs.as_ptr(), lhs.len()) };
        let kernel_equal = unsafe { kernel(lhs.as_ptr(), lhs.as_ptr(), lhs.len()) };
        assert_eq!(kernel_equal, scalar_equal);

        let scalar_diff = unsafe { row_equal_scalar(lhs.as_ptr(), rhs.as_ptr(), lhs.len()) };
        let kernel_diff = unsafe { kernel(lhs.as_ptr(), rhs.as_ptr(), lhs.len()) };
        assert_eq!(kernel_diff, scalar_diff);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unrolled_simd_row_compare_matches_legacy_kernels() {
        let mut lhs = vec![0u8; 8192];
        for (idx, value) in lhs.iter_mut().enumerate() {
            *value = (idx as u8).wrapping_mul(37).wrapping_add(11);
        }
        let mut rhs = lhs.clone();

        let test_lengths = [31usize, 64, 127, 255, 512, 1023, 4097, lhs.len()];
        for &len in &test_lengths {
            assert_eq!(
                unsafe { row_equal_sse2_unrolled(lhs.as_ptr(), rhs.as_ptr(), len) },
                unsafe { row_equal_sse2_legacy(lhs.as_ptr(), rhs.as_ptr(), len) }
            );
        }

        rhs[2027] ^= 0x5A;
        for &len in &test_lengths {
            assert_eq!(
                unsafe { row_equal_sse2_unrolled(lhs.as_ptr(), rhs.as_ptr(), len) },
                unsafe { row_equal_sse2_legacy(lhs.as_ptr(), rhs.as_ptr(), len) }
            );
        }

        if std::arch::is_x86_feature_detected!("avx2") {
            rhs.copy_from_slice(&lhs);
            for &len in &test_lengths {
                assert_eq!(
                    unsafe { row_equal_avx2_unrolled(lhs.as_ptr(), rhs.as_ptr(), len) },
                    unsafe { row_equal_avx2_legacy(lhs.as_ptr(), rhs.as_ptr(), len) }
                );
            }

            rhs[6015] ^= 0x31;
            for &len in &test_lengths {
                assert_eq!(
                    unsafe { row_equal_avx2_unrolled(lhs.as_ptr(), rhs.as_ptr(), len) },
                    unsafe { row_equal_avx2_legacy(lhs.as_ptr(), rhs.as_ptr(), len) }
                );
            }
        }
    }
}
