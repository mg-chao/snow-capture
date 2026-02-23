use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use windows::Graphics::{RectInt32, SizeInt32};
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
use super::dirty_rect::{
    self, DirtyCopyStrategy, DirtyCopyThresholds, DirtyRectDenseMergeThresholds,
};
use super::gpu_tonemap::{GpuF16Converter, GpuTonemapper};
use super::monitor::{HdrMonitorMetadata, MonitorResolver, hdr_to_sdr_params};
use super::region_pipeline::{self, RegionPipelineState, RegionSlot, RegionStagingSlotAccess};
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
const WGC_DIRTY_GPU_COPY_MAX_RECTS: usize = 64;
const WGC_DIRTY_GPU_COPY_MAX_AREA_PERCENT: u64 = 45;
const WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS: usize = 8;
const WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT: u64 = 18;
const WGC_FULL_DIRTY_DENSE_FALLBACK_MIN_RECTS: usize = WGC_DIRTY_COPY_MAX_RECTS + 32;
const WGC_FULL_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS: usize = WGC_DIRTY_COPY_MAX_RECTS * 4;
const WGC_FULL_DIRTY_DENSE_FALLBACK_AREA_PERCENT: u64 = 72;
const WGC_REGION_DIRTY_DENSE_FALLBACK_MIN_RECTS: usize = WGC_DIRTY_COPY_MAX_RECTS + 32;
const WGC_REGION_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS: usize = WGC_DIRTY_COPY_MAX_RECTS * 4;
const WGC_REGION_DIRTY_DENSE_FALLBACK_AREA_PERCENT: u64 = 72;
const WGC_DIRTY_REGION_FETCH_BATCH: usize = 64;
const WGC_STALE_POLL_SPIN_MIN: u32 = 16;
const WGC_STALE_POLL_SPIN_MAX: u32 = 256;
const WGC_STALE_POLL_SPIN_INITIAL: u32 = 64;
const WGC_STALE_POLL_SPIN_INCREASE_STEP: u32 = 16;
const WGC_QUERY_SPIN_MIN_POLLS: u32 = 2;
const WGC_QUERY_SPIN_MAX_POLLS: u32 = 64;
const WGC_QUERY_SPIN_INITIAL_POLLS: u32 = 4;
const WGC_QUERY_SPIN_INCREASE_STEP: u32 = 4;
const WGC_DIRTY_RECT_DENSE_MERGE_LEGACY_MIN_RECTS: usize = 64;
const WGC_DIRTY_RECT_DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN: u32 = 96;
const WGC_DIRTY_COPY_THRESHOLDS: DirtyCopyThresholds = DirtyCopyThresholds {
    max_rects: WGC_DIRTY_COPY_MAX_RECTS,
    max_area_percent: WGC_DIRTY_COPY_MAX_AREA_PERCENT,
    gpu_max_rects: WGC_DIRTY_GPU_COPY_MAX_RECTS,
    gpu_max_area_percent: WGC_DIRTY_GPU_COPY_MAX_AREA_PERCENT,
    gpu_low_latency_max_rects: WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS,
    gpu_low_latency_max_area_percent: WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT,
};
const WGC_DIRTY_RECT_DENSE_MERGE_THRESHOLDS: DirtyRectDenseMergeThresholds =
    DirtyRectDenseMergeThresholds {
        min_rects: WGC_DIRTY_RECT_DENSE_MERGE_LEGACY_MIN_RECTS,
        max_vertical_span: WGC_DIRTY_RECT_DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN,
    };

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RegionDirtyRectExtraction {
    available: bool,
    unchanged: bool,
    force_full_copy: bool,
}

#[derive(Clone, Copy, Debug)]
struct RegionClipBounds {
    source_width: u32,
    source_height: u32,
    region_left: u32,
    region_top: u32,
    region_right: u32,
    region_bottom: u32,
}

impl RegionClipBounds {
    #[inline(always)]
    fn new(source_width: u32, source_height: u32, region_bounds: DirtyRect) -> Option<Self> {
        let region_right = region_bounds.x.checked_add(region_bounds.width)?;
        let region_bottom = region_bounds.y.checked_add(region_bounds.height)?;
        Some(Self {
            source_width,
            source_height,
            region_left: region_bounds.x,
            region_top: region_bounds.y,
            region_right,
            region_bottom,
        })
    }
}

#[derive(Clone, Copy)]
struct WgcStaleTimeoutConfig {
    min: Duration,
    max: Duration,
    initial: Duration,
    decrease_step: Duration,
    increase_step: Duration,
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
    stale_timeout_config(true)
}

#[inline(always)]
fn duration_saturating_add_clamped(base: Duration, delta: Duration, max: Duration) -> Duration {
    base.checked_add(delta).unwrap_or(max).min(max)
}

#[inline(always)]
fn duration_saturating_sub_clamped(base: Duration, delta: Duration, min: Duration) -> Duration {
    base.saturating_sub(delta).max(min)
}



fn dirty_region_mode_supported(mode: GraphicsCaptureDirtyRegionMode) -> bool {
    mode == GraphicsCaptureDirtyRegionMode::ReportOnly
        || mode == GraphicsCaptureDirtyRegionMode::ReportAndRender
}

#[inline(always)]
fn clamp_dirty_region_rect(raw: RectInt32, width: u32, height: u32) -> Option<DirtyRect> {
    if raw.Width <= 0 || raw.Height <= 0 {
        return None;
    }

    let x = raw.X.max(0) as u32;
    let y = raw.Y.max(0) as u32;
    if x >= width || y >= height {
        return None;
    }

    let rect_width = raw.Width as u32;
    let rect_height = raw.Height as u32;
    let clamped_w = rect_width.min(width - x);
    let clamped_h = rect_height.min(height - y);
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

#[inline(always)]
fn clip_dirty_region_rect_to_region_fallback(
    raw: RectInt32,
    bounds: RegionClipBounds,
) -> Option<DirtyRect> {
    if raw.Width <= 0 || raw.Height <= 0 {
        return None;
    }

    let region_left = i64::from(bounds.region_left);
    let region_top = i64::from(bounds.region_top);
    let region_right = i64::from(bounds.region_right);
    let region_bottom = i64::from(bounds.region_bottom);
    if region_left >= region_right || region_top >= region_bottom {
        return None;
    }

    let source_right = i64::from(bounds.source_width);
    let source_bottom = i64::from(bounds.source_height);
    if source_right <= 0 || source_bottom <= 0 {
        return None;
    }

    let raw_left = i64::from(raw.X.max(0));
    let raw_top = i64::from(raw.Y.max(0));
    let raw_right = raw_left.saturating_add(i64::from(raw.Width));
    let raw_bottom = raw_top.saturating_add(i64::from(raw.Height));

    let clipped_left = raw_left.max(region_left);
    let clipped_top = raw_top.max(region_top);
    let clipped_right = raw_right.min(region_right).min(source_right);
    let clipped_bottom = raw_bottom.min(region_bottom).min(source_bottom);
    if clipped_right <= clipped_left || clipped_bottom <= clipped_top {
        return None;
    }

    let x = u32::try_from(clipped_left.saturating_sub(region_left)).ok()?;
    let y = u32::try_from(clipped_top.saturating_sub(region_top)).ok()?;
    let width = u32::try_from(clipped_right.saturating_sub(clipped_left)).ok()?;
    let height = u32::try_from(clipped_bottom.saturating_sub(clipped_top)).ok()?;
    if width == 0 || height == 0 {
        return None;
    }

    Some(DirtyRect {
        x,
        y,
        width,
        height,
    })
}

#[inline(always)]
fn clip_dirty_region_rect_to_region_with_bounds(
    raw: RectInt32,
    bounds: RegionClipBounds,
) -> Option<DirtyRect> {
    if raw.Width <= 0 || raw.Height <= 0 {
        return None;
    }

    if raw.X < 0 || raw.Y < 0 {
        return clip_dirty_region_rect_to_region_fallback(raw, bounds);
    }

    let raw_x = raw.X as u32;
    let raw_y = raw.Y as u32;
    if raw_x >= bounds.source_width || raw_y >= bounds.source_height {
        return None;
    }

    let raw_right = raw_x.saturating_add(raw.Width as u32);
    let raw_bottom = raw_y.saturating_add(raw.Height as u32);
    let clipped_left = raw_x.max(bounds.region_left);
    let clipped_top = raw_y.max(bounds.region_top);
    let clipped_right = raw_right.min(bounds.region_right).min(bounds.source_width);
    let clipped_bottom = raw_bottom
        .min(bounds.region_bottom)
        .min(bounds.source_height);
    if clipped_right <= clipped_left || clipped_bottom <= clipped_top {
        return None;
    }

    Some(DirtyRect {
        x: clipped_left.saturating_sub(bounds.region_left),
        y: clipped_top.saturating_sub(bounds.region_top),
        width: clipped_right.saturating_sub(clipped_left),
        height: clipped_bottom.saturating_sub(clipped_top),
    })
}

fn for_each_dirty_region(
    frame: &Direct3D11CaptureFrame,
    mut visit: impl FnMut(RectInt32) -> bool,
) -> Option<GraphicsCaptureDirtyRegionMode> {
    let mode = frame.DirtyRegionMode().ok()?;
    if !dirty_region_mode_supported(mode) {
        return None;
    }
    let regions = frame.DirtyRegions().ok()?;
    let count = regions.Size().ok()?;
    if count == 0 {
        return Some(mode);
    }

    let mut start_idx = 0u32;
    let mut batch = [RectInt32::default(); WGC_DIRTY_REGION_FETCH_BATCH];
    while start_idx < count {
        let remaining = usize::try_from(count - start_idx).ok()?;
        let batch_len = remaining.min(batch.len());
        let fetched = match regions.GetMany(start_idx, &mut batch[..batch_len]) {
            Ok(value) => value,
            Err(_) => break,
        };
        let fetched_len = usize::try_from(fetched).ok()?.min(batch_len);
        if fetched_len == 0 {
            break;
        }

        for raw in &batch[..fetched_len] {
            if !visit(*raw) {
                return Some(mode);
            }
        }
        start_idx = start_idx.saturating_add(u32::try_from(fetched_len).ok()?);
    }

    if start_idx < count {
        for idx in start_idx..count {
            if let Ok(raw) = regions.GetAt(idx) {
                if !visit(raw) {
                    return Some(mode);
                }
            }
        }
    }

    Some(mode)
}

fn extract_dirty_rects(
    frame: &Direct3D11CaptureFrame,
    width: u32,
    height: u32,
    out: &mut Vec<DirtyRect>,
) -> Option<GraphicsCaptureDirtyRegionMode> {
    out.clear();

    let total_pixels = (width as u64).saturating_mul(height as u64);
    let mut dense_fallback = false;
    let mut dirty_pixels = 0u64;
    let mut non_empty_rects = 0usize;

    let mode = for_each_dirty_region(frame, |raw| {
        if dense_fallback {
            return false;
        }

        if let Some(clamped) = clamp_dirty_region_rect(raw, width, height) {
            non_empty_rects = non_empty_rects.saturating_add(1);
            dirty_pixels = dirty_pixels
                .saturating_add((clamped.width as u64).saturating_mul(clamped.height as u64));
            let exceeds_dense_area = total_pixels > 0
                && dirty_pixels.saturating_mul(100)
                    > total_pixels.saturating_mul(WGC_FULL_DIRTY_DENSE_FALLBACK_AREA_PERCENT);
            let should_force_full_copy = non_empty_rects
                > WGC_FULL_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS
                || (non_empty_rects > WGC_FULL_DIRTY_DENSE_FALLBACK_MIN_RECTS
                    && exceeds_dense_area);
            if should_force_full_copy {
                dense_fallback = true;
                out.clear();
                if width > 0 && height > 0 {
                    out.push(DirtyRect {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    });
                }
                return false;
            }
            out.push(clamped);
        }
        true
    })?;

    if dense_fallback {
        return Some(mode);
    }

    normalize_dirty_rects_preclamped_in_place(out, width, height, false);
    Some(mode)
}

#[cfg(test)]
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

thread_local! {
    static DIRTY_RECT_MERGE_SCRATCH: RefCell<Vec<dirty_rect::DirtyRectMergeCandidate>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
#[inline(always)]
fn dirty_rect_order_non_decreasing(previous: DirtyRect, current: DirtyRect) -> bool {
    current.y > previous.y || (current.y == previous.y && current.x >= previous.x)
}

#[cfg(test)]
#[inline(always)]
fn update_dirty_rect_sort_hint(
    sorted: &mut bool,
    previous: &mut Option<DirtyRect>,
    current: DirtyRect,
) {
    if *sorted
        && let Some(prev) = previous.as_ref().copied()
        && !dirty_rect_order_non_decreasing(prev, current)
    {
        *sorted = false;
    }
    *previous = Some(current);
}

fn normalize_dirty_rects_after_prepare_in_place(
    pending: &mut Vec<DirtyRect>,
    already_sorted_hint: bool,
) {
    DIRTY_RECT_MERGE_SCRATCH.with(|scratch| {
        let mut merged = scratch.borrow_mut();
        dirty_rect::normalize_dirty_rects_after_prepare_in_place(
            pending,
            already_sorted_hint,
            WGC_DIRTY_RECT_DENSE_MERGE_THRESHOLDS,
            &mut merged,
        );
    });
}

#[inline(always)]
fn normalize_dirty_rects_preclamped_in_place(
    rects: &mut Vec<DirtyRect>,
    width: u32,
    height: u32,
    already_sorted_hint: bool,
) {
    if rects.is_empty() {
        return;
    }

    debug_assert!(rects.iter().all(|rect| {
        rect.width > 0
            && rect.height > 0
            && rect
                .x
                .checked_add(rect.width)
                .is_some_and(|right| right <= width)
            && rect
                .y
                .checked_add(rect.height)
                .is_some_and(|bottom| bottom <= height)
    }));

    let mut pending = std::mem::take(rects);
    normalize_dirty_rects_after_prepare_in_place(&mut pending, already_sorted_hint);
    *rects = pending;
}

#[cfg(test)]
fn normalize_dirty_rects_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    let mut merge_scratch = Vec::new();
    dirty_rect::normalize_dirty_rects_in_place(
        rects,
        width,
        height,
        false,
        WGC_DIRTY_RECT_DENSE_MERGE_THRESHOLDS,
        true,
        &mut merge_scratch,
    );
}

#[cfg(test)]
fn normalize_dirty_rects_reference_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    dirty_rect::normalize_dirty_rects_reference_in_place(
        rects,
        width,
        height,
        WGC_DIRTY_RECT_DENSE_MERGE_THRESHOLDS,
    );
}

fn extract_region_dirty_rects(
    frame: &Direct3D11CaptureFrame,
    source_width: u32,
    source_height: u32,
    blit: CaptureBlitRegion,
    out: &mut Vec<DirtyRect>,
) -> RegionDirtyRectExtraction {
    out.clear();
    let Some(region_bounds) = dirty_rect::clamp_dirty_rect(
        DirtyRect {
            x: blit.src_x,
            y: blit.src_y,
            width: blit.width,
            height: blit.height,
        },
        source_width,
        source_height,
    ) else {
        return RegionDirtyRectExtraction::default();
    };
    let Some(clip_bounds) = RegionClipBounds::new(source_width, source_height, region_bounds)
    else {
        return RegionDirtyRectExtraction::default();
    };

    let total_region_pixels =
        (region_bounds.width as u64).saturating_mul(region_bounds.height as u64);
    let mut dense_fallback = false;
    let mut dirty_pixels = 0u64;
    let mut non_empty_rects = 0usize;

    if for_each_dirty_region(frame, |raw| {
        if dense_fallback {
            return false;
        }

        if let Some(clipped) = clip_dirty_region_rect_to_region_with_bounds(raw, clip_bounds) {
            non_empty_rects = non_empty_rects.saturating_add(1);
            dirty_pixels = dirty_pixels
                .saturating_add((clipped.width as u64).saturating_mul(clipped.height as u64));

            let exceeds_dense_area = total_region_pixels > 0
                && dirty_pixels.saturating_mul(100)
                    > total_region_pixels
                        .saturating_mul(WGC_REGION_DIRTY_DENSE_FALLBACK_AREA_PERCENT);
            let should_force_full_copy = non_empty_rects
                > WGC_REGION_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS
                || (non_empty_rects > WGC_REGION_DIRTY_DENSE_FALLBACK_MIN_RECTS
                    && exceeds_dense_area);
            if should_force_full_copy {
                dense_fallback = true;
                out.clear();
                return false;
            }

            out.push(clipped);
        }
        true
    })
    .is_none()
    {
        return RegionDirtyRectExtraction::default();
    }

    if dense_fallback {
        return RegionDirtyRectExtraction {
            available: true,
            unchanged: false,
            force_full_copy: true,
        };
    }

    if non_empty_rects == 0 {
        return RegionDirtyRectExtraction {
            available: true,
            unchanged: true,
            force_full_copy: false,
        };
    }

    normalize_dirty_rects_preclamped_in_place(
        out,
        region_bounds.width,
        region_bounds.height,
        false,
    );
    RegionDirtyRectExtraction {
        available: true,
        unchanged: out.is_empty(),
        force_full_copy: false,
    }
}

#[inline(always)]
fn dirty_rect_conversion_hints(
    rects: &[DirtyRect],
    dirty_pixels: u64,
    trusted_bounds: bool,
) -> surface::DirtyRectConversionHints {
    surface::DirtyRectConversionHints {
        trusted_bounds,
        non_empty_rects: Some(rects.len()),
        total_dirty_pixels: usize::try_from(dirty_pixels).ok(),
    }
}

#[inline(always)]
fn dirty_rect_destination_bounds_trusted(
    dst_x: u32,
    dst_y: u32,
    copy_width: u32,
    copy_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> bool {
    let Some(dst_right) = dst_x.checked_add(copy_width) else {
        return false;
    };
    let Some(dst_bottom) = dst_y.checked_add(copy_height) else {
        return false;
    };
    dst_right <= dst_width && dst_bottom <= dst_height
}

fn evaluate_dirty_copy_strategy(rects: &[DirtyRect], width: u32, height: u32) -> DirtyCopyStrategy {
    dirty_rect::evaluate_dirty_copy_strategy(rects, width, height, WGC_DIRTY_COPY_THRESHOLDS)
}

#[cfg(test)]
fn should_use_dirty_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
    evaluate_dirty_copy_strategy(rects, width, height).cpu
}

#[cfg(test)]
fn should_use_dirty_gpu_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
    evaluate_dirty_copy_strategy(rects, width, height).gpu
}

#[cfg(test)]
fn should_use_low_latency_dirty_gpu_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
    evaluate_dirty_copy_strategy(rects, width, height).gpu_low_latency
}

#[inline(always)]
fn should_short_circuit_region_duplicate(
    capture_mode: CaptureMode,
    destination_has_history: bool,
    source_is_duplicate: bool,
    pending_slot_available: bool,
) -> bool {
    should_short_circuit_duplicate(
        capture_mode,
        destination_has_history,
        source_is_duplicate,
        pending_slot_available,
    )
}

#[inline(always)]
fn should_short_circuit_duplicate(
    capture_mode: CaptureMode,
    destination_has_history: bool,
    source_is_duplicate: bool,
    pending_slot_available: bool,
) -> bool {
    capture_mode == CaptureMode::ScreenRecording
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
    dirty_cpu_copy_preferred: bool,
    dirty_gpu_copy_preferred: bool,
    dirty_total_pixels: u64,
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
        self.dirty_cpu_copy_preferred = false;
        self.dirty_gpu_copy_preferred = false;
        self.dirty_total_pixels = 0;
        self.dirty_rects.clear();
        self.populated = false;
    }
}

impl RegionSlot for WgcStagingSlot {
    fn soft_reset(&mut self) {
        self.invalidate();
    }
    fn hard_reset(&mut self) {
        self.invalidate();
    }
}

impl RegionStagingSlotAccess for WgcStagingSlot {
    fn staging_resource(&self) -> Option<&ID3D11Resource> {
        self.staging_resource.as_ref()
    }
    fn set_staging(&mut self, tex: ID3D11Texture2D, res: ID3D11Resource) {
        self.staging = Some(tex);
        self.staging_resource = Some(res);
    }
    fn query(&self) -> Option<&ID3D11Query> {
        self.query.as_ref()
    }
    fn set_query(&mut self, q: ID3D11Query) {
        self.query = Some(q);
    }
    fn dirty_rects(&self) -> &[DirtyRect] {
        &self.dirty_rects
    }
    fn dirty_gpu_copy_preferred(&self) -> bool {
        self.dirty_gpu_copy_preferred
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
    region: RegionPipelineState<WgcStagingSlot, WGC_STAGING_SLOTS>,
    cached_src_desc: Option<D3D11_TEXTURE2D_DESC>,
    /// Last system-relative time, used for duplicate detection.
    last_present_time: i64,
    /// Last present timestamp emitted to callers.
    last_emitted_present_time: i64,
    stale_poll_spins: u32,
    stale_timeout_config: WgcStaleTimeoutConfig,
    stale_frame_timeout: Duration,
    adaptive_spin_polls: u32,
    capture_mode: CaptureMode,
    /// HDR-to-SDR tonemap parameters, `Some` when the monitor has HDR enabled.
    hdr_to_sdr: Option<HdrToSdrParams>,
    /// GPU compute-shader tonemapper (HDR F16 -> sRGB RGBA8).
    gpu_tonemapper: Option<GpuTonemapper>,
    /// GPU F16->sRGB converter for when source is F16 but no HDR tonemap needed.
    gpu_f16_converter: Option<GpuF16Converter>,
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
        // Screenshot mode favors single-frame latency; keep WGC dirty-region
        // mode at the lighter ReportOnly setting until recording requests
        // ReportAndRender.
        let _ = session.SetDirtyRegionMode(GraphicsCaptureDirtyRegionMode::ReportOnly);

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
            region: RegionPipelineState::new(),
            cached_src_desc: None,
            last_present_time: 0,
            last_emitted_present_time: 0,
            stale_poll_spins: WGC_STALE_POLL_SPIN_INITIAL,
            stale_timeout_config,
            stale_frame_timeout: stale_timeout_config.initial,
            adaptive_spin_polls: WGC_QUERY_SPIN_INITIAL_POLLS,
            capture_mode: CaptureMode::Screenshot,
            hdr_to_sdr,
            gpu_tonemapper,
            gpu_f16_converter,
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
            if self.pending_slot.is_some() {
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
            if self.region.pending_slot.is_some() {
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
        self.region.reset();
        self.has_frame_history = false;
        self.last_emitted_present_time = 0;
        self.stale_poll_spins = WGC_STALE_POLL_SPIN_INITIAL;
        self.stale_frame_timeout = self.stale_timeout_config.initial;
        self.adaptive_spin_polls = WGC_QUERY_SPIN_INITIAL_POLLS;
        self.source_dirty_rects_scratch.clear();
        self.region_dirty_rects_scratch.clear();
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

    fn ensure_region_slot(
        &mut self,
        slot_idx: usize,
        desc: &D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<()> {
        let slot = &mut self.region.slots[slot_idx];
        let needs_recreate = slot.source_desc.is_none_or(|cached| {
            cached.Width != desc.Width
                || cached.Height != desc.Height
                || cached.Format != desc.Format
                || cached.SampleDesc.Count != desc.SampleDesc.Count
                || cached.SampleDesc.Quality != desc.SampleDesc.Quality
        });

        if needs_recreate {
            slot.staging = None;
            slot.source_desc = Some(*desc);
            slot.populated = false;
        }
        region_pipeline::ensure_region_slot_texture(
            &self.device,
            slot,
            desc,
            needs_recreate,
            "failed to create WGC region staging texture",
        )
    }

    fn copy_region_source_to_slot(
        &self,
        slot_idx: usize,
        source_resource: &ID3D11Resource,
        blit: CaptureBlitRegion,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        region_pipeline::copy_region_source_to_slot(
            &self.context,
            &self.region.slots[slot_idx],
            source_resource,
            blit,
            can_use_dirty_gpu_copy,
        )
    }

    fn copy_source_to_slot(
        &mut self,
        slot_idx: usize,
        source_resource: &ID3D11Resource,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        let mut used_dirty_copy = false;
        {
            let slot = &self.staging_slots[slot_idx];
            let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                CaptureError::Platform(anyhow::anyhow!(
                    "failed to resolve WGC staging resource for slot {}",
                    slot_idx
                ))
            })?;

            if can_use_dirty_gpu_copy && slot.dirty_gpu_copy_preferred {
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
        region_pipeline::query_signaled(&self.context, query, flags, true)
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

    fn maybe_flush_region_after_submit(&self, write_slot: usize, read_slot: usize) {
        region_pipeline::maybe_flush_region_after_submit(
            &self.context,
            write_slot,
            read_slot,
            &self.region.slots[read_slot],
            true,
        );
    }

    fn wait_for_region_slot_copy(&mut self, slot_idx: usize) {
        self.region.adaptive_spin_polls = region_pipeline::wait_for_region_slot_copy(
            &self.context,
            &self.region.slots[slot_idx],
            self.region.adaptive_spin_polls,
            true,
        );
    }

    fn read_region_slot_into_output(
        &mut self,
        slot_idx: usize,
        out: &mut Frame,
        destination_has_history: bool,
        blit: CaptureBlitRegion,
    ) -> CaptureResult<CaptureSampleMetadata> {
        if !self.region.slots[slot_idx].populated {
            return Err(CaptureError::Timeout);
        }

        let slot = &self.region.slots[slot_idx];
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
            let slot = &self.region.slots[slot_idx];
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

            let use_dirty_copy = self.has_frame_history
                && destination_has_history
                && slot.dirty_cpu_copy_preferred
                && !slot.dirty_rects.is_empty();
            let write_full_slot_direct = blit.dst_x == 0
                && blit.dst_y == 0
                && out.width() == source_desc.Width
                && out.height() == source_desc.Height;
            let dirty_bounds_trusted = write_full_slot_direct
                || dirty_rect_destination_bounds_trusted(
                    blit.dst_x,
                    blit.dst_y,
                    source_desc.Width,
                    source_desc.Height,
                    out.width(),
                    out.height(),
                );
            let dirty_hints = dirty_rect_conversion_hints(
                &slot.dirty_rects,
                slot.dirty_total_pixels,
                dirty_bounds_trusted,
            );
            let map_full_slot = |out: &mut Frame| -> CaptureResult<()> {
                if write_full_slot_direct {
                    surface::map_staging_to_frame(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &source_desc,
                        out,
                        slot.hdr_to_sdr,
                        "failed to map WGC region staging texture",
                    )?;
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
                    )?;
                }
                Ok(())
            };

            if use_dirty_copy {
                let dirty_map_result = if write_full_slot_direct {
                    surface::map_staging_dirty_rects_to_frame(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &source_desc,
                        out,
                        &slot.dirty_rects,
                        true,
                        dirty_hints,
                        slot.hdr_to_sdr,
                        "failed to map WGC region staging texture (dirty regions)",
                    )
                } else {
                    surface::map_staging_dirty_rects_to_frame_with_offset(
                        &self.context,
                        staging,
                        Some(staging_resource),
                        &source_desc,
                        out,
                        &slot.dirty_rects,
                        blit.dst_x,
                        blit.dst_y,
                        true,
                        dirty_hints,
                        slot.hdr_to_sdr,
                        "failed to map WGC region staging texture (dirty regions)",
                    )
                };

                match dirty_map_result {
                    Ok(converted) if converted > 0 => Ok(()),
                    Ok(_) | Err(_) => map_full_slot(out),
                }
            } else {
                map_full_slot(out)
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
        let Some(slot_idx) = self.region.pending_slot else {
            return None;
        };
        if !self.region.slots[slot_idx].populated {
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

    fn try_short_circuit_duplicate(
        &mut self,
        capture_time: Instant,
        present_time_ticks: i64,
        out: &mut Frame,
    ) -> bool {
        let Some(slot_idx) = self.pending_slot else {
            return false;
        };
        if !self.staging_slots[slot_idx].populated {
            return false;
        }
        let out_width = out.width();
        let out_height = out.height();
        let out_matches_source = self.staging_slots[slot_idx]
            .source_desc
            .is_some_and(|desc| out_width == desc.Width && out_height == desc.Height);
        if !out_matches_source {
            return false;
        }

        {
            let slot = &mut self.staging_slots[slot_idx];
            slot.capture_time = Some(capture_time);
            slot.present_time_ticks = present_time_ticks;
            slot.is_duplicate = true;
            slot.dirty_mode_available = true;
            slot.dirty_cpu_copy_preferred = false;
            slot.dirty_gpu_copy_preferred = false;
            slot.dirty_total_pixels = 0;
            slot.dirty_rects.clear();
        }

        out.metadata.capture_time = Some(capture_time);
        out.metadata.present_time_qpc = if present_time_ticks != 0 {
            Some(present_time_ticks)
        } else {
            None
        };
        out.metadata.is_duplicate = true;
        out.metadata.dirty_rects.clear();

        if present_time_ticks != 0 {
            self.last_emitted_present_time = present_time_ticks;
        }
        true
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
                && slot.dirty_cpu_copy_preferred
                && !slot.dirty_rects.is_empty();
            let dirty_hints =
                dirty_rect_conversion_hints(&slot.dirty_rects, slot.dirty_total_pixels, true);

            if use_dirty_copy {
                match surface::map_staging_dirty_rects_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    &slot.dirty_rects,
                    true,
                    dirty_hints,
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
        self.region.ensure_blit(blit);

        let allow_stale_return =
            self.capture_mode == CaptureMode::ScreenRecording && destination_has_history;
        let capture_time = Instant::now();
        let single_shot_screenshot =
            self.capture_mode == CaptureMode::Screenshot && !destination_has_history;

        let maybe_capture = self.acquire_next_frame_or_stale_region(allow_stale_return)?;
        let (capture_frame, time_ticks) = if let Some(capture) = maybe_capture {
            capture
        } else if let Some(slot_idx) = self.region.pending_slot {
            match self.read_region_slot_into_output(slot_idx, out, destination_has_history, blit) {
                Ok(mut sample) => {
                    sample.capture_time = Some(capture_time);
                    sample.is_duplicate = true;
                    self.has_frame_history = true;
                    return Ok(sample);
                }
                Err(_) => {
                    self.region.reset();
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
            self.capture_mode,
            destination_has_history,
            source_is_duplicate,
            self.region.pending_slot.is_some(),
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

            let region_desc = surface::region_desc_for_blit(&effective_desc, blit);
            if single_shot_screenshot {
                let write_slot = 0usize;
                self.ensure_region_slot(write_slot, &region_desc)?;
                {
                    let slot = &self.region.slots[write_slot];
                    let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                        CaptureError::Platform(anyhow::anyhow!(
                            "failed to resolve WGC region staging resource for screenshot fast path"
                        ))
                    })?;
                    let source_box = D3D11_BOX {
                        left: blit.src_x,
                        top: blit.src_y,
                        front: 0,
                        right: src_right,
                        bottom: src_bottom,
                        back: 1,
                    };
                    d3d11::with_texture_resource(
                        &effective_source,
                        "failed to cast WGC region source texture to ID3D11Resource",
                        |source_resource| {
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
                            Ok(())
                        },
                    )?;
                }
                unsafe {
                    self.context.Flush();
                }
                {
                    let slot = &self.region.slots[write_slot];
                    let staging = slot.staging.as_ref().ok_or_else(|| {
                        CaptureError::Platform(anyhow::anyhow!(
                            "failed to resolve WGC region staging texture for screenshot fast path"
                        ))
                    })?;
                    surface::map_staging_to_frame(
                        &self.context,
                        staging,
                        slot.staging_resource.as_ref(),
                        &region_desc,
                        out,
                        effective_hdr,
                        "failed to map WGC region staging texture",
                    )?;
                }
                let sample = CaptureSampleMetadata {
                    capture_time: Some(capture_time),
                    present_time_qpc: if time_ticks != 0 {
                        Some(time_ticks)
                    } else {
                        None
                    },
                    is_duplicate: source_is_duplicate,
                };
                self.region.pending_slot = None;
                self.region.next_write_slot = 0;
                return Ok(sample);
            }
            let region_dirty_extraction = if source_is_duplicate {
                region_dirty_rects.clear();
                RegionDirtyRectExtraction {
                    available: true,
                    unchanged: true,
                    force_full_copy: false,
                }
            } else {
                let extraction = extract_region_dirty_rects(
                    &capture_frame,
                    effective_desc.Width,
                    effective_desc.Height,
                    blit,
                    &mut region_dirty_rects,
                );
                if !extraction.available {
                    region_dirty_rects.clear();
                }
                extraction
            };
            let region_dirty_available = region_dirty_extraction.available;
            let region_unchanged = region_dirty_extraction.unchanged;

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
            let region_dirty_strategy = if region_dirty_extraction.force_full_copy {
                DirtyCopyStrategy::default()
            } else {
                evaluate_dirty_copy_strategy(
                    &region_dirty_rects,
                    region_desc.Width,
                    region_desc.Height,
                )
            };
            let write_slot = if recording_mode {
                self.region.next_write_slot % WGC_STAGING_SLOTS
            } else {
                0
            };
            let read_slot = if recording_mode {
                self.region.pending_slot.unwrap_or(write_slot)
            } else {
                write_slot
            };

            let skip_submit_copy = recording_mode
                && destination_has_history
                && self.region.pending_slot.is_some()
                && (source_is_duplicate || region_unchanged);

            let read_slot = if skip_submit_copy {
                let slot_idx = self.region.pending_slot.unwrap_or(read_slot);
                let slot = &mut self.region.slots[slot_idx];
                slot.capture_time = Some(capture_time);
                slot.present_time_ticks = time_ticks;
                slot.is_duplicate = true;
                slot.dirty_mode_available = region_dirty_available;
                slot.dirty_cpu_copy_preferred = false;
                slot.dirty_gpu_copy_preferred = false;
                slot.dirty_total_pixels = 0;
                slot.dirty_rects.clear();
                slot.populated = true;
                slot_idx
            } else {
                self.ensure_region_slot(write_slot, &region_desc)?;
                let can_use_dirty_gpu_copy = destination_has_history && {
                    let slot = &self.region.slots[write_slot];
                    slot.populated
                        && slot.present_time_ticks != 0
                        && slot.present_time_ticks == previous_present_time
                };
                {
                    let slot = &mut self.region.slots[write_slot];
                    slot.capture_time = Some(capture_time);
                    slot.present_time_ticks = time_ticks;
                    slot.is_duplicate = source_is_duplicate || region_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(region_desc);
                    slot.dirty_mode_available = region_dirty_available;
                    std::mem::swap(&mut slot.dirty_rects, &mut region_dirty_rects);
                    slot.dirty_cpu_copy_preferred =
                        region_dirty_available && region_dirty_strategy.cpu;
                    slot.dirty_gpu_copy_preferred = can_use_dirty_gpu_copy
                        && region_dirty_available
                        && region_dirty_strategy.gpu;
                    slot.dirty_total_pixels = region_dirty_strategy.dirty_pixels;
                    slot.populated = true;
                }

                d3d11::with_texture_resource(
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
                    self.region.pending_slot = Some(write_slot);
                    self.region.next_write_slot = (write_slot + 1) % WGC_STAGING_SLOTS;
                }
            } else {
                self.region.pending_slot = None;
                self.region.next_write_slot = 0;
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
                self.region.reset();
                self.has_frame_history = false;
                Err(err)
            }
        }
    }

    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.region.pending_slot = None;
        self.region.next_write_slot = 0;
        self.region.blit = None;

        let mut out = reuse.unwrap_or_else(Frame::empty);
        let destination_has_history =
            out.metadata.capture_time.is_some() && !out.as_rgba_bytes().is_empty();
        let single_shot_screenshot =
            self.capture_mode == CaptureMode::Screenshot && !destination_has_history;
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

        if should_short_circuit_duplicate(
            self.capture_mode,
            destination_has_history,
            source_is_duplicate,
            self.pending_slot.is_some(),
        ) && self.try_short_circuit_duplicate(capture_time, time_ticks, &mut out)
        {
            let _ = capture_frame.Close();
            self.has_frame_history = true;
            return Ok(out);
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

            if single_shot_screenshot {
                let write_slot = 0usize;
                self.ensure_staging_slot(write_slot, &effective_desc)?;
                {
                    let slot = &self.staging_slots[write_slot];
                    let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
                        CaptureError::Platform(anyhow::anyhow!(
                            "failed to resolve WGC staging resource for screenshot fast path"
                        ))
                    })?;
                    d3d11::with_texture_resource(
                        &effective_source,
                        "failed to cast WGC frame texture to ID3D11Resource",
                        |source_resource| {
                            unsafe {
                                self.context.CopyResource(staging_resource, source_resource);
                            }
                            Ok(())
                        },
                    )?;
                }
                unsafe {
                    self.context.Flush();
                }
                {
                    let slot = &self.staging_slots[write_slot];
                    let staging = slot.staging.as_ref().ok_or_else(|| {
                        CaptureError::Platform(anyhow::anyhow!(
                            "failed to resolve WGC staging texture for screenshot fast path"
                        ))
                    })?;
                    surface::map_staging_to_frame(
                        &self.context,
                        staging,
                        slot.staging_resource.as_ref(),
                        &effective_desc,
                        &mut out,
                        effective_hdr,
                        "failed to map WGC staging texture",
                    )?;
                }
                out.metadata.capture_time = Some(capture_time);
                out.metadata.present_time_qpc = if time_ticks != 0 {
                    Some(time_ticks)
                } else {
                    None
                };
                out.metadata.is_duplicate = source_is_duplicate;
                out.metadata.dirty_rects.clear();
                if time_ticks != 0 {
                    self.last_emitted_present_time = time_ticks;
                }
                self.pending_slot = None;
                self.next_write_slot = 0;
                return Ok(());
            }

            let (source_dirty_available, source_unchanged) = if source_is_duplicate {
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
                slot.dirty_cpu_copy_preferred = false;
                slot.dirty_gpu_copy_preferred = false;
                slot.dirty_total_pixels = 0;
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
                    std::mem::swap(&mut slot.dirty_rects, &mut source_dirty_rects);
                    let strategy = evaluate_dirty_copy_strategy(
                        &slot.dirty_rects,
                        effective_desc.Width,
                        effective_desc.Height,
                    );
                    slot.dirty_cpu_copy_preferred = source_dirty_available && strategy.cpu;
                    slot.dirty_gpu_copy_preferred =
                        can_use_dirty_gpu_copy && source_dirty_available && strategy.gpu;
                    slot.dirty_total_pixels = strategy.dirty_pixels;
                    slot.capture_time = Some(capture_time);
                    slot.present_time_ticks = time_ticks;
                    slot.is_duplicate = source_is_duplicate || source_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(effective_desc);
                    slot.populated = true;
                }

                d3d11::with_texture_resource(
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
        let dirty_mode = if mode == CaptureMode::ScreenRecording {
            GraphicsCaptureDirtyRegionMode::ReportAndRender
        } else {
            GraphicsCaptureDirtyRegionMode::ReportOnly
        };
        let _ = self.session.SetDirtyRegionMode(dirty_mode);
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
    fn dirty_gpu_copy_heuristic_accepts_sparse_updates() {
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
        assert!(should_use_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_gpu_copy_heuristic_rejects_wide_damage() {
        let rects = vec![DirtyRect {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        }];
        assert!(!should_use_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_copy_strategy_separates_cpu_and_gpu_thresholds() {
        let rects = vec![DirtyRect {
            x: 0,
            y: 0,
            width: 1920,
            height: 540,
        }];
        let strategy = evaluate_dirty_copy_strategy(&rects, 1920, 1080);
        assert!(strategy.cpu);
        assert!(!strategy.gpu);
        assert!(!strategy.gpu_low_latency);
    }

    #[test]
    fn dirty_copy_strategy_applies_rect_count_limits_per_mode() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            };
            WGC_DIRTY_GPU_COPY_MAX_RECTS + 1
        ];
        let strategy = evaluate_dirty_copy_strategy(&rects, 1920, 1080);
        assert!(strategy.cpu);
        assert!(!strategy.gpu);
        assert!(!strategy.gpu_low_latency);
    }

    #[test]
    fn dirty_copy_strategy_reports_dirty_pixel_totals() {
        let rects = vec![
            DirtyRect {
                x: 10,
                y: 10,
                width: 32,
                height: 24,
            },
            DirtyRect {
                x: 100,
                y: 80,
                width: 16,
                height: 12,
            },
        ];
        let strategy = evaluate_dirty_copy_strategy(&rects, 1920, 1080);
        assert_eq!(strategy.dirty_pixels, (32_u64 * 24_u64) + (16_u64 * 12_u64));
    }

    #[test]
    fn low_latency_dirty_gpu_copy_heuristic_is_stricter() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            };
            WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS + 1
        ];
        assert!(!should_use_low_latency_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_rect_destination_bounds_trusted_requires_in_bounds_destination() {
        assert!(dirty_rect_destination_bounds_trusted(
            32, 48, 640, 360, 1920, 1080
        ));
        assert!(!dirty_rect_destination_bounds_trusted(
            1500, 900, 640, 360, 1920, 1080
        ));
    }

    #[test]
    fn dirty_rect_destination_bounds_trusted_rejects_coordinate_overflow() {
        assert!(!dirty_rect_destination_bounds_trusted(
            u32::MAX,
            12,
            8,
            8,
            u32::MAX,
            u32::MAX,
        ));
        assert!(!dirty_rect_destination_bounds_trusted(
            12,
            u32::MAX,
            8,
            8,
            u32::MAX,
            u32::MAX,
        ));
    }

    #[test]
    fn duplicate_short_circuit_requires_recording_mode() {
        assert!(!should_short_circuit_duplicate(
            CaptureMode::Screenshot,
            true,
            true,
            true,
        ));
        assert!(should_short_circuit_duplicate(
            CaptureMode::ScreenRecording,
            true,
            true,
            true,
        ));
    }

    #[test]
    fn duplicate_short_circuit_requires_duplicate_with_history() {
        assert!(!should_short_circuit_duplicate(
            CaptureMode::ScreenRecording,
            false,
            true,
            true,
        ));
        assert!(!should_short_circuit_duplicate(
            CaptureMode::ScreenRecording,
            true,
            false,
            true,
        ));
    }

    #[test]
    fn duplicate_short_circuit_requires_pending_slot() {
        assert!(!should_short_circuit_duplicate(
            CaptureMode::ScreenRecording,
            true,
            true,
            false,
        ));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_recording_mode() {
        assert!(!should_short_circuit_region_duplicate(
            CaptureMode::Screenshot,
            true,
            true,
            true,
        ));
        assert!(should_short_circuit_region_duplicate(
            CaptureMode::ScreenRecording,
            true,
            true,
            true,
        ));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_duplicate_with_history() {
        assert!(!should_short_circuit_region_duplicate(
            CaptureMode::ScreenRecording,
            false,
            true,
            true,
        ));
        assert!(!should_short_circuit_region_duplicate(
            CaptureMode::ScreenRecording,
            true,
            false,
            true,
        ));
    }

    #[test]
    fn region_duplicate_short_circuit_requires_pending_slot() {
        assert!(!should_short_circuit_region_duplicate(
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
    fn normalize_dirty_rects_matches_reference_algorithm_on_randomized_inputs() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;

        for _case in 0..256 {
            let mut input = Vec::with_capacity(160);
            let rect_count = 8 + ((state >> 5) as usize % 152);
            for _ in 0..rect_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = ((state >> 12) as u32) % 2600;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = ((state >> 20) as u32) % 1500;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let width = ((state >> 24) as u32) % 900;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let height = ((state >> 28) as u32) % 500;
                input.push(DirtyRect {
                    x,
                    y,
                    width,
                    height,
                });
            }

            let mut optimized = input.clone();
            let mut reference = input;
            normalize_dirty_rects_in_place(&mut optimized, 1920, 1080);
            normalize_dirty_rects_reference_in_place(&mut reference, 1920, 1080);
            assert_eq!(optimized, reference);
        }
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

    fn clamp_dirty_region_rect_legacy(
        raw: RectInt32,
        width: u32,
        height: u32,
    ) -> Option<DirtyRect> {
        if raw.Width <= 0 || raw.Height <= 0 {
            return None;
        }
        let x = raw.X.max(0) as u32;
        let y = raw.Y.max(0) as u32;
        let rect_width = raw.Width as u32;
        let rect_height = raw.Height as u32;
        dirty_rect::clamp_dirty_rect(
            DirtyRect {
                x,
                y,
                width: rect_width,
                height: rect_height,
            },
            width,
            height,
        )
    }

    fn normalize_raw_dirty_rects_legacy(
        raw_rects: &[RectInt32],
        width: u32,
        height: u32,
        out: &mut Vec<DirtyRect>,
    ) {
        out.clear();
        out.reserve(raw_rects.len());
        for raw in raw_rects {
            if let Some(clamped) = clamp_dirty_region_rect_legacy(*raw, width, height) {
                out.push(clamped);
            }
        }
        normalize_dirty_rects_in_place(out, width, height);
    }

    fn normalize_raw_dirty_rects_optimized(
        raw_rects: &[RectInt32],
        width: u32,
        height: u32,
        out: &mut Vec<DirtyRect>,
    ) {
        out.clear();
        out.reserve(raw_rects.len());
        let mut sorted_hint = true;
        let mut previous_rect = None;
        for raw in raw_rects {
            if let Some(clamped) = clamp_dirty_region_rect(*raw, width, height) {
                update_dirty_rect_sort_hint(&mut sorted_hint, &mut previous_rect, clamped);
                out.push(clamped);
            }
        }
        normalize_dirty_rects_preclamped_in_place(out, width, height, sorted_hint);
    }

    fn extract_region_raw_dirty_rects_legacy(
        raw_rects: &[RectInt32],
        source_width: u32,
        source_height: u32,
        blit: CaptureBlitRegion,
        out: &mut Vec<DirtyRect>,
    ) -> RegionDirtyRectExtraction {
        out.clear();
        let Some(region_bounds) = dirty_rect::clamp_dirty_rect(
            DirtyRect {
                x: blit.src_x,
                y: blit.src_y,
                width: blit.width,
                height: blit.height,
            },
            source_width,
            source_height,
        ) else {
            return RegionDirtyRectExtraction::default();
        };

        out.reserve(raw_rects.len());
        for raw in raw_rects {
            let Some(clamped) = clamp_dirty_region_rect_legacy(*raw, source_width, source_height)
            else {
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
        RegionDirtyRectExtraction {
            available: true,
            unchanged: out.is_empty(),
            force_full_copy: false,
        }
    }

    fn extract_region_raw_dirty_rects_optimized(
        raw_rects: &[RectInt32],
        source_width: u32,
        source_height: u32,
        blit: CaptureBlitRegion,
        dense_fallback_enabled: bool,
        out: &mut Vec<DirtyRect>,
    ) -> RegionDirtyRectExtraction {
        out.clear();
        let Some(region_bounds) = dirty_rect::clamp_dirty_rect(
            DirtyRect {
                x: blit.src_x,
                y: blit.src_y,
                width: blit.width,
                height: blit.height,
            },
            source_width,
            source_height,
        ) else {
            return RegionDirtyRectExtraction::default();
        };
        let Some(clip_bounds) = RegionClipBounds::new(source_width, source_height, region_bounds)
        else {
            return RegionDirtyRectExtraction::default();
        };

        let total_region_pixels =
            (region_bounds.width as u64).saturating_mul(region_bounds.height as u64);
        let mut dense_fallback = false;
        let mut dirty_pixels = 0u64;
        let mut non_empty_rects = 0usize;
        let track_sorted_hint = !dense_fallback_enabled;
        let mut sorted_hint = track_sorted_hint;
        let mut previous_rect = None;
        out.reserve(raw_rects.len());

        for raw in raw_rects {
            if dense_fallback {
                break;
            }
            let Some(clipped) = clip_dirty_region_rect_to_region_with_bounds(*raw, clip_bounds)
            else {
                continue;
            };
            non_empty_rects = non_empty_rects.saturating_add(1);
            dirty_pixels = dirty_pixels
                .saturating_add((clipped.width as u64).saturating_mul(clipped.height as u64));
            let exceeds_dense_area = total_region_pixels > 0
                && dirty_pixels.saturating_mul(100)
                    > total_region_pixels
                        .saturating_mul(WGC_REGION_DIRTY_DENSE_FALLBACK_AREA_PERCENT);
            if dense_fallback_enabled
                && (non_empty_rects > WGC_REGION_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS
                    || (non_empty_rects > WGC_REGION_DIRTY_DENSE_FALLBACK_MIN_RECTS
                        && exceeds_dense_area))
            {
                dense_fallback = true;
                out.clear();
                break;
            }
            if track_sorted_hint {
                update_dirty_rect_sort_hint(&mut sorted_hint, &mut previous_rect, clipped);
            }
            out.push(clipped);
        }

        if dense_fallback {
            return RegionDirtyRectExtraction {
                available: true,
                unchanged: false,
                force_full_copy: true,
            };
        }

        if non_empty_rects == 0 {
            return RegionDirtyRectExtraction {
                available: true,
                unchanged: true,
                force_full_copy: false,
            };
        }

        normalize_dirty_rects_preclamped_in_place(
            out,
            region_bounds.width,
            region_bounds.height,
            track_sorted_hint && sorted_hint,
        );
        RegionDirtyRectExtraction {
            available: true,
            unchanged: out.is_empty(),
            force_full_copy: false,
        }
    }

    fn extract_full_raw_dirty_rects_legacy(
        raw_rects: &[RectInt32],
        width: u32,
        height: u32,
        out: &mut Vec<DirtyRect>,
    ) -> RegionDirtyRectExtraction {
        out.clear();
        out.reserve(raw_rects.len());
        for raw in raw_rects {
            if let Some(clamped) = clamp_dirty_region_rect_legacy(*raw, width, height) {
                out.push(clamped);
            }
        }
        normalize_dirty_rects_in_place(out, width, height);
        RegionDirtyRectExtraction {
            available: true,
            unchanged: out.is_empty(),
            force_full_copy: false,
        }
    }

    fn extract_full_raw_dirty_rects_optimized(
        raw_rects: &[RectInt32],
        width: u32,
        height: u32,
        dense_fallback_enabled: bool,
        out: &mut Vec<DirtyRect>,
    ) -> RegionDirtyRectExtraction {
        out.clear();
        let total_pixels = (width as u64).saturating_mul(height as u64);
        let mut dense_fallback = false;
        let mut dirty_pixels = 0u64;
        let mut non_empty_rects = 0usize;
        let track_sorted_hint = !dense_fallback_enabled;
        let mut sorted_hint = track_sorted_hint;
        let mut previous_rect = None;
        out.reserve(raw_rects.len());

        for raw in raw_rects {
            if dense_fallback {
                break;
            }
            let Some(clamped) = clamp_dirty_region_rect(*raw, width, height) else {
                continue;
            };
            non_empty_rects = non_empty_rects.saturating_add(1);
            dirty_pixels = dirty_pixels
                .saturating_add((clamped.width as u64).saturating_mul(clamped.height as u64));

            let exceeds_dense_area = total_pixels > 0
                && dirty_pixels.saturating_mul(100)
                    > total_pixels.saturating_mul(WGC_FULL_DIRTY_DENSE_FALLBACK_AREA_PERCENT);
            if dense_fallback_enabled
                && (non_empty_rects > WGC_FULL_DIRTY_DENSE_FALLBACK_HARD_MAX_RECTS
                    || (non_empty_rects > WGC_FULL_DIRTY_DENSE_FALLBACK_MIN_RECTS
                        && exceeds_dense_area))
            {
                dense_fallback = true;
                out.clear();
                if width > 0 && height > 0 {
                    out.push(DirtyRect {
                        x: 0,
                        y: 0,
                        width,
                        height,
                    });
                }
                break;
            }

            if track_sorted_hint {
                update_dirty_rect_sort_hint(&mut sorted_hint, &mut previous_rect, clamped);
            }
            out.push(clamped);
        }

        if dense_fallback {
            return RegionDirtyRectExtraction {
                available: true,
                unchanged: false,
                force_full_copy: true,
            };
        }

        normalize_dirty_rects_preclamped_in_place(
            out,
            width,
            height,
            track_sorted_hint && sorted_hint,
        );
        RegionDirtyRectExtraction {
            available: true,
            unchanged: out.is_empty(),
            force_full_copy: false,
        }
    }

    #[test]
    fn dirty_region_rect_clamp_matches_legacy_logic() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..1024 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (((state >> 8) as i32) % 2800) - 400;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (((state >> 16) as i32) % 1700) - 300;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let width = (((state >> 24) as i32) % 1400) - 40;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let height = (((state >> 32) as i32) % 900) - 30;

            let raw = RectInt32 {
                X: x,
                Y: y,
                Width: width,
                Height: height,
            };
            let legacy = clamp_dirty_region_rect_legacy(raw, 1920, 1080);
            let optimized = clamp_dirty_region_rect(raw, 1920, 1080);
            assert_eq!(optimized, legacy);
        }
    }

    #[test]
    fn region_dirty_extraction_matches_legacy_when_dense_fallback_is_disabled() {
        let mut raw_rects = row_major_raw_rects(32, 24, 14, 6, 36, 24, 14, 12, 84);
        raw_rects.extend(row_major_raw_rects(680, 420, 5, 4, 22, 18, 18, 14, 20));

        let blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: 1920,
            height: 1080,
            dst_x: 0,
            dst_y: 0,
        };

        let mut legacy = Vec::new();
        let mut optimized = Vec::new();
        let legacy_state =
            extract_region_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, blit, &mut legacy);
        let optimized_state = extract_region_raw_dirty_rects_optimized(
            &raw_rects,
            1920,
            1080,
            blit,
            false,
            &mut optimized,
        );

        assert_eq!(legacy_state, optimized_state);
        assert_eq!(legacy, optimized);
    }

    #[test]
    fn region_dirty_extraction_matches_legacy_for_out_of_bounds_rects_when_dense_fallback_disabled()
    {
        let blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: 1920,
            height: 1080,
            dst_x: 0,
            dst_y: 0,
        };

        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for case_idx in 0..256 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rect_count = 6 + ((state >> 7) as usize % 190);
            let mut raw_rects = Vec::with_capacity(rect_count);
            for _ in 0..rect_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = (((state >> 8) as i32) % 3600) - 1200;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = (((state >> 20) as i32) % 2200) - 700;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let width = (((state >> 28) as i32) % 960) - 80;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let height = (((state >> 36) as i32) % 720) - 60;
                raw_rects.push(RectInt32 {
                    X: x,
                    Y: y,
                    Width: width,
                    Height: height,
                });
            }

            let mut legacy = Vec::new();
            let mut optimized = Vec::new();
            let legacy_state =
                extract_region_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, blit, &mut legacy);
            let optimized_state = extract_region_raw_dirty_rects_optimized(
                &raw_rects,
                1920,
                1080,
                blit,
                false,
                &mut optimized,
            );

            assert_eq!(
                legacy_state, optimized_state,
                "state mismatch in randomized case {case_idx}",
            );
            assert_eq!(
                legacy, optimized,
                "rect mismatch in randomized case {case_idx}"
            );
        }
    }

    #[test]
    fn region_dirty_extraction_dense_fallback_preserves_full_copy_decision() {
        let raw_rects = row_major_raw_rects(0, 0, 45, 20, 8, 6, 2, 2, 900);
        let blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: 1920,
            height: 1080,
            dst_x: 0,
            dst_y: 0,
        };

        let mut legacy = Vec::new();
        let mut optimized = Vec::new();
        let legacy_state =
            extract_region_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, blit, &mut legacy);
        let optimized_state = extract_region_raw_dirty_rects_optimized(
            &raw_rects,
            1920,
            1080,
            blit,
            true,
            &mut optimized,
        );

        assert!(legacy_state.available);
        assert!(!legacy_state.unchanged);
        let legacy_strategy = evaluate_dirty_copy_strategy(&legacy, blit.width, blit.height);
        assert_eq!(legacy_strategy, DirtyCopyStrategy::default());

        assert!(optimized_state.available);
        assert!(!optimized_state.unchanged);
        assert!(optimized_state.force_full_copy);
        assert!(optimized.is_empty());
    }

    #[test]
    fn full_dirty_extraction_matches_legacy_when_dense_fallback_is_disabled() {
        let mut raw_rects = row_major_raw_rects(24, 20, 16, 8, 24, 20, 12, 10, 128);
        raw_rects.extend(row_major_raw_rects(640, 360, 6, 5, 18, 14, 16, 12, 30));

        let mut legacy = Vec::new();
        let mut optimized = Vec::new();
        let legacy_state = extract_full_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, &mut legacy);
        let optimized_state =
            extract_full_raw_dirty_rects_optimized(&raw_rects, 1920, 1080, false, &mut optimized);

        assert_eq!(legacy_state, optimized_state);
        assert_eq!(legacy, optimized);
    }

    #[test]
    fn full_dirty_extraction_matches_legacy_for_out_of_bounds_rects_when_dense_fallback_disabled() {
        let mut state = 0x510e_527f_9b05_688c_u64;
        for case_idx in 0..256 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rect_count = 6 + ((state >> 7) as usize % 190);
            let mut raw_rects = Vec::with_capacity(rect_count);
            for _ in 0..rect_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = (((state >> 8) as i32) % 3600) - 1200;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = (((state >> 20) as i32) % 2200) - 700;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let width = (((state >> 28) as i32) % 960) - 80;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let height = (((state >> 36) as i32) % 720) - 60;
                raw_rects.push(RectInt32 {
                    X: x,
                    Y: y,
                    Width: width,
                    Height: height,
                });
            }

            let mut legacy = Vec::new();
            let mut optimized = Vec::new();
            let legacy_state =
                extract_full_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, &mut legacy);
            let optimized_state = extract_full_raw_dirty_rects_optimized(
                &raw_rects,
                1920,
                1080,
                false,
                &mut optimized,
            );

            assert_eq!(
                legacy_state, optimized_state,
                "state mismatch in randomized case {case_idx}",
            );
            assert_eq!(
                legacy, optimized,
                "rect mismatch in randomized case {case_idx}",
            );
        }
    }

    #[test]
    fn full_dirty_extraction_dense_fallback_preserves_full_copy_decision() {
        let raw_rects = row_major_raw_rects(0, 0, 45, 20, 8, 6, 2, 2, 900);

        let mut legacy = Vec::new();
        let mut optimized = Vec::new();
        let legacy_state = extract_full_raw_dirty_rects_legacy(&raw_rects, 1920, 1080, &mut legacy);
        let optimized_state =
            extract_full_raw_dirty_rects_optimized(&raw_rects, 1920, 1080, true, &mut optimized);

        assert!(legacy_state.available);
        assert!(!legacy_state.unchanged);
        let legacy_strategy = evaluate_dirty_copy_strategy(&legacy, 1920, 1080);
        assert_eq!(legacy_strategy, DirtyCopyStrategy::default());

        assert!(optimized_state.available);
        assert!(!optimized_state.unchanged);
        assert!(optimized_state.force_full_copy);
        assert_eq!(
            optimized,
            vec![DirtyRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }],
        );
        let optimized_strategy = evaluate_dirty_copy_strategy(&optimized, 1920, 1080);
        assert!(!optimized_strategy.cpu);
        assert!(!optimized_strategy.gpu);
        assert!(!optimized_strategy.gpu_low_latency);
    }

    fn should_use_dirty_copy_legacy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
        if rects.is_empty() || rects.len() > WGC_DIRTY_COPY_MAX_RECTS {
            return false;
        }
        let total_pixels = (width as u64).saturating_mul(height as u64);
        if total_pixels == 0 {
            return false;
        }
        let cpu_limit = total_pixels.saturating_mul(WGC_DIRTY_COPY_MAX_AREA_PERCENT);
        let mut dirty_pixels = 0u64;
        for rect in rects {
            dirty_pixels =
                dirty_pixels.saturating_add((rect.width as u64).saturating_mul(rect.height as u64));
            if dirty_pixels.saturating_mul(100) > cpu_limit {
                return false;
            }
        }
        true
    }

    fn should_use_dirty_gpu_copy_legacy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
        if rects.is_empty() || rects.len() > WGC_DIRTY_GPU_COPY_MAX_RECTS {
            return false;
        }
        let total_pixels = (width as u64).saturating_mul(height as u64);
        if total_pixels == 0 {
            return false;
        }
        let gpu_limit = total_pixels.saturating_mul(WGC_DIRTY_GPU_COPY_MAX_AREA_PERCENT);
        let mut dirty_pixels = 0u64;
        for rect in rects {
            dirty_pixels =
                dirty_pixels.saturating_add((rect.width as u64).saturating_mul(rect.height as u64));
            if dirty_pixels.saturating_mul(100) > gpu_limit {
                return false;
            }
        }
        true
    }

    fn should_use_low_latency_dirty_gpu_copy_legacy(
        rects: &[DirtyRect],
        width: u32,
        height: u32,
    ) -> bool {
        if rects.is_empty() || rects.len() > WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS {
            return false;
        }
        let total_pixels = (width as u64).saturating_mul(height as u64);
        if total_pixels == 0 {
            return false;
        }
        let limit = total_pixels.saturating_mul(WGC_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT);
        let mut dirty_pixels = 0u64;
        for rect in rects {
            dirty_pixels =
                dirty_pixels.saturating_add((rect.width as u64).saturating_mul(rect.height as u64));
            if dirty_pixels.saturating_mul(100) > limit {
                return false;
            }
        }
        true
    }

    fn row_major_raw_rects(
        start_x: i32,
        start_y: i32,
        cols: i32,
        rows: i32,
        rect_w: i32,
        rect_h: i32,
        gap_x: i32,
        gap_y: i32,
        limit: usize,
    ) -> Vec<RectInt32> {
        let mut out = Vec::with_capacity(limit);
        for row in 0..rows {
            for col in 0..cols {
                if out.len() == limit {
                    return out;
                }
                out.push(RectInt32 {
                    X: start_x + col * (rect_w + gap_x),
                    Y: start_y + row * (rect_h + gap_y),
                    Width: rect_w,
                    Height: rect_h,
                });
            }
        }
        out
    }

    #[test]
    fn region_clip_fastpath_matches_fallback() {
        let bounds = RegionClipBounds::new(
            3840,
            2160,
            DirtyRect {
                x: 320,
                y: 180,
                width: 1920,
                height: 1080,
            },
        )
        .expect("clip bounds must be valid");

        let mut state = 0x243f_6a88_85a3_08d3_u64;
        for case_idx in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (((state >> 9) as i32) % 5200) - 900;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (((state >> 21) as i32) % 3200) - 500;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let width = (((state >> 31) as i32) % 1800) - 120;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let height = (((state >> 41) as i32) % 1400) - 120;
            let raw = RectInt32 {
                X: x,
                Y: y,
                Width: width,
                Height: height,
            };
            let fallback = clip_dirty_region_rect_to_region_fallback(raw, bounds);
            let optimized = clip_dirty_region_rect_to_region_with_bounds(raw, bounds);
            assert_eq!(
                optimized, fallback,
                "clip mismatch in randomized case {case_idx}",
            );
        }
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_region_clip_fastpath_vs_fallback() {
        use std::hint::black_box;

        let bounds = RegionClipBounds::new(
            3840,
            2160,
            DirtyRect {
                x: 320,
                y: 180,
                width: 1920,
                height: 1080,
            },
        )
        .expect("clip bounds must be valid");

        let mut predominantly_positive = row_major_raw_rects(0, 0, 48, 18, 24, 20, 12, 10, 640);
        predominantly_positive.extend(row_major_raw_rects(1400, 620, 20, 10, 36, 24, 20, 12, 220));
        let mut mixed_edges = row_major_raw_rects(-120, -80, 38, 16, 26, 18, 10, 12, 560);
        mixed_edges.extend(row_major_raw_rects(3000, 1500, 16, 8, 64, 48, 28, 20, 220));

        let mut random_workload = Vec::with_capacity(980);
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..980 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (((state >> 7) as i32) % 5600) - 1000;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (((state >> 19) as i32) % 3400) - 700;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let width = (((state >> 29) as i32) % 2100) - 160;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let height = (((state >> 39) as i32) % 1500) - 140;
            random_workload.push(RectInt32 {
                X: x,
                Y: y,
                Width: width,
                Height: height,
            });
        }

        let workloads = vec![predominantly_positive, mixed_edges, random_workload];
        for rects in &workloads {
            for raw in rects {
                let fallback = clip_dirty_region_rect_to_region_fallback(*raw, bounds);
                let optimized = clip_dirty_region_rect_to_region_with_bounds(*raw, bounds);
                assert_eq!(optimized, fallback);
            }
        }

        let warmup_iters = 12_000usize;
        let measure_iters = 120_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            for raw in rects {
                let fallback = clip_dirty_region_rect_to_region_fallback(*raw, bounds);
                sink ^= fallback.map(|rect| rect.width as usize).unwrap_or(0);
                let optimized = clip_dirty_region_rect_to_region_with_bounds(*raw, bounds);
                sink ^= optimized.map(|rect| rect.height as usize).unwrap_or(0);
            }
        }

        let fallback_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            for raw in rects {
                let rect = clip_dirty_region_rect_to_region_fallback(*raw, bounds);
                sink ^= rect.map(|r| r.width as usize).unwrap_or(0);
            }
        }
        let fallback_elapsed = fallback_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            for raw in rects {
                let rect = clip_dirty_region_rect_to_region_with_bounds(*raw, bounds);
                sink ^= rect.map(|r| r.height as usize).unwrap_or(0);
            }
        }
        let optimized_elapsed = optimized_start.elapsed();

        black_box(sink);

        let fallback_ms = fallback_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            fallback_ms / optimized_ms
        } else {
            f64::INFINITY
        };
        println!(
            "wgc region clip benchmark: fallback={fallback_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= fallback_ms * 0.92,
            "region clip fast path did not improve enough: fallback={fallback_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_dirty_region_rect_clamp_and_normalize_vs_legacy() {
        use std::hint::black_box;

        let mut random_workload = Vec::with_capacity(220);
        let mut state = 0xfeed_f00d_cafe_babe_u64;
        for _ in 0..220 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = (((state >> 5) as i32) % 2600) - 280;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = (((state >> 15) as i32) % 1500) - 220;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let width = (((state >> 23) as i32) % 420) - 24;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let height = (((state >> 31) as i32) % 260) - 16;
            random_workload.push(RectInt32 {
                X: x,
                Y: y,
                Width: width,
                Height: height,
            });
        }

        let workloads = vec![
            row_major_raw_rects(12, 12, 12, 4, 20, 18, 10, 10, 48),
            row_major_raw_rects(4, 4, 22, 9, 8, 7, 5, 4, 180),
            random_workload,
        ];

        let mut legacy = Vec::<DirtyRect>::new();
        let mut optimized = Vec::<DirtyRect>::new();
        for rects in &workloads {
            normalize_raw_dirty_rects_legacy(rects, 1920, 1080, &mut legacy);
            normalize_raw_dirty_rects_optimized(rects, 1920, 1080, &mut optimized);
            assert_eq!(optimized, legacy);
        }

        let warmup_iters = 8_000usize;
        let measure_iters = 60_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            normalize_raw_dirty_rects_legacy(rects, 1920, 1080, &mut legacy);
            sink ^= legacy.len();
            normalize_raw_dirty_rects_optimized(rects, 1920, 1080, &mut optimized);
            sink ^= optimized.len() << 1;
        }

        let legacy_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            normalize_raw_dirty_rects_legacy(rects, 1920, 1080, &mut legacy);
            sink ^= legacy.len();
            sink ^= legacy.first().map(|rect| rect.width as usize).unwrap_or(0);
        }
        let legacy_elapsed = legacy_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            normalize_raw_dirty_rects_optimized(rects, 1920, 1080, &mut optimized);
            sink ^= optimized.len() << 1;
            sink ^= optimized
                .first()
                .map(|rect| rect.height as usize)
                .unwrap_or(0);
        }
        let optimized_elapsed = optimized_start.elapsed();

        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            f64::INFINITY
        };
        println!(
            "wgc dirty-rect clamp benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.05,
            "dirty-rect clamp/normalize regressed: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_region_dirty_dense_fallback_vs_legacy() {
        use std::hint::black_box;

        let blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: 1920,
            height: 1080,
            dst_x: 0,
            dst_y: 0,
        };
        let workloads = vec![
            row_major_raw_rects(0, 0, 45, 20, 8, 6, 2, 2, 900),
            row_major_raw_rects(0, 0, 20, 16, 80, 60, 16, 6, 320),
            row_major_raw_rects(4, 6, 40, 22, 10, 8, 2, 2, 820),
        ];

        let mut legacy_rects = Vec::<DirtyRect>::new();
        let mut optimized_rects = Vec::<DirtyRect>::new();
        for raw_rects in &workloads {
            let legacy_state = extract_region_raw_dirty_rects_legacy(
                raw_rects,
                1920,
                1080,
                blit,
                &mut legacy_rects,
            );
            let legacy_strategy =
                evaluate_dirty_copy_strategy(&legacy_rects, blit.width, blit.height);
            assert!(legacy_state.available);
            assert!(!legacy_state.unchanged);
            assert_eq!(
                legacy_strategy,
                DirtyCopyStrategy::default(),
                "workload unexpectedly eligible for dirty copy; benchmark expects full-copy fallback",
            );

            let optimized_state = extract_region_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                blit,
                true,
                &mut optimized_rects,
            );
            assert!(optimized_state.available);
            assert!(!optimized_state.unchanged);
            assert!(optimized_state.force_full_copy);
            assert!(optimized_rects.is_empty());
        }

        let warmup_iters = 8_000usize;
        let measure_iters = 70_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let legacy_state = extract_region_raw_dirty_rects_legacy(
                raw_rects,
                1920,
                1080,
                blit,
                &mut legacy_rects,
            );
            let legacy_strategy =
                evaluate_dirty_copy_strategy(&legacy_rects, blit.width, blit.height);
            sink ^= legacy_rects.len();
            sink ^= legacy_state.available as usize;
            sink ^= (legacy_strategy.cpu as usize) << 1;

            let optimized_state = extract_region_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                blit,
                true,
                &mut optimized_rects,
            );
            let optimized_strategy = if optimized_state.force_full_copy {
                DirtyCopyStrategy::default()
            } else {
                evaluate_dirty_copy_strategy(&optimized_rects, blit.width, blit.height)
            };
            sink ^= optimized_rects.len() << 1;
            sink ^= (optimized_state.force_full_copy as usize) << 2;
            sink ^= (optimized_strategy.gpu as usize) << 3;
        }

        let legacy_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let legacy_state = extract_region_raw_dirty_rects_legacy(
                raw_rects,
                1920,
                1080,
                blit,
                &mut legacy_rects,
            );
            let legacy_strategy =
                evaluate_dirty_copy_strategy(&legacy_rects, blit.width, blit.height);
            sink ^= legacy_rects.len();
            sink ^= legacy_state.available as usize;
            sink ^= (legacy_strategy.cpu as usize) << 1;
        }
        let legacy_elapsed = legacy_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let optimized_state = extract_region_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                blit,
                true,
                &mut optimized_rects,
            );
            let optimized_strategy = if optimized_state.force_full_copy {
                DirtyCopyStrategy::default()
            } else {
                evaluate_dirty_copy_strategy(&optimized_rects, blit.width, blit.height)
            };
            sink ^= optimized_rects.len() << 1;
            sink ^= (optimized_state.force_full_copy as usize) << 2;
            sink ^= (optimized_strategy.gpu as usize) << 3;
        }
        let optimized_elapsed = optimized_start.elapsed();

        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            f64::INFINITY
        };
        println!(
            "wgc region dense-fallback benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 0.90,
            "dense-fallback path did not improve enough: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_full_dirty_dense_fallback_vs_legacy() {
        use std::hint::black_box;

        let workloads = vec![
            row_major_raw_rects(0, 0, 45, 20, 8, 6, 2, 2, 900),
            row_major_raw_rects(4, 4, 44, 20, 9, 7, 2, 2, 880),
            row_major_raw_rects(8, 6, 41, 20, 10, 8, 2, 2, 820),
        ];

        let mut legacy_rects = Vec::<DirtyRect>::new();
        let mut optimized_rects = Vec::<DirtyRect>::new();
        for raw_rects in &workloads {
            let legacy_state =
                extract_full_raw_dirty_rects_legacy(raw_rects, 1920, 1080, &mut legacy_rects);
            let legacy_strategy = evaluate_dirty_copy_strategy(&legacy_rects, 1920, 1080);
            assert!(legacy_state.available);
            assert!(!legacy_state.unchanged);
            assert_eq!(
                legacy_strategy,
                DirtyCopyStrategy::default(),
                "workload unexpectedly eligible for dirty copy; benchmark expects full-copy fallback",
            );

            let optimized_state = extract_full_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                true,
                &mut optimized_rects,
            );
            let optimized_strategy = evaluate_dirty_copy_strategy(&optimized_rects, 1920, 1080);
            assert!(optimized_state.available);
            assert!(!optimized_state.unchanged);
            assert!(optimized_state.force_full_copy);
            assert!(!optimized_strategy.cpu);
            assert!(!optimized_strategy.gpu);
            assert!(!optimized_strategy.gpu_low_latency);
        }

        let warmup_iters = 8_000usize;
        let measure_iters = 70_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let legacy_state =
                extract_full_raw_dirty_rects_legacy(raw_rects, 1920, 1080, &mut legacy_rects);
            let legacy_strategy = evaluate_dirty_copy_strategy(&legacy_rects, 1920, 1080);
            sink ^= legacy_rects.len();
            sink ^= legacy_state.available as usize;
            sink ^= (legacy_strategy.cpu as usize) << 1;

            let optimized_state = extract_full_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                true,
                &mut optimized_rects,
            );
            let optimized_strategy = evaluate_dirty_copy_strategy(&optimized_rects, 1920, 1080);
            sink ^= optimized_rects.len() << 1;
            sink ^= (optimized_state.force_full_copy as usize) << 2;
            sink ^= (optimized_strategy.gpu as usize) << 3;
        }

        let legacy_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let legacy_state =
                extract_full_raw_dirty_rects_legacy(raw_rects, 1920, 1080, &mut legacy_rects);
            let legacy_strategy = evaluate_dirty_copy_strategy(&legacy_rects, 1920, 1080);
            sink ^= legacy_rects.len();
            sink ^= legacy_state.available as usize;
            sink ^= (legacy_strategy.cpu as usize) << 1;
        }
        let legacy_elapsed = legacy_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let optimized_state = extract_full_raw_dirty_rects_optimized(
                raw_rects,
                1920,
                1080,
                true,
                &mut optimized_rects,
            );
            let optimized_strategy = evaluate_dirty_copy_strategy(&optimized_rects, 1920, 1080);
            sink ^= optimized_rects.len() << 1;
            sink ^= (optimized_state.force_full_copy as usize) << 2;
            sink ^= (optimized_strategy.gpu as usize) << 3;
        }
        let optimized_elapsed = optimized_start.elapsed();

        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            f64::INFINITY
        };
        println!(
            "wgc full-frame dense-fallback benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 0.90,
            "full-frame dense-fallback path did not improve enough: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_dirty_copy_strategy_single_pass_vs_legacy() {
        use std::hint::black_box;

        let workloads = vec![
            vec![
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
            ],
            vec![
                DirtyRect {
                    x: 40,
                    y: 32,
                    width: 64,
                    height: 48,
                };
                WGC_DIRTY_GPU_COPY_MAX_RECTS + 4
            ],
            vec![
                DirtyRect {
                    x: 0,
                    y: 0,
                    width: 1920,
                    height: 540,
                },
                DirtyRect {
                    x: 0,
                    y: 540,
                    width: 1920,
                    height: 300,
                },
            ],
        ];

        for rects in &workloads {
            let legacy_cpu = should_use_dirty_copy_legacy(rects, 1920, 1080);
            let legacy_gpu = should_use_dirty_gpu_copy_legacy(rects, 1920, 1080);
            let legacy_gpu_low = should_use_low_latency_dirty_gpu_copy_legacy(rects, 1920, 1080);
            let optimized = evaluate_dirty_copy_strategy(rects, 1920, 1080);
            assert_eq!(optimized.cpu, legacy_cpu);
            assert_eq!(optimized.gpu, legacy_gpu);
            assert_eq!(optimized.gpu_low_latency, legacy_gpu_low);
        }

        let warmup_iters = 10_000usize;
        let measure_iters = 150_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            let width = black_box(1920u32);
            let height = black_box(1080u32);
            sink ^= should_use_dirty_copy_legacy(rects, width, height) as usize;
            sink ^= (should_use_dirty_gpu_copy_legacy(rects, width, height) as usize) << 1;
            sink ^=
                (should_use_low_latency_dirty_gpu_copy_legacy(rects, width, height) as usize) << 2;
            let strategy = evaluate_dirty_copy_strategy(rects, width, height);
            sink ^= (strategy.cpu as usize) << 3;
            sink ^= (strategy.gpu as usize) << 4;
            sink ^= (strategy.gpu_low_latency as usize) << 5;
        }

        let legacy_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            let width = black_box(1920u32);
            let height = black_box(1080u32);
            sink ^= should_use_dirty_copy_legacy(rects, width, height) as usize;
            sink ^= (should_use_dirty_gpu_copy_legacy(rects, width, height) as usize) << 1;
            sink ^=
                (should_use_low_latency_dirty_gpu_copy_legacy(rects, width, height) as usize) << 2;
        }
        let legacy_elapsed = legacy_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let rects = black_box(&workloads[iter % workloads.len()]);
            let width = black_box(1920u32);
            let height = black_box(1080u32);
            let strategy = evaluate_dirty_copy_strategy(rects, width, height);
            sink ^= strategy.cpu as usize;
            sink ^= (strategy.gpu as usize) << 1;
            sink ^= (strategy.gpu_low_latency as usize) << 2;
            sink ^= (strategy.dirty_pixels as usize) << 3;
        }
        let optimized_elapsed = optimized_start.elapsed();
        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            f64::INFINITY
        };
        println!(
            "wgc dirty strategy benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.05,
            "single-pass dirty strategy regressed: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
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
