use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_QUERY_DESC, D3D11_QUERY_EVENT, D3D11_TEXTURE2D_DESC, ID3D11Device,
    ID3D11DeviceContext, ID3D11Query, ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO,
    DXGI_OUTDUPL_MOVE_RECT, DXGI_OUTDUPL_POINTER_SHAPE_INFO, DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR,
    DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR, IDXGIOutput, IDXGIOutput1, IDXGIOutput5,
    IDXGIOutputDuplication, IDXGIResource,
};
use windows::core::Interface;

use crate::backend::{CaptureBlitRegion, CaptureMode, CaptureSampleMetadata, CursorCaptureConfig};
use crate::convert::HdrToSdrParams;
use crate::error::{CaptureError, CaptureResult};
use crate::frame::{CursorData, DirtyRect, Frame};
use crate::monitor::MonitorId;

use super::d3d11;
use super::dirty_rect::{
    self, DirtyCopyStrategy, DirtyCopyThresholds, DirtyRectDenseMergeThresholds,
};
use super::gpu_tonemap::{GpuF16Converter, GpuTonemapper};
use super::monitor::{MonitorResolver, ResolvedMonitor, hdr_to_sdr_params};
use super::surface::{self, StagingSampleDesc};

enum AcquireResult {
    Ok(ID3D11Texture2D, DXGI_OUTDUPL_FRAME_INFO),
    AccessLost,
}

enum TryAcquireResult {
    Ok(ID3D11Texture2D, DXGI_OUTDUPL_FRAME_INFO),
    AccessLost,
    Retry,
}

const PRESENT_ATTEMPTS: usize = 15;
const PRESENT_TIMEOUT_MS: u32 = 16;
const FALLBACK_TIMEOUT_MS: u32 = 250;
const STEADY_STATE_ATTEMPTS: usize = 20;
const STEADY_STATE_TIMEOUT_MS: u32 = 100;
const DXGI_DIRTY_COPY_MAX_RECTS: usize = 192;
const DXGI_DIRTY_COPY_MAX_AREA_PERCENT: u64 = 70;
const DXGI_DIRTY_GPU_COPY_MAX_RECTS: usize = 64;
const DXGI_DIRTY_GPU_COPY_MAX_AREA_PERCENT: u64 = 45;
const DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS: usize = 8;
const DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT: u64 = 18;
const DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT: u64 = 1_048_576;
const DXGI_REGION_STAGING_SLOTS: usize = 3;
const DXGI_REGION_DIRTY_TRACK_MAX_RECTS: usize = DXGI_DIRTY_COPY_MAX_RECTS + 1;
const DXGI_REGION_MOVE_TRACK_MAX_RECTS: usize = DXGI_REGION_DIRTY_TRACK_MAX_RECTS;
const DXGI_DIRTY_RECT_DENSE_MERGE_LEGACY_MIN_RECTS: usize = 64;
const DXGI_DIRTY_RECT_DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN: u32 = 96;
const DXGI_DIRTY_COPY_THRESHOLDS: DirtyCopyThresholds = DirtyCopyThresholds {
    max_rects: DXGI_DIRTY_COPY_MAX_RECTS,
    max_area_percent: DXGI_DIRTY_COPY_MAX_AREA_PERCENT,
    gpu_max_rects: DXGI_DIRTY_GPU_COPY_MAX_RECTS,
    gpu_max_area_percent: DXGI_DIRTY_GPU_COPY_MAX_AREA_PERCENT,
    gpu_low_latency_max_rects: DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS,
    gpu_low_latency_max_area_percent: DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT,
};
const DXGI_DIRTY_RECT_DENSE_MERGE_THRESHOLDS: DirtyRectDenseMergeThresholds =
    DirtyRectDenseMergeThresholds {
        min_rects: DXGI_DIRTY_RECT_DENSE_MERGE_LEGACY_MIN_RECTS,
        max_vertical_span: DXGI_DIRTY_RECT_DENSE_MERGE_LEGACY_MAX_VERTICAL_SPAN,
    };

#[inline]
fn window_low_latency_max_pixels() -> u64 {
    DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT
}

#[inline]
fn region_full_slot_map_fastpath_enabled() -> bool {
    true
}

#[derive(Default)]
struct RegionStagingSlot {
    staging: Option<ID3D11Texture2D>,
    staging_resource: Option<ID3D11Resource>,
    staging_key: Option<(u32, u32, DXGI_FORMAT)>,
    query: Option<ID3D11Query>,
    source_desc: Option<D3D11_TEXTURE2D_DESC>,
    hdr_to_sdr: Option<HdrToSdrParams>,
    capture_time: Option<Instant>,
    present_time_qpc: i64,
    is_duplicate: bool,
    dirty_mode_available: bool,
    move_mode_available: bool,
    dirty_cpu_copy_preferred: bool,
    dirty_gpu_copy_preferred: bool,
    dirty_total_pixels: u64,
    dirty_rects: Vec<DirtyRect>,
    move_rects: Vec<MoveRect>,
    populated: bool,
}

impl RegionStagingSlot {
    fn reset_runtime_state(&mut self) {
        self.source_desc = None;
        self.hdr_to_sdr = None;
        self.capture_time = None;
        self.present_time_qpc = 0;
        self.is_duplicate = false;
        self.dirty_mode_available = false;
        self.move_mode_available = false;
        self.dirty_cpu_copy_preferred = false;
        self.dirty_gpu_copy_preferred = false;
        self.dirty_total_pixels = 0;
        self.dirty_rects.clear();
        self.move_rects.clear();
        self.populated = false;
    }

    fn invalidate(&mut self) {
        self.staging = None;
        self.staging_resource = None;
        self.staging_key = None;
        self.query = None;
        self.reset_runtime_state();
    }
}



#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MoveRect {
    src_x: u32,
    src_y: u32,
    dst_x: u32,
    dst_y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct RegionDirtyBounds {
    source_width: u32,
    source_height: u32,
    region_x: u32,
    region_y: u32,
    region_right: u32,
    region_bottom: u32,
}

impl RegionDirtyBounds {
    #[inline]
    fn from_source_and_blit(
        source_width: u32,
        source_height: u32,
        blit: CaptureBlitRegion,
    ) -> Option<Self> {
        let region = dirty_rect::clamp_dirty_rect(
            DirtyRect {
                x: blit.src_x,
                y: blit.src_y,
                width: blit.width,
                height: blit.height,
            },
            source_width,
            source_height,
        )?;

        let region_right = region.x.saturating_add(region.width);
        let region_bottom = region.y.saturating_add(region.height);
        Some(Self {
            source_width,
            source_height,
            region_x: region.x,
            region_y: region.y,
            region_right,
            region_bottom,
        })
    }
}

#[inline(always)]
fn extract_region_dirty_rects_from_raw_slice(
    raw_rects: &[RECT],
    bounds: RegionDirtyBounds,
    out: &mut Vec<DirtyRect>,
) {
    out.clear();
    let max_rects = DXGI_REGION_DIRTY_TRACK_MAX_RECTS;
    let source_width = bounds.source_width;
    let source_height = bounds.source_height;
    let region_x = bounds.region_x;
    let region_y = bounds.region_y;
    let region_right = bounds.region_right;
    let region_bottom = bounds.region_bottom;

    for raw_rect in raw_rects {
        if out.len() == max_rects {
            break;
        }

        let width = (i64::from(raw_rect.right) - i64::from(raw_rect.left)).max(0) as u32;
        let height = (i64::from(raw_rect.bottom) - i64::from(raw_rect.top)).max(0) as u32;
        if width == 0 || height == 0 {
            continue;
        }

        let x = raw_rect.left.max(0) as u32;
        let y = raw_rect.top.max(0) as u32;
        if x >= source_width || y >= source_height {
            continue;
        }

        let right = x.saturating_add(width).min(source_width);
        let bottom = y.saturating_add(height).min(source_height);
        if right <= x || bottom <= y {
            continue;
        }

        if right <= region_x || x >= region_right || bottom <= region_y || y >= region_bottom {
            continue;
        }

        let clipped_x = x.max(region_x);
        let clipped_y = y.max(region_y);
        let clipped_right = right.min(region_right);
        let clipped_bottom = bottom.min(region_bottom);
        if clipped_right <= clipped_x || clipped_bottom <= clipped_y {
            continue;
        }

        out.push(DirtyRect {
            x: clipped_x.saturating_sub(region_x),
            y: clipped_y.saturating_sub(region_y),
            width: clipped_right.saturating_sub(clipped_x),
            height: clipped_bottom.saturating_sub(clipped_y),
        });
    }
}

#[inline(always)]
fn clamp_move_rect_to_surface(
    raw_rect: &DXGI_OUTDUPL_MOVE_RECT,
    source_width: u32,
    source_height: u32,
) -> Option<MoveRect> {
    let source_w = i64::from(source_width);
    let source_h = i64::from(source_height);
    if source_w <= 0 || source_h <= 0 {
        return None;
    }

    let dest_left = i64::from(raw_rect.DestinationRect.left);
    let dest_top = i64::from(raw_rect.DestinationRect.top);
    let dest_right = i64::from(raw_rect.DestinationRect.right);
    let dest_bottom = i64::from(raw_rect.DestinationRect.bottom);
    if dest_right <= dest_left || dest_bottom <= dest_top {
        return None;
    }

    let clipped_dest_left = dest_left.max(0);
    let clipped_dest_top = dest_top.max(0);
    let clipped_dest_right = dest_right.min(source_w);
    let clipped_dest_bottom = dest_bottom.min(source_h);
    if clipped_dest_right <= clipped_dest_left || clipped_dest_bottom <= clipped_dest_top {
        return None;
    }

    let source_left =
        i64::from(raw_rect.SourcePoint.x).saturating_add(clipped_dest_left - dest_left);
    let source_top = i64::from(raw_rect.SourcePoint.y).saturating_add(clipped_dest_top - dest_top);
    let source_right = source_left.saturating_add(clipped_dest_right - clipped_dest_left);
    let source_bottom = source_top.saturating_add(clipped_dest_bottom - clipped_dest_top);

    let clipped_source_left = source_left.max(0);
    let clipped_source_top = source_top.max(0);
    let clipped_source_right = source_right.min(source_w);
    let clipped_source_bottom = source_bottom.min(source_h);
    if clipped_source_right <= clipped_source_left || clipped_source_bottom <= clipped_source_top {
        return None;
    }

    let adjusted_dest_left =
        clipped_dest_left.saturating_add(clipped_source_left.saturating_sub(source_left));
    let adjusted_dest_top =
        clipped_dest_top.saturating_add(clipped_source_top.saturating_sub(source_top));
    let width = clipped_source_right.saturating_sub(clipped_source_left);
    let height = clipped_source_bottom.saturating_sub(clipped_source_top);
    if width <= 0 || height <= 0 {
        return None;
    }

    Some(MoveRect {
        src_x: clipped_source_left as u32,
        src_y: clipped_source_top as u32,
        dst_x: adjusted_dest_left as u32,
        dst_y: adjusted_dest_top as u32,
        width: width as u32,
        height: height as u32,
    })
}

#[inline(always)]
fn clip_and_rebase_region_move_rect(rect: MoveRect, bounds: RegionDirtyBounds) -> Option<MoveRect> {
    let width = i64::from(rect.width);
    let height = i64::from(rect.height);
    if width <= 0 || height <= 0 {
        return None;
    }

    let src_x = i64::from(rect.src_x);
    let src_y = i64::from(rect.src_y);
    let dst_x = i64::from(rect.dst_x);
    let dst_y = i64::from(rect.dst_y);
    let region_x = i64::from(bounds.region_x);
    let region_y = i64::from(bounds.region_y);
    let region_right = i64::from(bounds.region_right);
    let region_bottom = i64::from(bounds.region_bottom);

    let start_x = 0i64.max(region_x - dst_x).max(region_x - src_x);
    let start_y = 0i64.max(region_y - dst_y).max(region_y - src_y);
    let end_x = width.min(region_right - dst_x).min(region_right - src_x);
    let end_y = height.min(region_bottom - dst_y).min(region_bottom - src_y);
    if end_x <= start_x || end_y <= start_y {
        return None;
    }

    let rebased_src_x = src_x + start_x - region_x;
    let rebased_src_y = src_y + start_y - region_y;
    let rebased_dst_x = dst_x + start_x - region_x;
    let rebased_dst_y = dst_y + start_y - region_y;
    let rebased_width = end_x - start_x;
    let rebased_height = end_y - start_y;
    if rebased_width <= 0 || rebased_height <= 0 {
        return None;
    }

    Some(MoveRect {
        src_x: rebased_src_x as u32,
        src_y: rebased_src_y as u32,
        dst_x: rebased_dst_x as u32,
        dst_y: rebased_dst_y as u32,
        width: rebased_width as u32,
        height: rebased_height as u32,
    })
}

#[inline(always)]
fn extract_region_move_rects_from_move_slice(
    source_move_rects: &[MoveRect],
    bounds: RegionDirtyBounds,
    out: &mut Vec<MoveRect>,
) {
    out.clear();
    for rect in source_move_rects {
        if out.len() == DXGI_REGION_MOVE_TRACK_MAX_RECTS {
            break;
        }
        if let Some(clipped) = clip_and_rebase_region_move_rect(*rect, bounds) {
            out.push(clipped);
        }
    }
}

fn apply_move_rects_to_frame(
    frame: &mut Frame,
    move_rects: &[MoveRect],
    dst_origin_x: u32,
    dst_origin_y: u32,
) -> CaptureResult<()> {
    if move_rects.is_empty() {
        return Ok(());
    }

    let frame_width = usize::try_from(frame.width()).map_err(|_| CaptureError::BufferOverflow)?;
    let frame_height = usize::try_from(frame.height()).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_origin_x = usize::try_from(dst_origin_x).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_origin_y = usize::try_from(dst_origin_y).map_err(|_| CaptureError::BufferOverflow)?;
    let pitch = frame_width
        .checked_mul(4)
        .ok_or(CaptureError::BufferOverflow)?;
    let base = frame.as_mut_rgba_ptr();

    for rect in move_rects {
        let width = usize::try_from(rect.width).map_err(|_| CaptureError::BufferOverflow)?;
        let height = usize::try_from(rect.height).map_err(|_| CaptureError::BufferOverflow)?;
        if width == 0 || height == 0 {
            continue;
        }

        let src_x = dst_origin_x
            .checked_add(usize::try_from(rect.src_x).map_err(|_| CaptureError::BufferOverflow)?)
            .ok_or(CaptureError::BufferOverflow)?;
        let src_y = dst_origin_y
            .checked_add(usize::try_from(rect.src_y).map_err(|_| CaptureError::BufferOverflow)?)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_x = dst_origin_x
            .checked_add(usize::try_from(rect.dst_x).map_err(|_| CaptureError::BufferOverflow)?)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_y = dst_origin_y
            .checked_add(usize::try_from(rect.dst_y).map_err(|_| CaptureError::BufferOverflow)?)
            .ok_or(CaptureError::BufferOverflow)?;

        let src_right = src_x
            .checked_add(width)
            .ok_or(CaptureError::BufferOverflow)?;
        let src_bottom = src_y
            .checked_add(height)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_right = dst_x
            .checked_add(width)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_bottom = dst_y
            .checked_add(height)
            .ok_or(CaptureError::BufferOverflow)?;
        if src_right > frame_width
            || src_bottom > frame_height
            || dst_right > frame_width
            || dst_bottom > frame_height
        {
            return Err(CaptureError::BufferOverflow);
        }

        let row_bytes = width.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
        let src_row_start = src_y
            .checked_mul(pitch)
            .and_then(|base_offset| {
                src_x
                    .checked_mul(4)
                    .and_then(|xoff| base_offset.checked_add(xoff))
            })
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_row_start = dst_y
            .checked_mul(pitch)
            .and_then(|base_offset| {
                dst_x
                    .checked_mul(4)
                    .and_then(|xoff| base_offset.checked_add(xoff))
            })
            .ok_or(CaptureError::BufferOverflow)?;

        if src_x == 0 && dst_x == 0 && row_bytes == pitch {
            let total_bytes = row_bytes
                .checked_mul(height)
                .ok_or(CaptureError::BufferOverflow)?;
            unsafe {
                std::ptr::copy(
                    base.add(src_row_start),
                    base.add(dst_row_start),
                    total_bytes,
                );
            }
            continue;
        }

        if dst_y > src_y {
            for row in (0..height).rev() {
                let src_offset = src_row_start
                    .checked_add(row.checked_mul(pitch).ok_or(CaptureError::BufferOverflow)?)
                    .ok_or(CaptureError::BufferOverflow)?;
                let dst_offset = dst_row_start
                    .checked_add(row.checked_mul(pitch).ok_or(CaptureError::BufferOverflow)?)
                    .ok_or(CaptureError::BufferOverflow)?;
                unsafe {
                    std::ptr::copy(base.add(src_offset), base.add(dst_offset), row_bytes);
                }
            }
        } else {
            for row in 0..height {
                let src_offset = src_row_start
                    .checked_add(row.checked_mul(pitch).ok_or(CaptureError::BufferOverflow)?)
                    .ok_or(CaptureError::BufferOverflow)?;
                let dst_offset = dst_row_start
                    .checked_add(row.checked_mul(pitch).ok_or(CaptureError::BufferOverflow)?)
                    .ok_or(CaptureError::BufferOverflow)?;
                unsafe {
                    std::ptr::copy(base.add(src_offset), base.add(dst_offset), row_bytes);
                }
            }
        }
    }

    Ok(())
}

fn normalize_dirty_rects_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    let mut merge_scratch = Vec::new();
    dirty_rect::normalize_dirty_rects_in_place(
        rects,
        width,
        height,
        false,
        DXGI_DIRTY_RECT_DENSE_MERGE_THRESHOLDS,
        true,
        &mut merge_scratch,
    );
}

#[cfg(test)]
fn normalize_dirty_rects_reference_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut pending = std::mem::take(rects);
    let mut write = 0usize;
    for read in 0..pending.len() {
        if let Some(clamped) = dirty_rect::clamp_dirty_rect(pending[read], width, height) {
            pending[write] = clamped;
            write += 1;
        }
    }
    pending.truncate(write);
    if pending.len() <= 1 {
        *rects = pending;
        return;
    }

    if dirty_rect::should_use_legacy_dense_merge(&pending, DXGI_DIRTY_RECT_DENSE_MERGE_THRESHOLDS) {
        *rects = pending;
        dirty_rect::normalize_dirty_rects_legacy_after_clamp(rects);
        return;
    }

    pending.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));

    rects.reserve(pending.len());
    for rect in pending {
        let mut candidate = rect;
        loop {
            let mut merged_any = false;
            let candidate_bottom = candidate.y.saturating_add(candidate.height);
            let mut idx = 0usize;
            while idx < rects.len() {
                let existing = rects[idx];
                let existing_bottom = existing.y.saturating_add(existing.height);
                if existing_bottom < candidate.y {
                    idx += 1;
                    continue;
                }
                if existing.y > candidate_bottom {
                    break;
                }

                if dirty_rect::dirty_rects_can_merge(candidate, existing) {
                    candidate = dirty_rect::merge_dirty_rects(candidate, existing);
                    rects.remove(idx);
                    merged_any = true;
                } else {
                    idx += 1;
                }
            }

            if !merged_any {
                break;
            }
        }

        let insert_at = rects
            .binary_search_by(|probe| {
                probe
                    .y
                    .cmp(&candidate.y)
                    .then_with(|| probe.x.cmp(&candidate.x))
            })
            .unwrap_or_else(|pos| pos);
        rects.insert(insert_at, candidate);
    }
}

fn extract_region_dirty_rects_direct(
    duplication: &IDXGIOutputDuplication,
    info: &DXGI_OUTDUPL_FRAME_INFO,
    rect_buffer: &mut Vec<RECT>,
    source_width: u32,
    source_height: u32,
    blit: CaptureBlitRegion,
    out: &mut Vec<DirtyRect>,
) -> bool {
    let Some(bounds) = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
    else {
        out.clear();
        return false;
    };

    let dirty_bytes = info.TotalMetadataBufferSize as usize;
    if dirty_bytes == 0 {
        out.clear();
        return true;
    }

    let max_rects = dirty_bytes / std::mem::size_of::<RECT>();
    if max_rects == 0 {
        out.clear();
        return false;
    }
    if rect_buffer.len() < max_rects {
        rect_buffer.resize(max_rects, RECT::default());
    }

    let mut buf_size = info.TotalMetadataBufferSize;
    let hr = unsafe {
        duplication.GetFrameDirtyRects(buf_size, rect_buffer.as_mut_ptr(), &mut buf_size)
    };
    if hr.is_err() {
        out.clear();
        return false;
    }

    let actual_count = ((buf_size as usize) / std::mem::size_of::<RECT>()).min(rect_buffer.len());
    extract_region_dirty_rects_from_raw_slice(&rect_buffer[..actual_count], bounds, out);
    true
}

fn extract_move_rects(
    duplication: &IDXGIOutputDuplication,
    info: &DXGI_OUTDUPL_FRAME_INFO,
    move_buffer: &mut Vec<DXGI_OUTDUPL_MOVE_RECT>,
    source_width: u32,
    source_height: u32,
    out: &mut Vec<MoveRect>,
) -> bool {
    out.clear();
    let metadata_bytes = info.TotalMetadataBufferSize as usize;
    if metadata_bytes == 0 {
        return true;
    }

    let max_rects = metadata_bytes / std::mem::size_of::<DXGI_OUTDUPL_MOVE_RECT>();
    if max_rects == 0 {
        return true;
    }
    if move_buffer.len() < max_rects {
        move_buffer.resize(max_rects, DXGI_OUTDUPL_MOVE_RECT::default());
    }

    let mut buf_size = info.TotalMetadataBufferSize;
    let hr =
        unsafe { duplication.GetFrameMoveRects(buf_size, move_buffer.as_mut_ptr(), &mut buf_size) };
    if hr.is_err() {
        out.clear();
        return false;
    }

    let actual_count = ((buf_size as usize) / std::mem::size_of::<DXGI_OUTDUPL_MOVE_RECT>())
        .min(move_buffer.len());
    if actual_count > DXGI_REGION_MOVE_TRACK_MAX_RECTS {
        out.clear();
        return false;
    }
    for raw_rect in &move_buffer[..actual_count] {
        if let Some(clamped) = clamp_move_rect_to_surface(raw_rect, source_width, source_height) {
            out.push(clamped);
        }
    }
    true
}

fn extract_region_move_rects(
    source_move_rects: &[MoveRect],
    source_width: u32,
    source_height: u32,
    blit: CaptureBlitRegion,
    out: &mut Vec<MoveRect>,
) -> bool {
    let Some(bounds) = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
    else {
        out.clear();
        return false;
    };
    extract_region_move_rects_from_move_slice(source_move_rects, bounds, out);
    true
}

#[inline]
fn can_use_region_dirty_reconstruct(
    region_move_available: bool,
    region_has_moves: bool,
    move_reconstruct_enabled: bool,
) -> bool {
    region_move_available && (!region_has_moves || move_reconstruct_enabled)
}

fn create_duplication(
    output: &IDXGIOutput,
    device: &ID3D11Device,
) -> CaptureResult<IDXGIOutputDuplication> {
    if let Ok(output5) = output.cast::<IDXGIOutput5>() {
        let formats = [DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM];
        if let Ok(duplication) = unsafe { output5.DuplicateOutput1(device, 0, &formats) } {
            return Ok(duplication);
        }
    }

    let output1: IDXGIOutput1 = output
        .cast()
        .context("failed to query IDXGIOutput1")
        .map_err(CaptureError::Platform)?;
    unsafe { output1.DuplicateOutput(device) }
        .context("DuplicateOutput failed")
        .map_err(CaptureError::Platform)
}

fn try_acquire_frame(
    duplication: &IDXGIOutputDuplication,
    timeout_ms: u32,
    require_present_time: bool,
) -> CaptureResult<TryAcquireResult> {
    let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
    let mut resource: Option<IDXGIResource> = None;
    let acquired = unsafe { duplication.AcquireNextFrame(timeout_ms, &mut info, &mut resource) };
    if let Err(error) = acquired {
        if error.code() == DXGI_ERROR_WAIT_TIMEOUT {
            return Ok(TryAcquireResult::Retry);
        }
        if error.code() == DXGI_ERROR_ACCESS_LOST {
            return Ok(TryAcquireResult::AccessLost);
        }
        return Err(CaptureError::Platform(
            anyhow::Error::from(error).context("AcquireNextFrame failed"),
        ));
    }

    if require_present_time && info.LastPresentTime == 0 {
        unsafe { duplication.ReleaseFrame() }.ok();
        return Ok(TryAcquireResult::Retry);
    }

    let Some(resource) = resource else {
        unsafe { duplication.ReleaseFrame() }.ok();
        return Ok(TryAcquireResult::Retry);
    };

    let texture: ID3D11Texture2D = resource
        .cast()
        .context("failed to cast acquired IDXGIResource to ID3D11Texture2D")
        .map_err(CaptureError::Platform)?;
    Ok(TryAcquireResult::Ok(texture, info))
}

fn acquire_with_retries(
    duplication: &IDXGIOutputDuplication,
    attempts: usize,
    timeout_ms: u32,
    require_present_time: bool,
) -> CaptureResult<Option<AcquireResult>> {
    for _ in 0..attempts {
        match try_acquire_frame(duplication, timeout_ms, require_present_time)? {
            TryAcquireResult::Ok(texture, info) => {
                return Ok(Some(AcquireResult::Ok(texture, info)));
            }
            TryAcquireResult::AccessLost => return Ok(Some(AcquireResult::AccessLost)),
            TryAcquireResult::Retry => {}
        }
    }
    Ok(None)
}

fn acquire_frame(
    duplication: &IDXGIOutputDuplication,
    require_present_time: bool,
) -> CaptureResult<AcquireResult> {
    if require_present_time {
        if let Some(result) =
            acquire_with_retries(duplication, PRESENT_ATTEMPTS, PRESENT_TIMEOUT_MS, true)?
        {
            return Ok(result);
        }

        return match try_acquire_frame(duplication, FALLBACK_TIMEOUT_MS, false)? {
            TryAcquireResult::Ok(texture, info) => Ok(AcquireResult::Ok(texture, info)),
            TryAcquireResult::AccessLost => Ok(AcquireResult::AccessLost),
            TryAcquireResult::Retry => Err(CaptureError::Timeout),
        };
    }

    if let Some(result) = acquire_with_retries(
        duplication,
        STEADY_STATE_ATTEMPTS,
        STEADY_STATE_TIMEOUT_MS,
        false,
    )? {
        return Ok(result);
    }
    Err(CaptureError::Timeout)
}

fn with_monitor_context<T>(
    result: CaptureResult<T>,
    monitor: &MonitorId,
    action: &'static str,
) -> CaptureResult<T> {
    result.map_err(|error| match error {
        CaptureError::Platform(inner) => CaptureError::Platform(inner.context(format!(
            "failed to {action} capturer for {}",
            monitor.name()
        ))),
        other => other,
    })
}

/// Triple-buffered staging ring for overlapping GPU copies with CPU reads.
///
/// With three slots the GPU can be copying into slot C while the CPU reads
/// from slot A and slot B sits ready -- fully decoupling the GPU and CPU
/// timelines at high frame rates.  The cost is one extra staging texture
/// (~33 MB at 4K).
const STAGING_SLOTS: usize = 3;

struct StagingRing {
    slots: [Option<ID3D11Texture2D>; STAGING_SLOTS],
    /// Cached `ID3D11Resource` for each slot -- avoids a COM
    /// `QueryInterface` (`cast()`) on every `submit_copy` call.
    slot_resources: [Option<ID3D11Resource>; STAGING_SLOTS],
    queries: [Option<ID3D11Query>; STAGING_SLOTS],
    /// Index of the slot that was most recently submitted for GPU copy.
    /// The *other* slots are available for CPU reads or future writes.
    write_idx: usize,
    /// Whether a GPU copy is currently in-flight on `write_idx`.
    pending: bool,
    /// Index of the previous write slot (now available for CPU read).
    /// `None` when there is no pending read.
    read_idx: Option<usize>,
    /// Cached descriptor of the staging slots.  Avoids calling
    /// `GetDesc()` on every frame in `ensure_staging_texture` when the
    /// resolution hasn't changed.
    cached_desc: Option<(
        u32,
        u32,
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    )>,
    /// Adaptive spin count for GPU readback polling.  Starts at
    /// `INITIAL_SPIN_POLLS` and adjusts based on whether the GPU copy
    /// completes within the spin window.
    adaptive_spin_polls: u32,
    /// Number of staging slots currently allocated for the active mode.
    allocated_slots: usize,
}

impl StagingRing {
    /// Initial spin count -- conservative starting point.
    const INITIAL_SPIN_POLLS: u32 = 4;
    /// Minimum spin count to avoid degenerating to pure blocking.
    const MIN_SPIN_POLLS: u32 = 2;
    /// Maximum spin count -- cap to avoid burning too many cycles.
    const MAX_SPIN_POLLS: u32 = 64;
    /// Additive step for increasing the spin count on a miss.
    /// More conservative than multiplicative (x2) growth to avoid
    /// overshooting and wasting cycles when the GPU is consistently
    /// slower than the spin window.
    const SPIN_INCREASE_STEP: u32 = 4;

    fn new() -> Self {
        Self {
            slots: [None, None, None],
            slot_resources: [None, None, None],
            queries: [None, None, None],
            write_idx: 0,
            pending: false,
            read_idx: None,
            cached_desc: None,
            adaptive_spin_polls: Self::INITIAL_SPIN_POLLS,
            allocated_slots: 0,
        }
    }

    fn invalidate(&mut self) {
        self.slots = [None, None, None];
        self.slot_resources = [None, None, None];
        self.queries = [None, None, None];
        self.pending = false;
        self.read_idx = None;
        self.cached_desc = None;
        self.adaptive_spin_polls = Self::INITIAL_SPIN_POLLS;
        self.allocated_slots = 0;
    }

    fn reset_pipeline(&mut self) {
        self.pending = false;
        self.read_idx = None;
        self.write_idx = 0;
        self.adaptive_spin_polls = Self::INITIAL_SPIN_POLLS;
    }

    /// Ensure the active staging slots match the given texture description.
    /// Skips the per-slot `GetDesc()` COM call when the cached
    /// (width, height, format) triple already matches.
    fn ensure_slots(
        &mut self,
        device: &ID3D11Device,
        desc: &D3D11_TEXTURE2D_DESC,
        requested_slots: usize,
    ) -> CaptureResult<()> {
        let target_slots = requested_slots.clamp(1, STAGING_SLOTS);
        let key = (desc.Width, desc.Height, desc.Format);
        let desc_changed = self.cached_desc != Some(key);

        if desc_changed || self.allocated_slots < target_slots {
            for i in 0..target_slots {
                surface::ensure_staging_texture(
                    device,
                    &mut self.slots[i],
                    desc,
                    StagingSampleDesc::SingleSample,
                    "failed to create staging texture",
                )?;
                // Cache the ID3D11Resource cast alongside the texture.
                self.slot_resources[i] = self.slots[i]
                    .as_ref()
                    .map(|tex| tex.cast::<ID3D11Resource>().unwrap());
            }
        }

        if desc_changed || self.allocated_slots > target_slots {
            for i in target_slots..STAGING_SLOTS {
                self.slots[i] = None;
                self.slot_resources[i] = None;
                self.queries[i] = None;
            }
        }

        // Create event queries for active slots only.
        for i in 0..target_slots {
            if self.queries[i].is_none() {
                let query_desc = D3D11_QUERY_DESC {
                    Query: D3D11_QUERY_EVENT,
                    ..Default::default()
                };
                let mut query: Option<ID3D11Query> = None;
                unsafe { device.CreateQuery(&query_desc, Some(&mut query)) }
                    .context("CreateQuery for staging ring failed")
                    .map_err(CaptureError::Platform)?;
                self.queries[i] = query;
            }
        }

        self.cached_desc = Some(key);
        self.allocated_slots = target_slots;
        Ok(())
    }

    fn copy_source_to_slot(
        &self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
        slot: usize,
        dirty_rects: &[DirtyRect],
        use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        let staging_res = self.slot_resources[slot].as_ref().unwrap();
        d3d11::with_texture_resource(
            source,
            "failed to cast DXGI source texture to ID3D11Resource",
            |source_res| {
                if use_dirty_gpu_copy {
                    for rect in dirty_rects {
                        if rect.width == 0 || rect.height == 0 {
                            continue;
                        }
                        let source_right = rect
                            .x
                            .checked_add(rect.width)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_bottom = rect
                            .y
                            .checked_add(rect.height)
                            .ok_or(CaptureError::BufferOverflow)?;
                        let source_box = D3D11_BOX {
                            left: rect.x,
                            top: rect.y,
                            front: 0,
                            right: source_right,
                            bottom: source_bottom,
                            back: 1,
                        };
                        unsafe {
                            context.CopySubresourceRegion(
                                staging_res,
                                0,
                                rect.x,
                                rect.y,
                                0,
                                source_res,
                                0,
                                Some(&source_box),
                            );
                        }
                    }
                } else {
                    unsafe {
                        context.CopyResource(staging_res, source_res);
                    }
                }
                Ok(())
            },
        )?;

        if let Some(ref query) = self.queries[slot] {
            unsafe { context.End(query) };
        }
        Ok(())
    }

    /// Submit a GPU copy from `source` into the current write slot.
    /// Returns the *read* slot index if there was a previous pending
    /// copy that can now be consumed.
    fn submit_copy(
        &mut self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
    ) -> CaptureResult<Option<usize>> {
        let prev_pending = self.pending;
        let read_idx = if prev_pending {
            // The previous write slot becomes the new read slot.
            Some(self.write_idx)
        } else {
            None
        };
        let write_idx = if prev_pending {
            (self.write_idx + 1) % STAGING_SLOTS
        } else {
            self.write_idx
        };

        self.copy_source_to_slot(context, source, write_idx, &[], false)?;

        // Only flush when there is a pending read slot whose query
        // hasn't completed yet.  This lets the driver batch the copy
        // with subsequent work when the GPU is keeping up, while still
        // ensuring the copy starts promptly when we need the result
        // on the next call.
        if let Some(ridx) = read_idx {
            let needs_flush = self.queries[ridx].as_ref().is_none_or(|q| {
                let mut data: u32 = 0;
                // D3D11_ASYNC_GETDATA_DONOTFLUSH = 0x1
                unsafe {
                    context.GetData(
                        q,
                        Some(&mut data as *mut u32 as *mut _),
                        std::mem::size_of::<u32>() as u32,
                        0x1,
                    )
                }
                .is_err()
            });
            if needs_flush {
                unsafe { context.Flush() };
            }
        } else {
            // First frame -- flush to kick off the copy immediately.
            unsafe { context.Flush() };
        }

        self.write_idx = write_idx;
        self.pending = true;
        self.read_idx = read_idx;
        Ok(read_idx)
    }

    #[inline(always)]
    fn latest_write_slot(&self) -> usize {
        self.write_idx
    }

    /// Wait for the copy on the given slot to complete, then map and
    /// convert into the frame.
    ///
    /// Because the copy was submitted on the *previous* capture call,
    /// it is almost always finished by the time we get here.  We do a
    /// short bounded spin-wait (up to `MAX_SPIN_POLLS` non-blocking
    /// polls with `spin_loop` hints) using `D3D11_ASYNC_GETDATA_DONOTFLUSH`
    /// to avoid stalling the GPU command queue.  If the GPU still isn't
    /// done after the micro-spin, we fall through to the blocking `Map`
    /// call.  The spin is cheaper than the driver's internal kernel wait
    /// for the common case where the copy finishes within a few hundred
    /// nanoseconds of our check.
    fn read_slot_with_strategy(
        &mut self,
        context: &ID3D11DeviceContext,
        slot: usize,
        desc: &D3D11_TEXTURE2D_DESC,
        frame: &mut Frame,
        hdr_to_sdr: Option<HdrToSdrParams>,
        dirty_rects: &[DirtyRect],
        dirty_total_pixels: u64,
        use_dirty_copy: bool,
        skip_readback: bool,
    ) -> CaptureResult<()> {
        // The caller keeps the previous output pixels for duplicate frames,
        // so we can skip both query polling and CPU mapping entirely.
        if skip_readback {
            return Ok(());
        }

        // D3D11_ASYNC_GETDATA_DONOTFLUSH = 0x1 -- avoids an implicit
        // Flush() inside GetData which would stall the GPU pipeline.
        const DO_NOT_FLUSH: u32 = 0x1;

        if let Some(ref query) = self.queries[slot] {
            let mut data: u32 = 0;
            let mut completed_in_spin = false;
            let polls = self.adaptive_spin_polls;
            for _ in 0..polls {
                let hr = unsafe {
                    context.GetData(
                        query,
                        Some(&mut data as *mut u32 as *mut _),
                        std::mem::size_of::<u32>() as u32,
                        DO_NOT_FLUSH,
                    )
                };
                if hr.is_ok() {
                    completed_in_spin = true;
                    break;
                }
                std::hint::spin_loop();
            }
            // Adapt spin count: if the GPU finished within the spin
            // window, we can afford fewer polls next time (the pipeline
            // is keeping up).  If it didn't, increase the window so we
            // have a better chance of catching it before the blocking
            // Map() call.
            if completed_in_spin {
                self.adaptive_spin_polls =
                    (self.adaptive_spin_polls.saturating_sub(1)).max(Self::MIN_SPIN_POLLS);
            } else {
                self.adaptive_spin_polls = (self
                    .adaptive_spin_polls
                    .saturating_add(Self::SPIN_INCREASE_STEP))
                .min(Self::MAX_SPIN_POLLS);
            }
        }

        let staging = self.slots[slot].as_ref().unwrap();
        let staging_res = self.slot_resources[slot].as_ref();
        if use_dirty_copy {
            let hints = surface::DirtyRectConversionHints {
                trusted_bounds: true,
                non_empty_rects: Some(dirty_rects.len()),
                total_dirty_pixels: usize::try_from(dirty_total_pixels).ok(),
            };
            match surface::map_staging_dirty_rects_to_frame(
                context,
                staging,
                staging_res,
                desc,
                frame,
                dirty_rects,
                true,
                hints,
                hdr_to_sdr,
                "failed to map staging texture (dirty regions)",
            ) {
                Ok(converted) if converted > 0 => Ok(()),
                Ok(_) | Err(_) => surface::map_staging_to_frame(
                    context,
                    staging,
                    staging_res,
                    desc,
                    frame,
                    hdr_to_sdr,
                    "failed to map staging texture",
                ),
            }
        } else {
            surface::map_staging_to_frame(
                context,
                staging,
                staging_res,
                desc,
                frame,
                hdr_to_sdr,
                "failed to map staging texture",
            )
        }
    }

    /// Synchronous single-shot capture.
    ///
    /// The call submits a GPU copy to the current slot, flushes the
    /// command stream, then maps and converts into `frame`.
    ///
    /// `use_dirty_gpu_copy` controls whether the source upload uses
    /// per-rect `CopySubresourceRegion` updates instead of `CopyResource`.
    /// `use_dirty_cpu_copy` controls whether the CPU conversion path maps
    /// only the provided dirty rectangles.
    fn copy_and_read_with_strategy(
        &mut self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
        desc: &D3D11_TEXTURE2D_DESC,
        frame: &mut Frame,
        hdr_to_sdr: Option<HdrToSdrParams>,
        dirty_rects: &[DirtyRect],
        dirty_total_pixels: u64,
        use_dirty_cpu_copy: bool,
        use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        let slot = self.write_idx;
        self.copy_source_to_slot(context, source, slot, dirty_rects, use_dirty_gpu_copy)?;
        unsafe { context.Flush() };
        self.read_slot_with_strategy(
            context,
            slot,
            desc,
            frame,
            hdr_to_sdr,
            dirty_rects,
            dirty_total_pixels,
            use_dirty_cpu_copy,
            false,
        )?;
        self.pending = false;
        self.read_idx = None;
        Ok(())
    }

    /// Synchronous single-shot: full copy + map + convert.
    fn copy_and_read(
        &mut self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
        desc: &D3D11_TEXTURE2D_DESC,
        frame: &mut Frame,
        hdr_to_sdr: Option<HdrToSdrParams>,
    ) -> CaptureResult<()> {
        self.copy_and_read_with_strategy(
            context,
            source,
            desc,
            frame,
            hdr_to_sdr,
            &[],
            0,
            false,
            false,
        )
    }
}

/// Extract dirty rectangles from the DXGI duplication frame.
/// Returns whether dirty metadata was available.
fn extract_dirty_rects(
    duplication: &IDXGIOutputDuplication,
    info: &DXGI_OUTDUPL_FRAME_INFO,
    rect_buffer: &mut Vec<RECT>,
    out: &mut Vec<DirtyRect>,
) -> bool {
    out.clear();
    let dirty_bytes = info.TotalMetadataBufferSize as usize;
    if dirty_bytes == 0 {
        return true;
    }

    // Query dirty rects. The buffer size is in bytes; each RECT is 16 bytes.
    let max_rects = dirty_bytes / std::mem::size_of::<RECT>();
    if max_rects == 0 {
        return false;
    }
    if rect_buffer.len() < max_rects {
        rect_buffer.resize(max_rects, RECT::default());
    }

    let mut buf_size = info.TotalMetadataBufferSize;
    let hr = unsafe {
        duplication.GetFrameDirtyRects(buf_size, rect_buffer.as_mut_ptr(), &mut buf_size)
    };
    if hr.is_err() {
        return false;
    }
    let actual_count = ((buf_size as usize) / std::mem::size_of::<RECT>()).min(rect_buffer.len());
    for rect in &rect_buffer[..actual_count] {
        let x = rect.left.max(0) as u32;
        let y = rect.top.max(0) as u32;
        let w = (rect.right - rect.left).max(0) as u32;
        let h = (rect.bottom - rect.top).max(0) as u32;
        if w > 0 && h > 0 {
            out.push(DirtyRect {
                x,
                y,
                width: w,
                height: h,
            });
        }
    }
    true
}

fn evaluate_dirty_copy_strategy(rects: &[DirtyRect], width: u32, height: u32) -> DirtyCopyStrategy {
    dirty_rect::evaluate_dirty_copy_strategy(rects, width, height, DXGI_DIRTY_COPY_THRESHOLDS)
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
fn choose_monitor_low_latency_dirty_gpu_mode(
    recording_mode: bool,
    feature_enabled: bool,
    force_enabled: bool,
    mode_active: bool,
    output_matches_source: bool,
    frame_pixels_unchanged: bool,
    dirty_gpu_preferred: bool,
    low_latency_dirty_gpu_preferred: bool,
) -> bool {
    if !recording_mode || !feature_enabled || !output_matches_source {
        return false;
    }

    if frame_pixels_unchanged {
        return mode_active;
    }

    if force_enabled {
        return dirty_gpu_preferred;
    }

    low_latency_dirty_gpu_preferred
}

#[inline(always)]
fn should_use_monitor_low_latency_dirty_gpu_mode(
    capture_mode: CaptureMode,
    mode_active: bool,
    output_matches_source: bool,
    frame_pixels_unchanged: bool,
    dirty_strategy: DirtyCopyStrategy,
) -> bool {
    choose_monitor_low_latency_dirty_gpu_mode(
        capture_mode == CaptureMode::ScreenRecording,
        true,
        false,
        mode_active,
        output_matches_source,
        frame_pixels_unchanged,
        dirty_strategy.gpu,
        dirty_strategy.gpu_low_latency,
    )
}

#[inline(always)]
fn choose_window_low_latency_region_mode(
    recording_mode: bool,
    prefer_low_latency: bool,
    low_latency_enabled: bool,
    force_low_latency: bool,
    max_low_latency_pixels: u64,
    blit: CaptureBlitRegion,
    low_latency_dirty_gpu_preferred: bool,
) -> bool {
    if !recording_mode || !prefer_low_latency || !low_latency_enabled {
        return false;
    }

    if force_low_latency {
        return true;
    }

    let region_pixels = u64::from(blit.width).saturating_mul(u64::from(blit.height));
    if region_pixels == 0 {
        return false;
    }

    region_pixels <= max_low_latency_pixels || low_latency_dirty_gpu_preferred
}

#[inline(always)]
fn should_use_window_low_latency_region_path(
    recording_mode: bool,
    prefer_low_latency: bool,
    blit: CaptureBlitRegion,
    low_latency_dirty_gpu_preferred: bool,
) -> bool {
    choose_window_low_latency_region_mode(
        recording_mode,
        prefer_low_latency,
        true,
        false,
        window_low_latency_max_pixels(),
        blit,
        low_latency_dirty_gpu_preferred,
    )
}

#[inline(always)]
fn should_prefer_monitor_region_low_latency(
    _capture_mode: CaptureMode,
    _blit: CaptureBlitRegion,
) -> bool {
    false
}

#[inline(always)]
fn should_skip_screenrecord_submit_copy(
    fastpath_enabled: bool,
    has_pending_submission: bool,
    source_is_duplicate: bool,
    source_unchanged: bool,
) -> bool {
    fastpath_enabled && has_pending_submission && (source_is_duplicate || source_unchanged)
}

fn convert_cursor_shape_bgra_to_rgba(
    shape_buf: &[u8],
    shape_info: &DXGI_OUTDUPL_POINTER_SHAPE_INFO,
) -> Vec<u8> {
    let width = shape_info.Width as usize;
    let height = shape_info.Height as usize;
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let row_bytes = match width.checked_mul(4) {
        Some(bytes) => bytes,
        None => return Vec::new(),
    };
    let pitch = shape_info.Pitch as usize;

    let pixel_count = match width.checked_mul(height) {
        Some(count) => count,
        None => return Vec::new(),
    };
    let rgba_len = match pixel_count.checked_mul(4) {
        Some(len) => len,
        None => return Vec::new(),
    };
    let mut rgba = vec![0u8; rgba_len];

    if pitch == row_bytes {
        let required = match row_bytes.checked_mul(height) {
            Some(bytes) => bytes,
            None => return Vec::new(),
        };
        if shape_buf.len() >= required {
            unsafe {
                crate::convert::convert_bgra_to_rgba_unchecked(
                    shape_buf.as_ptr(),
                    rgba.as_mut_ptr(),
                    pixel_count,
                );
            }
            return rgba;
        }
    }

    if pitch < 4 {
        return rgba;
    }

    for row in 0..height {
        let src_offset = match row.checked_mul(pitch) {
            Some(offset) => offset,
            None => break,
        };
        if src_offset >= shape_buf.len() {
            break;
        }

        let available_bytes = (shape_buf.len() - src_offset).min(pitch).min(row_bytes);
        let available_pixels = available_bytes / 4;
        if available_pixels == 0 {
            continue;
        }

        let dst_offset = match row.checked_mul(row_bytes) {
            Some(offset) => offset,
            None => return Vec::new(),
        };

        unsafe {
            crate::convert::convert_bgra_to_rgba_unchecked(
                shape_buf.as_ptr().add(src_offset),
                rgba.as_mut_ptr().add(dst_offset),
                available_pixels,
            );
        }
    }

    rgba
}

/// Extract cursor shape and position from the DXGI duplication frame.
/// Returns `None` if cursor data is unavailable or extraction fails.
fn extract_cursor_data(
    duplication: &IDXGIOutputDuplication,
    info: &DXGI_OUTDUPL_FRAME_INFO,
) -> Option<CursorData> {
    let visible = info.PointerPosition.Visible.as_bool();
    let position_x = info.PointerPosition.Position.x;
    let position_y = info.PointerPosition.Position.y;

    // Try to get pointer shape if it was updated this frame.
    let (hotspot_x, hotspot_y, shape_width, shape_height, shape_rgba) =
        if info.PointerShapeBufferSize > 0 {
            let buf_size = info.PointerShapeBufferSize as usize;
            let mut shape_buf = vec![0u8; buf_size];
            let mut shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
            let mut required_size = 0u32;
            let hr = unsafe {
                duplication.GetFramePointerShape(
                    buf_size as u32,
                    shape_buf.as_mut_ptr() as *mut _,
                    &mut required_size,
                    &mut shape_info,
                )
            };
            if hr.is_ok() {
                let w = shape_info.Width;
                let h = shape_info.Height;
                let hotx = shape_info.HotSpot.x as u32;
                let hoty = shape_info.HotSpot.y as u32;

                // Convert to RGBA based on shape type.
                let rgba = match shape_info.Type {
                    t if t == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR.0 as u32
                        || t == DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR.0 as u32 =>
                    {
                        convert_cursor_shape_bgra_to_rgba(&shape_buf, &shape_info)
                    }
                    _ => {
                        // Monochrome or unknown -- skip shape data.
                        Vec::new()
                    }
                };
                (hotx, hoty, w, h, rgba)
            } else {
                (0, 0, 0, 0, Vec::new())
            }
        } else {
            (0, 0, 0, 0, Vec::new())
        };

    Some(CursorData {
        hotspot_x,
        hotspot_y,
        position_x,
        position_y,
        visible,
        shape_width,
        shape_height,
        shape_rgba,
    })
}

struct OutputCapturer {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    duplication: IDXGIOutputDuplication,
    staging_ring: StagingRing,
    /// Cached descriptor of the last successfully read frame, used to
    /// read back the pipelined staging slot on the next capture call.
    pending_desc: Option<D3D11_TEXTURE2D_DESC>,
    pending_hdr: Option<HdrToSdrParams>,
    /// Whether the pending pipelined frame was marked duplicate by DXGI.
    pending_is_duplicate: bool,
    /// Dirty rectangles associated with the pending pipelined frame.
    pending_dirty_rects: Vec<DirtyRect>,
    /// Pre-computed dirty-copy strategy for `pending_dirty_rects`.
    pending_dirty_cpu_copy_preferred: bool,
    pending_dirty_total_pixels: u64,
    /// Whether monitor capture is currently using the single-slot
    /// low-latency dirty GPU copy path.
    monitor_low_latency_dirty_gpu_active: bool,
    /// Whether slot 0 contains a complete monitor frame for incremental
    /// dirty GPU updates.
    monitor_low_latency_dirty_gpu_primed: bool,
    /// Descriptor key for the primed low-latency monitor slot.
    monitor_low_latency_dirty_gpu_desc: Option<(u32, u32, DXGI_FORMAT)>,
    /// Cached descriptor of the desktop texture from the duplication
    /// interface.  DXGI duplication textures don't change format/size
    /// mid-session, so we only need to query once (and re-query after
    /// `AccessLost` recreation).
    cached_src_desc: Option<D3D11_TEXTURE2D_DESC>,
    /// Internal frame buffer reused across captures when the caller
    /// doesn't pass one via `capture_frame_reuse`.  Avoids repeated
    /// large-page VirtualAlloc/VirtualFree cycles.
    spare_frame: Option<Frame>,
    /// Dedicated staging ring for sub-rect readback (window/region capture).
    /// Keeps window/region capture fully pipelined in screen-recording mode.
    region_slots: [RegionStagingSlot; DXGI_REGION_STAGING_SLOTS],
    region_pending_slot: Option<usize>,
    region_next_write_slot: usize,
    region_adaptive_spin_polls: u32,
    region_blit: Option<CaptureBlitRegion>,
    /// Scratch buffers reused for dirty-rect extraction to avoid per-frame allocations.
    dxgi_rect_buffer: Vec<RECT>,
    dxgi_move_rect_buffer: Vec<DXGI_OUTDUPL_MOVE_RECT>,
    source_dirty_rects_scratch: Vec<DirtyRect>,
    source_move_rects_scratch: Vec<MoveRect>,
    region_dirty_rects_scratch: Vec<DirtyRect>,
    region_move_rects_scratch: Vec<MoveRect>,
    output: IDXGIOutput,
    hdr_to_sdr: Option<HdrToSdrParams>,
    gpu_tonemapper: Option<GpuTonemapper>,
    /// GPU-side F16->sRGB converter for when the source is F16 but no
    /// HDR tonemapping is needed.  Avoids the expensive CPU-side SIMD
    /// F16 conversion path.
    gpu_f16_converter: Option<GpuF16Converter>,
    needs_presented_first_frame: bool,
    /// Last present time from DXGI, used for duplicate frame detection.
    last_present_time: i64,
    /// Whether to capture cursor shape and position data.
    cursor_config: CursorCaptureConfig,
    /// Capture intent controls whether recording-oriented buffering
    /// should be enabled.
    capture_mode: CaptureMode,
}

impl OutputCapturer {
    fn new(resolved: &ResolvedMonitor) -> CaptureResult<Self> {
        let (device, context) = d3d11::create_d3d11_device_for_adapter(&resolved.adapter, true)
            .map_err(CaptureError::Platform)?;
        let duplication = create_duplication(&resolved.output, &device)?;
        let hdr_to_sdr = hdr_to_sdr_params(resolved.hdr_metadata);
        let gpu_tonemapper = if hdr_to_sdr.is_some() {
            Some(GpuTonemapper::new(&device)?)
        } else {
            None
        };
        // Create the F16 converter for non-HDR F16 sources.
        // Non-fatal if it fails -- we fall back to CPU conversion.
        let gpu_f16_converter = GpuF16Converter::new(&device).ok();
        Ok(Self {
            device,
            context,
            duplication,
            staging_ring: StagingRing::new(),
            pending_desc: None,
            pending_hdr: None,
            pending_is_duplicate: false,
            pending_dirty_rects: Vec::new(),
            pending_dirty_cpu_copy_preferred: false,
            pending_dirty_total_pixels: 0,
            monitor_low_latency_dirty_gpu_active: false,
            monitor_low_latency_dirty_gpu_primed: false,
            monitor_low_latency_dirty_gpu_desc: None,
            cached_src_desc: None,
            spare_frame: None,
            region_slots: std::array::from_fn(|_| RegionStagingSlot::default()),
            region_pending_slot: None,
            region_next_write_slot: 0,
            region_adaptive_spin_polls: StagingRing::INITIAL_SPIN_POLLS,
            region_blit: None,
            dxgi_rect_buffer: Vec::new(),
            dxgi_move_rect_buffer: Vec::new(),
            source_dirty_rects_scratch: Vec::new(),
            source_move_rects_scratch: Vec::new(),
            region_dirty_rects_scratch: Vec::new(),
            region_move_rects_scratch: Vec::new(),
            output: resolved.output.clone(),
            hdr_to_sdr,
            gpu_tonemapper,
            gpu_f16_converter,
            needs_presented_first_frame: true,
            last_present_time: 0,
            cursor_config: CursorCaptureConfig::default(),
            capture_mode: CaptureMode::Screenshot,
        })
    }

    fn recreate_duplication(&mut self) -> CaptureResult<()> {
        self.staging_ring.invalidate();
        self.pending_desc = None;
        self.pending_hdr = None;
        self.pending_is_duplicate = false;
        self.pending_dirty_rects.clear();
        self.pending_dirty_cpu_copy_preferred = false;
        self.pending_dirty_total_pixels = 0;
        self.monitor_low_latency_dirty_gpu_active = false;
        self.monitor_low_latency_dirty_gpu_primed = false;
        self.monitor_low_latency_dirty_gpu_desc = None;
        self.cached_src_desc = None;
        self.invalidate_region_pipeline();
        self.dxgi_rect_buffer.clear();
        self.dxgi_move_rect_buffer.clear();
        self.source_dirty_rects_scratch.clear();
        self.source_move_rects_scratch.clear();
        self.region_dirty_rects_scratch.clear();
        self.region_move_rects_scratch.clear();
        self.duplication = create_duplication(&self.output, &self.device)?;
        self.needs_presented_first_frame = true;
        Ok(())
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        if self.capture_mode == mode {
            return;
        }
        self.capture_mode = mode;
        // Drop any in-flight pipeline state when switching modes.
        self.pending_desc = None;
        self.pending_hdr = None;
        self.pending_is_duplicate = false;
        self.pending_dirty_rects.clear();
        self.pending_dirty_cpu_copy_preferred = false;
        self.pending_dirty_total_pixels = 0;
        self.monitor_low_latency_dirty_gpu_active = false;
        self.monitor_low_latency_dirty_gpu_primed = false;
        self.monitor_low_latency_dirty_gpu_desc = None;
        self.staging_ring.reset_pipeline();
        self.reset_region_pipeline();
    }

    fn effective_source(
        &mut self,
        desktop_texture: &ID3D11Texture2D,
        src_desc: D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<(
        ID3D11Texture2D,
        D3D11_TEXTURE2D_DESC,
        Option<HdrToSdrParams>,
    )> {
        if src_desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT {
            return Ok((desktop_texture.clone(), src_desc, self.hdr_to_sdr));
        }

        if let (Some(params), Some(tonemapper)) = (self.hdr_to_sdr, self.gpu_tonemapper.as_mut()) {
            let output = tonemapper.tonemap(
                &self.device,
                &self.context,
                desktop_texture,
                &src_desc,
                params.sanitized(),
            )?;
            let mut out_desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { output.GetDesc(&mut out_desc) };
            return Ok((output.clone(), out_desc, None));
        }

        if let Some(converter) = self.gpu_f16_converter.as_mut() {
            let output =
                converter.convert(&self.device, &self.context, desktop_texture, &src_desc)?;
            let mut out_desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { output.GetDesc(&mut out_desc) };
            return Ok((output.clone(), out_desc, None));
        }

        Ok((desktop_texture.clone(), src_desc, self.hdr_to_sdr))
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

    fn reset_region_pipeline(&mut self) {
        self.region_pending_slot = None;
        self.region_next_write_slot = 0;
        self.region_adaptive_spin_polls = StagingRing::INITIAL_SPIN_POLLS;
        self.region_blit = None;
        for slot in &mut self.region_slots {
            slot.reset_runtime_state();
        }
    }

    fn invalidate_region_pipeline(&mut self) {
        self.region_pending_slot = None;
        self.region_next_write_slot = 0;
        self.region_adaptive_spin_polls = StagingRing::INITIAL_SPIN_POLLS;
        self.region_blit = None;
        for slot in &mut self.region_slots {
            slot.invalidate();
        }
    }

    fn ensure_region_pipeline_for_blit(&mut self, blit: CaptureBlitRegion) {
        if self.region_blit == Some(blit) {
            return;
        }
        self.reset_region_pipeline();
        self.region_blit = Some(blit);
    }

    fn ensure_region_slot(
        &mut self,
        slot_idx: usize,
        region_desc: &D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<()> {
        let slot = &mut self.region_slots[slot_idx];
        let key = (region_desc.Width, region_desc.Height, region_desc.Format);
        let slot_ready = slot.staging_key == Some(key)
            && slot.staging.is_some()
            && slot.staging_resource.is_some();
        if !slot_ready {
            let staging = surface::ensure_staging_texture(
                &self.device,
                &mut slot.staging,
                region_desc,
                StagingSampleDesc::SingleSample,
                "failed to create region staging texture",
            )?;
            slot.staging_resource = Some(
                staging
                    .cast::<ID3D11Resource>()
                    .context("failed to cast region staging texture to ID3D11Resource")
                    .map_err(CaptureError::Platform)?,
            );
            slot.staging_key = Some(key);
        }

        if slot.query.is_none() {
            let query_desc = D3D11_QUERY_DESC {
                Query: D3D11_QUERY_EVENT,
                ..Default::default()
            };
            let mut query: Option<ID3D11Query> = None;
            unsafe { self.device.CreateQuery(&query_desc, Some(&mut query)) }
                .context("CreateQuery for DXGI region staging slot failed")
                .map_err(CaptureError::Platform)?;
            slot.query = query;
        }

        Ok(())
    }

    fn query_signaled(&self, query: &ID3D11Query, flags: u32) -> bool {
        let mut data: u32 = 0;
        unsafe {
            self.context.GetData(
                query,
                Some(&mut data as *mut u32 as *mut _),
                std::mem::size_of::<u32>() as u32,
                flags,
            )
        }
        .is_ok()
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
                .max(StagingRing::MIN_SPIN_POLLS);
        } else {
            self.region_adaptive_spin_polls = self
                .region_adaptive_spin_polls
                .saturating_add(StagingRing::SPIN_INCREASE_STEP)
                .min(StagingRing::MAX_SPIN_POLLS);
        }
    }

    fn copy_region_source_to_slot(
        &self,
        slot_idx: usize,
        source: &ID3D11Texture2D,
        blit: CaptureBlitRegion,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        let slot = &self.region_slots[slot_idx];
        let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "DXGI region slot missing staging resource after initialization"
            ))
        })?;
        d3d11::with_texture_resource(
            source,
            "failed to cast region source texture to ID3D11Resource",
            |source_resource| {
                let mut used_dirty_copy = false;
                if can_use_dirty_gpu_copy && slot.dirty_gpu_copy_preferred {
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
                Ok(())
            },
        )?;

        unsafe {
            if let Some(query) = slot.query.as_ref() {
                self.context.End(query);
            }
        }
        Ok(())
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
                "DXGI region slot is populated but missing source descriptor"
            ))
        })?;
        let sample = CaptureSampleMetadata {
            capture_time: Some(slot.capture_time.unwrap_or_else(Instant::now)),
            present_time_qpc: if slot.present_time_qpc != 0 {
                Some(slot.present_time_qpc)
            } else {
                None
            },
            is_duplicate: slot.is_duplicate,
        };

        let moves_only = {
            let slot = &self.region_slots[slot_idx];
            let has_moves =
                destination_has_history && slot.move_mode_available && !slot.move_rects.is_empty();
            if has_moves {
                apply_move_rects_to_frame(out, &slot.move_rects, blit.dst_x, blit.dst_y)?;
            }
            has_moves && slot.dirty_mode_available && slot.dirty_rects.is_empty()
        };

        if destination_has_history && sample.is_duplicate {
            return Ok(sample);
        }
        if moves_only {
            return Ok(sample);
        }

        self.wait_for_region_slot_copy(slot_idx);

        let slot = &self.region_slots[slot_idx];
        let staging = slot.staging.as_ref().ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "DXGI region slot is populated but missing staging texture"
            ))
        })?;
        let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "DXGI region slot is populated but missing staging resource"
            ))
        })?;
        let staging_blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: source_desc.Width,
            height: source_desc.Height,
            dst_x: blit.dst_x,
            dst_y: blit.dst_y,
        };

        let use_dirty_copy = destination_has_history
            && slot.dirty_mode_available
            && !slot.dirty_rects.is_empty()
            && slot.dirty_cpu_copy_preferred;
        let write_full_slot_direct = region_full_slot_map_fastpath_enabled()
            && blit.dst_x == 0
            && blit.dst_y == 0
            && out.width() == source_desc.Width
            && out.height() == source_desc.Height;
        let map_full_slot = |out: &mut Frame| -> CaptureResult<()> {
            if write_full_slot_direct {
                surface::map_staging_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    slot.hdr_to_sdr,
                    "failed to map DXGI region staging texture",
                )?;
            } else {
                surface::map_staging_rect_to_frame(
                    &self.context,
                    staging,
                    Some(staging_resource),
                    &source_desc,
                    out,
                    staging_blit,
                    slot.hdr_to_sdr,
                    "failed to map DXGI region staging texture",
                )?;
            }
            Ok(())
        };

        if use_dirty_copy {
            let dirty_hints = surface::DirtyRectConversionHints {
                trusted_bounds: true,
                non_empty_rects: Some(slot.dirty_rects.len()),
                total_dirty_pixels: usize::try_from(slot.dirty_total_pixels).ok(),
            };
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
                    "failed to map DXGI region staging texture (dirty regions)",
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
                    "failed to map DXGI region staging texture (dirty regions)",
                )
            };

            match dirty_map_result {
                Ok(converted) if converted > 0 => Ok(sample),
                Ok(_) | Err(_) => {
                    map_full_slot(out)?;
                    Ok(sample)
                }
            }
        } else {
            map_full_slot(out)?;
            Ok(sample)
        }
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        destination_has_history: bool,
        prefer_low_latency: bool,
    ) -> CaptureResult<CaptureSampleMetadata> {
        if blit.width == 0 || blit.height == 0 {
            return Err(CaptureError::InvalidConfig(
                "capture region dimensions must be non-zero".into(),
            ));
        }

        // Region capture uses its own sub-rect staging path.
        // Reset full-frame pipeline state so monitor capture and region
        // capture don't consume stale pending slots when callers switch targets.
        self.pending_desc = None;
        self.pending_hdr = None;
        self.pending_is_duplicate = false;
        self.pending_dirty_rects.clear();
        self.pending_dirty_cpu_copy_preferred = false;
        self.pending_dirty_total_pixels = 0;
        self.monitor_low_latency_dirty_gpu_active = false;
        self.monitor_low_latency_dirty_gpu_primed = false;
        self.monitor_low_latency_dirty_gpu_desc = None;
        self.staging_ring.reset_pipeline();

        let mut destination_has_history = destination_has_history;
        if self.region_blit != Some(blit) {
            // Callers may reuse a frame across different window/region targets.
            // Even if dimensions match, the previous pixels are stale when the
            // source blit changes or the region pipeline was reset.
            destination_has_history = false;
        }
        self.ensure_region_pipeline_for_blit(blit);

        let capture_time = Instant::now();
        let single_shot_screenshot =
            self.capture_mode == CaptureMode::Screenshot && !destination_has_history;
        let (desktop_texture, frame_info) =
            match acquire_frame(&self.duplication, self.needs_presented_first_frame)? {
                AcquireResult::Ok(texture, info) => (texture, info),
                AcquireResult::AccessLost => {
                    self.recreate_duplication()?;
                    destination_has_history = false;
                    self.ensure_region_pipeline_for_blit(blit);
                    match acquire_frame(&self.duplication, self.needs_presented_first_frame)? {
                        AcquireResult::Ok(texture, info) => (texture, info),
                        AcquireResult::AccessLost => return Err(CaptureError::AccessLost),
                    }
                }
            };

        let source_present_time_qpc = frame_info.LastPresentTime;
        let source_is_duplicate =
            source_present_time_qpc != 0 && source_present_time_qpc == self.last_present_time;
        if source_present_time_qpc != 0 {
            self.last_present_time = source_present_time_qpc;
        }

        let mut region_dirty_rects = std::mem::take(&mut self.region_dirty_rects_scratch);
        let mut region_move_rects = std::mem::take(&mut self.region_move_rects_scratch);
        let capture_result = (|| -> CaptureResult<CaptureSampleMetadata> {
            let src_desc = match self.cached_src_desc {
                Some(desc) => desc,
                None => {
                    let mut desc = D3D11_TEXTURE2D_DESC::default();
                    unsafe { desktop_texture.GetDesc(&mut desc) };
                    self.cached_src_desc = Some(desc);
                    desc
                }
            };

            let (effective_source, effective_desc, effective_hdr) =
                self.effective_source(&desktop_texture, src_desc)?;

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

            if single_shot_screenshot {
                let write_slot = 0usize;
                self.ensure_region_slot(write_slot, &region_desc)?;
                {
                    let slot = &mut self.region_slots[write_slot];
                    slot.capture_time = Some(capture_time);
                    slot.present_time_qpc = source_present_time_qpc;
                    slot.is_duplicate = source_is_duplicate;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(region_desc);
                    slot.dirty_mode_available = false;
                    slot.move_mode_available = false;
                    slot.dirty_cpu_copy_preferred = false;
                    slot.dirty_gpu_copy_preferred = false;
                    slot.dirty_total_pixels = 0;
                    slot.dirty_rects.clear();
                    slot.move_rects.clear();
                    slot.populated = true;
                }

                self.copy_region_source_to_slot(write_slot, &effective_source, blit, false)?;
                self.maybe_flush_region_after_submit(write_slot, write_slot);
                let sample =
                    self.read_region_slot_into_output(write_slot, destination, false, blit)?;
                self.region_pending_slot = None;
                self.region_next_write_slot = 0;
                return Ok(sample);
            }

            let (region_dirty_available, region_move_available, region_has_moves, region_unchanged) =
                if source_is_duplicate {
                    region_dirty_rects.clear();
                    region_move_rects.clear();
                    (true, true, false, true)
                } else {
                    let region_dirty_available = extract_region_dirty_rects_direct(
                        &self.duplication,
                        &frame_info,
                        &mut self.dxgi_rect_buffer,
                        effective_desc.Width,
                        effective_desc.Height,
                        blit,
                        &mut region_dirty_rects,
                    );
                    let region_move_available = {
                        let source_move_available = extract_move_rects(
                            &self.duplication,
                            &frame_info,
                            &mut self.dxgi_move_rect_buffer,
                            effective_desc.Width,
                            effective_desc.Height,
                            &mut self.source_move_rects_scratch,
                        );
                        let region_move_available = source_move_available
                            && extract_region_move_rects(
                                &self.source_move_rects_scratch,
                                effective_desc.Width,
                                effective_desc.Height,
                                blit,
                                &mut region_move_rects,
                            );
                        if !region_move_available {
                            region_move_rects.clear();
                        }
                        region_move_available
                    };

                    if region_dirty_available
                        && region_dirty_rects.len() <= DXGI_DIRTY_COPY_MAX_RECTS
                    {
                        normalize_dirty_rects_in_place(
                            &mut region_dirty_rects,
                            blit.width,
                            blit.height,
                        );
                    }

                    let region_has_moves = region_move_available && !region_move_rects.is_empty();
                    (
                        region_dirty_available,
                        region_move_available,
                        region_has_moves,
                        region_dirty_available
                            && region_move_available
                            && region_dirty_rects.is_empty()
                            && !region_has_moves,
                    )
                };
            if self.capture_mode != CaptureMode::ScreenRecording
                && destination_has_history
                && (source_is_duplicate || region_unchanged)
            {
                return Ok(CaptureSampleMetadata {
                    capture_time: Some(capture_time),
                    present_time_qpc: if source_present_time_qpc != 0 {
                        Some(source_present_time_qpc)
                    } else {
                        None
                    },
                    is_duplicate: true,
                });
            }

            let recording_mode = self.capture_mode == CaptureMode::ScreenRecording;
            let can_use_dirty_reconstruct =
                can_use_region_dirty_reconstruct(region_move_available, region_has_moves, true);
            let dirty_copy_strategy = if region_dirty_available && can_use_dirty_reconstruct {
                evaluate_dirty_copy_strategy(
                    &region_dirty_rects,
                    region_desc.Width,
                    region_desc.Height,
                )
            } else {
                DirtyCopyStrategy::default()
            };
            let low_latency_dirty_gpu_preferred = dirty_copy_strategy.gpu_low_latency;
            let regular_dirty_gpu_preferred = dirty_copy_strategy.gpu;
            let low_latency_recording = should_use_window_low_latency_region_path(
                recording_mode,
                prefer_low_latency,
                blit,
                low_latency_dirty_gpu_preferred,
            );
            let write_slot = if low_latency_recording {
                self.region_pending_slot.unwrap_or(0)
            } else if recording_mode {
                self.region_next_write_slot % DXGI_REGION_STAGING_SLOTS
            } else {
                0
            };
            let read_slot = if low_latency_recording {
                write_slot
            } else if recording_mode {
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
                slot.present_time_qpc = source_present_time_qpc;
                slot.is_duplicate = true;
                slot.hdr_to_sdr = effective_hdr;
                slot.source_desc = Some(region_desc);
                slot.dirty_mode_available = region_dirty_available;
                slot.move_mode_available = region_move_available;
                slot.dirty_cpu_copy_preferred = false;
                slot.dirty_gpu_copy_preferred = false;
                slot.dirty_total_pixels = 0;
                slot.dirty_rects.clear();
                slot.move_rects.clear();
                slot.populated = true;
                slot_idx
            } else {
                self.ensure_region_slot(write_slot, &region_desc)?;
                let can_use_dirty_gpu_copy = destination_has_history
                    && (self.capture_mode != CaptureMode::ScreenRecording || low_latency_recording);
                {
                    let slot = &mut self.region_slots[write_slot];
                    slot.capture_time = Some(capture_time);
                    slot.present_time_qpc = source_present_time_qpc;
                    slot.is_duplicate = source_is_duplicate || region_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(region_desc);
                    slot.dirty_mode_available = region_dirty_available;
                    slot.move_mode_available = region_move_available;
                    std::mem::swap(&mut slot.dirty_rects, &mut region_dirty_rects);
                    std::mem::swap(&mut slot.move_rects, &mut region_move_rects);
                    slot.dirty_cpu_copy_preferred =
                        region_dirty_available && dirty_copy_strategy.cpu;
                    let dirty_gpu_copy_preferred = if low_latency_recording {
                        low_latency_dirty_gpu_preferred
                    } else {
                        regular_dirty_gpu_preferred
                    };
                    slot.dirty_gpu_copy_preferred = can_use_dirty_gpu_copy
                        && region_dirty_available
                        && dirty_gpu_copy_preferred;
                    slot.dirty_total_pixels = dirty_copy_strategy.dirty_pixels;
                    slot.populated = true;
                }

                self.copy_region_source_to_slot(
                    write_slot,
                    &effective_source,
                    blit,
                    can_use_dirty_gpu_copy,
                )?;
                self.maybe_flush_region_after_submit(write_slot, read_slot);
                read_slot
            };

            let sample = self.read_region_slot_into_output(
                read_slot,
                destination,
                destination_has_history,
                blit,
            )?;

            if recording_mode {
                if !skip_submit_copy {
                    self.region_pending_slot = Some(write_slot);
                    if low_latency_recording {
                        self.region_next_write_slot = write_slot;
                    } else {
                        self.region_next_write_slot = (write_slot + 1) % DXGI_REGION_STAGING_SLOTS;
                    }
                }
            } else {
                self.region_pending_slot = None;
                self.region_next_write_slot = 0;
            }

            Ok(sample)
        })();

        region_dirty_rects.clear();
        self.region_dirty_rects_scratch = region_dirty_rects;
        region_move_rects.clear();
        self.region_move_rects_scratch = region_move_rects;

        unsafe {
            self.duplication.ReleaseFrame().ok();
        }
        self.needs_presented_first_frame = false;

        if capture_result.is_err() {
            self.reset_region_pipeline();
        }

        capture_result
    }

    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        // Full-frame capture and region/window capture keep independent
        // pipelines. Reset region state when callers switch back to full
        // monitor capture to avoid consuming stale region slots later.
        if self.region_blit.is_some() || self.region_pending_slot.is_some() {
            self.reset_region_pipeline();
        }

        // Reuse caller-provided frame, or fall back to our internal spare,
        // or create a new empty frame as last resort.
        let mut frame = reuse
            .or_else(|| self.spare_frame.take())
            .unwrap_or_else(Frame::empty);
        let has_frame_history =
            frame.metadata.capture_time.is_some() && !frame.as_rgba_bytes().is_empty();
        let single_shot_screenshot =
            self.capture_mode == CaptureMode::Screenshot && !has_frame_history;
        frame.reset_metadata();

        let capture_time = Instant::now();

        let (desktop_texture, frame_info) =
            match acquire_frame(&self.duplication, self.needs_presented_first_frame)? {
                AcquireResult::Ok(texture, info) => (texture, info),
                AcquireResult::AccessLost => {
                    self.recreate_duplication()?;
                    match acquire_frame(&self.duplication, self.needs_presented_first_frame)? {
                        AcquireResult::Ok(texture, info) => (texture, info),
                        AcquireResult::AccessLost => return Err(CaptureError::AccessLost),
                    }
                }
            };

        // Populate frame metadata from DXGI frame info.
        frame.metadata.capture_time = Some(capture_time);
        frame.metadata.present_time_qpc = if frame_info.LastPresentTime != 0 {
            Some(frame_info.LastPresentTime)
        } else {
            None
        };
        frame.metadata.is_duplicate =
            frame_info.LastPresentTime != 0 && frame_info.LastPresentTime == self.last_present_time;
        if frame_info.LastPresentTime != 0 {
            self.last_present_time = frame_info.LastPresentTime;
        }

        // Duplicate frames have no new desktop damage. Skip the COM metadata
        // query on this fast path.
        let source_unchanged = if single_shot_screenshot {
            frame.metadata.dirty_rects.clear();
            false
        } else if frame.metadata.is_duplicate {
            frame.metadata.dirty_rects.clear();
            true
        } else if extract_dirty_rects(
            &self.duplication,
            &frame_info,
            &mut self.dxgi_rect_buffer,
            &mut frame.metadata.dirty_rects,
        ) {
            frame.metadata.dirty_rects.is_empty()
        } else {
            false
        };

        // Extract cursor data if configured.
        if self.cursor_config.capture_cursor {
            frame.metadata.cursor = extract_cursor_data(&self.duplication, &frame_info);
        }

        // Use cached source descriptor when available -- avoids a COM
        // GetDesc() call on every frame.  Invalidated on AccessLost.
        let src_desc = match self.cached_src_desc {
            Some(desc) => desc,
            None => {
                let mut desc = D3D11_TEXTURE2D_DESC::default();
                unsafe { desktop_texture.GetDesc(&mut desc) };
                self.cached_src_desc = Some(desc);
                desc
            }
        };

        let (effective_source, effective_desc, effective_hdr) =
            self.effective_source(&desktop_texture, src_desc)?;

        let frame_pixels_unchanged = frame.metadata.is_duplicate || source_unchanged;
        let output_matches_source = has_frame_history
            && frame.width() == effective_desc.Width
            && frame.height() == effective_desc.Height;

        let mut normalized_dirty_rects = std::mem::take(&mut self.source_dirty_rects_scratch);
        normalized_dirty_rects.clear();
        let mut current_dirty_strategy = DirtyCopyStrategy::default();
        if !single_shot_screenshot
            && !frame_pixels_unchanged
            && !frame.metadata.dirty_rects.is_empty()
            && frame.metadata.dirty_rects.len() <= DXGI_DIRTY_COPY_MAX_RECTS
        {
            normalized_dirty_rects.extend_from_slice(&frame.metadata.dirty_rects);
            normalize_dirty_rects_in_place(
                &mut normalized_dirty_rects,
                effective_desc.Width,
                effective_desc.Height,
            );
            current_dirty_strategy = evaluate_dirty_copy_strategy(
                &normalized_dirty_rects,
                effective_desc.Width,
                effective_desc.Height,
            );
        }

        let convert_result = (|| -> CaptureResult<()> {
            if self.capture_mode == CaptureMode::ScreenRecording {
                let use_monitor_low_latency_dirty_gpu =
                    should_use_monitor_low_latency_dirty_gpu_mode(
                        self.capture_mode,
                        self.monitor_low_latency_dirty_gpu_active,
                        output_matches_source,
                        frame_pixels_unchanged,
                        current_dirty_strategy,
                    );

                if use_monitor_low_latency_dirty_gpu {
                    if !self.monitor_low_latency_dirty_gpu_active {
                        self.pending_desc = None;
                        self.pending_hdr = None;
                        self.pending_is_duplicate = false;
                        self.pending_dirty_rects.clear();
                        self.pending_dirty_cpu_copy_preferred = false;
                        self.pending_dirty_total_pixels = 0;
                        self.staging_ring.reset_pipeline();
                        self.monitor_low_latency_dirty_gpu_primed = false;
                        self.monitor_low_latency_dirty_gpu_desc = None;
                    }
                    self.monitor_low_latency_dirty_gpu_active = true;

                    self.staging_ring
                        .ensure_slots(&self.device, &effective_desc, 1)?;
                    let desc_key = (
                        effective_desc.Width,
                        effective_desc.Height,
                        effective_desc.Format,
                    );
                    if self.monitor_low_latency_dirty_gpu_desc != Some(desc_key) {
                        self.monitor_low_latency_dirty_gpu_desc = Some(desc_key);
                        self.monitor_low_latency_dirty_gpu_primed = false;
                    }

                    let skip_readback = output_matches_source && frame_pixels_unchanged;
                    if !skip_readback {
                        let use_dirty_gpu_copy = self.monitor_low_latency_dirty_gpu_primed
                            && !normalized_dirty_rects.is_empty()
                            && current_dirty_strategy.gpu;
                        let use_dirty_cpu_copy = output_matches_source
                            && !normalized_dirty_rects.is_empty()
                            && current_dirty_strategy.cpu;
                        self.staging_ring.copy_and_read_with_strategy(
                            &self.context,
                            &effective_source,
                            &effective_desc,
                            &mut frame,
                            effective_hdr,
                            &normalized_dirty_rects,
                            current_dirty_strategy.dirty_pixels,
                            use_dirty_cpu_copy,
                            use_dirty_gpu_copy,
                        )?;
                        self.monitor_low_latency_dirty_gpu_primed = true;
                    }

                    self.pending_desc = None;
                    self.pending_hdr = None;
                    self.pending_is_duplicate = false;
                    self.pending_dirty_rects.clear();
                    self.pending_dirty_cpu_copy_preferred = false;
                    self.pending_dirty_total_pixels = 0;
                } else {
                    if self.monitor_low_latency_dirty_gpu_active {
                        self.monitor_low_latency_dirty_gpu_active = false;
                        self.monitor_low_latency_dirty_gpu_primed = false;
                        self.monitor_low_latency_dirty_gpu_desc = None;
                        self.staging_ring.reset_pipeline();
                    }

                    self.staging_ring
                        .ensure_slots(&self.device, &effective_desc, STAGING_SLOTS)?;
                    let pending_slot_compatible = self.pending_desc.as_ref().is_some_and(|desc| {
                        desc.Width == effective_desc.Width
                            && desc.Height == effective_desc.Height
                            && desc.Format == effective_desc.Format
                    }) && self.pending_hdr == effective_hdr;
                    let skip_submit_copy = should_skip_screenrecord_submit_copy(
                        true,
                        pending_slot_compatible,
                        frame.metadata.is_duplicate,
                        source_unchanged,
                    );

                    let mut read_slot = self.staging_ring.latest_write_slot();
                    let mut read_desc = effective_desc;
                    let mut read_hdr = effective_hdr;
                    let mut read_is_duplicate = frame_pixels_unchanged;
                    let mut read_dirty_rects: &[DirtyRect] = &[];
                    let mut read_dirty_cpu_copy_preferred = false;
                    let mut read_dirty_total_pixels = 0u64;

                    if skip_submit_copy {
                        if let Some(prev_desc) = self.pending_desc.as_ref() {
                            read_desc = *prev_desc;
                            read_hdr = self.pending_hdr;
                            // The pending slot may still contain new pixels that
                            // haven't been consumed yet (pipeline catch-up after a
                            // non-duplicate frame). Respect its duplicate state.
                            read_is_duplicate = self.pending_is_duplicate;
                            read_dirty_rects = &self.pending_dirty_rects;
                            read_dirty_cpu_copy_preferred = self.pending_dirty_cpu_copy_preferred;
                            read_dirty_total_pixels = self.pending_dirty_total_pixels;
                        }
                    } else {
                        let submitted_read_slot = self
                            .staging_ring
                            .submit_copy(&self.context, &effective_source)?;
                        if let (Some(slot), Some(prev_desc)) =
                            (submitted_read_slot, self.pending_desc.as_ref())
                        {
                            // Read back the previous slot while the next copy is in flight.
                            read_slot = slot;
                            read_desc = *prev_desc;
                            read_hdr = self.pending_hdr;
                            read_is_duplicate = self.pending_is_duplicate;
                            read_dirty_rects = &self.pending_dirty_rects;
                            read_dirty_cpu_copy_preferred = self.pending_dirty_cpu_copy_preferred;
                            read_dirty_total_pixels = self.pending_dirty_total_pixels;
                        } else {
                            // Bootstrap/desync path: read the freshly submitted slot.
                            read_slot = self.staging_ring.latest_write_slot();
                            read_desc = effective_desc;
                            read_hdr = effective_hdr;
                            read_is_duplicate = frame_pixels_unchanged;
                        }
                    }

                    let read_output_matches_source = has_frame_history
                        && frame.width() == read_desc.Width
                        && frame.height() == read_desc.Height;
                    let skip_readback = read_output_matches_source && read_is_duplicate;
                    let use_dirty_copy = read_output_matches_source
                        && !skip_readback
                        && !read_dirty_rects.is_empty()
                        && read_dirty_cpu_copy_preferred;
                    self.staging_ring.read_slot_with_strategy(
                        &self.context,
                        read_slot,
                        &read_desc,
                        &mut frame,
                        read_hdr,
                        read_dirty_rects,
                        read_dirty_total_pixels,
                        use_dirty_copy,
                        skip_readback,
                    )?;

                    if skip_submit_copy {
                        // We intentionally kept the previous pending slot alive;
                        // keep metadata aligned with that slot's unchanged contents.
                        self.pending_is_duplicate = true;
                        self.pending_dirty_rects.clear();
                        self.pending_dirty_cpu_copy_preferred = false;
                        self.pending_dirty_total_pixels = 0;
                    } else {
                        self.pending_desc = Some(effective_desc);
                        self.pending_hdr = effective_hdr;
                        self.pending_is_duplicate = frame_pixels_unchanged;
                        self.pending_dirty_rects.clear();
                        self.pending_dirty_cpu_copy_preferred = false;
                        self.pending_dirty_total_pixels = 0;
                        if !frame_pixels_unchanged && !normalized_dirty_rects.is_empty() {
                            self.pending_dirty_rects
                                .extend_from_slice(&normalized_dirty_rects);
                            self.pending_dirty_cpu_copy_preferred = current_dirty_strategy.cpu;
                            self.pending_dirty_total_pixels = current_dirty_strategy.dirty_pixels;
                        }
                    }
                }
            } else {
                // Screenshot mode avoids recording-only buffering.
                self.staging_ring
                    .ensure_slots(&self.device, &effective_desc, 1)?;
                self.pending_desc = None;
                self.pending_hdr = None;
                self.pending_is_duplicate = false;
                self.pending_dirty_rects.clear();
                self.pending_dirty_cpu_copy_preferred = false;
                self.pending_dirty_total_pixels = 0;
                self.monitor_low_latency_dirty_gpu_active = false;
                self.monitor_low_latency_dirty_gpu_primed = false;
                self.monitor_low_latency_dirty_gpu_desc = None;
                self.staging_ring.reset_pipeline();
                self.staging_ring.copy_and_read(
                    &self.context,
                    &effective_source,
                    &effective_desc,
                    &mut frame,
                    effective_hdr,
                )?;
            }
            Ok(())
        })();

        normalized_dirty_rects.clear();
        self.source_dirty_rects_scratch = normalized_dirty_rects;
        unsafe {
            self.duplication.ReleaseFrame().ok();
        }
        self.needs_presented_first_frame = false;
        if let Err(err) = convert_result {
            self.pending_desc = None;
            self.pending_hdr = None;
            self.pending_is_duplicate = false;
            self.pending_dirty_rects.clear();
            self.pending_dirty_cpu_copy_preferred = false;
            self.pending_dirty_total_pixels = 0;
            self.monitor_low_latency_dirty_gpu_active = false;
            self.monitor_low_latency_dirty_gpu_primed = false;
            self.monitor_low_latency_dirty_gpu_desc = None;
            self.staging_ring.reset_pipeline();
            return Err(err);
        }

        Ok(frame)
    }
}

pub(crate) struct WindowsMonitorCapturer {
    monitor: MonitorId,
    resolver: Arc<MonitorResolver>,
    _com: super::com::CoInitGuard,
    output: OutputCapturer,
    cursor_config: CursorCaptureConfig,
    capture_mode: CaptureMode,
}

impl WindowsMonitorCapturer {
    pub(crate) fn new(monitor: &MonitorId, resolver: Arc<MonitorResolver>) -> CaptureResult<Self> {
        let com = super::com::CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
        let resolved = resolver.resolve_monitor(monitor)?;
        let output = with_monitor_context(OutputCapturer::new(&resolved), monitor, "initialize")?;
        Ok(Self {
            monitor: monitor.clone(),
            resolver,
            _com: com,
            output,
            cursor_config: CursorCaptureConfig::default(),
            capture_mode: CaptureMode::Screenshot,
        })
    }
}

impl crate::backend::MonitorCapturer for WindowsMonitorCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        let result = self.output.capture(reuse);
        match result {
            Ok(frame) => {
                // Stash a spare frame internally so the next call without
                // an explicit reuse frame can skip allocation.
                if self.output.spare_frame.is_none() {
                    // We can't keep *this* frame (we're returning it), so
                    // create a detached spare buffer with matching geometry.
                    let mut spare = Frame::empty();
                    if spare
                        .ensure_rgba_capacity(frame.width(), frame.height())
                        .is_ok()
                    {
                        self.output.spare_frame = Some(spare);
                    }
                }
                Ok(frame)
            }
            Err(CaptureError::MonitorLost) => {
                let resolved = self.resolver.resolve_monitor(&self.monitor)?;
                self.output = with_monitor_context(
                    OutputCapturer::new(&resolved),
                    &self.monitor,
                    "reinitialize",
                )?;
                self.output.cursor_config = self.cursor_config;
                self.output.set_capture_mode(self.capture_mode);
                self.output.capture(None)
            }
            Err(e) => Err(e),
        }
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        destination_has_history: bool,
    ) -> CaptureResult<Option<CaptureSampleMetadata>> {
        let prefer_low_latency = should_prefer_monitor_region_low_latency(self.capture_mode, blit);
        let result = self.output.capture_region_into(
            blit,
            destination,
            destination_has_history,
            prefer_low_latency,
        );
        match result {
            Ok(sample) => Ok(Some(sample)),
            Err(CaptureError::MonitorLost) | Err(CaptureError::AccessLost) => {
                let resolved = self.resolver.resolve_monitor(&self.monitor)?;
                self.output = with_monitor_context(
                    OutputCapturer::new(&resolved),
                    &self.monitor,
                    "reinitialize",
                )?;
                self.output.cursor_config = self.cursor_config;
                self.output.set_capture_mode(self.capture_mode);
                let retry_prefer_low_latency =
                    should_prefer_monitor_region_low_latency(self.capture_mode, blit);
                self.output
                    .capture_region_into(
                        blit,
                        destination,
                        destination_has_history,
                        retry_prefer_low_latency,
                    )
                    .map(Some)
            }
            Err(e) => Err(e),
        }
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
        self.output.set_capture_mode(mode);
    }

    fn set_cursor_config(&mut self, config: CursorCaptureConfig) {
        self.cursor_config = config;
        self.output.cursor_config = config;
    }
}

// ---------------------------------------------------------------------------
// DXGI-based window capture: captures only the window's visible monitor
// sub-rectangle via CopySubresourceRegion, avoiding full-monitor readback
// and CPU-side cropping.
// ---------------------------------------------------------------------------

use crate::window::WindowId;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::HMONITOR;
use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, IsIconic, IsWindow};

/// Find the monitor that contains the majority of the given window.
fn monitor_from_window(hwnd: HWND) -> Option<HMONITOR> {
    use windows::Win32::Graphics::Gdi::MONITOR_DEFAULTTONULL;
    use windows::Win32::Graphics::Gdi::MonitorFromWindow;
    let hmon = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL) };
    if hmon.0.is_null() { None } else { Some(hmon) }
}

/// Resolve a `HMONITOR` handle to a `MonitorId` by matching against the
/// known monitor set from the resolver.
fn hmonitor_to_monitor_id(hmon: HMONITOR, resolver: &MonitorResolver) -> CaptureResult<MonitorId> {
    let monitors = resolver.enumerate_monitors()?;
    let target_handle = hmon.0 as isize;
    monitors
        .into_iter()
        .find(|m| m.raw_handle() == target_handle)
        .ok_or_else(|| {
            CaptureError::BackendUnavailable(
                "could not resolve HMONITOR to a known MonitorId".into(),
            )
        })
}

/// Get the monitor's desktop rectangle via `GetMonitorInfoW`.
fn monitor_rect(hmon: HMONITOR) -> CaptureResult<RECT> {
    use std::mem::size_of;
    use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITORINFO};

    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    if !unsafe { GetMonitorInfoW(hmon, &mut info) }.as_bool() {
        return Err(CaptureError::Platform(anyhow::anyhow!(
            "GetMonitorInfoW failed for DXGI window capture"
        )));
    }
    Ok(info.rcMonitor)
}

#[inline(always)]
fn rect_within_rect(inner: &RECT, outer: &RECT) -> bool {
    inner.left >= outer.left
        && inner.top >= outer.top
        && inner.right <= outer.right
        && inner.bottom <= outer.bottom
}

/// Wrapper around `HWND` to satisfy `Send`.  Window handles are
/// plain integer-sized values that are safe to use from any thread
/// (Win32 window messages are dispatched by the OS regardless of
/// calling thread).
#[derive(Clone, Copy)]
struct SendHwnd(HWND);
unsafe impl Send for SendHwnd {}

/// Same wrapper for `HMONITOR`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SendHmon(HMONITOR);
unsafe impl Send for SendHmon {}

pub(crate) struct WindowsDxgiWindowCapturer {
    hwnd: SendHwnd,
    resolver: Arc<MonitorResolver>,
    _com: super::com::CoInitGuard,
    /// The DXGI output capturer for the monitor the window is on.
    output: OutputCapturer,
    /// Cached monitor handle so we can detect when the window moves
    /// to a different monitor.
    current_hmon: SendHmon,
    /// Cached monitor desktop bounds. Avoids `GetMonitorInfoW` on every frame.
    current_monitor_rect: RECT,
    cursor_config: CursorCaptureConfig,
    capture_mode: CaptureMode,
}

impl WindowsDxgiWindowCapturer {
    pub(crate) fn new(window: &WindowId, resolver: Arc<MonitorResolver>) -> CaptureResult<Self> {
        let com = super::com::CoInitGuard::init_multithreaded().map_err(CaptureError::Platform)?;
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

        let hmon = monitor_from_window(hwnd).ok_or_else(|| {
            CaptureError::BackendUnavailable("window is not on any monitor".into())
        })?;

        let monitor_id = hmonitor_to_monitor_id(hmon, &resolver)?;
        let resolved = resolver.resolve_monitor(&monitor_id)?;
        let output = OutputCapturer::new(&resolved).map_err(|e| {
            CaptureError::BackendUnavailable(format!(
                "failed to create DXGI duplication for window's monitor: {e}"
            ))
        })?;
        let current_monitor_rect = monitor_rect(hmon)?;

        Ok(Self {
            hwnd: SendHwnd(hwnd),
            resolver,
            _com: com,
            output,
            current_hmon: SendHmon(hmon),
            current_monitor_rect,
            cursor_config: CursorCaptureConfig::default(),
            capture_mode: CaptureMode::Screenshot,
        })
    }

    /// Re-create the DXGI output capturer when the window moves to a
    /// different monitor or after access-lost recovery.
    fn reinit_for_monitor(&mut self, hmon: HMONITOR) -> CaptureResult<()> {
        let monitor_id = hmonitor_to_monitor_id(hmon, &self.resolver)?;
        let resolved = self.resolver.resolve_monitor(&monitor_id)?;
        self.output = OutputCapturer::new(&resolved)?;
        self.output.cursor_config = self.cursor_config;
        self.output.set_capture_mode(self.capture_mode);
        self.current_hmon = SendHmon(hmon);
        self.current_monitor_rect = monitor_rect(hmon)?;
        Ok(())
    }

    fn window_blit_on_monitor(
        mon_rect: &RECT,
        win_rect: &RECT,
    ) -> CaptureResult<CaptureBlitRegion> {
        let mon_w = (mon_rect.right - mon_rect.left) as u32;
        let mon_h = (mon_rect.bottom - mon_rect.top) as u32;

        let src_x = (win_rect.left.max(mon_rect.left) - mon_rect.left) as u32;
        let src_y = (win_rect.top.max(mon_rect.top) - mon_rect.top) as u32;
        let right = (win_rect.right.min(mon_rect.right) - mon_rect.left) as u32;
        let bottom = (win_rect.bottom.min(mon_rect.bottom) - mon_rect.top) as u32;

        if src_x >= right || src_y >= bottom || right > mon_w || bottom > mon_h {
            return Err(CaptureError::InvalidTarget(
                "window has no visible area on its monitor".into(),
            ));
        }

        Ok(CaptureBlitRegion {
            src_x,
            src_y,
            width: right - src_x,
            height: bottom - src_y,
            dst_x: 0,
            dst_y: 0,
        })
    }

    fn resolve_window_blit(
        &mut self,
        hwnd: HWND,
        win_rect: &RECT,
    ) -> CaptureResult<CaptureBlitRegion> {
        if !rect_within_rect(win_rect, &self.current_monitor_rect) {
            let hmon = monitor_from_window(hwnd).ok_or_else(|| {
                CaptureError::BackendUnavailable("window is not on any monitor".into())
            })?;
            if SendHmon(hmon) != self.current_hmon {
                self.reinit_for_monitor(hmon)?;
            }
        }

        let blit = match Self::window_blit_on_monitor(&self.current_monitor_rect, win_rect) {
            Ok(blit) => blit,
            Err(error) => {
                self.current_monitor_rect = monitor_rect(self.current_hmon.0)?;
                match Self::window_blit_on_monitor(&self.current_monitor_rect, win_rect) {
                    Ok(blit) => blit,
                    Err(_) => return Err(error),
                }
            }
        };

        Ok(blit)
    }

    fn capture_internal(
        &mut self,
        reuse: Option<Frame>,
        destination_history_hint: Option<bool>,
    ) -> CaptureResult<Frame> {
        let hwnd = self.hwnd.0;

        if !unsafe { IsWindow(hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window no longer exists".into(),
            ));
        }
        if unsafe { IsIconic(hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget("window is minimized".into()));
        }

        let mut win_rect = RECT::default();
        unsafe { GetWindowRect(hwnd, &mut win_rect) }
            .ok()
            .context("GetWindowRect failed")
            .map_err(CaptureError::Platform)?;

        let win_w = win_rect.right - win_rect.left;
        let win_h = win_rect.bottom - win_rect.top;
        if win_w <= 0 || win_h <= 0 {
            return Err(CaptureError::InvalidTarget(
                "window has empty bounds".into(),
            ));
        }

        let blit = self.resolve_window_blit(hwnd, &win_rect)?;

        let mut frame = reuse.unwrap_or_else(Frame::empty);
        let has_pixels = !frame.as_rgba_bytes().is_empty();
        let history_enabled = destination_history_hint.unwrap_or(has_pixels) && has_pixels;
        let destination_has_history =
            history_enabled && frame.width() == blit.width && frame.height() == blit.height;
        frame.ensure_rgba_capacity(blit.width, blit.height)?;
        frame.reset_metadata();

        let sample =
            match self
                .output
                .capture_region_into(blit, &mut frame, destination_has_history, true)
            {
                Ok(sample) => sample,
                Err(CaptureError::MonitorLost) | Err(CaptureError::AccessLost) => {
                    let retry_hmon = monitor_from_window(hwnd).ok_or_else(|| {
                        CaptureError::BackendUnavailable("window is not on any monitor".into())
                    })?;
                    self.reinit_for_monitor(retry_hmon)?;
                    win_rect = RECT::default();
                    unsafe { GetWindowRect(hwnd, &mut win_rect) }
                        .ok()
                        .context("GetWindowRect failed during DXGI window capture recovery")
                        .map_err(CaptureError::Platform)?;
                    let retry_blit = self.resolve_window_blit(hwnd, &win_rect)?;
                    let retry_has_history = history_enabled
                        && frame.width() == retry_blit.width
                        && frame.height() == retry_blit.height
                        && !frame.as_rgba_bytes().is_empty();
                    frame.ensure_rgba_capacity(retry_blit.width, retry_blit.height)?;
                    self.output.capture_region_into(
                        retry_blit,
                        &mut frame,
                        retry_has_history,
                        true,
                    )?
                }
                Err(e) => return Err(e),
            };

        frame.metadata.capture_time = sample.capture_time;
        frame.metadata.present_time_qpc = sample.present_time_qpc;
        frame.metadata.is_duplicate = sample.is_duplicate;
        Ok(frame)
    }
}

impl crate::backend::MonitorCapturer for WindowsDxgiWindowCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        self.capture_internal(reuse, None)
    }

    fn capture_with_history_hint(
        &mut self,
        reuse: Option<Frame>,
        destination_has_history: bool,
    ) -> CaptureResult<Frame> {
        self.capture_internal(reuse, Some(destination_has_history))
    }

    fn set_capture_mode(&mut self, mode: CaptureMode) {
        self.capture_mode = mode;
        self.output.set_capture_mode(mode);
    }

    fn set_cursor_config(&mut self, config: CursorCaptureConfig) {
        self.cursor_config = config;
        self.output.cursor_config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_submit_copy_requires_fastpath_and_pending_state() {
        assert!(!should_skip_screenrecord_submit_copy(
            false, true, true, false
        ));
        assert!(!should_skip_screenrecord_submit_copy(
            true, false, true, false
        ));
    }

    #[test]
    fn skip_submit_copy_triggers_for_duplicate_frames() {
        assert!(should_skip_screenrecord_submit_copy(
            true, true, true, false
        ));
    }

    #[test]
    fn skip_submit_copy_triggers_for_empty_dirty_updates() {
        assert!(should_skip_screenrecord_submit_copy(
            true, true, false, true
        ));
    }

    #[test]
    fn dirty_copy_heuristic_accepts_small_sparse_updates() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 120,
                height: 80,
            },
            DirtyRect {
                x: 600,
                y: 420,
                width: 96,
                height: 64,
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
            height: 900,
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
            DXGI_DIRTY_COPY_MAX_RECTS + 1
        ];
        assert!(!should_use_dirty_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_gpu_copy_heuristic_accepts_sparse_updates() {
        let rects = vec![
            DirtyRect {
                x: 64,
                y: 64,
                width: 128,
                height: 96,
            },
            DirtyRect {
                x: 640,
                y: 400,
                width: 96,
                height: 72,
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
            height: 1000,
        }];
        assert!(!should_use_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn dirty_gpu_copy_heuristic_rejects_excessive_rect_count() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            };
            DXGI_DIRTY_GPU_COPY_MAX_RECTS + 1
        ];
        assert!(!should_use_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn low_latency_dirty_gpu_copy_heuristic_is_stricter() {
        let rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 64,
                height: 64,
            };
            DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS + 1
        ];
        assert!(!should_use_low_latency_dirty_gpu_copy(&rects, 1920, 1080));
    }

    #[test]
    fn monitor_low_latency_dirty_gpu_mode_requires_recording_and_size_match() {
        assert!(!choose_monitor_low_latency_dirty_gpu_mode(
            false, true, false, false, true, false, true, true
        ));
        assert!(!choose_monitor_low_latency_dirty_gpu_mode(
            true, true, false, false, false, false, true, true
        ));
    }

    #[test]
    fn monitor_low_latency_dirty_gpu_mode_sticks_for_duplicate_frames_when_active() {
        assert!(choose_monitor_low_latency_dirty_gpu_mode(
            true, true, false, true, true, true, false, false
        ));
        assert!(!choose_monitor_low_latency_dirty_gpu_mode(
            true, true, false, false, true, true, false, false
        ));
    }

    #[test]
    fn monitor_low_latency_dirty_gpu_mode_force_flag_uses_regular_gpu_threshold() {
        assert!(choose_monitor_low_latency_dirty_gpu_mode(
            true, true, true, false, true, false, true, false
        ));
        assert!(!choose_monitor_low_latency_dirty_gpu_mode(
            true, true, true, false, true, false, false, true
        ));
    }

    fn test_blit(width: u32, height: u32) -> CaptureBlitRegion {
        CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width,
            height,
            dst_x: 0,
            dst_y: 0,
        }
    }

    #[test]
    fn window_low_latency_mode_prefers_small_regions() {
        assert!(choose_window_low_latency_region_mode(
            true,
            true,
            true,
            false,
            1_048_576,
            test_blit(1280, 720),
            false
        ));
    }

    #[test]
    fn window_low_latency_mode_requires_recording_preference_and_enablement() {
        assert!(!choose_window_low_latency_region_mode(
            false,
            true,
            true,
            true,
            DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT,
            test_blit(1280, 720),
            true
        ));
        assert!(!choose_window_low_latency_region_mode(
            true,
            false,
            true,
            true,
            DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT,
            test_blit(1280, 720),
            true
        ));
        assert!(!choose_window_low_latency_region_mode(
            true,
            true,
            false,
            true,
            DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT,
            test_blit(1280, 720),
            true
        ));
    }

    #[test]
    fn window_low_latency_mode_uses_pipeline_for_large_regions_without_sparse_damage() {
        assert!(!choose_window_low_latency_region_mode(
            true,
            true,
            true,
            false,
            1_048_576,
            test_blit(1920, 1080),
            false
        ));
    }

    #[test]
    fn window_low_latency_mode_rejects_zero_sized_regions_without_force() {
        assert!(!choose_window_low_latency_region_mode(
            true,
            true,
            true,
            false,
            DXGI_WINDOW_LOW_LATENCY_MAX_PIXELS_DEFAULT,
            test_blit(0, 720),
            false
        ));
    }

    #[test]
    fn window_low_latency_mode_keeps_low_latency_for_sparse_large_updates() {
        assert!(choose_window_low_latency_region_mode(
            true,
            true,
            true,
            false,
            1_048_576,
            test_blit(1920, 1080),
            true
        ));
    }

    #[test]
    fn window_low_latency_mode_force_flag_overrides_thresholds() {
        assert!(choose_window_low_latency_region_mode(
            true,
            true,
            true,
            true,
            1,
            test_blit(3840, 2160),
            false
        ));
    }

    #[test]
    fn normalize_dirty_rects_merges_touching_spans() {
        let mut rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
            DirtyRect {
                x: 10,
                y: 0,
                width: 5,
                height: 10,
            },
            DirtyRect {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
            },
            DirtyRect {
                x: 0,
                y: 10,
                width: 15,
                height: 5,
            },
        ];
        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);
        assert_eq!(
            rects,
            vec![DirtyRect {
                x: 0,
                y: 0,
                width: 15,
                height: 15,
            }]
        );
    }

    #[test]
    fn normalize_dirty_rects_clamps_and_filters_invalid_input() {
        let mut rects = vec![
            DirtyRect {
                x: 1900,
                y: 1000,
                width: 100,
                height: 100,
            },
            DirtyRect {
                x: 2500,
                y: 100,
                width: 40,
                height: 40,
            },
            DirtyRect {
                x: 100,
                y: 100,
                width: 0,
                height: 12,
            },
            DirtyRect {
                x: 4,
                y: 5,
                width: 10,
                height: 10,
            },
        ];
        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);
        assert_eq!(
            rects,
            vec![
                DirtyRect {
                    x: 4,
                    y: 5,
                    width: 10,
                    height: 10,
                },
                DirtyRect {
                    x: 1900,
                    y: 1000,
                    width: 20,
                    height: 80,
                },
            ]
        );
    }

    #[test]
    fn normalize_dirty_rects_does_not_merge_corner_only_contact() {
        let mut rects = vec![
            DirtyRect {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
            DirtyRect {
                x: 10,
                y: 10,
                width: 10,
                height: 10,
            },
        ];
        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);
        assert_eq!(
            rects,
            vec![
                DirtyRect {
                    x: 0,
                    y: 0,
                    width: 10,
                    height: 10,
                },
                DirtyRect {
                    x: 10,
                    y: 10,
                    width: 10,
                    height: 10,
                },
            ]
        );
    }

    #[test]
    fn normalize_dirty_rects_merges_transitive_neighbors() {
        let mut rects = vec![
            DirtyRect {
                x: 40,
                y: 0,
                width: 10,
                height: 10,
            },
            DirtyRect {
                x: 0,
                y: 5,
                width: 60,
                height: 10,
            },
            DirtyRect {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
        ];
        normalize_dirty_rects_in_place(&mut rects, 1920, 1080);
        assert_eq!(
            rects,
            vec![DirtyRect {
                x: 0,
                y: 0,
                width: 60,
                height: 15,
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
    fn cursor_shape_bgra_to_rgba_converts_contiguous_rows() {
        let shape = DXGI_OUTDUPL_POINTER_SHAPE_INFO {
            Width: 2,
            Height: 1,
            Pitch: 8,
            ..Default::default()
        };
        let src = vec![
            0x10, 0x20, 0x30, 0x40, // BGRA
            0x50, 0x60, 0x70, 0x80, // BGRA
        ];
        let rgba = convert_cursor_shape_bgra_to_rgba(&src, &shape);
        assert_eq!(
            rgba,
            vec![
                0x30, 0x20, 0x10, 0x40, //
                0x70, 0x60, 0x50, 0x80
            ]
        );
    }

    #[test]
    fn cursor_shape_bgra_to_rgba_converts_with_row_padding() {
        let shape = DXGI_OUTDUPL_POINTER_SHAPE_INFO {
            Width: 2,
            Height: 2,
            Pitch: 12,
            ..Default::default()
        };
        let src = vec![
            0x10, 0x20, 0x30, 0xFF, 0x40, 0x50, 0x60, 0xEE, 0, 0, 0, 0, // row 0
            0x70, 0x80, 0x90, 0xDD, 0xA0, 0xB0, 0xC0, 0xCC, 0, 0, 0, 0, // row 1
        ];
        let rgba = convert_cursor_shape_bgra_to_rgba(&src, &shape);
        assert_eq!(
            rgba,
            vec![
                0x30, 0x20, 0x10, 0xFF, 0x60, 0x50, 0x40, 0xEE, //
                0x90, 0x80, 0x70, 0xDD, 0xC0, 0xB0, 0xA0, 0xCC
            ]
        );
    }

    #[test]
    fn cursor_shape_bgra_to_rgba_handles_truncated_buffer() {
        let shape = DXGI_OUTDUPL_POINTER_SHAPE_INFO {
            Width: 2,
            Height: 1,
            Pitch: 8,
            ..Default::default()
        };
        let src = vec![
            0x10, 0x20, 0x30, 0xAA, // full pixel
            0x40, 0x50, // partial pixel (ignored)
        ];
        let rgba = convert_cursor_shape_bgra_to_rgba(&src, &shape);
        assert_eq!(rgba, vec![0x30, 0x20, 0x10, 0xAA, 0, 0, 0, 0]);
    }

    #[test]
    fn cursor_shape_bgra_to_rgba_handles_pitch_smaller_than_row() {
        let shape = DXGI_OUTDUPL_POINTER_SHAPE_INFO {
            Width: 2,
            Height: 2,
            Pitch: 4,
            ..Default::default()
        };
        let src = vec![
            0x10, 0x20, 0x30, 0xAA, // row 0: one pixel available
            0x40, 0x50, 0x60, 0xBB, // row 1: one pixel available
        ];
        let rgba = convert_cursor_shape_bgra_to_rgba(&src, &shape);
        assert_eq!(
            rgba,
            vec![
                0x30, 0x20, 0x10, 0xAA, 0, 0, 0, 0, //
                0x60, 0x50, 0x40, 0xBB, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn move_rect_clamp_trims_source_and_destination_bounds() {
        let raw = DXGI_OUTDUPL_MOVE_RECT {
            SourcePoint: windows::Win32::Foundation::POINT { x: 50, y: 10 },
            DestinationRect: RECT {
                left: -10,
                top: 5,
                right: 30,
                bottom: 25,
            },
        };
        let clamped = clamp_move_rect_to_surface(&raw, 100, 80).expect("expected clamped rect");
        assert_eq!(
            clamped,
            MoveRect {
                src_x: 60,
                src_y: 10,
                dst_x: 0,
                dst_y: 5,
                width: 30,
                height: 20,
            }
        );

        let raw_overflow = DXGI_OUTDUPL_MOVE_RECT {
            SourcePoint: windows::Win32::Foundation::POINT { x: 90, y: 40 },
            DestinationRect: RECT {
                left: 70,
                top: 40,
                right: 100,
                bottom: 60,
            },
        };
        let clipped =
            clamp_move_rect_to_surface(&raw_overflow, 100, 80).expect("expected clipped rect");
        assert_eq!(
            clipped,
            MoveRect {
                src_x: 90,
                src_y: 40,
                dst_x: 70,
                dst_y: 40,
                width: 10,
                height: 20,
            }
        );
    }

    #[test]
    fn region_move_rects_clip_and_rebase_require_shared_overlap() {
        let bounds = RegionDirtyBounds::from_source_and_blit(
            200,
            120,
            CaptureBlitRegion {
                src_x: 40,
                src_y: 20,
                width: 80,
                height: 60,
                dst_x: 0,
                dst_y: 0,
            },
        )
        .expect("expected valid region bounds");
        let source_moves = vec![
            MoveRect {
                src_x: 30,
                src_y: 30,
                dst_x: 70,
                dst_y: 40,
                width: 40,
                height: 20,
            },
            MoveRect {
                src_x: 0,
                src_y: 0,
                dst_x: 150,
                dst_y: 10,
                width: 20,
                height: 20,
            },
        ];
        let mut out = Vec::new();
        extract_region_move_rects_from_move_slice(&source_moves, bounds, &mut out);
        assert_eq!(
            out,
            vec![MoveRect {
                src_x: 0,
                src_y: 10,
                dst_x: 40,
                dst_y: 20,
                width: 30,
                height: 20,
            }]
        );
    }

    #[test]
    fn region_dirty_reconstruct_requires_move_metadata_for_incremental_copy() {
        assert!(can_use_region_dirty_reconstruct(true, false, true));
        assert!(can_use_region_dirty_reconstruct(true, false, false));
        assert!(can_use_region_dirty_reconstruct(true, true, true));
        assert!(!can_use_region_dirty_reconstruct(true, true, false));
        assert!(!can_use_region_dirty_reconstruct(false, false, true));
        assert!(!can_use_region_dirty_reconstruct(false, true, true));
    }

    #[test]
    fn apply_move_rects_handles_downward_overlap() {
        let width = 8u32;
        let height = 5u32;
        let mut frame = Frame::empty();
        frame
            .ensure_rgba_capacity(width, height)
            .expect("failed to allocate frame buffer");

        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel = y * width as usize + x;
                let idx = pixel * 4;
                frame.as_mut_rgba_bytes()[idx] = pixel as u8;
                frame.as_mut_rgba_bytes()[idx + 1] = (pixel.wrapping_mul(3)) as u8;
                frame.as_mut_rgba_bytes()[idx + 2] = (pixel.wrapping_mul(7)) as u8;
                frame.as_mut_rgba_bytes()[idx + 3] = 0xFF;
            }
        }

        let before = frame.as_rgba_bytes().to_vec();
        let move_rects = [MoveRect {
            src_x: 1,
            src_y: 1,
            dst_x: 1,
            dst_y: 2,
            width: 4,
            height: 2,
        }];
        apply_move_rects_to_frame(&mut frame, &move_rects, 0, 0).expect("move apply failed");

        let mut expected = before.clone();
        let pitch = width as usize * 4;
        for row in 0..move_rects[0].height as usize {
            let src_offset =
                (move_rects[0].src_y as usize + row) * pitch + move_rects[0].src_x as usize * 4;
            let dst_offset =
                (move_rects[0].dst_y as usize + row) * pitch + move_rects[0].dst_x as usize * 4;
            let row_bytes = move_rects[0].width as usize * 4;
            expected[dst_offset..dst_offset + row_bytes]
                .copy_from_slice(&before[src_offset..src_offset + row_bytes]);
        }

        assert_eq!(frame.as_rgba_bytes(), expected.as_slice());
    }

    #[test]
    fn direct_region_dirty_extract_matches_legacy_clipping() {
        let source_width = 2560u32;
        let source_height = 1440u32;
        let workloads = make_direct_dirty_extract_inputs();
        let blits = direct_dirty_extract_blits();
        let mut legacy = Vec::new();
        let mut optimized = Vec::new();

        for (idx, raw_rects) in workloads.iter().enumerate() {
            let blit = blits[idx % blits.len()];
            assert!(extract_region_dirty_rects_from_raw_slice_legacy(
                raw_rects,
                source_width,
                source_height,
                blit,
                &mut legacy,
            ));
            let bounds = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
                .expect("expected valid blit bounds");
            extract_region_dirty_rects_from_raw_slice(raw_rects, bounds, &mut optimized);
            assert_eq!(optimized, legacy);
        }
    }

    #[test]
    fn direct_region_dirty_extract_handles_extreme_raw_coordinates() {
        let source_width = 2560u32;
        let source_height = 1440u32;
        let blit = CaptureBlitRegion {
            src_x: 0,
            src_y: 0,
            width: source_width,
            height: source_height,
            dst_x: 0,
            dst_y: 0,
        };
        let raw_rects = vec![RECT {
            left: i32::MIN,
            top: i32::MIN,
            right: i32::MAX,
            bottom: i32::MAX,
        }];

        let mut legacy = Vec::new();
        assert!(extract_region_dirty_rects_from_raw_slice_legacy(
            &raw_rects,
            source_width,
            source_height,
            blit,
            &mut legacy,
        ));

        let bounds = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
            .expect("expected valid blit bounds");
        let mut optimized = Vec::new();
        extract_region_dirty_rects_from_raw_slice(&raw_rects, bounds, &mut optimized);

        assert_eq!(optimized, legacy);
        assert_eq!(
            optimized,
            vec![DirtyRect {
                x: 0,
                y: 0,
                width: source_width,
                height: source_height,
            }]
        );
    }

    #[test]
    fn rect_within_rect_detects_containment() {
        let inner = RECT {
            left: 100,
            top: 100,
            right: 200,
            bottom: 180,
        };
        let outer = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1080,
        };
        assert!(rect_within_rect(&inner, &outer));
    }

    #[test]
    fn rect_within_rect_detects_out_of_bounds() {
        let inner = RECT {
            left: -10,
            top: 0,
            right: 300,
            bottom: 200,
        };
        let outer = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1080,
        };
        assert!(!rect_within_rect(&inner, &outer));
    }

    fn row_major_rects(
        start_x: u32,
        start_y: u32,
        cols: u32,
        rows: u32,
        rect_w: u32,
        rect_h: u32,
        gap_x: u32,
        gap_y: u32,
        limit: usize,
    ) -> Vec<DirtyRect> {
        let mut out = Vec::with_capacity(limit);
        for row in 0..rows {
            for col in 0..cols {
                if out.len() == limit {
                    return out;
                }
                let x = start_x + col * (rect_w + gap_x);
                let y = start_y + row * (rect_h + gap_y);
                out.push(DirtyRect {
                    x,
                    y,
                    width: rect_w,
                    height: rect_h,
                });
            }
        }
        out
    }

    fn make_dirty_strategy_workloads() -> Vec<Vec<DirtyRect>> {
        vec![
            row_major_rects(24, 18, 8, 4, 32, 28, 24, 18, 32),
            row_major_rects(8, 8, 16, 6, 10, 8, 8, 6, 96),
            row_major_rects(32, 24, 2, 10, 640, 18, 96, 10, 20),
            row_major_rects(6, 6, 30, 6, 8, 6, 4, 3, 180),
        ]
    }

    fn make_normalize_bench_inputs() -> Vec<Vec<DirtyRect>> {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut out = Vec::with_capacity(24);
        for _ in 0..24 {
            let mut rects = Vec::with_capacity(200);
            let rect_count = 96 + ((state >> 5) as usize % 96);
            for _ in 0..rect_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = ((state >> 12) as u32) % 2600;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = ((state >> 20) as u32) % 1500;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let width = 1 + ((state >> 24) as u32) % 900;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let height = 1 + ((state >> 28) as u32) % 500;
                rects.push(DirtyRect {
                    x,
                    y,
                    width,
                    height,
                });
            }
            out.push(rects);
        }
        out
    }

    fn make_direct_dirty_extract_inputs() -> Vec<Vec<RECT>> {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut workloads = Vec::with_capacity(20);
        for _ in 0..20 {
            let mut rects = Vec::with_capacity(260);
            let rect_count = 80 + ((state >> 5) as usize % 180);
            for _ in 0..rect_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let left = ((state >> 18) as i32 & 0x1fff) - 1024;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let top = ((state >> 22) as i32 & 0x0fff) - 512;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let width = ((state >> 24) as i32 & 0x03ff) - 64;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let height = ((state >> 28) as i32 & 0x01ff) - 32;
                let right = left.saturating_add(width);
                let bottom = top.saturating_add(height);
                rects.push(RECT {
                    left,
                    top,
                    right,
                    bottom,
                });
            }
            workloads.push(rects);
        }
        workloads
    }

    fn direct_dirty_extract_blits() -> [CaptureBlitRegion; 5] {
        [
            CaptureBlitRegion {
                src_x: 0,
                src_y: 0,
                width: 2560,
                height: 1440,
                dst_x: 0,
                dst_y: 0,
            },
            CaptureBlitRegion {
                src_x: 120,
                src_y: 80,
                width: 1920,
                height: 1080,
                dst_x: 0,
                dst_y: 0,
            },
            CaptureBlitRegion {
                src_x: 320,
                src_y: 160,
                width: 1280,
                height: 720,
                dst_x: 0,
                dst_y: 0,
            },
            CaptureBlitRegion {
                src_x: 960,
                src_y: 540,
                width: 800,
                height: 450,
                dst_x: 0,
                dst_y: 0,
            },
            CaptureBlitRegion {
                src_x: 100,
                src_y: 200,
                width: 640,
                height: 640,
                dst_x: 0,
                dst_y: 0,
            },
        ]
    }

    fn extract_region_dirty_rects_from_raw_slice_legacy(
        raw_rects: &[RECT],
        source_width: u32,
        source_height: u32,
        blit: CaptureBlitRegion,
        out: &mut Vec<DirtyRect>,
    ) -> bool {
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
            return false;
        };
        let region_right = region_bounds.x.saturating_add(region_bounds.width);
        let region_bottom = region_bounds.y.saturating_add(region_bounds.height);

        for raw_rect in raw_rects {
            if out.len() == DXGI_REGION_DIRTY_TRACK_MAX_RECTS {
                break;
            }

            let width = (i64::from(raw_rect.right) - i64::from(raw_rect.left)).max(0) as u32;
            let height = (i64::from(raw_rect.bottom) - i64::from(raw_rect.top)).max(0) as u32;
            if width == 0 || height == 0 {
                continue;
            }

            let dirty = DirtyRect {
                x: raw_rect.left.max(0) as u32,
                y: raw_rect.top.max(0) as u32,
                width,
                height,
            };
            let Some(clamped) = dirty_rect::clamp_dirty_rect(dirty, source_width, source_height) else {
                continue;
            };

            let clamped_right = clamped.x.saturating_add(clamped.width);
            let clamped_bottom = clamped.y.saturating_add(clamped.height);
            let x = clamped.x.max(region_bounds.x);
            let y = clamped.y.max(region_bounds.y);
            let right = clamped_right.min(region_right);
            let bottom = clamped_bottom.min(region_bottom);
            if right <= x || bottom <= y {
                continue;
            }

            out.push(DirtyRect {
                x: x.saturating_sub(region_bounds.x),
                y: y.saturating_sub(region_bounds.y),
                width: right.saturating_sub(x),
                height: bottom.saturating_sub(y),
            });
        }

        true
    }

    fn should_use_dirty_copy_legacy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
        if rects.is_empty() || rects.len() > DXGI_DIRTY_COPY_MAX_RECTS {
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
                > total_pixels.saturating_mul(DXGI_DIRTY_COPY_MAX_AREA_PERCENT)
            {
                return false;
            }
        }

        true
    }

    fn should_use_dirty_gpu_copy_legacy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
        if rects.is_empty() || rects.len() > DXGI_DIRTY_GPU_COPY_MAX_RECTS {
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
                > total_pixels.saturating_mul(DXGI_DIRTY_GPU_COPY_MAX_AREA_PERCENT)
            {
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
        if rects.is_empty() || rects.len() > DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_RECTS {
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
                > total_pixels.saturating_mul(DXGI_DIRTY_GPU_COPY_LOW_LATENCY_MAX_AREA_PERCENT)
            {
                return false;
            }
        }

        true
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_dirty_copy_strategy_single_pass_vs_legacy() {
        use std::hint::black_box;

        let workloads = make_dirty_strategy_workloads();
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
        let measure_iters = 120_000usize;

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
        }
        let optimized_elapsed = optimized_start.elapsed();
        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            0.0
        };

        println!(
            "dxgi dirty strategy benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.03,
            "single-pass dirty strategy regressed: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_normalize_dirty_rects_merge_candidate_vs_reference() {
        use std::hint::black_box;

        let inputs = make_normalize_bench_inputs();
        for input in &inputs {
            let mut optimized = input.clone();
            let mut reference = input.clone();
            normalize_dirty_rects_in_place(&mut optimized, 1920, 1080);
            normalize_dirty_rects_reference_in_place(&mut reference, 1920, 1080);
            assert_eq!(optimized, reference);
        }

        let warmup_iters = 1_000usize;
        let measure_iters = 8_000usize;
        let mut working = Vec::new();
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            working.clear();
            working.extend_from_slice(&inputs[iter % inputs.len()]);
            normalize_dirty_rects_reference_in_place(&mut working, 1920, 1080);
            sink ^= working.len();

            working.clear();
            working.extend_from_slice(&inputs[iter % inputs.len()]);
            normalize_dirty_rects_in_place(&mut working, 1920, 1080);
            sink ^= working.len() << 1;
        }

        let reference_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            working.clear();
            working.extend_from_slice(&inputs[iter % inputs.len()]);
            normalize_dirty_rects_reference_in_place(&mut working, 1920, 1080);
            sink ^= working.len();
        }
        let reference_elapsed = reference_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            working.clear();
            working.extend_from_slice(&inputs[iter % inputs.len()]);
            normalize_dirty_rects_in_place(&mut working, 1920, 1080);
            sink ^= working.len() << 1;
        }
        let optimized_elapsed = optimized_start.elapsed();
        black_box(sink);

        let reference_ms = reference_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            reference_ms / optimized_ms
        } else {
            0.0
        };

        println!(
            "dxgi normalize benchmark: reference={reference_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= reference_ms * 1.03,
            "merge-candidate normalize path regressed: reference={reference_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_direct_region_dirty_extract_clip_vs_legacy() {
        use std::hint::black_box;

        let source_width = 2560u32;
        let source_height = 1440u32;
        let workloads = make_direct_dirty_extract_inputs();
        let blits = direct_dirty_extract_blits();

        let mut legacy_output = Vec::with_capacity(DXGI_REGION_DIRTY_TRACK_MAX_RECTS);
        let mut optimized_output = Vec::with_capacity(DXGI_REGION_DIRTY_TRACK_MAX_RECTS);
        for (idx, raw_rects) in workloads.iter().enumerate() {
            let blit = blits[idx % blits.len()];
            assert!(extract_region_dirty_rects_from_raw_slice_legacy(
                raw_rects,
                source_width,
                source_height,
                blit,
                &mut legacy_output,
            ));
            let bounds = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
                .expect("expected valid blit bounds");
            extract_region_dirty_rects_from_raw_slice(raw_rects, bounds, &mut optimized_output);
            assert_eq!(optimized_output, legacy_output);
        }

        let warmup_iters = 8_000usize;
        let measure_iters = 180_000usize;
        let mut sink = 0usize;

        for iter in 0..warmup_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let blit = black_box(blits[iter % blits.len()]);

            extract_region_dirty_rects_from_raw_slice_legacy(
                raw_rects,
                source_width,
                source_height,
                blit,
                &mut legacy_output,
            );
            sink ^= legacy_output.len();

            let bounds = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
                .expect("expected valid blit bounds");
            extract_region_dirty_rects_from_raw_slice(raw_rects, bounds, &mut optimized_output);
            sink ^= optimized_output.len() << 1;
        }

        let legacy_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let blit = black_box(blits[iter % blits.len()]);
            extract_region_dirty_rects_from_raw_slice_legacy(
                raw_rects,
                source_width,
                source_height,
                blit,
                &mut legacy_output,
            );
            sink ^= legacy_output.len();
        }
        let legacy_elapsed = legacy_start.elapsed();

        let optimized_start = std::time::Instant::now();
        for iter in 0..measure_iters {
            let raw_rects = black_box(&workloads[iter % workloads.len()]);
            let blit = black_box(blits[iter % blits.len()]);
            let bounds = RegionDirtyBounds::from_source_and_blit(source_width, source_height, blit)
                .expect("expected valid blit bounds");
            extract_region_dirty_rects_from_raw_slice(raw_rects, bounds, &mut optimized_output);
            sink ^= optimized_output.len() << 1;
        }
        let optimized_elapsed = optimized_start.elapsed();
        black_box(sink);

        let legacy_ms = legacy_elapsed.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_elapsed.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            0.0
        };

        println!(
            "dxgi direct dirty extract benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.03,
            "direct dirty extract path regressed: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_monitor_dirty_gpu_copy_emulation_vs_full_copy() {
        use std::hint::black_box;
        use std::time::Duration;

        const WIDTH: usize = 2560;
        const HEIGHT: usize = 1440;
        const WARMUP_ITERS: usize = 48;
        const MEASURE_ITERS: usize = 280;

        let pitch = WIDTH * 4;
        let buffer_len = pitch * HEIGHT;
        let dirty_rects = [
            DirtyRect {
                x: 0,
                y: 0,
                width: WIDTH as u32,
                height: 24,
            },
            DirtyRect {
                x: 0,
                y: (HEIGHT - 24) as u32,
                width: WIDTH as u32,
                height: 24,
            },
            DirtyRect {
                x: 0,
                y: 32,
                width: 24,
                height: (HEIGHT - 64) as u32,
            },
            DirtyRect {
                x: (WIDTH - 24) as u32,
                y: 32,
                width: 24,
                height: (HEIGHT - 64) as u32,
            },
            DirtyRect {
                x: 960,
                y: 620,
                width: 640,
                height: 180,
            },
        ];

        let converter = crate::convert::SurfaceRowConverter::new(
            crate::convert::SurfacePixelFormat::Bgra8,
            crate::convert::SurfaceConversionOptions::default(),
        );

        let mut previous_rgba = vec![0u8; buffer_len];
        for (idx, byte) in previous_rgba.iter_mut().enumerate() {
            *byte = (idx as u8).wrapping_mul(17).wrapping_add(91);
        }

        let mut source_rgba = vec![0u8; buffer_len];
        let mut source_bgra = vec![0u8; buffer_len];
        let mut staging_legacy = vec![0u8; buffer_len];
        let mut staging_optimized = vec![0u8; buffer_len];
        let mut output_legacy = previous_rgba.clone();
        let mut output_optimized = previous_rgba.clone();

        let seed_to_bgra = |src_rgba: &[u8], dst_bgra: &mut [u8]| {
            dst_bgra.copy_from_slice(src_rgba);
            for pixel in dst_bgra.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        };
        seed_to_bgra(&previous_rgba, &mut staging_legacy);
        staging_optimized.copy_from_slice(&staging_legacy);

        let mut legacy_total = Duration::ZERO;
        let mut optimized_total = Duration::ZERO;
        let total_iters = WARMUP_ITERS + MEASURE_ITERS;
        let mut sink = 0u64;

        for iter in 0..total_iters {
            source_rgba.copy_from_slice(output_legacy.as_slice());
            for rect in &dirty_rects {
                let x = rect.x as usize;
                let y = rect.y as usize;
                let width = rect.width as usize;
                let height = rect.height as usize;
                for row in 0..height {
                    let row_offset = (y + row) * pitch + x * 4;
                    for col in 0..width {
                        let idx = row_offset + col * 4;
                        source_rgba[idx] =
                            (iter as u8).wrapping_mul(19).wrapping_add((idx / 4) as u8);
                        source_rgba[idx + 1] =
                            (iter as u8).wrapping_mul(11).wrapping_add((idx / 8) as u8);
                        source_rgba[idx + 2] =
                            (iter as u8).wrapping_mul(7).wrapping_add((idx / 16) as u8);
                        source_rgba[idx + 3] = 0xFF;
                    }
                }
            }
            seed_to_bgra(&source_rgba, &mut source_bgra);

            let legacy_start = std::time::Instant::now();
            staging_legacy.copy_from_slice(&source_bgra);
            unsafe {
                let src_ptr = staging_legacy.as_ptr();
                let dst_ptr = output_legacy.as_mut_ptr();
                for rect in &dirty_rects {
                    let x = rect.x as usize;
                    let y = rect.y as usize;
                    let width = rect.width as usize;
                    let height = rect.height as usize;
                    let src_offset = y * pitch + x * 4;
                    let dst_offset = src_offset;
                    converter.convert_rows_unchecked(
                        src_ptr.add(src_offset),
                        pitch,
                        dst_ptr.add(dst_offset),
                        pitch,
                        width,
                        height,
                    );
                }
            }
            let legacy_elapsed = legacy_start.elapsed();

            let optimized_start = std::time::Instant::now();
            unsafe {
                let src_ptr = source_bgra.as_ptr();
                let staging_ptr = staging_optimized.as_mut_ptr();
                for rect in &dirty_rects {
                    let x = rect.x as usize;
                    let y = rect.y as usize;
                    let width = rect.width as usize;
                    let height = rect.height as usize;
                    let row_bytes = width * 4;
                    for row in 0..height {
                        let offset = (y + row) * pitch + x * 4;
                        std::ptr::copy_nonoverlapping(
                            src_ptr.add(offset),
                            staging_ptr.add(offset),
                            row_bytes,
                        );
                    }
                }

                let stage_ptr = staging_optimized.as_ptr();
                let dst_ptr = output_optimized.as_mut_ptr();
                for rect in &dirty_rects {
                    let x = rect.x as usize;
                    let y = rect.y as usize;
                    let width = rect.width as usize;
                    let height = rect.height as usize;
                    let src_offset = y * pitch + x * 4;
                    let dst_offset = src_offset;
                    converter.convert_rows_unchecked(
                        stage_ptr.add(src_offset),
                        pitch,
                        dst_ptr.add(dst_offset),
                        pitch,
                        width,
                        height,
                    );
                }
            }
            let optimized_elapsed = optimized_start.elapsed();

            if iter >= WARMUP_ITERS {
                legacy_total += legacy_elapsed;
                optimized_total += optimized_elapsed;
            }

            assert_eq!(
                output_optimized, output_legacy,
                "monitor dirty-gpu emulation mismatch at iteration {iter}"
            );
            assert_eq!(
                staging_optimized, staging_legacy,
                "staging dirty-gpu emulation mismatch at iteration {iter}"
            );

            sink ^= output_legacy[(iter * 131) % output_legacy.len()] as u64;
            sink ^= staging_optimized[(iter * 89) % staging_optimized.len()] as u64;
        }
        black_box(sink);

        let legacy_ms = legacy_total.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_total.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            0.0
        };

        println!(
            "dxgi monitor dirty-gpu benchmark: full-copy={legacy_ms:.3} ms dirty-copy={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.05,
            "monitor dirty-gpu path regressed beyond tolerance: full-copy={legacy_ms:.3}ms dirty-copy={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }

    #[test]
    #[ignore = "performance benchmark guard; run explicitly with --ignored --nocapture"]
    fn bench_region_move_apply_vs_full_convert() {
        use std::hint::black_box;
        use std::time::Duration;

        const WIDTH: usize = 1920;
        const HEIGHT: usize = 1080;
        const WARMUP_ITERS: usize = 80;
        const MEASURE_ITERS: usize = 420;

        let pitch = WIDTH * 4;
        let buffer_len = pitch * HEIGHT;
        let move_rects = [MoveRect {
            src_x: 0,
            src_y: 0,
            dst_x: 0,
            dst_y: 1,
            width: WIDTH as u32,
            height: (HEIGHT - 1) as u32,
        }];
        let dirty_rects = [DirtyRect {
            x: 0,
            y: 0,
            width: WIDTH as u32,
            height: 1,
        }];

        let mut seed_rgba = vec![0u8; buffer_len];
        for (idx, byte) in seed_rgba.iter_mut().enumerate() {
            *byte = (idx as u8).wrapping_mul(29).wrapping_add(11);
        }

        let mut source_frame = Frame::empty();
        source_frame
            .ensure_rgba_capacity(WIDTH as u32, HEIGHT as u32)
            .expect("failed to allocate source frame");
        let mut state_frame = Frame::empty();
        state_frame
            .ensure_rgba_capacity(WIDTH as u32, HEIGHT as u32)
            .expect("failed to allocate state frame");
        state_frame.as_mut_rgba_bytes().copy_from_slice(&seed_rgba);
        let mut optimized_frame = Frame::empty();
        optimized_frame
            .ensure_rgba_capacity(WIDTH as u32, HEIGHT as u32)
            .expect("failed to allocate optimized frame");
        optimized_frame
            .as_mut_rgba_bytes()
            .copy_from_slice(&seed_rgba);

        let converter = crate::convert::SurfaceRowConverter::new(
            crate::convert::SurfacePixelFormat::Bgra8,
            crate::convert::SurfaceConversionOptions::default(),
        );
        let mut source_bgra = vec![0u8; buffer_len];
        let mut sink = 0u64;
        let mut benchmark_legacy = vec![0u8; buffer_len];
        let mut legacy_total = Duration::ZERO;
        let mut optimized_total = Duration::ZERO;
        let total_iters = WARMUP_ITERS + MEASURE_ITERS;

        for iter in 0..total_iters {
            source_frame
                .as_mut_rgba_bytes()
                .copy_from_slice(state_frame.as_rgba_bytes());
            apply_move_rects_to_frame(&mut source_frame, &move_rects, 0, 0)
                .expect("failed to synthesize source move");
            for rect in &dirty_rects {
                let row_start = rect.y as usize * pitch;
                let row_end = row_start + rect.width as usize * 4;
                for idx in (row_start..row_end).step_by(4) {
                    source_frame.as_mut_rgba_bytes()[idx] =
                        (iter as u8).wrapping_mul(13).wrapping_add((idx / 4) as u8);
                    source_frame.as_mut_rgba_bytes()[idx + 1] =
                        (iter as u8).wrapping_mul(7).wrapping_add((idx / 8) as u8);
                    source_frame.as_mut_rgba_bytes()[idx + 2] =
                        (iter as u8).wrapping_mul(3).wrapping_add((idx / 16) as u8);
                    source_frame.as_mut_rgba_bytes()[idx + 3] = 0xFF;
                }
            }

            source_bgra.copy_from_slice(source_frame.as_rgba_bytes());
            for pixel in source_bgra.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }

            let legacy_start = std::time::Instant::now();
            crate::convert::convert_surface_to_rgba(
                crate::convert::SurfacePixelFormat::Bgra8,
                &source_bgra,
                pitch,
                &mut benchmark_legacy,
                pitch,
                WIDTH,
                HEIGHT,
                crate::convert::SurfaceConversionOptions::default(),
            );
            let legacy_elapsed = legacy_start.elapsed();

            let optimized_start = std::time::Instant::now();
            apply_move_rects_to_frame(&mut optimized_frame, &move_rects, 0, 0)
                .expect("move apply failed during benchmark");
            unsafe {
                let dst_ptr = optimized_frame.as_mut_rgba_ptr();
                let src_ptr = source_bgra.as_ptr();
                for rect in &dirty_rects {
                    let x = rect.x as usize;
                    let y = rect.y as usize;
                    let width = rect.width as usize;
                    let height = rect.height as usize;
                    let src_offset = y * pitch + x * 4;
                    let dst_offset = y * pitch + x * 4;
                    converter.convert_rows_unchecked(
                        src_ptr.add(src_offset),
                        pitch,
                        dst_ptr.add(dst_offset),
                        pitch,
                        width,
                        height,
                    );
                }
            }
            let optimized_elapsed = optimized_start.elapsed();

            if iter >= WARMUP_ITERS {
                legacy_total += legacy_elapsed;
                optimized_total += optimized_elapsed;
            }

            assert_eq!(
                optimized_frame.as_rgba_bytes(),
                benchmark_legacy.as_slice(),
                "move+dirty reconstruction mismatch at iteration {iter}",
            );
            state_frame
                .as_mut_rgba_bytes()
                .copy_from_slice(benchmark_legacy.as_slice());
            sink ^= benchmark_legacy[(iter * 131) % benchmark_legacy.len()] as u64;
            sink ^= optimized_frame.as_rgba_bytes()[(iter * 97) % buffer_len] as u64;
        }
        black_box(sink);

        let legacy_ms = legacy_total.as_secs_f64() * 1000.0;
        let optimized_ms = optimized_total.as_secs_f64() * 1000.0;
        let speedup = if optimized_ms > 0.0 {
            legacy_ms / optimized_ms
        } else {
            0.0
        };

        println!(
            "dxgi region move benchmark: legacy={legacy_ms:.3} ms optimized={optimized_ms:.3} ms speedup={speedup:.2}x"
        );

        assert!(
            optimized_ms <= legacy_ms * 1.05,
            "move+dirty path regressed beyond tolerance: legacy={legacy_ms:.3}ms optimized={optimized_ms:.3}ms ({speedup:.2}x)"
        );
    }
}
