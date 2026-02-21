use anyhow::Context;
use std::cell::{Cell, RefCell};
use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext, ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_WAS_STILL_DRAWING;
use windows::core::Interface;

use crate::backend::CaptureBlitRegion;
use crate::convert::{
    self, HdrToSdrParams, SurfaceConversionOptions, SurfaceLayout, SurfacePixelFormat,
};
use crate::error::{CaptureError, CaptureResult};
use crate::frame::{DirtyRect, Frame};

pub(crate) enum StagingSampleDesc {
    Source,
    SingleSample,
}

fn source_surface_format(format: DXGI_FORMAT) -> Option<SurfacePixelFormat> {
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => {
            Some(SurfacePixelFormat::Bgra8)
        }
        DXGI_FORMAT_R8G8B8A8_UNORM => Some(SurfacePixelFormat::Rgba8),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(SurfacePixelFormat::Rgba16Float),
        _ => None,
    }
}

fn source_row_bytes(format: SurfacePixelFormat, pixel_count: usize) -> CaptureResult<usize> {
    let bytes_per_pixel = source_bytes_per_pixel(format);
    pixel_count
        .checked_mul(bytes_per_pixel)
        .ok_or(CaptureError::BufferOverflow)
}

fn source_bytes_per_pixel(format: SurfacePixelFormat) -> usize {
    match format {
        SurfacePixelFormat::Bgra8 | SurfacePixelFormat::Rgba8 => 4usize,
        SurfacePixelFormat::Rgba16Float => 8usize,
    }
}

const DIRTY_RECT_PARALLEL_MIN_PIXELS: usize = 131_072;
const DIRTY_RECT_PARALLEL_MIN_CHUNK_PIXELS: usize = 32_768;
const DIRTY_RECT_PARALLEL_MAX_WORKERS: usize = 9;
const DIRTY_RECT_PARALLEL_MIN_RECTS: usize = 4;
const D3D11_MAP_FLAG_DO_NOT_WAIT: u32 = 0x100000;
const D3D11_MAP_SPIN_POLLS_DEFAULT: usize = 6;
const D3D11_MAP_SPIN_POLLS_MIN: usize = 1;
const D3D11_MAP_SPIN_POLLS_MAX: usize = 64;

thread_local! {
    static MAP_SPIN_POLLS_ADAPTIVE: Cell<usize> = const { Cell::new(D3D11_MAP_SPIN_POLLS_DEFAULT) };
    static DIRTY_WORK_ITEMS_SCRATCH: RefCell<Vec<DirtyRectWorkItem>> = const { RefCell::new(Vec::new()) };
}

#[inline(always)]
fn adaptive_map_spin_polls(max_polls: usize) -> usize {
    MAP_SPIN_POLLS_ADAPTIVE.with(|state| {
        let current = state.get().clamp(
            D3D11_MAP_SPIN_POLLS_MIN,
            max_polls.max(D3D11_MAP_SPIN_POLLS_MIN),
        );
        if current != state.get() {
            state.set(current);
        }
        current
    })
}

#[inline(always)]
fn update_adaptive_map_spin_polls(max_polls: usize, success_attempt: Option<usize>) {
    MAP_SPIN_POLLS_ADAPTIVE.with(|state| {
        let current = state.get().clamp(
            D3D11_MAP_SPIN_POLLS_MIN,
            max_polls.max(D3D11_MAP_SPIN_POLLS_MIN),
        );
        let next = match success_attempt {
            Some(0) => current.saturating_sub(1).max(D3D11_MAP_SPIN_POLLS_MIN),
            Some(attempt) => attempt.saturating_add(2).clamp(
                D3D11_MAP_SPIN_POLLS_MIN,
                max_polls.max(D3D11_MAP_SPIN_POLLS_MIN),
            ),
            None => current.saturating_sub(2).max(D3D11_MAP_SPIN_POLLS_MIN),
        };
        state.set(next);
    });
}

#[inline(always)]
fn map_resource_read_with_spin(
    context: &ID3D11DeviceContext,
    resource: &ID3D11Resource,
    map_context: &'static str,
) -> CaptureResult<D3D11_MAPPED_SUBRESOURCE> {
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();

    if map_spin_wait_enabled() {
        let max_polls = map_spin_poll_count();
        let polls = adaptive_map_spin_polls(max_polls);
        let mut exhausted_spin = true;
        for attempt in 0..polls {
            let map_result = unsafe {
                context.Map(
                    resource,
                    0,
                    D3D11_MAP_READ,
                    D3D11_MAP_FLAG_DO_NOT_WAIT,
                    Some(&mut mapped),
                )
            };
            match map_result {
                Ok(()) => {
                    update_adaptive_map_spin_polls(max_polls, Some(attempt));
                    return Ok(mapped);
                }
                Err(error) if error.code() == DXGI_ERROR_WAS_STILL_DRAWING => {
                    if attempt + 1 < polls {
                        std::hint::spin_loop();
                    }
                }
                Err(_) => {
                    exhausted_spin = false;
                    break;
                }
            }
        }
        if exhausted_spin {
            update_adaptive_map_spin_polls(max_polls, None);
        }
    } else {
        let hr = unsafe {
            context.Map(
                resource,
                0,
                D3D11_MAP_READ,
                D3D11_MAP_FLAG_DO_NOT_WAIT,
                Some(&mut mapped),
            )
        };
        if hr.is_ok() {
            return Ok(mapped);
        }
    }

    mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe { context.Map(resource, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
        .context(map_context)
        .map_err(CaptureError::Platform)?;
    Ok(mapped)
}

#[derive(Clone, Copy)]
struct DirtyRectWorkItem {
    src_offset: usize,
    dst_offset: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[inline(always)]
fn work_items_overlap(a: DirtyRectWorkItem, b: DirtyRectWorkItem) -> bool {
    let a_right = a.x.saturating_add(a.width);
    let a_bottom = a.y.saturating_add(a.height);
    let b_right = b.x.saturating_add(b.width);
    let b_bottom = b.y.saturating_add(b.height);
    a.x < b_right && b.x < a_right && a.y < b_bottom && b.y < a_bottom
}

fn work_items_non_overlapping(work_items: &[DirtyRectWorkItem]) -> bool {
    if work_items.len() <= 1 {
        return true;
    }

    for i in 0..work_items.len() {
        for j in (i + 1)..work_items.len() {
            if work_items_overlap(work_items[i], work_items[j]) {
                return false;
            }
        }
    }
    true
}

#[inline(always)]
fn dirty_rects_overlap(a: DirtyRect, b: DirtyRect) -> bool {
    if a.width == 0 || a.height == 0 || b.width == 0 || b.height == 0 {
        return false;
    }

    let Some(a_right) = a.x.checked_add(a.width) else {
        return true;
    };
    let Some(a_bottom) = a.y.checked_add(a.height) else {
        return true;
    };
    let Some(b_right) = b.x.checked_add(b.width) else {
        return true;
    };
    let Some(b_bottom) = b.y.checked_add(b.height) else {
        return true;
    };

    a.x < b_right && b.x < a_right && a.y < b_bottom && b.y < a_bottom
}

fn dirty_rects_non_overlapping_checked(dirty_rects: &[DirtyRect]) -> bool {
    if dirty_rects.len() <= 1 {
        return true;
    }

    for i in 0..dirty_rects.len() {
        for j in (i + 1)..dirty_rects.len() {
            if dirty_rects_overlap(dirty_rects[i], dirty_rects[j]) {
                return false;
            }
        }
    }
    true
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

fn dirty_rect_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_DIRTY_RECT_PARALLEL"))
}

fn dirty_rect_fastpath_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_DIRTY_RECT_FASTPATH"))
}

fn dirty_rect_non_overlap_shortcut_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_DIRTY_RECT_NON_OVERLAP_SHORTCUT"))
}

fn dirty_rect_trusted_fastpath_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_DIRTY_RECT_TRUSTED_FASTPATH"))
}

fn dirty_rect_trusted_direct_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_DIRTY_RECT_TRUSTED_DIRECT"))
}

fn map_spin_wait_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DISABLE_D3D11_MAP_SPIN_WAIT"))
}

fn map_spin_poll_count() -> usize {
    static POLLS: OnceLock<usize> = OnceLock::new();
    *POLLS.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_D3D11_MAP_SPIN_POLLS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(D3D11_MAP_SPIN_POLLS_DEFAULT)
            .clamp(D3D11_MAP_SPIN_POLLS_MIN, D3D11_MAP_SPIN_POLLS_MAX)
    })
}

#[inline(always)]
fn build_dirty_rect_work_item(
    rect: &DirtyRect,
    width: usize,
    height: usize,
    dst_origin_x: usize,
    dst_origin_y: usize,
    dst_width: usize,
    dst_height: usize,
    src_pitch: usize,
    src_bytes_per_pixel: usize,
    dst_pitch: usize,
) -> CaptureResult<Option<DirtyRectWorkItem>> {
    let x = usize::try_from(rect.x).map_err(|_| CaptureError::BufferOverflow)?;
    let y = usize::try_from(rect.y).map_err(|_| CaptureError::BufferOverflow)?;
    let rect_w = usize::try_from(rect.width).map_err(|_| CaptureError::BufferOverflow)?;
    let rect_h = usize::try_from(rect.height).map_err(|_| CaptureError::BufferOverflow)?;

    if rect_w == 0 || rect_h == 0 || x >= width || y >= height {
        return Ok(None);
    }

    let end_x = x.saturating_add(rect_w).min(width);
    let end_y = y.saturating_add(rect_h).min(height);
    if end_x <= x || end_y <= y {
        return Ok(None);
    }

    let copy_w = end_x - x;
    let copy_h = end_y - y;

    let src_row_bytes = copy_w
        .checked_mul(src_bytes_per_pixel)
        .ok_or(CaptureError::BufferOverflow)?;
    let src_end_in_row = x
        .checked_mul(src_bytes_per_pixel)
        .and_then(|offset| offset.checked_add(src_row_bytes))
        .ok_or(CaptureError::BufferOverflow)?;
    if src_end_in_row > src_pitch {
        return Err(CaptureError::BufferOverflow);
    }

    let dst_x = dst_origin_x
        .checked_add(x)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_y = dst_origin_y
        .checked_add(y)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_right = dst_x
        .checked_add(copy_w)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_bottom = dst_y
        .checked_add(copy_h)
        .ok_or(CaptureError::BufferOverflow)?;
    if dst_right > dst_width || dst_bottom > dst_height {
        return Err(CaptureError::BufferOverflow);
    }

    let dst_row_bytes = copy_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
    let dst_end_in_row = dst_x
        .checked_mul(4)
        .and_then(|offset| offset.checked_add(dst_row_bytes))
        .ok_or(CaptureError::BufferOverflow)?;
    if dst_end_in_row > dst_pitch {
        return Err(CaptureError::BufferOverflow);
    }

    let src_offset = y
        .checked_mul(src_pitch)
        .and_then(|base| {
            x.checked_mul(src_bytes_per_pixel)
                .and_then(|xoff| base.checked_add(xoff))
        })
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_offset = dst_y
        .checked_mul(dst_pitch)
        .and_then(|base| dst_x.checked_mul(4).and_then(|xoff| base.checked_add(xoff)))
        .ok_or(CaptureError::BufferOverflow)?;

    Ok(Some(DirtyRectWorkItem {
        src_offset,
        dst_offset,
        x,
        y,
        width: copy_w,
        height: copy_h,
    }))
}

#[inline(always)]
fn dirty_rects_fit_bounds(
    dirty_rects: &[DirtyRect],
    src_width: u32,
    src_height: u32,
    dst_origin_x: u32,
    dst_origin_y: u32,
    dst_width: u32,
    dst_height: u32,
) -> bool {
    for rect in dirty_rects {
        if rect.width == 0 || rect.height == 0 {
            continue;
        }

        let Some(src_right) = rect.x.checked_add(rect.width) else {
            return false;
        };
        let Some(src_bottom) = rect.y.checked_add(rect.height) else {
            return false;
        };
        if src_right > src_width || src_bottom > src_height {
            return false;
        }

        let Some(dst_x) = dst_origin_x.checked_add(rect.x) else {
            return false;
        };
        let Some(dst_y) = dst_origin_y.checked_add(rect.y) else {
            return false;
        };
        let Some(dst_right) = dst_x.checked_add(rect.width) else {
            return false;
        };
        let Some(dst_bottom) = dst_y.checked_add(rect.height) else {
            return false;
        };
        if dst_right > dst_width || dst_bottom > dst_height {
            return false;
        }
    }
    true
}

#[inline(always)]
unsafe fn build_dirty_rect_work_item_trusted(
    rect: DirtyRect,
    dst_origin_x: usize,
    dst_origin_y: usize,
    src_pitch: usize,
    src_bytes_per_pixel: usize,
    dst_pitch: usize,
) -> Option<DirtyRectWorkItem> {
    if rect.width == 0 || rect.height == 0 {
        return None;
    }

    let x = rect.x as usize;
    let y = rect.y as usize;
    let copy_w = rect.width as usize;
    let copy_h = rect.height as usize;
    let dst_x = dst_origin_x + x;
    let dst_y = dst_origin_y + y;

    let src_offset = y * src_pitch + x * src_bytes_per_pixel;
    let dst_offset = dst_y * dst_pitch + dst_x * 4;

    Some(DirtyRectWorkItem {
        src_offset,
        dst_offset,
        x,
        y,
        width: copy_w,
        height: copy_h,
    })
}

#[inline(always)]
unsafe fn convert_dirty_rects_trusted_direct_unchecked(
    converter: convert::SurfaceRowConverter,
    src_base: *const u8,
    src_pitch: usize,
    src_bytes_per_pixel: usize,
    dst_base: *mut u8,
    dst_pitch: usize,
    dirty_rects: &[DirtyRect],
    dst_origin_x: usize,
    dst_origin_y: usize,
) -> CaptureResult<usize> {
    let mut converted = 0usize;
    let mut total_dirty_pixels = 0usize;
    for rect in dirty_rects {
        let width = rect.width as usize;
        let height = rect.height as usize;
        if width == 0 || height == 0 {
            continue;
        }
        let dirty_pixels = width
            .checked_mul(height)
            .ok_or(CaptureError::BufferOverflow)?;
        total_dirty_pixels = total_dirty_pixels
            .checked_add(dirty_pixels)
            .ok_or(CaptureError::BufferOverflow)?;
        converted = converted.saturating_add(1);
    }
    if converted == 0 {
        return Ok(0);
    }

    let should_parallelize = convert::should_parallelize_work(
        total_dirty_pixels,
        DIRTY_RECT_PARALLEL_MIN_PIXELS,
        DIRTY_RECT_PARALLEL_MIN_CHUNK_PIXELS,
        DIRTY_RECT_PARALLEL_MAX_WORKERS,
    );
    let can_parallel = dirty_rect_parallel_enabled()
        && converted >= DIRTY_RECT_PARALLEL_MIN_RECTS
        && should_parallelize
        && (dirty_rect_non_overlap_shortcut_enabled()
            || dirty_rects_non_overlapping_checked(dirty_rects));

    if can_parallel {
        use rayon::prelude::*;

        let src_addr = src_base as usize;
        let dst_addr = dst_base as usize;
        convert::with_conversion_pool(DIRTY_RECT_PARALLEL_MAX_WORKERS, || {
            dirty_rects.par_iter().for_each(|rect| {
                let width = rect.width as usize;
                let height = rect.height as usize;
                if width == 0 || height == 0 {
                    return;
                }

                let x = rect.x as usize;
                let y = rect.y as usize;
                let src_offset = y * src_pitch + x * src_bytes_per_pixel;
                let dst_offset = (dst_origin_y + y) * dst_pitch + (dst_origin_x + x) * 4;
                unsafe {
                    converter.convert_rows_unchecked(
                        (src_addr + src_offset) as *const u8,
                        src_pitch,
                        (dst_addr + dst_offset) as *mut u8,
                        dst_pitch,
                        width,
                        height,
                    );
                }
            });
        });
        return Ok(converted);
    }

    let allow_inner_parallel = converted == 1;
    for rect in dirty_rects {
        let width = rect.width as usize;
        let height = rect.height as usize;
        if width == 0 || height == 0 {
            continue;
        }

        let x = rect.x as usize;
        let y = rect.y as usize;
        let src_offset = y * src_pitch + x * src_bytes_per_pixel;
        let dst_offset = (dst_origin_y + y) * dst_pitch + (dst_origin_x + x) * 4;
        unsafe {
            if allow_inner_parallel {
                converter.convert_rows_maybe_parallel_unchecked(
                    src_base.add(src_offset),
                    src_pitch,
                    dst_base.add(dst_offset),
                    dst_pitch,
                    width,
                    height,
                );
            } else {
                converter.convert_rows_unchecked(
                    src_base.add(src_offset),
                    src_pitch,
                    dst_base.add(dst_offset),
                    dst_pitch,
                    width,
                    height,
                );
            }
        }
    }

    Ok(converted)
}

pub(crate) fn ensure_staging_texture<'a>(
    device: &ID3D11Device,
    staging: &'a mut Option<ID3D11Texture2D>,
    src: &D3D11_TEXTURE2D_DESC,
    sample_desc: StagingSampleDesc,
    create_context: &'static str,
) -> CaptureResult<&'a ID3D11Texture2D> {
    let needs_new_staging = match staging {
        Some(existing) => {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe { existing.GetDesc(&mut desc) };
            desc.Width != src.Width || desc.Height != src.Height || desc.Format != src.Format
        }
        None => true,
    };

    if needs_new_staging {
        let sample_desc = match sample_desc {
            StagingSampleDesc::Source => src.SampleDesc,
            StagingSampleDesc::SingleSample => DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        };

        let desc = D3D11_TEXTURE2D_DESC {
            Width: src.Width,
            Height: src.Height,
            MipLevels: 1,
            ArraySize: 1,
            Format: src.Format,
            SampleDesc: sample_desc,
            Usage: D3D11_USAGE_STAGING,
            BindFlags: Default::default(),
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: Default::default(),
        };

        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }
            .context(create_context)
            .map_err(CaptureError::Platform)?;
        *staging = texture;
    }

    Ok(staging.as_ref().unwrap())
}

pub(crate) fn copy_mapped_surface_to_frame(
    frame: &mut Frame,
    desc: &D3D11_TEXTURE2D_DESC,
    mapped: &D3D11_MAPPED_SUBRESOURCE,
    hdr_to_sdr: Option<HdrToSdrParams>,
) -> CaptureResult<()> {
    let width_u32 = desc.Width;
    let height_u32 = desc.Height;
    frame.ensure_rgba_capacity(width_u32, height_u32)?;

    let width = usize::try_from(width_u32).map_err(|_| CaptureError::BufferOverflow)?;
    let height = usize::try_from(height_u32).map_err(|_| CaptureError::BufferOverflow)?;

    let src_pitch = mapped.RowPitch as usize;
    let src_base = mapped.pData as *const u8;
    let format = source_surface_format(desc.Format)
        .ok_or_else(|| CaptureError::UnsupportedFormat(format!("{:?}", desc.Format)))?;

    let src_row_len = source_row_bytes(format, width)?;
    src_pitch
        .checked_mul(height.saturating_sub(1))
        .and_then(|base| base.checked_add(src_row_len))
        .ok_or(CaptureError::BufferOverflow)?;

    let dst_pitch = width.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
    unsafe {
        convert::convert_surface_to_rgba_unchecked(
            format,
            SurfaceLayout::new(
                src_base,
                src_pitch,
                frame.as_mut_rgba_ptr(),
                dst_pitch,
                width,
                height,
            ),
            SurfaceConversionOptions { hdr_to_sdr },
        );
    }

    Ok(())
}

/// Map an already-populated staging texture and convert its contents into
/// the output frame. This assumes the GPU copy into `staging` has already
/// been submitted by the caller.
///
/// When `staging_resource` is `Some`, the pre-cached `ID3D11Resource` is
/// used directly, avoiding a COM `QueryInterface` (`cast()`) on every call.
pub(crate) fn map_staging_to_frame(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    staging_resource: Option<&ID3D11Resource>,
    desc: &D3D11_TEXTURE2D_DESC,
    frame: &mut Frame,
    hdr_to_sdr: Option<HdrToSdrParams>,
    map_context: &'static str,
) -> CaptureResult<()> {
    let owned_resource;
    let resource = match staging_resource {
        Some(r) => r,
        None => {
            owned_resource = staging
                .cast::<ID3D11Resource>()
                .context("failed to cast staging texture to ID3D11Resource")
                .map_err(CaptureError::Platform)?;
            &owned_resource
        }
    };

    let mapped = map_resource_read_with_spin(context, resource, map_context)?;

    let result = copy_mapped_surface_to_frame(frame, desc, &mapped, hdr_to_sdr);
    unsafe {
        context.Unmap(resource, 0);
    }
    result
}

/// Map a populated staging texture and convert a source sub-rectangle into
/// the destination frame at `blit.dst_x`/`blit.dst_y`.
pub(crate) fn map_staging_rect_to_frame(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    staging_resource: Option<&ID3D11Resource>,
    desc: &D3D11_TEXTURE2D_DESC,
    frame: &mut Frame,
    blit: CaptureBlitRegion,
    hdr_to_sdr: Option<HdrToSdrParams>,
    map_context: &'static str,
) -> CaptureResult<()> {
    if blit.width == 0 || blit.height == 0 {
        return Ok(());
    }

    let src_width = usize::try_from(desc.Width).map_err(|_| CaptureError::BufferOverflow)?;
    let src_height = usize::try_from(desc.Height).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_width = usize::try_from(frame.width()).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_height = usize::try_from(frame.height()).map_err(|_| CaptureError::BufferOverflow)?;

    let src_x = usize::try_from(blit.src_x).map_err(|_| CaptureError::BufferOverflow)?;
    let src_y = usize::try_from(blit.src_y).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_x = usize::try_from(blit.dst_x).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_y = usize::try_from(blit.dst_y).map_err(|_| CaptureError::BufferOverflow)?;
    let copy_w = usize::try_from(blit.width).map_err(|_| CaptureError::BufferOverflow)?;
    let copy_h = usize::try_from(blit.height).map_err(|_| CaptureError::BufferOverflow)?;

    let src_right = src_x
        .checked_add(copy_w)
        .ok_or(CaptureError::BufferOverflow)?;
    let src_bottom = src_y
        .checked_add(copy_h)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_right = dst_x
        .checked_add(copy_w)
        .ok_or(CaptureError::BufferOverflow)?;
    let dst_bottom = dst_y
        .checked_add(copy_h)
        .ok_or(CaptureError::BufferOverflow)?;
    if src_right > src_width
        || src_bottom > src_height
        || dst_right > dst_width
        || dst_bottom > dst_height
    {
        return Err(CaptureError::BufferOverflow);
    }

    let owned_resource;
    let resource = match staging_resource {
        Some(r) => r,
        None => {
            owned_resource = staging
                .cast::<ID3D11Resource>()
                .context("failed to cast staging texture to ID3D11Resource")
                .map_err(CaptureError::Platform)?;
            &owned_resource
        }
    };

    let mapped = map_resource_read_with_spin(context, resource, map_context)?;

    let convert_result = (|| -> CaptureResult<()> {
        let src_pitch = mapped.RowPitch as usize;
        let src_base = mapped.pData as *const u8;
        let format = source_surface_format(desc.Format)
            .ok_or_else(|| CaptureError::UnsupportedFormat(format!("{:?}", desc.Format)))?;
        let src_row_len = source_row_bytes(format, src_width)?;
        src_pitch
            .checked_mul(src_height.saturating_sub(1))
            .and_then(|base| base.checked_add(src_row_len))
            .ok_or(CaptureError::BufferOverflow)?;

        let src_bytes_per_pixel = source_bytes_per_pixel(format);
        let src_copy_row_bytes = source_row_bytes(format, copy_w)?;
        let src_end_in_row = src_x
            .checked_mul(src_bytes_per_pixel)
            .and_then(|offset| offset.checked_add(src_copy_row_bytes))
            .ok_or(CaptureError::BufferOverflow)?;
        if src_end_in_row > src_pitch {
            return Err(CaptureError::BufferOverflow);
        }

        let dst_pitch = dst_width
            .checked_mul(4)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_copy_row_bytes = copy_w.checked_mul(4).ok_or(CaptureError::BufferOverflow)?;
        let dst_end_in_row = dst_x
            .checked_mul(4)
            .and_then(|offset| offset.checked_add(dst_copy_row_bytes))
            .ok_or(CaptureError::BufferOverflow)?;
        if dst_end_in_row > dst_pitch {
            return Err(CaptureError::BufferOverflow);
        }

        let src_offset = src_y
            .checked_mul(src_pitch)
            .and_then(|base| {
                src_x
                    .checked_mul(src_bytes_per_pixel)
                    .and_then(|xoff| base.checked_add(xoff))
            })
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_offset = dst_y
            .checked_mul(dst_pitch)
            .and_then(|base| dst_x.checked_mul(4).and_then(|xoff| base.checked_add(xoff)))
            .ok_or(CaptureError::BufferOverflow)?;

        unsafe {
            convert::convert_surface_to_rgba_unchecked(
                format,
                SurfaceLayout::new(
                    src_base.add(src_offset),
                    src_pitch,
                    frame.as_mut_rgba_ptr().add(dst_offset),
                    dst_pitch,
                    copy_w,
                    copy_h,
                ),
                SurfaceConversionOptions { hdr_to_sdr },
            );
        }
        Ok(())
    })();

    unsafe {
        context.Unmap(resource, 0);
    }
    convert_result
}

/// Map an already-populated staging texture and apply only the provided
/// dirty rectangles to the output frame.
///
/// Returns the number of dirty rectangles that were actually converted.
pub(crate) fn map_staging_dirty_rects_to_frame(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    staging_resource: Option<&ID3D11Resource>,
    desc: &D3D11_TEXTURE2D_DESC,
    frame: &mut Frame,
    dirty_rects: &[DirtyRect],
    dirty_rects_non_overlapping: bool,
    hdr_to_sdr: Option<HdrToSdrParams>,
    map_context: &'static str,
) -> CaptureResult<usize> {
    frame.ensure_rgba_capacity(desc.Width, desc.Height)?;
    map_staging_dirty_rects_to_frame_with_offset(
        context,
        staging,
        staging_resource,
        desc,
        frame,
        dirty_rects,
        0,
        0,
        dirty_rects_non_overlapping,
        hdr_to_sdr,
        map_context,
    )
}

/// Map an already-populated staging texture and apply dirty rectangles to the
/// output frame at an offset destination origin.
///
/// Dirty rectangles are in source texture coordinates. They are written to
/// `frame` at `(dst_origin_x + rect.x, dst_origin_y + rect.y)`.
pub(crate) fn map_staging_dirty_rects_to_frame_with_offset(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    staging_resource: Option<&ID3D11Resource>,
    desc: &D3D11_TEXTURE2D_DESC,
    frame: &mut Frame,
    dirty_rects: &[DirtyRect],
    dst_origin_x: u32,
    dst_origin_y: u32,
    dirty_rects_non_overlapping: bool,
    hdr_to_sdr: Option<HdrToSdrParams>,
    map_context: &'static str,
) -> CaptureResult<usize> {
    if dirty_rects.is_empty() {
        return Ok(0);
    }

    let width_u32 = desc.Width;
    let height_u32 = desc.Height;
    let width = usize::try_from(width_u32).map_err(|_| CaptureError::BufferOverflow)?;
    let height = usize::try_from(height_u32).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_origin_x = usize::try_from(dst_origin_x).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_origin_y = usize::try_from(dst_origin_y).map_err(|_| CaptureError::BufferOverflow)?;

    let dst_width_u32 = frame.width();
    let dst_height_u32 = frame.height();
    let dst_width = usize::try_from(dst_width_u32).map_err(|_| CaptureError::BufferOverflow)?;
    let dst_height = usize::try_from(dst_height_u32).map_err(|_| CaptureError::BufferOverflow)?;

    let owned_resource;
    let resource = match staging_resource {
        Some(r) => r,
        None => {
            owned_resource = staging
                .cast::<ID3D11Resource>()
                .context("failed to cast staging texture to ID3D11Resource")
                .map_err(CaptureError::Platform)?;
            &owned_resource
        }
    };

    let mapped = map_resource_read_with_spin(context, resource, map_context)?;

    let convert_result = (|| -> CaptureResult<usize> {
        let src_pitch = mapped.RowPitch as usize;
        let src_base = mapped.pData as *const u8;
        let format = source_surface_format(desc.Format)
            .ok_or_else(|| CaptureError::UnsupportedFormat(format!("{:?}", desc.Format)))?;
        let src_row_len = source_row_bytes(format, width)?;
        src_pitch
            .checked_mul(height.saturating_sub(1))
            .and_then(|base| base.checked_add(src_row_len))
            .ok_or(CaptureError::BufferOverflow)?;

        let src_bytes_per_pixel = source_bytes_per_pixel(format);
        let dst_pitch = dst_width
            .checked_mul(4)
            .ok_or(CaptureError::BufferOverflow)?;
        let dst_base = frame.as_mut_rgba_ptr();
        let options = SurfaceConversionOptions { hdr_to_sdr };
        let converter = convert::SurfaceRowConverter::new(format, options);
        let trusted_fastpath = dirty_rect_trusted_fastpath_enabled()
            && dirty_rects_non_overlapping
            && dirty_rects_fit_bounds(
                dirty_rects,
                width_u32,
                height_u32,
                dst_origin_x as u32,
                dst_origin_y as u32,
                dst_width_u32,
                dst_height_u32,
            );

        if trusted_fastpath && dirty_rect_trusted_direct_enabled() {
            // SAFETY: `trusted_fastpath` guarantees all dirty rectangles are
            // in-bounds for both source and destination surfaces.
            return unsafe {
                convert_dirty_rects_trusted_direct_unchecked(
                    converter,
                    src_base,
                    src_pitch,
                    src_bytes_per_pixel,
                    dst_base,
                    dst_pitch,
                    dirty_rects,
                    dst_origin_x,
                    dst_origin_y,
                )
            };
        }

        if dirty_rect_fastpath_enabled() && dirty_rects.len() < DIRTY_RECT_PARALLEL_MIN_RECTS {
            let allow_inner_parallel = dirty_rects.len() == 1;
            let mut converted = 0usize;
            for rect in dirty_rects {
                let item = if trusted_fastpath {
                    // SAFETY: trusted mode is reserved for normalized,
                    // in-bounds dirty rect lists produced by capture backends.
                    unsafe {
                        build_dirty_rect_work_item_trusted(
                            *rect,
                            dst_origin_x,
                            dst_origin_y,
                            src_pitch,
                            src_bytes_per_pixel,
                            dst_pitch,
                        )
                    }
                } else {
                    build_dirty_rect_work_item(
                        rect,
                        width,
                        height,
                        dst_origin_x,
                        dst_origin_y,
                        dst_width,
                        dst_height,
                        src_pitch,
                        src_bytes_per_pixel,
                        dst_pitch,
                    )?
                };
                let Some(item) = item else {
                    continue;
                };
                unsafe {
                    if allow_inner_parallel {
                        converter.convert_rows_maybe_parallel_unchecked(
                            src_base.add(item.src_offset),
                            src_pitch,
                            dst_base.add(item.dst_offset),
                            dst_pitch,
                            item.width,
                            item.height,
                        );
                    } else {
                        converter.convert_rows_unchecked(
                            src_base.add(item.src_offset),
                            src_pitch,
                            dst_base.add(item.dst_offset),
                            dst_pitch,
                            item.width,
                            item.height,
                        );
                    }
                }
                converted = converted.saturating_add(1);
            }
            return Ok(converted);
        }

        DIRTY_WORK_ITEMS_SCRATCH.with(|scratch| -> CaptureResult<usize> {
            let mut work_items = scratch.borrow_mut();
            work_items.clear();
            let additional_capacity = dirty_rects.len().saturating_sub(work_items.capacity());
            if additional_capacity > 0 {
                work_items.reserve(additional_capacity);
            }

            let mut total_dirty_pixels = 0usize;
            if trusted_fastpath {
                for rect in dirty_rects {
                    // SAFETY: trusted mode is reserved for normalized,
                    // in-bounds dirty rect lists produced by capture backends.
                    let Some(item) = (unsafe {
                        build_dirty_rect_work_item_trusted(
                            *rect,
                            dst_origin_x,
                            dst_origin_y,
                            src_pitch,
                            src_bytes_per_pixel,
                            dst_pitch,
                        )
                    }) else {
                        continue;
                    };
                    let dirty_pixels = item
                        .width
                        .checked_mul(item.height)
                        .ok_or(CaptureError::BufferOverflow)?;
                    total_dirty_pixels = total_dirty_pixels
                        .checked_add(dirty_pixels)
                        .ok_or(CaptureError::BufferOverflow)?;
                    work_items.push(item);
                }
            } else {
                for rect in dirty_rects {
                    let Some(item) = build_dirty_rect_work_item(
                        rect,
                        width,
                        height,
                        dst_origin_x,
                        dst_origin_y,
                        dst_width,
                        dst_height,
                        src_pitch,
                        src_bytes_per_pixel,
                        dst_pitch,
                    )?
                    else {
                        continue;
                    };

                    let dirty_pixels = item
                        .width
                        .checked_mul(item.height)
                        .ok_or(CaptureError::BufferOverflow)?;
                    total_dirty_pixels = total_dirty_pixels
                        .checked_add(dirty_pixels)
                        .ok_or(CaptureError::BufferOverflow)?;
                    work_items.push(item);
                }
            }

            if work_items.is_empty() {
                return Ok(0);
            }

            let should_parallelize = convert::should_parallelize_work(
                total_dirty_pixels,
                DIRTY_RECT_PARALLEL_MIN_PIXELS,
                DIRTY_RECT_PARALLEL_MIN_CHUNK_PIXELS,
                DIRTY_RECT_PARALLEL_MAX_WORKERS,
            );
            let work_items_slice = &work_items[..];
            let non_overlapping =
                if dirty_rects_non_overlapping && dirty_rect_non_overlap_shortcut_enabled() {
                    debug_assert!(work_items_non_overlapping(work_items_slice));
                    true
                } else {
                    work_items_non_overlapping(work_items_slice)
                };
            let can_parallel = dirty_rect_parallel_enabled()
                && work_items_slice.len() >= DIRTY_RECT_PARALLEL_MIN_RECTS
                && should_parallelize
                && non_overlapping;

            if can_parallel {
                use rayon::prelude::*;

                let src_addr = src_base as usize;
                let dst_addr = dst_base as usize;
                convert::with_conversion_pool(DIRTY_RECT_PARALLEL_MAX_WORKERS, || {
                    work_items_slice.par_iter().for_each(|item| unsafe {
                        converter.convert_rows_unchecked(
                            (src_addr + item.src_offset) as *const u8,
                            src_pitch,
                            (dst_addr + item.dst_offset) as *mut u8,
                            dst_pitch,
                            item.width,
                            item.height,
                        );
                    });
                });
            } else {
                let allow_inner_parallel = work_items_slice.len() == 1;
                for item in work_items_slice {
                    unsafe {
                        if allow_inner_parallel {
                            converter.convert_rows_maybe_parallel_unchecked(
                                src_base.add(item.src_offset),
                                src_pitch,
                                dst_base.add(item.dst_offset),
                                dst_pitch,
                                item.width,
                                item.height,
                            );
                        } else {
                            converter.convert_rows_unchecked(
                                src_base.add(item.src_offset),
                                src_pitch,
                                dst_base.add(item.dst_offset),
                                dst_pitch,
                                item.width,
                                item.height,
                            );
                        }
                    }
                }
            }

            Ok(work_items_slice.len())
        })
    })();

    unsafe {
        context.Unmap(resource, 0);
    }
    convert_result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(x: usize, y: usize, width: usize, height: usize) -> DirtyRectWorkItem {
        DirtyRectWorkItem {
            src_offset: 0,
            dst_offset: 0,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn work_items_overlap_detects_intersection() {
        let a = item(10, 10, 40, 30);
        let b = item(30, 20, 20, 20);
        assert!(work_items_overlap(a, b));
    }

    #[test]
    fn work_items_overlap_rejects_edge_touch() {
        let a = item(10, 10, 20, 20);
        let b = item(30, 10, 20, 20);
        assert!(!work_items_overlap(a, b));
    }

    #[test]
    fn work_items_non_overlapping_detects_collision() {
        let work_items = vec![item(0, 0, 32, 32), item(31, 4, 16, 16), item(80, 5, 8, 8)];
        assert!(!work_items_non_overlapping(&work_items));
    }

    #[test]
    fn work_items_non_overlapping_accepts_disjoint_list() {
        let work_items = vec![item(0, 0, 16, 16), item(20, 0, 12, 16), item(0, 20, 8, 8)];
        assert!(work_items_non_overlapping(&work_items));
    }

    #[test]
    fn dirty_rects_non_overlapping_checked_detects_overlap() {
        let dirty_rects = vec![
            DirtyRect {
                x: 4,
                y: 8,
                width: 16,
                height: 12,
            },
            DirtyRect {
                x: 12,
                y: 10,
                width: 8,
                height: 8,
            },
        ];
        assert!(!dirty_rects_non_overlapping_checked(&dirty_rects));
    }

    #[test]
    fn dirty_rects_non_overlapping_checked_ignores_zero_sized_rects() {
        let dirty_rects = vec![
            DirtyRect {
                x: 2,
                y: 2,
                width: 24,
                height: 8,
            },
            DirtyRect {
                x: 10,
                y: 4,
                width: 0,
                height: 8,
            },
            DirtyRect {
                x: 30,
                y: 2,
                width: 12,
                height: 8,
            },
        ];
        assert!(dirty_rects_non_overlapping_checked(&dirty_rects));
    }

    #[test]
    fn build_dirty_rect_work_item_clamps_to_source_bounds() {
        let rect = DirtyRect {
            x: 1910,
            y: 1075,
            width: 32,
            height: 20,
        };
        let item = build_dirty_rect_work_item(&rect, 1920, 1080, 0, 0, 1920, 1080, 7680, 4, 7680)
            .expect("work item build failed")
            .expect("expected dirty work item");
        assert_eq!(item.width, 10);
        assert_eq!(item.height, 5);
        assert_eq!(item.x, 1910);
        assert_eq!(item.y, 1075);
    }

    #[test]
    fn build_dirty_rect_work_item_rejects_destination_overflow() {
        let rect = DirtyRect {
            x: 40,
            y: 10,
            width: 32,
            height: 16,
        };
        let result = build_dirty_rect_work_item(&rect, 128, 128, 100, 0, 128, 128, 512, 4, 512);
        assert!(matches!(result, Err(CaptureError::BufferOverflow)));
    }

    #[test]
    fn dirty_rects_fit_bounds_accepts_valid_rects() {
        let rects = vec![
            DirtyRect {
                x: 4,
                y: 8,
                width: 12,
                height: 16,
            },
            DirtyRect {
                x: 64,
                y: 20,
                width: 24,
                height: 12,
            },
        ];
        assert!(dirty_rects_fit_bounds(&rects, 128, 96, 10, 6, 192, 128));
    }

    #[test]
    fn dirty_rects_fit_bounds_rejects_destination_overflow() {
        let rects = vec![DirtyRect {
            x: 120,
            y: 0,
            width: 16,
            height: 4,
        }];
        assert!(!dirty_rects_fit_bounds(&rects, 256, 64, 80, 0, 200, 100));
    }

    #[test]
    fn trusted_dirty_rect_work_item_matches_checked_builder() {
        let rect = DirtyRect {
            x: 25,
            y: 40,
            width: 10,
            height: 12,
        };
        let checked = build_dirty_rect_work_item(&rect, 256, 256, 3, 7, 400, 400, 1024, 4, 1600)
            .expect("checked builder failed")
            .expect("checked builder returned None");
        let trusted = unsafe { build_dirty_rect_work_item_trusted(rect, 3, 7, 1024, 4, 1600) }
            .expect("trusted builder returned None");
        assert_eq!(checked.src_offset, trusted.src_offset);
        assert_eq!(checked.dst_offset, trusted.dst_offset);
        assert_eq!(checked.width, trusted.width);
        assert_eq!(checked.height, trusted.height);
    }

    #[test]
    fn trusted_direct_dirty_rect_conversion_matches_work_item_path() {
        let src_width = 160usize;
        let src_height = 96usize;
        let src_pitch = src_width * 4 + 64;
        let src_len = src_pitch * (src_height - 1) + src_width * 4;
        let mut src = vec![0u8; src_len];
        for (idx, byte) in src.iter_mut().enumerate() {
            *byte = (idx as u8).wrapping_mul(29).wrapping_add(7);
        }

        let dst_origin_x = 7usize;
        let dst_origin_y = 11usize;
        let dst_width = 220usize;
        let dst_height = 170usize;
        let dst_pitch = dst_width * 4;
        let dst_len = dst_pitch * dst_height;

        let dirty_rects = vec![
            DirtyRect {
                x: 2,
                y: 3,
                width: 24,
                height: 12,
            },
            DirtyRect {
                x: 40,
                y: 10,
                width: 20,
                height: 8,
            },
            DirtyRect {
                x: 90,
                y: 22,
                width: 30,
                height: 10,
            },
            DirtyRect {
                x: 18,
                y: 48,
                width: 42,
                height: 16,
            },
            DirtyRect {
                x: 130,
                y: 60,
                width: 18,
                height: 14,
            },
            DirtyRect {
                x: 12,
                y: 80,
                width: 0,
                height: 4,
            },
        ];

        assert!(dirty_rects_fit_bounds(
            &dirty_rects,
            src_width as u32,
            src_height as u32,
            dst_origin_x as u32,
            dst_origin_y as u32,
            dst_width as u32,
            dst_height as u32
        ));

        let converter = convert::SurfaceRowConverter::new(
            SurfacePixelFormat::Bgra8,
            SurfaceConversionOptions { hdr_to_sdr: None },
        );

        let mut expected = vec![0u8; dst_len];
        let mut expected_converted = 0usize;
        for rect in &dirty_rects {
            let item = build_dirty_rect_work_item(
                rect,
                src_width,
                src_height,
                dst_origin_x,
                dst_origin_y,
                dst_width,
                dst_height,
                src_pitch,
                4,
                dst_pitch,
            )
            .expect("checked work-item conversion failed");
            let Some(item) = item else {
                continue;
            };
            expected_converted = expected_converted.saturating_add(1);
            unsafe {
                converter.convert_rows_unchecked(
                    src.as_ptr().add(item.src_offset),
                    src_pitch,
                    expected.as_mut_ptr().add(item.dst_offset),
                    dst_pitch,
                    item.width,
                    item.height,
                );
            }
        }

        let mut direct = vec![0u8; dst_len];
        let converted = unsafe {
            convert_dirty_rects_trusted_direct_unchecked(
                converter,
                src.as_ptr(),
                src_pitch,
                4,
                direct.as_mut_ptr(),
                dst_pitch,
                &dirty_rects,
                dst_origin_x,
                dst_origin_y,
            )
        }
        .expect("trusted direct conversion failed");

        assert_eq!(converted, expected_converted);
        assert_eq!(direct, expected);
    }
}
