use anyhow::Context;
use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext, ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};
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

fn dirty_rect_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SNOW_CAPTURE_DISABLE_DIRTY_RECT_PARALLEL")
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

    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // Try a non-blocking map first to catch the common case where the GPU
    // finishes right before mapping.
    // D3D11_MAP_FLAG_DO_NOT_WAIT = 0x100000
    const DO_NOT_WAIT: u32 = 0x100000;
    let non_blocking =
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, DO_NOT_WAIT, Some(&mut mapped)) };
    if non_blocking.is_err() {
        mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .context(map_context)
            .map_err(CaptureError::Platform)?;
    }

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

    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // D3D11_MAP_FLAG_DO_NOT_WAIT = 0x100000
    const DO_NOT_WAIT: u32 = 0x100000;
    let non_blocking =
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, DO_NOT_WAIT, Some(&mut mapped)) };
    if non_blocking.is_err() {
        mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .context(map_context)
            .map_err(CaptureError::Platform)?;
    }

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

    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // D3D11_MAP_FLAG_DO_NOT_WAIT = 0x100000
    const DO_NOT_WAIT: u32 = 0x100000;
    let non_blocking =
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, DO_NOT_WAIT, Some(&mut mapped)) };
    if non_blocking.is_err() {
        mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(resource, 0, D3D11_MAP_READ, 0, Some(&mut mapped)) }
            .context(map_context)
            .map_err(CaptureError::Platform)?;
    }

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

        let mut work_items = Vec::with_capacity(dirty_rects.len());
        let mut total_dirty_pixels = 0usize;

        for rect in dirty_rects {
            let x = usize::try_from(rect.x).map_err(|_| CaptureError::BufferOverflow)?;
            let y = usize::try_from(rect.y).map_err(|_| CaptureError::BufferOverflow)?;
            let rect_w = usize::try_from(rect.width).map_err(|_| CaptureError::BufferOverflow)?;
            let rect_h = usize::try_from(rect.height).map_err(|_| CaptureError::BufferOverflow)?;

            if rect_w == 0 || rect_h == 0 || x >= width || y >= height {
                continue;
            }

            let end_x = x.saturating_add(rect_w).min(width);
            let end_y = y.saturating_add(rect_h).min(height);
            if end_x <= x || end_y <= y {
                continue;
            }

            let copy_w = end_x - x;
            let copy_h = end_y - y;

            let src_row_bytes = source_row_bytes(format, copy_w)?;
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

            let dirty_pixels = copy_w
                .checked_mul(copy_h)
                .ok_or(CaptureError::BufferOverflow)?;
            total_dirty_pixels = total_dirty_pixels
                .checked_add(dirty_pixels)
                .ok_or(CaptureError::BufferOverflow)?;

            work_items.push(DirtyRectWorkItem {
                src_offset,
                dst_offset,
                x,
                y,
                width: copy_w,
                height: copy_h,
            });
        }

        if work_items.is_empty() {
            return Ok(0);
        }

        let options = SurfaceConversionOptions { hdr_to_sdr };
        let should_parallelize = convert::should_parallelize_work(
            total_dirty_pixels,
            DIRTY_RECT_PARALLEL_MIN_PIXELS,
            DIRTY_RECT_PARALLEL_MIN_CHUNK_PIXELS,
            DIRTY_RECT_PARALLEL_MAX_WORKERS,
        );
        let can_parallel = dirty_rect_parallel_enabled()
            && work_items.len() >= DIRTY_RECT_PARALLEL_MIN_RECTS
            && should_parallelize
            && work_items_non_overlapping(&work_items);

        if can_parallel {
            use rayon::prelude::*;

            let src_addr = src_base as usize;
            let dst_addr = dst_base as usize;
            convert::with_conversion_pool(DIRTY_RECT_PARALLEL_MAX_WORKERS, || {
                work_items.par_iter().for_each(|item| unsafe {
                    convert::convert_surface_to_rgba_unchecked(
                        format,
                        SurfaceLayout::new(
                            (src_addr + item.src_offset) as *const u8,
                            src_pitch,
                            (dst_addr + item.dst_offset) as *mut u8,
                            dst_pitch,
                            item.width,
                            item.height,
                        ),
                        options,
                    );
                });
            });
        } else {
            for item in &work_items {
                unsafe {
                    convert::convert_surface_to_rgba_unchecked(
                        format,
                        SurfaceLayout::new(
                            src_base.add(item.src_offset),
                            src_pitch,
                            dst_base.add(item.dst_offset),
                            dst_pitch,
                            item.width,
                            item.height,
                        ),
                        options,
                    );
                }
            }
        }

        Ok(work_items.len())
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
}
