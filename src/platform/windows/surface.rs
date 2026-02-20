use anyhow::Context;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_STAGING, ID3D11Device, ID3D11DeviceContext, ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::core::Interface;

use crate::convert::{
    self, HdrToSdrParams, SurfaceConversionOptions, SurfaceLayout, SurfacePixelFormat,
};
use crate::error::{CaptureError, CaptureResult};
use crate::frame::Frame;

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
    let bytes_per_pixel = match format {
        SurfacePixelFormat::Bgra8 | SurfacePixelFormat::Rgba8 => 4usize,
        SurfacePixelFormat::Rgba16Float => 8usize,
    };
    pixel_count
        .checked_mul(bytes_per_pixel)
        .ok_or(CaptureError::BufferOverflow)
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
