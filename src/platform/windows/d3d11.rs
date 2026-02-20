use anyhow::{Context, Result};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_SINGLETHREADED, D3D11_SDK_VERSION,
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
};
use windows::Win32::Graphics::Dxgi::IDXGIAdapter;

/// Create a D3D11 device on the given adapter.
///
/// When `single_threaded` is true the device is created with
/// `D3D11_CREATE_DEVICE_SINGLETHREADED`, which removes internal driver
/// locking overhead.  This is safe for backends that only access the
/// device from a single thread (e.g. DXGI duplication), but must NOT
/// be used for WGC whose `CreateFreeThreaded` frame pool accesses the
/// device from multiple threads.
pub(crate) fn create_d3d11_device_for_adapter(
    adapter: &IDXGIAdapter,
    single_threaded: bool,
) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    create_d3d11_device(Some(adapter), single_threaded)
}

/// Create a D3D11 device on the default hardware adapter.
pub(crate) fn create_d3d11_device_default(
    single_threaded: bool,
) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    create_d3d11_device(None, single_threaded)
}

fn create_d3d11_device(
    adapter: Option<&IDXGIAdapter>,
    single_threaded: bool,
) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device: Option<ID3D11Device> = None;
    let mut context: Option<ID3D11DeviceContext> = None;
    let feature_levels = [D3D_FEATURE_LEVEL_11_0];

    let mut flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if single_threaded {
        flags |= D3D11_CREATE_DEVICE_SINGLETHREADED;
    }

    unsafe {
        D3D11CreateDevice(
            adapter,
            if adapter.is_some() {
                D3D_DRIVER_TYPE_UNKNOWN
            } else {
                D3D_DRIVER_TYPE_HARDWARE
            },
            None,
            flags,
            Some(&feature_levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    }
    .context("D3D11CreateDevice failed")?;

    let device = device.context("D3D11CreateDevice did not return a device")?;
    let context = context.context("D3D11CreateDevice did not return a device context")?;
    Ok((device, context))
}
