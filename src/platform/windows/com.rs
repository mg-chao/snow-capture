use anyhow::{Context, Result};
use windows::Win32::Foundation::RPC_E_CHANGED_MODE;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};

pub(crate) struct CoInitGuard {
    should_uninit: bool,
}

impl CoInitGuard {
    pub fn init_multithreaded() -> Result<Self> {
        let hr = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
        if hr == RPC_E_CHANGED_MODE {
            return Ok(Self {
                should_uninit: false,
            });
        }

        hr.ok()
            .context("failed to initialize COM with CoInitializeEx(COINIT_MULTITHREADED)")?;
        Ok(Self {
            should_uninit: true,
        })
    }
}

impl Drop for CoInitGuard {
    fn drop(&mut self) {
        if self.should_uninit {
            unsafe {
                CoUninitialize();
            }
        }
    }
}
