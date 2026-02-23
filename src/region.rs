//! Multi-monitor region capture support.
//!
//! [`MonitorLayout`] snapshots the virtual desktop geometry at startup.
//! [`CaptureRegion`] describes an arbitrary rectangle in virtual desktop
//! coordinates that may span multiple monitors.

use crate::error::{CaptureError, CaptureResult};
use crate::monitor::MonitorId;

/// A rectangle in virtual desktop coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRegion {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl CaptureRegion {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> CaptureResult<Self> {
        if width == 0 || height == 0 {
            return Err(CaptureError::InvalidConfig(
                "region width and height must be > 0".into(),
            ));
        }
        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Right edge (exclusive) in virtual desktop coordinates.
    pub fn right(&self) -> i32 {
        self.x.saturating_add(self.width as i32)
    }

    /// Bottom edge (exclusive) in virtual desktop coordinates.
    pub fn bottom(&self) -> i32 {
        self.y.saturating_add(self.height as i32)
    }
}

/// Geometry of a single monitor in the virtual desktop.
#[derive(Clone, Debug)]
pub struct MonitorGeometry {
    pub monitor: MonitorId,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// A snapshot of the virtual desktop monitor layout, captured once at
/// session creation. Used to determine which monitors overlap a
/// [`CaptureRegion`] and how to composite the final frame.
#[derive(Clone, Debug)]
pub struct MonitorLayout {
    pub monitors: Vec<MonitorGeometry>,
    /// Bounding box of the entire virtual desktop.
    pub virtual_left: i32,
    pub virtual_top: i32,
    pub virtual_width: u32,
    pub virtual_height: u32,
}

impl MonitorLayout {
    /// Snapshot the current monitor layout from the OS.
    ///
    /// This queries monitor positions once and caches them. Call this
    /// at session startup — the layout is not refreshed automatically.
    pub fn snapshot() -> CaptureResult<Self> {
        let monitors = crate::monitor::enumerate_monitors()?;
        Self::snapshot_from_monitors(monitors)
    }

    /// Snapshot monitor geometry for a known monitor set.
    ///
    /// `monitors` should come from the same backend/session that will be
    /// used for capture so monitor keys remain consistent.
    pub fn snapshot_from_monitors(monitors: Vec<MonitorId>) -> CaptureResult<Self> {
        #[cfg(target_os = "windows")]
        {
            snapshot_windows_from_monitors(monitors)
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = monitors;
            Err(CaptureError::Platform(anyhow::anyhow!(
                "monitor layout snapshot is only supported on Windows"
            )))
        }
    }

    /// Return the monitors that overlap the given region, along with
    /// the intersection rectangle in virtual desktop coordinates.
    pub fn overlapping_monitors(
        &self,
        region: &CaptureRegion,
    ) -> Vec<(MonitorGeometry, CaptureRegion)> {
        let mut result = Vec::new();
        let r_right = region.right();
        let r_bottom = region.bottom();

        for mon in &self.monitors {
            let m_right = mon.x.saturating_add(mon.width as i32);
            let m_bottom = mon.y.saturating_add(mon.height as i32);

            let ix = region.x.max(mon.x);
            let iy = region.y.max(mon.y);
            let ix2 = r_right.min(m_right);
            let iy2 = r_bottom.min(m_bottom);

            if ix < ix2 && iy < iy2 {
                let intersection = CaptureRegion {
                    x: ix,
                    y: iy,
                    width: (ix2 - ix) as u32,
                    height: (iy2 - iy) as u32,
                };
                result.push((mon.clone(), intersection));
            }
        }
        result
    }
}

#[cfg(target_os = "windows")]
fn snapshot_windows_from_monitors(monitors: Vec<MonitorId>) -> CaptureResult<MonitorLayout> {
    use std::mem::size_of;
    use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, HMONITOR, MONITORINFO, MONITORINFOEXW};

    if monitors.is_empty() {
        return Err(CaptureError::NoPrimaryMonitor);
    }

    let mut geometries = Vec::with_capacity(monitors.len());
    for monitor in monitors {
        let hmon = HMONITOR(monitor.raw_handle() as *mut std::ffi::c_void);
        if hmon.0.is_null() {
            return Err(CaptureError::MonitorLost);
        }

        let mut info = MONITORINFOEXW {
            monitorInfo: MONITORINFO {
                cbSize: size_of::<MONITORINFOEXW>() as u32,
                ..Default::default()
            },
            ..Default::default()
        };

        if !unsafe { GetMonitorInfoW(hmon, (&mut info as *mut MONITORINFOEXW).cast()) }.as_bool() {
            return Err(CaptureError::MonitorLost);
        }

        let rect = info.monitorInfo.rcMonitor;
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 0 || h <= 0 {
            continue;
        }

        geometries.push(MonitorGeometry {
            monitor,
            x: rect.left,
            y: rect.top,
            width: w as u32,
            height: h as u32,
        });
    }

    if geometries.is_empty() {
        return Err(CaptureError::MonitorLost);
    }

    // Compute virtual desktop bounding box.
    let mut vl = i32::MAX;
    let mut vt = i32::MAX;
    let mut vr = i32::MIN;
    let mut vb = i32::MIN;
    for m in &geometries {
        vl = vl.min(m.x);
        vt = vt.min(m.y);
        vr = vr.max(m.x + m.width as i32);
        vb = vb.max(m.y + m.height as i32);
    }

    Ok(MonitorLayout {
        monitors: geometries,
        virtual_left: vl,
        virtual_top: vt,
        virtual_width: (vr - vl) as u32,
        virtual_height: (vb - vt) as u32,
    })
}
