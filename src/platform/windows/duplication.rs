use std::sync::{Arc, OnceLock};
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
    DXGI_OUTDUPL_POINTER_SHAPE_INFO, DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR,
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
use super::gpu_tonemap::{GpuF16Converter, GpuTonemapper};
use super::monitor::{HdrMonitorMetadata, MonitorResolver, ResolvedMonitor};
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
const DXGI_REGION_STAGING_SLOTS: usize = 3;

#[inline]
fn env_var_truthy(var_name: &'static str) -> bool {
    std::env::var(var_name)
        .map(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        })
        .unwrap_or(false)
}

#[inline]
fn region_dirty_gpu_copy_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DXGI_DISABLE_REGION_DIRTY_GPU_COPY"))
}

#[inline]
fn duplicate_dirty_fastpath_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| !env_var_truthy("SNOW_CAPTURE_DXGI_DISABLE_DUPLICATE_DIRTY_FASTPATH"))
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
    dirty_copy_preferred: bool,
    dirty_rects: Vec<DirtyRect>,
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
        self.dirty_copy_preferred = false;
        self.dirty_rects.clear();
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

fn clamp_dirty_rect(rect: DirtyRect, width: u32, height: u32) -> Option<DirtyRect> {
    let x = rect.x.min(width);
    let y = rect.y.min(height);
    if x >= width || y >= height {
        return None;
    }

    let max_w = width - x;
    let max_h = height - y;
    let clamped_w = rect.width.min(max_w);
    let clamped_h = rect.height.min(max_h);
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

fn dirty_rect_bounds(rect: DirtyRect) -> (u32, u32) {
    (
        rect.x.saturating_add(rect.width),
        rect.y.saturating_add(rect.height),
    )
}

fn intervals_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start < b_end && b_start < a_end
}

fn intervals_touch_or_overlap(a_start: u32, a_end: u32, b_start: u32, b_end: u32) -> bool {
    a_start <= b_end && b_start <= a_end
}

fn dirty_rects_can_merge(a: DirtyRect, b: DirtyRect) -> bool {
    let (a_right, a_bottom) = dirty_rect_bounds(a);
    let (b_right, b_bottom) = dirty_rect_bounds(b);

    let horizontal_overlap = intervals_overlap(a.x, a_right, b.x, b_right);
    let vertical_overlap = intervals_overlap(a.y, a_bottom, b.y, b_bottom);
    let horizontal_touch_or_overlap = intervals_touch_or_overlap(a.x, a_right, b.x, b_right);
    let vertical_touch_or_overlap = intervals_touch_or_overlap(a.y, a_bottom, b.y, b_bottom);

    (horizontal_overlap && vertical_touch_or_overlap)
        || (vertical_overlap && horizontal_touch_or_overlap)
}

fn merge_dirty_rects(a: DirtyRect, b: DirtyRect) -> DirtyRect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let (a_right, a_bottom) = dirty_rect_bounds(a);
    let (b_right, b_bottom) = dirty_rect_bounds(b);
    let right = a_right.max(b_right);
    let bottom = a_bottom.max(b_bottom);
    DirtyRect {
        x,
        y,
        width: right.saturating_sub(x),
        height: bottom.saturating_sub(y),
    }
}

fn normalize_dirty_rects_in_place(rects: &mut Vec<DirtyRect>, width: u32, height: u32) {
    if rects.is_empty() {
        return;
    }

    let mut write = 0usize;
    for read in 0..rects.len() {
        if let Some(clamped) = clamp_dirty_rect(rects[read], width, height) {
            rects[write] = clamped;
            write += 1;
        }
    }
    rects.truncate(write);
    if rects.len() <= 1 {
        return;
    }

    let mut i = 0usize;
    while i < rects.len() {
        let mut j = i + 1;
        while j < rects.len() {
            if dirty_rects_can_merge(rects[i], rects[j]) {
                rects[i] = merge_dirty_rects(rects[i], rects[j]);
                rects.swap_remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }

    rects.sort_unstable_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));
}

fn extract_region_dirty_rects(
    source_dirty_rects: &[DirtyRect],
    source_width: u32,
    source_height: u32,
    blit: CaptureBlitRegion,
    out: &mut Vec<DirtyRect>,
) -> bool {
    out.clear();

    let Some(region_bounds) = clamp_dirty_rect(
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

    for rect in source_dirty_rects {
        let Some(clamped) = clamp_dirty_rect(*rect, source_width, source_height) else {
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

    true
}

fn hdr_to_sdr_params(hdr: HdrMonitorMetadata) -> Option<HdrToSdrParams> {
    if !hdr.advanced_color_enabled {
        return None;
    }

    if !hdr.hdr_enabled {
        return None;
    }

    let sdr_white_level_nits = hdr.sdr_white_level_nits.unwrap_or(80.0);
    let hdr_paper_white_nits = hdr.hdr_paper_white_nits.unwrap_or(80.0);

    Some(HdrToSdrParams {
        hdr_paper_white_nits,
        hdr_maximum_nits: hdr.hdr_maximum_nits.unwrap_or(1000.0),
        sdr_white_level_nits,
    })
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
    }

    fn reset_pipeline(&mut self) {
        self.pending = false;
        self.read_idx = None;
        self.write_idx = 0;
        self.adaptive_spin_polls = Self::INITIAL_SPIN_POLLS;
    }

    /// Ensure both staging slots match the given texture description.
    /// Skips the per-slot `GetDesc()` COM call when the cached
    /// (width, height, format) triple already matches.
    fn ensure_slots(
        &mut self,
        device: &ID3D11Device,
        desc: &D3D11_TEXTURE2D_DESC,
    ) -> CaptureResult<()> {
        let key = (desc.Width, desc.Height, desc.Format);
        let needs_recreate = self.cached_desc != Some(key);

        if needs_recreate {
            for i in 0..STAGING_SLOTS {
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
            self.cached_desc = Some(key);
        }
        // Create event queries for async readback
        for i in 0..STAGING_SLOTS {
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
        Ok(())
    }

    /// Submit a GPU copy from `source` into the current write slot.
    /// Returns the *read* slot index if there was a previous pending
    /// copy that can now be consumed.
    fn submit_copy(
        &mut self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
    ) -> Option<usize> {
        let prev_pending = self.pending;
        let read_idx = if prev_pending {
            // The previous write slot becomes the new read slot.
            Some(self.write_idx)
        } else {
            None
        };

        // Advance to the next slot in the ring for the new write.
        if prev_pending {
            self.write_idx = (self.write_idx + 1) % STAGING_SLOTS;
        }

        let staging_res = self.slot_resources[self.write_idx].as_ref().unwrap();
        let source_res: ID3D11Resource = source.cast().unwrap();

        unsafe {
            context.CopyResource(staging_res, &source_res);
        }

        // Signal the event query so we can poll for completion.
        if let Some(ref query) = self.queries[self.write_idx] {
            unsafe { context.End(query) };
        }

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

        self.pending = true;
        self.read_idx = read_idx;
        read_idx
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
            match surface::map_staging_dirty_rects_to_frame(
                context,
                staging,
                staging_res,
                desc,
                frame,
                dirty_rects,
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

    /// Synchronous single-shot: copy + flush + spin-wait + map + convert.
    /// Used for the first frame or when we can't pipeline.  We still do
    /// a short spin-wait after flushing to give the GPU a chance to
    /// finish the copy before the blocking `Map()` call.
    fn copy_and_read(
        &mut self,
        context: &ID3D11DeviceContext,
        source: &ID3D11Texture2D,
        desc: &D3D11_TEXTURE2D_DESC,
        frame: &mut Frame,
        hdr_to_sdr: Option<HdrToSdrParams>,
    ) -> CaptureResult<()> {
        let staging_res = self.slot_resources[self.write_idx].as_ref().unwrap();
        let source_res: ID3D11Resource = source.cast().unwrap();

        unsafe {
            context.CopyResource(staging_res, &source_res);
        }

        // Signal the event query and flush to kick off the copy.
        if let Some(ref query) = self.queries[self.write_idx] {
            unsafe { context.End(query) };
        }
        unsafe { context.Flush() };

        // Short spin-wait to avoid going straight into the blocking Map.
        const FIRST_FRAME_SPIN: u32 = 8;
        if let Some(ref query) = self.queries[self.write_idx] {
            let mut data: u32 = 0;
            for _ in 0..FIRST_FRAME_SPIN {
                let hr = unsafe {
                    context.GetData(
                        query,
                        Some(&mut data as *mut u32 as *mut _),
                        std::mem::size_of::<u32>() as u32,
                        0x1, // DO_NOT_FLUSH
                    )
                };
                if hr.is_ok() {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        let staging = self.slots[self.write_idx].as_ref().unwrap();
        let staging_res = self.slot_resources[self.write_idx].as_ref();
        surface::map_staging_to_frame(
            context,
            staging,
            staging_res,
            desc,
            frame,
            hdr_to_sdr,
            "failed to map staging texture",
        )?;
        self.pending = false;
        Ok(())
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

fn should_use_dirty_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
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

fn should_use_dirty_gpu_copy(rects: &[DirtyRect], width: u32, height: u32) -> bool {
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

#[inline(always)]
fn should_skip_screenrecord_submit_copy(
    fastpath_enabled: bool,
    has_pending_submission: bool,
    source_is_duplicate: bool,
    source_unchanged: bool,
) -> bool {
    fastpath_enabled && has_pending_submission && (source_is_duplicate || source_unchanged)
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
                        // Already BGRA -- swizzle to RGBA.
                        let pixel_count = (w * h) as usize;
                        let mut rgba = vec![0u8; pixel_count * 4];
                        let pitch = shape_info.Pitch as usize;
                        for row in 0..h as usize {
                            let src_offset = row * pitch;
                            let dst_offset = row * (w as usize) * 4;
                            for col in 0..w as usize {
                                let si = src_offset + col * 4;
                                let di = dst_offset + col * 4;
                                if si + 3 < shape_buf.len() && di + 3 < rgba.len() {
                                    rgba[di] = shape_buf[si + 2]; // R
                                    rgba[di + 1] = shape_buf[si + 1]; // G
                                    rgba[di + 2] = shape_buf[si]; // B
                                    rgba[di + 3] = shape_buf[si + 3]; // A
                                }
                            }
                        }
                        rgba
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
    source_dirty_rects_scratch: Vec<DirtyRect>,
    region_dirty_rects_scratch: Vec<DirtyRect>,
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
            cached_src_desc: None,
            spare_frame: None,
            region_slots: std::array::from_fn(|_| RegionStagingSlot::default()),
            region_pending_slot: None,
            region_next_write_slot: 0,
            region_adaptive_spin_polls: StagingRing::INITIAL_SPIN_POLLS,
            region_blit: None,
            dxgi_rect_buffer: Vec::new(),
            source_dirty_rects_scratch: Vec::new(),
            region_dirty_rects_scratch: Vec::new(),
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
        self.cached_src_desc = None;
        self.invalidate_region_pipeline();
        self.dxgi_rect_buffer.clear();
        self.source_dirty_rects_scratch.clear();
        self.region_dirty_rects_scratch.clear();
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
        let staging = surface::ensure_staging_texture(
            &self.device,
            &mut slot.staging,
            region_desc,
            StagingSampleDesc::SingleSample,
            "failed to create region staging texture",
        )?;

        let key = (region_desc.Width, region_desc.Height, region_desc.Format);
        if slot.staging_key != Some(key) || slot.staging_resource.is_none() {
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
        source_resource: &ID3D11Resource,
        blit: CaptureBlitRegion,
        can_use_dirty_gpu_copy: bool,
    ) -> CaptureResult<()> {
        let slot = &self.region_slots[slot_idx];
        let staging_resource = slot.staging_resource.as_ref().ok_or_else(|| {
            CaptureError::Platform(anyhow::anyhow!(
                "DXGI region slot missing staging resource after initialization"
            ))
        })?;

        let mut used_dirty_copy = false;
        if can_use_dirty_gpu_copy && region_dirty_gpu_copy_enabled() && slot.dirty_copy_preferred {
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

        if destination_has_history && sample.is_duplicate {
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
            && (slot.dirty_copy_preferred
                || should_use_dirty_copy(&slot.dirty_rects, source_desc.Width, source_desc.Height));

        if use_dirty_copy {
            match surface::map_staging_dirty_rects_to_frame_with_offset(
                &self.context,
                staging,
                Some(staging_resource),
                &source_desc,
                out,
                &slot.dirty_rects,
                blit.dst_x,
                blit.dst_y,
                slot.hdr_to_sdr,
                "failed to map DXGI region staging texture (dirty regions)",
            ) {
                Ok(converted) if converted > 0 => Ok(sample),
                Ok(_) | Err(_) => {
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
                    Ok(sample)
                }
            }
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
            Ok(sample)
        }
    }

    fn capture_region_into(
        &mut self,
        blit: CaptureBlitRegion,
        destination: &mut Frame,
        destination_has_history: bool,
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

            let (region_dirty_available, region_unchanged) = if source_is_duplicate {
                region_dirty_rects.clear();
                (true, true)
            } else {
                let source_dirty_available = extract_dirty_rects(
                    &self.duplication,
                    &frame_info,
                    &mut self.dxgi_rect_buffer,
                    &mut self.source_dirty_rects_scratch,
                );
                let region_dirty_available = if source_dirty_available {
                    extract_region_dirty_rects(
                        &self.source_dirty_rects_scratch,
                        effective_desc.Width,
                        effective_desc.Height,
                        blit,
                        &mut region_dirty_rects,
                    )
                } else {
                    region_dirty_rects.clear();
                    false
                };

                if region_dirty_available {
                    normalize_dirty_rects_in_place(
                        &mut region_dirty_rects,
                        blit.width,
                        blit.height,
                    );
                }

                (
                    region_dirty_available,
                    region_dirty_available && region_dirty_rects.is_empty(),
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

            let write_slot = if self.capture_mode == CaptureMode::ScreenRecording {
                self.region_next_write_slot % DXGI_REGION_STAGING_SLOTS
            } else {
                0
            };
            let read_slot = if self.capture_mode == CaptureMode::ScreenRecording {
                self.region_pending_slot.unwrap_or(write_slot)
            } else {
                write_slot
            };

            let skip_submit_copy = self.capture_mode == CaptureMode::ScreenRecording
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
                slot.dirty_copy_preferred = false;
                slot.dirty_rects.clear();
                slot.populated = true;
                slot_idx
            } else {
                self.ensure_region_slot(write_slot, &region_desc)?;
                // Dirty GPU copy needs the slot to already hold the previous frame.
                // The recording pipeline rotates slots, so only the single-slot
                // screenshot path can safely preserve unchanged pixels this way.
                let can_use_dirty_gpu_copy =
                    destination_has_history && self.capture_mode != CaptureMode::ScreenRecording;
                {
                    let slot = &mut self.region_slots[write_slot];
                    slot.capture_time = Some(capture_time);
                    slot.present_time_qpc = source_present_time_qpc;
                    slot.is_duplicate = source_is_duplicate || region_unchanged;
                    slot.hdr_to_sdr = effective_hdr;
                    slot.source_desc = Some(region_desc);
                    slot.dirty_mode_available = region_dirty_available;
                    slot.dirty_rects.clear();
                    slot.dirty_rects.extend_from_slice(&region_dirty_rects);
                    slot.dirty_copy_preferred = can_use_dirty_gpu_copy
                        && region_dirty_available
                        && should_use_dirty_gpu_copy(
                            &slot.dirty_rects,
                            region_desc.Width,
                            region_desc.Height,
                        );
                    slot.populated = true;
                }

                let source_resource: ID3D11Resource = effective_source
                    .cast()
                    .context("failed to cast region source texture to ID3D11Resource")
                    .map_err(CaptureError::Platform)?;
                self.copy_region_source_to_slot(
                    write_slot,
                    &source_resource,
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

            if self.capture_mode == CaptureMode::ScreenRecording {
                if !skip_submit_copy {
                    self.region_pending_slot = Some(write_slot);
                    self.region_next_write_slot = (write_slot + 1) % DXGI_REGION_STAGING_SLOTS;
                }
            } else {
                self.region_pending_slot = None;
                self.region_next_write_slot = 0;
            }

            Ok(sample)
        })();

        region_dirty_rects.clear();
        self.region_dirty_rects_scratch = region_dirty_rects;

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

        let duplicate_fastpath = duplicate_dirty_fastpath_enabled();
        // Duplicate frames have no new desktop damage. Skip the COM metadata
        // query on this fast path.
        let source_unchanged = if frame.metadata.is_duplicate {
            frame.metadata.dirty_rects.clear();
            duplicate_fastpath
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

        // Ensure staging ring has matching textures.
        self.staging_ring
            .ensure_slots(&self.device, &effective_desc)?;
        if self.capture_mode == CaptureMode::ScreenRecording {
            let frame_pixels_unchanged = frame.metadata.is_duplicate || source_unchanged;
            let pending_slot_compatible = self.pending_desc.as_ref().is_some_and(|desc| {
                desc.Width == effective_desc.Width
                    && desc.Height == effective_desc.Height
                    && desc.Format == effective_desc.Format
            }) && self.pending_hdr == effective_hdr;
            let skip_submit_copy = should_skip_screenrecord_submit_copy(
                duplicate_fastpath,
                pending_slot_compatible,
                frame.metadata.is_duplicate,
                source_unchanged,
            );

            let mut read_slot = self.staging_ring.latest_write_slot();
            let mut read_desc = effective_desc;
            let mut read_hdr = effective_hdr;
            let mut read_is_duplicate = frame_pixels_unchanged;
            let mut read_dirty_rects: &[DirtyRect] = &[];

            if skip_submit_copy {
                if let Some(prev_desc) = self.pending_desc.as_ref() {
                    read_desc = *prev_desc;
                    read_hdr = self.pending_hdr;
                    // The pending slot may still contain new pixels that
                    // haven't been consumed yet (pipeline catch-up after a
                    // non-duplicate frame). Respect its duplicate state.
                    read_is_duplicate = self.pending_is_duplicate;
                    read_dirty_rects = &self.pending_dirty_rects;
                }
            } else {
                let submitted_read_slot = self
                    .staging_ring
                    .submit_copy(&self.context, &effective_source);
                if let (Some(slot), Some(prev_desc)) =
                    (submitted_read_slot, self.pending_desc.as_ref())
                {
                    // Read back the previous slot while the next copy is in flight.
                    read_slot = slot;
                    read_desc = *prev_desc;
                    read_hdr = self.pending_hdr;
                    read_is_duplicate = self.pending_is_duplicate;
                    read_dirty_rects = &self.pending_dirty_rects;
                } else {
                    // Bootstrap/desync path: read the freshly submitted slot.
                    read_slot = self.staging_ring.latest_write_slot();
                    read_desc = effective_desc;
                    read_hdr = effective_hdr;
                    read_is_duplicate = frame_pixels_unchanged;
                }
            }

            let output_matches_source = has_frame_history
                && frame.width() == read_desc.Width
                && frame.height() == read_desc.Height;
            let skip_readback = output_matches_source && read_is_duplicate;
            let use_dirty_copy = output_matches_source
                && !skip_readback
                && !read_dirty_rects.is_empty()
                && should_use_dirty_copy(read_dirty_rects, read_desc.Width, read_desc.Height);
            self.staging_ring.read_slot_with_strategy(
                &self.context,
                read_slot,
                &read_desc,
                &mut frame,
                read_hdr,
                read_dirty_rects,
                use_dirty_copy,
                skip_readback,
            )?;

            if skip_submit_copy {
                // We intentionally kept the previous pending slot alive;
                // keep metadata aligned with that slot's unchanged contents.
                self.pending_is_duplicate = true;
                self.pending_dirty_rects.clear();
            } else {
                self.pending_desc = Some(effective_desc);
                self.pending_hdr = effective_hdr;
                self.pending_is_duplicate = frame_pixels_unchanged;
                self.pending_dirty_rects.clear();
                if !frame_pixels_unchanged {
                    self.pending_dirty_rects
                        .extend_from_slice(&frame.metadata.dirty_rects);
                    normalize_dirty_rects_in_place(
                        &mut self.pending_dirty_rects,
                        effective_desc.Width,
                        effective_desc.Height,
                    );
                }
            }
        } else {
            // Screenshot mode avoids recording-only buffering.
            self.pending_desc = None;
            self.pending_hdr = None;
            self.pending_is_duplicate = false;
            self.pending_dirty_rects.clear();
            self.staging_ring.reset_pipeline();
            self.staging_ring.copy_and_read(
                &self.context,
                &effective_source,
                &effective_desc,
                &mut frame,
                effective_hdr,
            )?;
        }
        unsafe {
            self.duplication.ReleaseFrame().ok();
        }

        self.needs_presented_first_frame = false;
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
                    // We can't keep *this* frame (we're returning it), but
                    // we pre-allocate a spare with matching capacity.
                    let spare = Frame::empty();
                    self.output.spare_frame = Some(spare);
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
        let result = self
            .output
            .capture_region_into(blit, destination, destination_has_history);
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
                self.output
                    .capture_region_into(blit, destination, destination_has_history)
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

        Ok(Self {
            hwnd: SendHwnd(hwnd),
            resolver,
            _com: com,
            output,
            current_hmon: SendHmon(hmon),
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
}

impl crate::backend::MonitorCapturer for WindowsDxgiWindowCapturer {
    fn capture(&mut self, reuse: Option<Frame>) -> CaptureResult<Frame> {
        let hwnd = self.hwnd.0;

        // Validate the window is still alive and visible.
        if !unsafe { IsWindow(hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget(
                "window no longer exists".into(),
            ));
        }
        if unsafe { IsIconic(hwnd) }.as_bool() {
            return Err(CaptureError::InvalidTarget("window is minimized".into()));
        }

        // Get current window bounds.
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

        // Check if the window moved to a different monitor.
        let hmon = monitor_from_window(hwnd).ok_or_else(|| {
            CaptureError::BackendUnavailable("window is not on any monitor".into())
        })?;

        if SendHmon(hmon) != self.current_hmon {
            self.reinit_for_monitor(hmon)?;
        }

        let mon_rect = monitor_rect(self.current_hmon.0)?;
        let blit = Self::window_blit_on_monitor(&mon_rect, &win_rect)?;

        let mut frame = reuse.unwrap_or_else(Frame::empty);
        let destination_has_history = frame.width() == blit.width
            && frame.height() == blit.height
            && !frame.as_rgba_bytes().is_empty();
        frame.ensure_rgba_capacity(blit.width, blit.height)?;
        frame.reset_metadata();

        let sample =
            match self
                .output
                .capture_region_into(blit, &mut frame, destination_has_history)
            {
                Ok(sample) => sample,
                Err(CaptureError::MonitorLost) | Err(CaptureError::AccessLost) => {
                    self.reinit_for_monitor(hmon)?;
                    self.output
                        .capture_region_into(blit, &mut frame, destination_has_history)?
                }
                Err(e) => return Err(e),
            };

        frame.metadata.capture_time = sample.capture_time;
        frame.metadata.present_time_qpc = sample.present_time_qpc;
        frame.metadata.is_duplicate = sample.is_duplicate;
        Ok(frame)
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
    fn region_dirty_rects_intersect_and_rebase() {
        let source = vec![
            DirtyRect {
                x: 10,
                y: 10,
                width: 30,
                height: 20,
            },
            DirtyRect {
                x: 60,
                y: 40,
                width: 20,
                height: 20,
            },
        ];
        let blit = CaptureBlitRegion {
            src_x: 20,
            src_y: 15,
            width: 50,
            height: 40,
            dst_x: 0,
            dst_y: 0,
        };
        let mut out = Vec::new();
        assert!(extract_region_dirty_rects(
            &source, 1920, 1080, blit, &mut out
        ));
        assert_eq!(
            out,
            vec![
                DirtyRect {
                    x: 0,
                    y: 0,
                    width: 20,
                    height: 15,
                },
                DirtyRect {
                    x: 40,
                    y: 25,
                    width: 10,
                    height: 15,
                },
            ]
        );
    }

    #[test]
    fn region_dirty_rects_detect_unchanged_region() {
        let source = vec![DirtyRect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        }];
        let blit = CaptureBlitRegion {
            src_x: 100,
            src_y: 100,
            width: 80,
            height: 60,
            dst_x: 0,
            dst_y: 0,
        };
        let mut out = Vec::new();
        assert!(extract_region_dirty_rects(
            &source, 1920, 1080, blit, &mut out
        ));
        assert!(out.is_empty());
    }

    #[test]
    fn region_dirty_rects_reject_invalid_blit() {
        let source = vec![DirtyRect {
            x: 100,
            y: 100,
            width: 30,
            height: 30,
        }];
        let blit = CaptureBlitRegion {
            src_x: 3000,
            src_y: 2000,
            width: 40,
            height: 40,
            dst_x: 0,
            dst_y: 0,
        };
        let mut out = Vec::new();
        assert!(!extract_region_dirty_rects(
            &source, 1920, 1080, blit, &mut out
        ));
        assert!(out.is_empty());
    }
}
