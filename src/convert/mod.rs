mod f16;
mod parallel;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod simd_x86;

use parallel::{install_conversion_pool, parallel_chunk_pixels, should_parallelize};
use std::sync::OnceLock;

/// Pre-initialize expensive one-time resources (rayon thread pool, F16 LUT,
/// SIMD kernel selection) so the first capture doesn't pay the cost.
/// Safe to call multiple times — only the first call does real work.
pub fn warmup() {
    // Thread pool
    parallel::warmup_pool(CONVERSION_PARALLEL_MAX_WORKERS);
    // F16→sRGB lookup table (64 KB, ~1-2 ms to build)
    f16::warmup_lut();
    // Force kernel selection OnceLocks
    let _ = bgra_kernel();
    let _ = bgra_kernel_nt();
    let _ = f16_kernel();
    let _ = f16_kernel_nt();
    let _ = f16_hdr_kernel();
}

#[inline]
pub(crate) fn with_conversion_pool<F>(max_workers: usize, job: F)
where
    F: FnOnce() + Send,
{
    install_conversion_pool(max_workers, job);
}

#[inline(always)]
pub(crate) fn should_parallelize_work(
    pixel_count: usize,
    min_pixels: usize,
    min_chunk_pixels: usize,
    max_workers: usize,
) -> bool {
    should_parallelize(pixel_count, min_pixels, min_chunk_pixels, max_workers)
}

const BGRA_PARALLEL_MIN_PIXELS: usize = 524_288;
const BGRA_PARALLEL_MIN_CHUNK_PIXELS: usize = 131_072;
const BGRA_PARALLEL_MAX_WORKERS: usize = 9;

/// Lower threshold for the non-temporal path used where `src != dst` is
/// guaranteed (GDI capture, DXGI staging→frame).  256K pixels lets us
/// parallelise earlier while still keeping chunks large enough to
/// amortise rayon overhead.
const BGRA_NT_PARALLEL_MIN_PIXELS: usize = 262_144;
const BGRA_NT_PARALLEL_MIN_CHUNK_PIXELS: usize = 65_536;

const F16_PARALLEL_MIN_PIXELS: usize = 262_144;
const F16_PARALLEL_MIN_CHUNK_PIXELS: usize = 65_536;
const F16_PARALLEL_MAX_WORKERS: usize = 9;
const CONVERSION_PARALLEL_MAX_WORKERS: usize =
    if BGRA_PARALLEL_MAX_WORKERS > F16_PARALLEL_MAX_WORKERS {
        BGRA_PARALLEL_MAX_WORKERS
    } else {
        F16_PARALLEL_MAX_WORKERS
    };

type PixelKernel = unsafe fn(*const u8, *mut u8, usize);
type HdrPixelKernel = unsafe fn(*const u8, *mut u8, usize, HdrToSdrParams);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfacePixelFormat {
    Bgra8,
    Rgba8,
    Rgba16Float,
}

/// Parameters controlling the HDR→SDR tonemapping pipeline.
///
/// The conversion is a two-step process:
///
///   1. White-point normalisation — rescale linear RGB so that the HDR
///      "paper white" level (`hdr_paper_white_nits`) maps to the SDR
///      display's white level (`sdr_white_level_nits`).
///   2. Peak-luminance limiting — encode the brightest channel into the
///      SMPTE ST 2084 PQ curve, clamp to `hdr_maximum_nits`, decode
///      back to linear, and scale all channels uniformly.  Because the
///      PQ helpers are monotonic inverses, this is equivalent to a
///      maxRGB linear clamp with hue-preserving uniform scaling
///      (channel-max style, per ITU-R BT.2408-7 Annex 5 guidance).  This
///      simple limiter can still produce perceptual artifacts in some
///      scenes, as noted in BT.2408.
///
/// After tonemapping, the result is encoded with the IEC 61966-2-1 sRGB
/// gamma curve and quantised to 8-bit RGBA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrToSdrParams {
    /// The luminance (in cd/m²) that the HDR content considers "paper white"
    /// — i.e. the brightness of a white UI element or document background.
    /// In desktop capture this is often an assumption from the caller; when
    /// explicit content metadata is unavailable we fall back to 80 nits.
    /// Typical values: 80 (SDR reference) to 200+ (bright HDR UIs).
    pub hdr_paper_white_nits: f32,
    /// The peak luminance (in cd/m²) of the HDR display used as the
    /// channel-max limiter target in step 2.
    /// Common values: 1000 (entry-level HDR) to 4000+ (high-end panels).
    pub hdr_maximum_nits: f32,
    /// The luminance (in cd/m²) that the SDR output considers "white".
    /// 80 nits is the SDR reference white (IEC 61966-2-1), but in Windows HDR
    /// mode this is user/display dependent (`DISPLAYCONFIG_SDR_WHITE_LEVEL`).
    pub sdr_white_level_nits: f32,
}

impl HdrToSdrParams {
    pub(crate) fn sanitized(self) -> Self {
        let hdr_paper_white_nits = if self.hdr_paper_white_nits.is_finite() {
            self.hdr_paper_white_nits.max(1.0)
        } else {
            80.0
        };
        let hdr_maximum_nits = if self.hdr_maximum_nits.is_finite() {
            self.hdr_maximum_nits.max(1000.0)
        } else {
            1000.0
        };
        let sdr_white_level_nits = if self.sdr_white_level_nits.is_finite() {
            self.sdr_white_level_nits.max(1.0)
        } else {
            80.0
        };
        Self {
            hdr_paper_white_nits,
            hdr_maximum_nits,
            sdr_white_level_nits,
        }
    }
}

impl Default for HdrToSdrParams {
    /// Conservative fallback defaults for HDR→SDR conversion:
    ///   - `hdr_paper_white_nits = 80.0` — the sRGB reference white
    ///     (IEC 61966-2-1); used as a fallback assumption when no explicit
    ///     content paper-white metadata is available.
    ///   - `hdr_maximum_nits = 1000.0` — a typical entry-level HDR
    ///     display peak and common HDR10 mastering target.
    ///   - `sdr_white_level_nits = 80.0` — SDR reference white fallback.
    fn default() -> Self {
        Self {
            hdr_paper_white_nits: 80.0,
            hdr_maximum_nits: 1000.0,
            sdr_white_level_nits: 80.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceConversionOptions {
    pub hdr_to_sdr: Option<HdrToSdrParams>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceLayout {
    pub(crate) src: *const u8,
    pub(crate) src_pitch: usize,
    pub(crate) dst: *mut u8,
    pub(crate) dst_pitch: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl SurfaceLayout {
    pub(crate) const fn new(
        src: *const u8,
        src_pitch: usize,
        dst: *mut u8,
        dst_pitch: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            src,
            src_pitch,
            dst,
            dst_pitch,
            width,
            height,
        }
    }

    fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    fn total_pixels(self) -> usize {
        self.width
            .checked_mul(self.height)
            .expect("surface pixel count overflow")
    }

    fn source_row_bytes(self, src_bytes_per_pixel: usize) -> usize {
        self.width
            .checked_mul(src_bytes_per_pixel)
            .expect("surface row byte overflow")
    }

    fn destination_row_bytes(self) -> usize {
        self.width
            .checked_mul(4)
            .expect("surface row byte overflow")
    }

    fn assert_pitches(self, src_bytes_per_pixel: usize) -> (usize, usize) {
        let src_row_bytes = self.source_row_bytes(src_bytes_per_pixel);
        let dst_row_bytes = self.destination_row_bytes();
        assert!(
            self.src_pitch >= src_row_bytes,
            "source pitch too small: pitch={}, required={}",
            self.src_pitch,
            src_row_bytes
        );
        assert!(
            self.dst_pitch >= dst_row_bytes,
            "destination pitch too small: pitch={}, required={}",
            self.dst_pitch,
            dst_row_bytes
        );
        (src_row_bytes, dst_row_bytes)
    }

    fn is_contiguous(self, src_row_bytes: usize, dst_row_bytes: usize) -> bool {
        self.src_pitch == src_row_bytes && self.dst_pitch == dst_row_bytes
    }

    fn allow_parallel_rows(self) -> bool {
        let maybe_src_total = self.src_pitch.checked_mul(self.height);
        let maybe_dst_total = self.dst_pitch.checked_mul(self.height);
        maybe_src_total
            .zip(maybe_dst_total)
            .map(|(src_total, dst_total)| {
                !parallel::ranges_overlap(self.src, src_total, self.dst, dst_total)
            })
            .unwrap_or(false)
    }
}

#[derive(Clone, Copy)]
struct ParallelConfig {
    min_pixels: usize,
    min_chunk_pixels: usize,
    max_workers: usize,
}

#[derive(Clone, Copy)]
struct SurfaceFormatPlan {
    src_bytes_per_pixel: usize,
    contiguous_kernel: PixelKernel,
    row_kernel: PixelKernel,
    /// Non-temporal row kernel for non-overlapping buffers where the
    /// destination won't be read back before the next capture.
    row_kernel_nt: PixelKernel,
    parallel: ParallelConfig,
}

#[derive(Clone, Copy)]
struct RowChunkPlan {
    chunk_rows: usize,
    chunk_count: usize,
}

pub fn convert_row_to_rgba(
    format: SurfacePixelFormat,
    src_row: &[u8],
    dst_row: &mut [u8],
    pixel_count: usize,
) {
    convert_row_to_rgba_with_options(
        format,
        src_row,
        dst_row,
        pixel_count,
        SurfaceConversionOptions::default(),
    );
}

pub fn convert_row_to_rgba_with_options(
    format: SurfacePixelFormat,
    src_row: &[u8],
    dst_row: &mut [u8],
    pixel_count: usize,
    options: SurfaceConversionOptions,
) {
    match format {
        SurfacePixelFormat::Bgra8 => convert_bgra_to_rgba(src_row, dst_row, pixel_count),
        SurfacePixelFormat::Rgba8 => {
            let byte_count = pixel_count * 4;
            dst_row[..byte_count].copy_from_slice(&src_row[..byte_count]);
        }
        SurfacePixelFormat::Rgba16Float => {
            if let Some(params) = options.hdr_to_sdr {
                let required_src = pixel_count
                    .checked_mul(8)
                    .expect("pixel_count overflow when converting HDR RGBA16F to sRGB");
                let required_dst = pixel_count
                    .checked_mul(4)
                    .expect("pixel_count overflow when converting HDR RGBA16F to sRGB");
                assert!(
                    src_row.len() >= required_src,
                    "RGBA16F source buffer too small: got {}, need at least {} bytes",
                    src_row.len(),
                    required_src
                );
                assert!(
                    dst_row.len() >= required_dst,
                    "RGBA destination buffer too small: got {}, need at least {} bytes",
                    dst_row.len(),
                    required_dst
                );
                unsafe {
                    f16_hdr_kernel()(
                        src_row.as_ptr(),
                        dst_row.as_mut_ptr(),
                        pixel_count,
                        params.sanitized(),
                    );
                }
            } else {
                convert_f16_rgba_to_srgb(src_row, dst_row, pixel_count);
            }
        }
    }
}

fn surface_format_plan(format: SurfacePixelFormat) -> SurfaceFormatPlan {
    match format {
        SurfacePixelFormat::Bgra8 => SurfaceFormatPlan {
            src_bytes_per_pixel: 4,
            contiguous_kernel: convert_bgra_to_rgba_unchecked,
            row_kernel: bgra_kernel(),
            row_kernel_nt: bgra_kernel_nt(),
            parallel: ParallelConfig {
                min_pixels: BGRA_PARALLEL_MIN_PIXELS,
                min_chunk_pixels: BGRA_PARALLEL_MIN_CHUNK_PIXELS,
                max_workers: BGRA_PARALLEL_MAX_WORKERS,
            },
        },
        SurfacePixelFormat::Rgba8 => SurfaceFormatPlan {
            src_bytes_per_pixel: 4,
            contiguous_kernel: memcpy_rgba_unchecked,
            row_kernel: memcpy_rgba_unchecked,
            row_kernel_nt: memcpy_rgba_nt_unchecked,
            parallel: ParallelConfig {
                min_pixels: usize::MAX, // never parallelise a plain memcpy
                min_chunk_pixels: usize::MAX,
                max_workers: 1,
            },
        },
        SurfacePixelFormat::Rgba16Float => SurfaceFormatPlan {
            src_bytes_per_pixel: 8,
            contiguous_kernel: convert_f16_rgba_to_srgb_unchecked,
            row_kernel: f16_kernel(),
            row_kernel_nt: f16_kernel_nt(),
            parallel: ParallelConfig {
                min_pixels: F16_PARALLEL_MIN_PIXELS,
                min_chunk_pixels: F16_PARALLEL_MIN_CHUNK_PIXELS,
                max_workers: F16_PARALLEL_MAX_WORKERS,
            },
        },
    }
}

fn maybe_parallel_row_chunks(
    layout: SurfaceLayout,
    parallel: ParallelConfig,
    total_pixels: usize,
) -> Option<RowChunkPlan> {
    if !layout.allow_parallel_rows()
        || !should_parallelize(
            total_pixels,
            parallel.min_pixels,
            parallel.min_chunk_pixels,
            parallel.max_workers,
        )
    {
        return None;
    }
    row_chunk_plan(
        layout,
        parallel.min_chunk_pixels,
        parallel.max_workers,
        total_pixels,
    )
}

fn row_chunk_plan(
    layout: SurfaceLayout,
    min_chunk_pixels: usize,
    max_workers: usize,
    total_pixels: usize,
) -> Option<RowChunkPlan> {
    let chunk_pixels = parallel_chunk_pixels(total_pixels, min_chunk_pixels, max_workers)?;
    let chunk_rows = (chunk_pixels / layout.width).max(1);
    Some(RowChunkPlan {
        chunk_rows,
        chunk_count: layout.height.div_ceil(chunk_rows),
    })
}

unsafe fn run_rows_serial_with<F>(layout: SurfaceLayout, mut row_fn: F)
where
    F: FnMut(*const u8, *mut u8, usize),
{
    for row in 0..layout.height {
        let src = unsafe { layout.src.add(row * layout.src_pitch) };
        let dst = unsafe { layout.dst.add(row * layout.dst_pitch) };
        row_fn(src, dst, layout.width);
    }
}

unsafe fn run_rows_parallel_with<F>(
    layout: SurfaceLayout,
    chunks: RowChunkPlan,
    max_workers: usize,
    row_fn: F,
) where
    F: Fn(*const u8, *mut u8, usize) + Send + Sync,
{
    let src_addr = layout.src as usize;
    let dst_addr = layout.dst as usize;

    use rayon::prelude::*;
    install_conversion_pool(max_workers, || {
        (0..chunks.chunk_count)
            .into_par_iter()
            .for_each(|chunk_idx| {
                let start_row = chunk_idx * chunks.chunk_rows;
                let rows = (layout.height - start_row).min(chunks.chunk_rows);
                for row_offset in 0..rows {
                    let row = start_row + row_offset;
                    row_fn(
                        (src_addr + row * layout.src_pitch) as *const u8,
                        (dst_addr + row * layout.dst_pitch) as *mut u8,
                        layout.width,
                    );
                }
            });
    });
}

unsafe fn run_rows_serial(layout: SurfaceLayout, kernel: PixelKernel) {
    unsafe {
        run_rows_serial_with(layout, |src, dst, width| kernel(src, dst, width));
    }
}

unsafe fn run_rows_parallel(
    layout: SurfaceLayout,
    kernel: PixelKernel,
    chunks: RowChunkPlan,
    max_workers: usize,
) {
    unsafe {
        run_rows_parallel_with(layout, chunks, max_workers, move |src, dst, width| {
            kernel(src, dst, width);
        });
    }
}

/// Minimum pixel count for non-temporal stores to be beneficial.
/// Below this threshold the destination buffer likely fits in L3 cache
/// and temporal stores are faster (avoids write-combine overhead).
/// ~128K pixels ≈ 512 KB at 4 bytes/pixel — well below typical L3
/// sizes but large enough that write-allocate traffic starts to hurt.
/// Decoupled from the parallelisation thresholds so that medium-
/// resolution captures (e.g. 720p–1080p) still benefit from NT stores
/// in the staging→frame path where src != dst is guaranteed.
const NT_STORE_MIN_PIXELS: usize = 131_072;

#[inline(always)]
fn nt_store_alignment_bytes() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            return 64;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return 32;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            return 16;
        }
        1
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1
    }
}

#[inline(always)]
fn ptr_is_aligned(ptr: *const u8, alignment: usize) -> bool {
    alignment <= 1 || ((ptr as usize) & (alignment - 1)) == 0
}

#[inline(always)]
fn nt_destination_is_aligned(dst: *const u8) -> bool {
    ptr_is_aligned(dst, nt_store_alignment_bytes())
}

#[inline(always)]
fn nt_rows_are_aligned(dst: *const u8, dst_pitch: usize) -> bool {
    let alignment = nt_store_alignment_bytes();
    alignment <= 1 || (ptr_is_aligned(dst, alignment) && dst_pitch.is_multiple_of(alignment))
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct BgraNtKernelSet {
    avx512: Option<PixelKernel>,
    avx2: Option<PixelKernel>,
    ssse3: Option<PixelKernel>,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn bgra_nt_kernel_set() -> &'static BgraNtKernelSet {
    static KERNELS: OnceLock<BgraNtKernelSet> = OnceLock::new();
    KERNELS.get_or_init(|| BgraNtKernelSet {
        avx512: if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            Some(simd_x86::convert_bgra_to_rgba_avx512_nt_unchecked)
        } else {
            None
        },
        avx2: if std::arch::is_x86_feature_detected!("avx2") {
            Some(simd_x86::convert_bgra_to_rgba_avx2_nt_unchecked)
        } else {
            None
        },
        ssse3: if std::arch::is_x86_feature_detected!("ssse3") {
            Some(simd_x86::convert_bgra_to_rgba_ssse3_nt_unchecked)
        } else {
            None
        },
    })
}

#[inline(always)]
fn bgra_nt_kernel_for_destination(dst: *const u8) -> Option<PixelKernel> {
    #[cfg(target_arch = "x86_64")]
    {
        let kernels = bgra_nt_kernel_set();
        if let Some(kernel) = kernels.avx512
            && ptr_is_aligned(dst, 64)
        {
            return Some(kernel);
        }
        if let Some(kernel) = kernels.avx2
            && ptr_is_aligned(dst, 32)
        {
            return Some(kernel);
        }
        if let Some(kernel) = kernels.ssse3
            && ptr_is_aligned(dst, 16)
        {
            return Some(kernel);
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = dst;
        None
    }
}

#[inline(always)]
fn bgra_nt_kernel_for_rows(dst: *const u8, dst_pitch: usize) -> Option<PixelKernel> {
    #[cfg(target_arch = "x86_64")]
    {
        let kernels = bgra_nt_kernel_set();
        if let Some(kernel) = kernels.avx512
            && ptr_is_aligned(dst, 64)
            && dst_pitch.is_multiple_of(64)
        {
            return Some(kernel);
        }
        if let Some(kernel) = kernels.avx2
            && ptr_is_aligned(dst, 32)
            && dst_pitch.is_multiple_of(32)
        {
            return Some(kernel);
        }
        if let Some(kernel) = kernels.ssse3
            && ptr_is_aligned(dst, 16)
            && dst_pitch.is_multiple_of(16)
        {
            return Some(kernel);
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, dst_pitch);
        None
    }
}

pub(crate) unsafe fn convert_surface_to_rgba_unchecked(
    format: SurfacePixelFormat,
    layout: SurfaceLayout,
    options: SurfaceConversionOptions,
) {
    if layout.is_empty() {
        return;
    }

    if format == SurfacePixelFormat::Rgba16Float
        && let Some(params) = options.hdr_to_sdr
    {
        unsafe {
            convert_f16_surface_to_srgb_hdr_unchecked(layout, params.sanitized());
        }
        return;
    }

    let plan = surface_format_plan(format);
    let (src_row_bytes, dst_row_bytes) = layout.assert_pitches(plan.src_bytes_per_pixel);
    let total_pixels = layout.total_pixels();

    if layout.is_contiguous(src_row_bytes, dst_row_bytes) {
        let non_overlapping = !parallel::ranges_overlap(
            layout.src,
            total_pixels * plan.src_bytes_per_pixel,
            layout.dst,
            total_pixels * 4,
        );
        // Only use NT stores when the buffer is large enough that it
        // would thrash the L3 cache with temporal writes.  For smaller
        // surfaces the regular (temporal) path is faster.
        let use_nt = non_overlapping
            && total_pixels >= NT_STORE_MIN_PIXELS
            && match format {
                SurfacePixelFormat::Bgra8 => {
                    bgra_nt_kernel_for_destination(layout.dst as *const u8).is_some()
                }
                _ => nt_destination_is_aligned(layout.dst as *const u8),
            };
        if use_nt {
            if format == SurfacePixelFormat::Bgra8 {
                unsafe {
                    convert_bgra_to_rgba_nt_unchecked(layout.src, layout.dst, total_pixels);
                }
                return;
            }
            if format == SurfacePixelFormat::Rgba16Float {
                unsafe {
                    convert_f16_rgba_to_srgb_nt_unchecked(layout.src, layout.dst, total_pixels);
                }
                return;
            }
        }
        unsafe {
            (plan.contiguous_kernel)(layout.src, layout.dst, total_pixels);
        }
        return;
    }

    if let Some(chunks) = maybe_parallel_row_chunks(layout, plan.parallel, total_pixels) {
        // Use NT row kernel when buffers don't overlap and the surface
        // is large enough — avoids cache pollution from write-allocate
        // traffic in the row-based path (common for padded staging textures).
        let bgra_row_nt_kernel = if format == SurfacePixelFormat::Bgra8 {
            bgra_nt_kernel_for_rows(layout.dst as *const u8, layout.dst_pitch)
        } else {
            None
        };
        let use_nt = layout.allow_parallel_rows()
            && total_pixels >= NT_STORE_MIN_PIXELS
            && match format {
                SurfacePixelFormat::Bgra8 => bgra_row_nt_kernel.is_some(),
                _ => nt_rows_are_aligned(layout.dst as *const u8, layout.dst_pitch),
            };
        let kernel = if use_nt {
            bgra_row_nt_kernel.unwrap_or(plan.row_kernel_nt)
        } else {
            plan.row_kernel
        };
        unsafe {
            run_rows_parallel(layout, kernel, chunks, plan.parallel.max_workers);
        }
        return;
    }

    // Serial row path — use NT stores when non-overlapping and large enough.
    let bgra_row_nt_kernel = if format == SurfacePixelFormat::Bgra8 {
        bgra_nt_kernel_for_rows(layout.dst as *const u8, layout.dst_pitch)
    } else {
        None
    };
    let use_nt = layout.allow_parallel_rows()
        && total_pixels >= NT_STORE_MIN_PIXELS
        && match format {
            SurfacePixelFormat::Bgra8 => bgra_row_nt_kernel.is_some(),
            _ => nt_rows_are_aligned(layout.dst as *const u8, layout.dst_pitch),
        };
    let kernel = if use_nt {
        bgra_row_nt_kernel.unwrap_or(plan.row_kernel_nt)
    } else {
        plan.row_kernel
    };
    unsafe {
        run_rows_serial(layout, kernel);
    }
}

unsafe fn convert_f16_surface_to_srgb_hdr_unchecked(layout: SurfaceLayout, params: HdrToSdrParams) {
    let (src_row_bytes, dst_row_bytes) = layout.assert_pitches(8);
    let total_pixels = layout.total_pixels();
    let kernel = f16_hdr_kernel();

    if layout.is_contiguous(src_row_bytes, dst_row_bytes) {
        unsafe {
            kernel(layout.src, layout.dst, total_pixels, params);
        }
        return;
    }

    let parallel = ParallelConfig {
        min_pixels: F16_PARALLEL_MIN_PIXELS,
        min_chunk_pixels: F16_PARALLEL_MIN_CHUNK_PIXELS,
        max_workers: F16_PARALLEL_MAX_WORKERS,
    };
    if let Some(chunks) = maybe_parallel_row_chunks(layout, parallel, total_pixels) {
        unsafe {
            run_rows_parallel_with(
                layout,
                chunks,
                parallel.max_workers,
                move |src, dst, width| {
                    kernel(src, dst, width, params);
                },
            );
        }
        return;
    }

    unsafe {
        run_rows_serial_with(layout, move |src, dst, width| {
            kernel(src, dst, width, params);
        });
    }
}

/// Passthrough copy for data that is already in RGBA8 layout.
/// Matches the `PixelKernel` signature so it can be used in
/// `SurfaceFormatPlan`.
unsafe fn memcpy_rgba_unchecked(src: *const u8, dst: *mut u8, pixel_count: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, pixel_count * 4);
    }
}

/// Non-temporal passthrough copy for RGBA8 data.  Uses streaming stores
/// to avoid polluting the cache when the destination won't be read back
/// before the next capture (staging→frame path).
unsafe fn memcpy_rgba_nt_unchecked(src: *const u8, dst: *mut u8, pixel_count: usize) {
    if !nt_destination_is_aligned(dst as *const u8) {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, pixel_count * 4);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                memcpy_rgba_nt_avx2(src, dst, pixel_count);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("sse2") {
            unsafe {
                memcpy_rgba_nt_sse2(src, dst, pixel_count);
            }
            return;
        }
    }
    // Fallback: regular copy
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, pixel_count * 4);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn memcpy_rgba_nt_avx2(src: *const u8, dst: *mut u8, pixel_count: usize) {
    use std::arch::x86_64::{__m256i, _mm_sfence, _mm256_loadu_si256, _mm256_stream_si256};
    let total_bytes = pixel_count * 4;
    let mut offset = 0usize;
    while offset + 32 <= total_bytes {
        unsafe {
            let v = _mm256_loadu_si256(src.add(offset) as *const __m256i);
            _mm256_stream_si256(dst.add(offset) as *mut __m256i, v);
        }
        offset += 32;
    }
    unsafe { _mm_sfence() };
    // Copy remaining bytes
    if offset < total_bytes {
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), total_bytes - offset);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn memcpy_rgba_nt_sse2(src: *const u8, dst: *mut u8, pixel_count: usize) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    let total_bytes = pixel_count * 4;
    let mut offset = 0usize;
    while offset + 16 <= total_bytes {
        unsafe {
            let v = _mm_loadu_si128(src.add(offset) as *const __m128i);
            _mm_stream_si128(dst.add(offset) as *mut __m128i, v);
        }
        offset += 16;
    }
    unsafe { _mm_sfence() };
    if offset < total_bytes {
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), total_bytes - offset);
        }
    }
}

pub fn convert_bgra_to_rgba(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    let required = pixel_count
        .checked_mul(4)
        .expect("pixel_count overflow when converting BGRA to RGBA");
    assert!(
        src.len() >= required,
        "BGRA source buffer too small: got {}, need at least {} bytes",
        src.len(),
        required
    );
    assert!(
        dst.len() >= required,
        "RGBA destination buffer too small: got {}, need at least {} bytes",
        dst.len(),
        required
    );
    unsafe {
        convert_bgra_to_rgba_unchecked(src.as_ptr(), dst.as_mut_ptr(), pixel_count);
    }
}

pub fn convert_f16_rgba_to_srgb(src: &[u8], dst: &mut [u8], pixel_count: usize) {
    let required_src = pixel_count
        .checked_mul(8)
        .expect("pixel_count overflow when converting RGBA16F to sRGB");
    let required_dst = pixel_count
        .checked_mul(4)
        .expect("pixel_count overflow when converting RGBA16F to sRGB");
    assert!(
        src.len() >= required_src,
        "RGBA16F source buffer too small: got {}, need at least {} bytes",
        src.len(),
        required_src
    );
    assert!(
        dst.len() >= required_dst,
        "RGBA destination buffer too small: got {}, need at least {} bytes",
        dst.len(),
        required_dst
    );
    unsafe {
        convert_f16_rgba_to_srgb_unchecked(src.as_ptr(), dst.as_mut_ptr(), pixel_count);
    }
}

#[inline(always)]
fn bgra_kernel() -> PixelKernel {
    static KERNEL: OnceLock<PixelKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_bgra_kernel)
}

/// Non-temporal (streaming-store) variant of the BGRA kernel.
/// Falls back to the regular kernel on platforms without NT intrinsics.
#[inline(always)]
fn bgra_kernel_nt() -> PixelKernel {
    static KERNEL: OnceLock<PixelKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_bgra_kernel_nt)
}

/// Best-available F16→sRGB kernel (SIMD when possible, scalar fallback).
#[inline(always)]
fn f16_kernel() -> PixelKernel {
    static KERNEL: OnceLock<PixelKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_f16_kernel)
}

/// Non-temporal (streaming-store) variant of the F16→sRGB kernel.
/// Uses NT stores for the output to avoid cache pollution on large surfaces.
#[inline(always)]
fn f16_kernel_nt() -> PixelKernel {
    static KERNEL: OnceLock<PixelKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_f16_kernel_nt)
}

fn select_f16_kernel() -> PixelKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_avx512_unchecked;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_f16c_unchecked;
        }
    }

    f16::convert_f16_rgba_to_srgb_scalar_unchecked
}

fn select_f16_kernel_nt() -> PixelKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_avx512_nt_unchecked;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_f16c_nt_unchecked;
        }
    }

    // Scalar path has no NT variant; fall back to the regular kernel.
    f16::convert_f16_rgba_to_srgb_scalar_unchecked
}

/// Best-available F16 HDR→sRGB kernel (SIMD when possible, scalar fallback).
#[inline(always)]
fn f16_hdr_kernel() -> HdrPixelKernel {
    static KERNEL: OnceLock<HdrPixelKernel> = OnceLock::new();
    *KERNEL.get_or_init(select_f16_hdr_kernel)
}

fn select_f16_hdr_kernel() -> HdrPixelKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_hdr_avx512_unchecked;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return simd_x86::convert_f16_rgba_to_srgb_hdr_f16c_unchecked;
        }
    }

    f16::convert_f16_rgba_to_srgb_hdr_scalar_unchecked
}

fn select_bgra_kernel() -> PixelKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            return simd_x86::convert_bgra_to_rgba_avx512_unchecked;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return simd_x86::convert_bgra_to_rgba_avx2_unchecked;
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return simd_x86::convert_bgra_to_rgba_ssse3_unchecked;
        }
    }

    scalar::convert_bgra_to_rgba_scalar_unchecked
}

fn select_bgra_kernel_nt() -> PixelKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            return simd_x86::convert_bgra_to_rgba_avx512_nt_unchecked;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return simd_x86::convert_bgra_to_rgba_avx2_nt_unchecked;
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return simd_x86::convert_bgra_to_rgba_ssse3_nt_unchecked;
        }
    }

    // Scalar path has no NT variant; fall back to the regular kernel.
    scalar::convert_bgra_to_rgba_scalar_unchecked
}

pub(crate) unsafe fn convert_bgra_to_rgba_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    if should_parallelize(
        pixel_count,
        BGRA_PARALLEL_MIN_PIXELS,
        BGRA_PARALLEL_MIN_CHUNK_PIXELS,
        BGRA_PARALLEL_MAX_WORKERS,
    ) {
        unsafe {
            convert_bgra_to_rgba_parallel(src, dst, pixel_count);
        }
        return;
    }
    unsafe {
        bgra_kernel()(src, dst, pixel_count);
    }
}

/// Non-temporal BGRA→RGBA conversion optimised for the GDI capture path.
///
/// Differences from `convert_bgra_to_rgba_unchecked`:
///   1. Always uses streaming (NT) stores — the destination buffer will
///      not be read back before the next capture, so polluting the cache
///      with write-allocate traffic is pure waste.
///   2. Uses a lower parallelisation threshold so that 1080p captures
///      (≈ 2 MP) reliably hit the multi-threaded path.
///
/// # Safety
///
/// * `src` and `dst` must not overlap.
/// * Both buffers must be at least `pixel_count * 4` bytes.
pub(crate) unsafe fn convert_bgra_to_rgba_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    let Some(nt_kernel) = bgra_nt_kernel_for_destination(dst as *const u8) else {
        unsafe {
            convert_bgra_to_rgba_unchecked(src, dst, pixel_count);
        }
        return;
    };

    if should_parallelize(
        pixel_count,
        BGRA_NT_PARALLEL_MIN_PIXELS,
        BGRA_NT_PARALLEL_MIN_CHUNK_PIXELS,
        BGRA_PARALLEL_MAX_WORKERS,
    ) {
        unsafe {
            convert_bgra_to_rgba_parallel_inner(
                src,
                dst,
                pixel_count,
                BGRA_NT_PARALLEL_MIN_CHUNK_PIXELS,
                BGRA_PARALLEL_MAX_WORKERS,
                Some(nt_kernel),
            );
        }
        return;
    }
    // Serial path — still use NT stores since src != dst is guaranteed
    // by the caller and the working set far exceeds the cache.
    unsafe {
        nt_kernel(src, dst, pixel_count);
    }
}

pub(crate) unsafe fn convert_f16_rgba_to_srgb_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    if should_parallelize(
        pixel_count,
        F16_PARALLEL_MIN_PIXELS,
        F16_PARALLEL_MIN_CHUNK_PIXELS,
        F16_PARALLEL_MAX_WORKERS,
    ) {
        unsafe {
            convert_f16_rgba_to_srgb_parallel(src, dst, pixel_count);
        }
        return;
    }
    unsafe {
        f16_kernel()(src, dst, pixel_count);
    }
}

/// Non-temporal F16→sRGB conversion for non-overlapping buffers.
///
/// Uses the lower NT parallelisation thresholds since the destination
/// buffer won't be read back before the next capture.
///
/// # Safety
///
/// * `src` and `dst` must not overlap.
/// * `src` must be at least `pixel_count * 8` bytes.
/// * `dst` must be at least `pixel_count * 4` bytes.
pub(crate) unsafe fn convert_f16_rgba_to_srgb_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    if !nt_destination_is_aligned(dst as *const u8) {
        unsafe {
            convert_f16_rgba_to_srgb_unchecked(src, dst, pixel_count);
        }
        return;
    }

    // F16 conversion is compute-heavy enough that NT stores don't help as
    // much as for BGRA (the bottleneck is ALU, not memory bandwidth), but
    // we still benefit from the lower parallelisation thresholds.
    if should_parallelize(
        pixel_count,
        BGRA_NT_PARALLEL_MIN_PIXELS,
        BGRA_NT_PARALLEL_MIN_CHUNK_PIXELS,
        F16_PARALLEL_MAX_WORKERS,
    ) {
        unsafe {
            convert_f16_rgba_to_srgb_parallel(src, dst, pixel_count);
        }
        return;
    }
    unsafe {
        f16_kernel()(src, dst, pixel_count);
    }
}

unsafe fn convert_bgra_to_rgba_parallel(src: *const u8, dst: *mut u8, pixel_count: usize) {
    let nt_kernel = bgra_nt_kernel_for_destination(dst as *const u8);
    unsafe {
        convert_bgra_to_rgba_parallel_inner(
            src,
            dst,
            pixel_count,
            BGRA_PARALLEL_MIN_CHUNK_PIXELS,
            BGRA_PARALLEL_MAX_WORKERS,
            nt_kernel,
        );
    }
}

unsafe fn convert_bgra_to_rgba_parallel_inner(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    min_chunk_pixels: usize,
    max_workers: usize,
    nt_kernel: Option<PixelKernel>,
) {
    let Some(chunk_pixels) = parallel_chunk_pixels(pixel_count, min_chunk_pixels, max_workers)
    else {
        unsafe {
            if let Some(kernel) = nt_kernel {
                kernel(src, dst, pixel_count);
            } else {
                bgra_kernel()(src, dst, pixel_count);
            }
        };
        return;
    };

    let total_bytes = pixel_count
        .checked_mul(4)
        .expect("pixel_count overflow while converting BGRA to RGBA");
    if parallel::ranges_overlap(src, total_bytes, dst, total_bytes) {
        unsafe { bgra_kernel()(src, dst, pixel_count) };
        return;
    }

    let chunk_count = pixel_count.div_ceil(chunk_pixels);
    // Use non-temporal stores in the parallel path — each chunk is large
    // enough that the destination lines won't be read back before the
    // full conversion completes, so bypassing the cache is a net win.
    let kernel = nt_kernel.unwrap_or_else(bgra_kernel);
    let src_addr = src as usize;
    let dst_addr = dst as usize;

    use rayon::prelude::*;
    install_conversion_pool(max_workers, || {
        (0..chunk_count)
            .into_par_iter()
            .for_each(|chunk_idx| unsafe {
                let start = chunk_idx * chunk_pixels;
                let len = (pixel_count - start).min(chunk_pixels);
                kernel(
                    (src_addr + start * 4) as *const u8,
                    (dst_addr + start * 4) as *mut u8,
                    len,
                );
            });
    });
}

unsafe fn convert_f16_rgba_to_srgb_parallel(src: *const u8, dst: *mut u8, pixel_count: usize) {
    let Some(chunk_pixels) = parallel_chunk_pixels(
        pixel_count,
        F16_PARALLEL_MIN_CHUNK_PIXELS,
        F16_PARALLEL_MAX_WORKERS,
    ) else {
        unsafe { f16_kernel()(src, dst, pixel_count) };
        return;
    };

    let src_total = pixel_count.checked_mul(8).expect("pixel_count overflow");
    let dst_total = pixel_count.checked_mul(4).expect("pixel_count overflow");
    if parallel::ranges_overlap(src, src_total, dst, dst_total) {
        unsafe { f16_kernel()(src, dst, pixel_count) };
        return;
    }

    let chunk_count = pixel_count.div_ceil(chunk_pixels);
    let kernel = f16_kernel();
    let src_addr = src as usize;
    let dst_addr = dst as usize;

    use rayon::prelude::*;
    install_conversion_pool(CONVERSION_PARALLEL_MAX_WORKERS, || {
        (0..chunk_count)
            .into_par_iter()
            .for_each(|chunk_idx| unsafe {
                let start = chunk_idx * chunk_pixels;
                let len = (pixel_count - start).min(chunk_pixels);
                kernel(
                    (src_addr + start * 8) as *const u8,
                    (dst_addr + start * 4) as *mut u8,
                    len,
                );
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_nt_fallback_handles_unaligned_destination() {
        let pixel_count = NT_STORE_MIN_PIXELS + 64;
        let mut src = vec![0u8; pixel_count * 4];
        for i in 0..pixel_count {
            let idx = i * 4;
            src[idx] = (i & 0xFF) as u8;
            src[idx + 1] = ((i >> 1) & 0xFF) as u8;
            src[idx + 2] = ((i >> 2) & 0xFF) as u8;
            src[idx + 3] = 0x2A;
        }

        let mut dst_storage = vec![0u8; pixel_count * 4 + 1];
        let dst_ptr = unsafe { dst_storage.as_mut_ptr().add(1) };

        unsafe {
            convert_bgra_to_rgba_nt_unchecked(src.as_ptr(), dst_ptr, pixel_count);
        }

        let dst = unsafe { std::slice::from_raw_parts(dst_ptr, pixel_count * 4) };
        for i in 0..pixel_count {
            let idx = i * 4;
            assert_eq!(dst[idx], src[idx + 2]);
            assert_eq!(dst[idx + 1], src[idx + 1]);
            assert_eq!(dst[idx + 2], src[idx]);
            assert_eq!(dst[idx + 3], 0x2A);
        }
    }
}
