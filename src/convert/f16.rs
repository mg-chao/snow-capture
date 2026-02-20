use super::HdrToSdrParams;
use half::f16;
use std::sync::OnceLock;

/// Convert a linear-light value in [0, 1] to an sRGB-encoded byte in [0, 255].
///
/// Implements the sRGB electro-optical transfer function (EOTF⁻¹) defined in
/// IEC 61966-2-1:1999, Section 4.7:
///
///   - Linear segment:  C_srgb = 12.92 · C_linear          when C_linear ≤ 0.0031308
///   - Gamma segment:   C_srgb = 1.055 · C_linear^(1/2.4) − 0.055   otherwise
///
/// The threshold 0.0031308 and the constants 12.92, 1.055, 0.055, and the
/// exponent 1/2.4 are all specified by the standard to ensure a smooth
/// transition between the two segments at the junction point.
pub(crate) fn linear_to_srgb_u8(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Constants for the SMPTE ST 2084 Perceptual Quantizer (PQ) transfer function.
///
/// Defined by SMPTE ST 2084:2014, the PQ EOTF maps a non-linear signal value
/// V to absolute luminance L (in cd/m²) as:
///
///   Y  = max(V^(1/m2) − c1, 0) / (c2 − c3 · V^(1/m2))
///   L  = 10000 · Y^(1/m1)
///
/// The inverse (OETF) used by `linear_to_st2084` is:
///
///   Y  = (L / 10000)^m1
///   V  = ((c1 + c2 · Y) / (1 + c3 · Y))^m2
///
/// The constants below are the exact rational values from the spec, expressed
/// as their f32 approximations:
///
///   m1 = 2610 / 16384       = 0.1593017578125
///   m2 = 2523 / 4096 × 128  = 78.84375
///   c1 = 3424 / 4096         = 0.8359375          (also: c3 − c2 + 1)
///   c2 = 2413 / 4096 × 32   = 18.8515625
///   c3 = 2392 / 4096 × 32   = 18.6875
const PQ_M1: f32 = 0.159_301_758;
const PQ_M2: f32 = 78.843_75;
const PQ_C1: f32 = 0.835_937_5;
const PQ_C2: f32 = 18.851_562_5;
const PQ_C3: f32 = 18.687_5;

/// Working reference luminance used to normalize PQ input in this pipeline.
/// ST 2084 itself is absolute (0..10,000 cd/m²); this code chooses
/// 1.0 = 1000 nits internally, so the PQ helpers operate on
/// `L / HDR_NITS_REFERENCE`.
const HDR_NITS_REFERENCE: f32 = 1000.0;

/// Encode a normalised linear luminance (1.0 = `HDR_NITS_REFERENCE` nits)
/// into a PQ non-linear signal value using the SMPTE ST 2084 EOTF⁻¹.
///
/// See SMPTE ST 2084:2014, Section 5.1 (Equation 1).
fn linear_to_st2084(v: f32) -> f32 {
    let p = v.max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p)).powf(PQ_M2)
}

/// Decode a PQ non-linear signal value back to normalised linear luminance
/// using the SMPTE ST 2084 EOTF.
///
/// See SMPTE ST 2084:2014, Section 5.2 (Equation 2).
fn st2084_to_linear(v: f32) -> f32 {
    let p = v.max(0.0).powf(1.0 / PQ_M2);
    let numerator = (p - PQ_C1).max(0.0);
    let denominator = PQ_C2 - PQ_C3 * p;
    (numerator / denominator).powf(1.0 / PQ_M1)
}

/// Step 1 of the HDR→SDR tonemap: white-point adjustment.
///
/// The HDR surface is authored with a "paper white" reference level
/// (`hdr_paper_white_nits`), while the SDR display expects a different
/// white level (`sdr_white_level_nits`).  Dividing by their ratio
/// re-normalises the scene so that SDR-range content appears at the
/// correct brightness on the target display.
///
/// This is the inverse of the SDR white-level boost used by Windows
/// Advanced Color composition. On Windows this level is exposed via
/// `DISPLAYCONFIG_SDR_WHITE_LEVEL`, converted to nits as
/// `SDRWhiteLevel * 80 / 1000`.
#[inline(always)]
fn adjust_hdr_whites(rgb: &mut [f32; 3], params: HdrToSdrParams) {
    let white_adjust = (params.sdr_white_level_nits / params.hdr_paper_white_nits).max(0.01);
    rgb[0] /= white_adjust;
    rgb[1] /= white_adjust;
    rgb[2] /= white_adjust;
}

/// Step 2 of the HDR→SDR tonemap: channel-max peak limiting.
///
/// The brightest channel is encoded with SMPTE ST 2084, clamped to the
/// configured peak (`hdr_maximum_nits`), decoded back to linear, and the
/// resulting ratio is applied uniformly to all channels.  That preserves
/// hue/saturation while limiting peak luminance.
///
/// Because `linear_to_st2084` and `st2084_to_linear` are monotonic inverses,
/// this is equivalent to clamping maxRGB in linear space and scaling all
/// channels by the same factor.  This aligns with channel-max guidance in
/// ITU-R BT.2408-7 (2023, Annex 5).  BT.2408 also notes that maxRGB-style
/// limiting can introduce perceptual artifacts in some scenes; this path
/// intentionally favors a simple hue-preserving limiter.
#[inline(always)]
fn adjust_hdr_maximum_nits(rgb: &mut [f32; 3], params: HdrToSdrParams) {
    let color_max = rgb[0].max(rgb[1]).max(rgb[2]);
    if color_max <= 1e-6 {
        return;
    }

    let color_2084 = linear_to_st2084(color_max);
    let max_2084 = linear_to_st2084(params.hdr_maximum_nits / HDR_NITS_REFERENCE);
    let limited_linear = st2084_to_linear(color_2084.min(max_2084));
    let scale = limited_linear / color_max;
    rgb[0] *= scale;
    rgb[1] *= scale;
    rgb[2] *= scale;
}

/// 64 KB lookup table mapping every possible IEEE 754 binary16 bit pattern
/// (0x0000..0xFFFF) to its sRGB-encoded u8 value.
///
/// For each 16-bit index `i`, the table stores:
///   `linear_to_srgb_u8(f16::from_bits(i).to_f32())`
///
/// This trades memory for speed: a single table lookup replaces the
/// per-channel `powf(1/2.4)` call in the scalar F16→sRGB path, turning
/// the conversion into three byte loads per pixel (plus one for alpha).
fn f16_to_srgb_lut() -> &'static [u8; 65_536] {
    static LUT: OnceLock<[u8; 65_536]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0u8; 65_536];
        let mut i = 0usize;
        while i < lut.len() {
            let linear = f16::from_bits(i as u16).to_f32();
            lut[i] = linear_to_srgb_u8(linear);
            i += 1;
        }
        lut
    })
}

/// Force-initialize the F16鈫抯RGB LUT so the first HDR capture doesn't
/// pay the ~1-2 ms build cost.
pub(crate) fn warmup_lut() {
    let _ = f16_to_srgb_lut();
}

#[inline(always)]
unsafe fn pack_f16_rgba_to_srgb(src_words: *const u16, lut: &[u8; 65_536]) -> u32 {
    let packed = unsafe { std::ptr::read_unaligned(src_words as *const u64) };
    #[cfg(target_endian = "big")]
    let packed = packed.swap_bytes();

    let r_bits = (packed & 0xFFFF) as usize;
    let g_bits = ((packed >> 16) & 0xFFFF) as usize;
    let b_bits = ((packed >> 32) & 0xFFFF) as usize;
    let a_bits = ((packed >> 48) & 0xFFFF) as usize;

    // Alpha is linear 0..1 鈫?0..255, clamped.
    let a = f16::from_bits(a_bits as u16).to_f32().clamp(0.0, 1.0);
    let a_byte = (a * 255.0 + 0.5) as u32;

    unsafe {
        (*lut.get_unchecked(r_bits) as u32)
            | ((*lut.get_unchecked(g_bits) as u32) << 8)
            | ((*lut.get_unchecked(b_bits) as u32) << 16)
            | (a_byte << 24)
    }
}

pub(crate) unsafe fn convert_f16_rgba_to_srgb_scalar_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    let lut = f16_to_srgb_lut();
    let lut_ptr = lut.as_ptr();
    let mut src_words = src as *const u16;
    let mut dst_px = dst as *mut u32;
    let mut remaining = pixel_count;

    // Software prefetch helper 鈥?prefetch the LUT entries for the next
    // batch of pixels so they're in L1/L2 by the time we need them.
    // Uses NTA hint since the LUT is 64 KB and we don't want to evict
    // other hot data from the cache hierarchy.
    macro_rules! prefetch_lut_entries {
        ($base:expr) => {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
                let packed = std::ptr::read_unaligned($base as *const u64);
                let r_bits = (packed & 0xFFFF) as usize;
                let g_bits = ((packed >> 16) & 0xFFFF) as usize;
                let b_bits = ((packed >> 32) & 0xFFFF) as usize;
                _mm_prefetch(lut_ptr.add(r_bits) as *const i8, _MM_HINT_NTA);
                _mm_prefetch(lut_ptr.add(g_bits) as *const i8, _MM_HINT_NTA);
                _mm_prefetch(lut_ptr.add(b_bits) as *const i8, _MM_HINT_NTA);
            }
        };
    }

    while remaining >= 16 {
        // Prefetch LUT entries for the *next* batch of 16 pixels
        // (4 channels * 2 bytes = 8 bytes per pixel, so 16 pixels ahead
        // is 64 u16 words = 128 bytes of source data).
        if remaining >= 32 {
            unsafe {
                prefetch_lut_entries!(src_words.add(64));
                prefetch_lut_entries!(src_words.add(68));
                prefetch_lut_entries!(src_words.add(72));
                prefetch_lut_entries!(src_words.add(76));
            }
        }

        unsafe {
            let c0 = pack_f16_rgba_to_srgb(src_words, lut);
            let c1 = pack_f16_rgba_to_srgb(src_words.add(4), lut);
            let c2 = pack_f16_rgba_to_srgb(src_words.add(8), lut);
            let c3 = pack_f16_rgba_to_srgb(src_words.add(12), lut);
            let c4 = pack_f16_rgba_to_srgb(src_words.add(16), lut);
            let c5 = pack_f16_rgba_to_srgb(src_words.add(20), lut);
            let c6 = pack_f16_rgba_to_srgb(src_words.add(24), lut);
            let c7 = pack_f16_rgba_to_srgb(src_words.add(28), lut);
            let c8 = pack_f16_rgba_to_srgb(src_words.add(32), lut);
            let c9 = pack_f16_rgba_to_srgb(src_words.add(36), lut);
            let c10 = pack_f16_rgba_to_srgb(src_words.add(40), lut);
            let c11 = pack_f16_rgba_to_srgb(src_words.add(44), lut);
            let c12 = pack_f16_rgba_to_srgb(src_words.add(48), lut);
            let c13 = pack_f16_rgba_to_srgb(src_words.add(52), lut);
            let c14 = pack_f16_rgba_to_srgb(src_words.add(56), lut);
            let c15 = pack_f16_rgba_to_srgb(src_words.add(60), lut);

            std::ptr::write_unaligned(dst_px, c0);
            std::ptr::write_unaligned(dst_px.add(1), c1);
            std::ptr::write_unaligned(dst_px.add(2), c2);
            std::ptr::write_unaligned(dst_px.add(3), c3);
            std::ptr::write_unaligned(dst_px.add(4), c4);
            std::ptr::write_unaligned(dst_px.add(5), c5);
            std::ptr::write_unaligned(dst_px.add(6), c6);
            std::ptr::write_unaligned(dst_px.add(7), c7);
            std::ptr::write_unaligned(dst_px.add(8), c8);
            std::ptr::write_unaligned(dst_px.add(9), c9);
            std::ptr::write_unaligned(dst_px.add(10), c10);
            std::ptr::write_unaligned(dst_px.add(11), c11);
            std::ptr::write_unaligned(dst_px.add(12), c12);
            std::ptr::write_unaligned(dst_px.add(13), c13);
            std::ptr::write_unaligned(dst_px.add(14), c14);
            std::ptr::write_unaligned(dst_px.add(15), c15);
        }

        src_words = unsafe { src_words.add(64) };
        dst_px = unsafe { dst_px.add(16) };
        remaining -= 16;
    }

    while remaining >= 8 {
        unsafe {
            let c0 = pack_f16_rgba_to_srgb(src_words, lut);
            let c1 = pack_f16_rgba_to_srgb(src_words.add(4), lut);
            let c2 = pack_f16_rgba_to_srgb(src_words.add(8), lut);
            let c3 = pack_f16_rgba_to_srgb(src_words.add(12), lut);
            let c4 = pack_f16_rgba_to_srgb(src_words.add(16), lut);
            let c5 = pack_f16_rgba_to_srgb(src_words.add(20), lut);
            let c6 = pack_f16_rgba_to_srgb(src_words.add(24), lut);
            let c7 = pack_f16_rgba_to_srgb(src_words.add(28), lut);

            std::ptr::write_unaligned(dst_px, c0);
            std::ptr::write_unaligned(dst_px.add(1), c1);
            std::ptr::write_unaligned(dst_px.add(2), c2);
            std::ptr::write_unaligned(dst_px.add(3), c3);
            std::ptr::write_unaligned(dst_px.add(4), c4);
            std::ptr::write_unaligned(dst_px.add(5), c5);
            std::ptr::write_unaligned(dst_px.add(6), c6);
            std::ptr::write_unaligned(dst_px.add(7), c7);
        }

        src_words = unsafe { src_words.add(32) };
        dst_px = unsafe { dst_px.add(8) };
        remaining -= 8;
    }

    while remaining != 0 {
        unsafe {
            let color = pack_f16_rgba_to_srgb(src_words, lut);
            std::ptr::write_unaligned(dst_px, color);
        }

        src_words = unsafe { src_words.add(4) };
        dst_px = unsafe { dst_px.add(1) };
        remaining -= 1;
    }
}

/// Scalar HDR→SDR conversion for a row of RGBA16Float pixels.
///
/// Pipeline per pixel:
///   1. Decode four IEEE 754 half-precision floats (R, G, B, A) to f32.
///   2. White-point adjustment — rescale linear RGB so that the HDR
///      "paper white" level maps to the SDR display's white level
///      (see `adjust_hdr_whites`).
///   3. Peak-luminance limiting — map the brightest channel through the
///      SMPTE ST 2084 PQ curve, clamp to `hdr_maximum_nits`, and scale
///      all channels by the resulting ratio to preserve hue
///      (see `adjust_hdr_maximum_nits`).
///   4. Apply the IEC 61966-2-1 sRGB gamma curve to each channel.
///   5. Quantise to 8-bit RGBA and write out.
///
/// This is the reference (non-SIMD) implementation; the AVX2/AVX-512
/// paths in `simd_x86.rs` perform the same operations in 8-wide lanes.
pub(crate) unsafe fn convert_f16_rgba_to_srgb_hdr_scalar_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    params: HdrToSdrParams,
) {
    let params = params.sanitized();
    let mut src_words = src as *const u16;
    let mut dst_px = dst as *mut u32;
    let mut remaining = pixel_count;

    while remaining != 0 {
        let packed = unsafe { std::ptr::read_unaligned(src_words as *const u64) };
        #[cfg(target_endian = "big")]
        let packed = packed.swap_bytes();

        let r = f16::from_bits((packed & 0xFFFF) as u16).to_f32().max(0.0);
        let g = f16::from_bits(((packed >> 16) & 0xFFFF) as u16)
            .to_f32()
            .max(0.0);
        let b = f16::from_bits(((packed >> 32) & 0xFFFF) as u16)
            .to_f32()
            .max(0.0);
        let a = f16::from_bits(((packed >> 48) & 0xFFFF) as u16)
            .to_f32()
            .clamp(0.0, 1.0);

        let mut rgb = [r, g, b];
        adjust_hdr_whites(&mut rgb, params);
        adjust_hdr_maximum_nits(&mut rgb, params);

        let a_byte = ((a * 255.0 + 0.5) as u32) & 0xFF;
        let color = u32::from(linear_to_srgb_u8(rgb[0]))
            | (u32::from(linear_to_srgb_u8(rgb[1])) << 8)
            | (u32::from(linear_to_srgb_u8(rgb[2])) << 16)
            | (a_byte << 24);
        unsafe { std::ptr::write_unaligned(dst_px, color) };

        src_words = unsafe { src_words.add(4) };
        dst_px = unsafe { dst_px.add(1) };
        remaining -= 1;
    }
}
