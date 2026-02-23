use super::HdrToSdrParams;
use super::f16::{
    convert_f16_rgba_to_srgb_hdr_scalar_unchecked, convert_f16_rgba_to_srgb_scalar_unchecked,
};
use super::scalar::convert_bgra_to_rgba_scalar_unchecked;

#[inline(always)]
fn nt_prefix_pixels(dst: *mut u8, pixel_count: usize, alignment: usize) -> usize {
    if pixel_count == 0 || alignment <= 1 {
        return 0;
    }
    let misalign = (dst as usize) & (alignment - 1);
    if misalign == 0 {
        return 0;
    }
    let bytes_to_align = alignment - misalign;
    if !bytes_to_align.is_multiple_of(4) {
        return pixel_count;
    }
    (bytes_to_align / 4).min(pixel_count)
}

// ---------------------------------------------------------------------------
// AVX-512
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx512_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx512_bgra_core(src, dst, pixel_count, false, false) }
}

/// Streaming-store variant — uses non-temporal writes to bypass the cache.
/// Caller must ensure `dst` will not be read back immediately (or issue an
/// `_mm_sfence` afterwards).
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx512_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx512_bgra_core(src, dst, pixel_count, true, true) }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx512_nt_nofence_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx512_bgra_core(src, dst, pixel_count, true, false) }
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn avx512_bgra_core(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    nontemporal: bool,
    fence: bool,
) {
    use std::arch::x86_64::{
        __m512i, _MM_HINT_T0, _mm_prefetch, _mm_sfence, _mm512_loadu_si512, _mm512_shuffle_epi8,
        _mm512_storeu_si512, _mm512_stream_si512,
    };

    let shuffle = unsafe {
        let pattern: [i8; 64] = [
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8,
            11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3,
            6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
        ];
        _mm512_loadu_si512(pattern.as_ptr() as *const __m512i)
    };

    macro_rules! store512 {
        ($ptr:expr, $val:expr) => {
            if nontemporal {
                _mm512_stream_si512($ptr as *mut __m512i, $val);
            } else {
                _mm512_storeu_si512($ptr as *mut __m512i, $val);
            }
        };
    }

    let mut x = 0usize;
    if nontemporal {
        let prefix = nt_prefix_pixels(dst, pixel_count, 64);
        if prefix > 0 {
            unsafe {
                convert_bgra_to_rgba_scalar_unchecked(src, dst, prefix);
            }
            x = prefix;
            if x == pixel_count {
                return;
            }
        }
    }

    // Process 128 pixels (8×16) per iteration to better amortise loop
    // overhead and improve instruction-level parallelism.
    while x + 128 <= pixel_count {
        let offset = x * 4;
        // Prefetch source data 1 iteration ahead (512 bytes).
        if x + 256 <= pixel_count {
            unsafe {
                _mm_prefetch(src.add(offset + 512) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 576) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 640) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 704) as *const i8, _MM_HINT_T0);
            }
        }
        let input0 = unsafe { _mm512_loadu_si512(src.add(offset) as *const __m512i) };
        let input1 = unsafe { _mm512_loadu_si512(src.add(offset + 64) as *const __m512i) };
        let input2 = unsafe { _mm512_loadu_si512(src.add(offset + 128) as *const __m512i) };
        let input3 = unsafe { _mm512_loadu_si512(src.add(offset + 192) as *const __m512i) };
        let input4 = unsafe { _mm512_loadu_si512(src.add(offset + 256) as *const __m512i) };
        let input5 = unsafe { _mm512_loadu_si512(src.add(offset + 320) as *const __m512i) };
        let input6 = unsafe { _mm512_loadu_si512(src.add(offset + 384) as *const __m512i) };
        let input7 = unsafe { _mm512_loadu_si512(src.add(offset + 448) as *const __m512i) };
        let output0 = _mm512_shuffle_epi8(input0, shuffle);
        let output1 = _mm512_shuffle_epi8(input1, shuffle);
        let output2 = _mm512_shuffle_epi8(input2, shuffle);
        let output3 = _mm512_shuffle_epi8(input3, shuffle);
        let output4 = _mm512_shuffle_epi8(input4, shuffle);
        let output5 = _mm512_shuffle_epi8(input5, shuffle);
        let output6 = _mm512_shuffle_epi8(input6, shuffle);
        let output7 = _mm512_shuffle_epi8(input7, shuffle);
        unsafe {
            store512!(dst.add(offset), output0);
            store512!(dst.add(offset + 64), output1);
            store512!(dst.add(offset + 128), output2);
            store512!(dst.add(offset + 192), output3);
            store512!(dst.add(offset + 256), output4);
            store512!(dst.add(offset + 320), output5);
            store512!(dst.add(offset + 384), output6);
            store512!(dst.add(offset + 448), output7);
        }
        x += 128;
    }

    while x + 16 <= pixel_count {
        let offset = x * 4;
        let input = unsafe { _mm512_loadu_si512(src.add(offset) as *const __m512i) };
        let output = _mm512_shuffle_epi8(input, shuffle);
        unsafe {
            store512!(dst.add(offset), output);
        }
        x += 16;
    }

    if nontemporal && fence {
        _mm_sfence();
    }

    if x < pixel_count {
        unsafe {
            convert_bgra_to_rgba_scalar_unchecked(src.add(x * 4), dst.add(x * 4), pixel_count - x);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx2_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx2_bgra_core(src, dst, pixel_count, false, false) }
}

/// Streaming-store variant for AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx2_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx2_bgra_core(src, dst, pixel_count, true, true) }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn convert_bgra_to_rgba_avx2_nt_nofence_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { avx2_bgra_core(src, dst, pixel_count, true, false) }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_bgra_core(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    nontemporal: bool,
    fence: bool,
) {
    use std::arch::x86_64::{
        __m256i, _MM_HINT_T0, _mm_prefetch, _mm_sfence, _mm256_loadu_si256, _mm256_setr_epi8,
        _mm256_shuffle_epi8, _mm256_storeu_si256, _mm256_stream_si256,
    };

    let shuffle = _mm256_setr_epi8(
        2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11,
        14, 13, 12, 15,
    );

    macro_rules! store256 {
        ($ptr:expr, $val:expr) => {
            if nontemporal {
                _mm256_stream_si256($ptr as *mut __m256i, $val);
            } else {
                _mm256_storeu_si256($ptr as *mut __m256i, $val);
            }
        };
    }

    let mut x = 0usize;
    if nontemporal {
        let prefix = nt_prefix_pixels(dst, pixel_count, 32);
        if prefix > 0 {
            unsafe {
                convert_bgra_to_rgba_scalar_unchecked(src, dst, prefix);
            }
            x = prefix;
            if x == pixel_count {
                return;
            }
        }
    }
    while x + 32 <= pixel_count {
        let offset = x * 4;
        // Prefetch source data ~4 iterations ahead (512 bytes) to hide
        // memory latency on modern out-of-order cores.
        if x + 128 <= pixel_count {
            unsafe {
                _mm_prefetch(src.add(offset + 256) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 320) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 384) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src.add(offset + 448) as *const i8, _MM_HINT_T0);
            }
        }
        let input0 = unsafe { _mm256_loadu_si256(src.add(offset) as *const __m256i) };
        let input1 = unsafe { _mm256_loadu_si256(src.add(offset + 32) as *const __m256i) };
        let input2 = unsafe { _mm256_loadu_si256(src.add(offset + 64) as *const __m256i) };
        let input3 = unsafe { _mm256_loadu_si256(src.add(offset + 96) as *const __m256i) };
        let output0 = _mm256_shuffle_epi8(input0, shuffle);
        let output1 = _mm256_shuffle_epi8(input1, shuffle);
        let output2 = _mm256_shuffle_epi8(input2, shuffle);
        let output3 = _mm256_shuffle_epi8(input3, shuffle);
        unsafe {
            store256!(dst.add(offset), output0);
            store256!(dst.add(offset + 32), output1);
            store256!(dst.add(offset + 64), output2);
            store256!(dst.add(offset + 96), output3);
        }
        x += 32;
    }

    while x + 8 <= pixel_count {
        let offset = x * 4;
        let input = unsafe { _mm256_loadu_si256(src.add(offset) as *const __m256i) };
        let output = _mm256_shuffle_epi8(input, shuffle);
        unsafe {
            store256!(dst.add(offset), output);
        }
        x += 8;
    }

    if nontemporal && fence {
        _mm_sfence();
    }

    if x < pixel_count {
        unsafe {
            convert_bgra_to_rgba_scalar_unchecked(src.add(x * 4), dst.add(x * 4), pixel_count - x);
        }
    }
}

// ---------------------------------------------------------------------------
// SSSE3
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn convert_bgra_to_rgba_ssse3_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { ssse3_bgra_core(src, dst, pixel_count, false, false) }
}

/// Streaming-store variant for SSSE3.
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn convert_bgra_to_rgba_ssse3_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { ssse3_bgra_core(src, dst, pixel_count, true, true) }
}

#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn convert_bgra_to_rgba_ssse3_nt_nofence_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { ssse3_bgra_core(src, dst, pixel_count, true, false) }
}

#[target_feature(enable = "ssse3")]
unsafe fn ssse3_bgra_core(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    nontemporal: bool,
    fence: bool,
) {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_setr_epi8, _mm_sfence, _mm_shuffle_epi8, _mm_storeu_si128,
        _mm_stream_si128,
    };

    let shuffle = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);

    macro_rules! store128 {
        ($ptr:expr, $val:expr) => {
            if nontemporal {
                _mm_stream_si128($ptr as *mut __m128i, $val);
            } else {
                _mm_storeu_si128($ptr as *mut __m128i, $val);
            }
        };
    }

    let mut x = 0usize;
    if nontemporal {
        let prefix = nt_prefix_pixels(dst, pixel_count, 16);
        if prefix > 0 {
            unsafe {
                convert_bgra_to_rgba_scalar_unchecked(src, dst, prefix);
            }
            x = prefix;
            if x == pixel_count {
                return;
            }
        }
    }
    while x + 16 <= pixel_count {
        let offset = x * 4;
        let input0 = unsafe { _mm_loadu_si128(src.add(offset) as *const __m128i) };
        let input1 = unsafe { _mm_loadu_si128(src.add(offset + 16) as *const __m128i) };
        let input2 = unsafe { _mm_loadu_si128(src.add(offset + 32) as *const __m128i) };
        let input3 = unsafe { _mm_loadu_si128(src.add(offset + 48) as *const __m128i) };
        let output0 = _mm_shuffle_epi8(input0, shuffle);
        let output1 = _mm_shuffle_epi8(input1, shuffle);
        let output2 = _mm_shuffle_epi8(input2, shuffle);
        let output3 = _mm_shuffle_epi8(input3, shuffle);
        unsafe {
            store128!(dst.add(offset), output0);
            store128!(dst.add(offset + 16), output1);
            store128!(dst.add(offset + 32), output2);
            store128!(dst.add(offset + 48), output3);
        }
        x += 16;
    }

    while x + 4 <= pixel_count {
        let offset = x * 4;
        let input = unsafe { _mm_loadu_si128(src.add(offset) as *const __m128i) };
        let output = _mm_shuffle_epi8(input, shuffle);
        unsafe {
            store128!(dst.add(offset), output);
        }
        x += 4;
    }

    if nontemporal && fence {
        _mm_sfence();
    }

    if x < pixel_count {
        unsafe {
            convert_bgra_to_rgba_scalar_unchecked(src.add(x * 4), dst.add(x * 4), pixel_count - x);
        }
    }
}

// ---------------------------------------------------------------------------
// F16->sRGB via AVX2 + F16C
// ---------------------------------------------------------------------------
//
// Uses `vcvtph2ps` (F16C) to bulk-convert half-floats to f32, then applies
// a polynomial sRGB gamma approximation entirely in SIMD, packs to u8, and
// writes out RGBA preserving the source alpha channel.
//
// The sRGB transfer function (IEC 61966-2-1:1999, Section 4.7) is:
//   srgb(x) = 1.055 · x^(1/2.4) − 0.055   for x > 0.0031308
//   srgb(x) = 12.92 · x                     for x ≤ 0.0031308
//
// We approximate x^(1/2.4) via the classic "fast-pow" IEEE 754 bit trick:
//
//   reinterpret_as_int(x^p) ≈ p · reinterpret_as_int(x) + 0x3F800000 · (1 − p)
//
// This exploits the fact that the integer representation of an IEEE 754
// float is roughly proportional to its log₂.  The technique is described
// in:
//   - Schraudolph, N. N. (1999). "A Fast, Compact Approximation of the
//     Exponential Function." Neural Computation, 11(4), 853–862.
//
// This is a speed/accuracy tradeoff; exact error depends on the input
// distribution and platform math behavior.

#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_f16c_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe {
        convert_f16_rgba_to_srgb_f16c_inner(src, dst, pixel_count, false, false);
    }
}

/// Non-temporal store variant — uses streaming writes to bypass the cache.
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_f16c_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe {
        convert_f16_rgba_to_srgb_f16c_inner(src, dst, pixel_count, true, true);
    }
}

#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_f16c_nt_nofence_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe {
        convert_f16_rgba_to_srgb_f16c_inner(src, dst, pixel_count, true, false);
    }
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn convert_f16_rgba_to_srgb_f16c_inner(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    nontemporal: bool,
    fence: bool,
) {
    use std::arch::x86_64::*;

    unsafe {
        // sRGB constants
        let threshold = _mm256_set1_ps(0.003_130_8_f32);
        let linear_scale = _mm256_set1_ps(12.92);
        let a = _mm256_set1_ps(1.055);
        let b = _mm256_set1_ps(-0.055);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let scale255 = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);

        // Fast pow constants for x^(1/2.4):
        //   as_int(x^p) ~ p * as_int(x) + 0x3F800000 * (1 - p)
        let pow_scale = _mm256_set1_ps(1.0 / 2.4);
        let pow_bias_f = 0x3F80_0000u32 as f32;
        let pow_offset = _mm256_set1_ps(pow_bias_f * (1.0 - 1.0 / 2.4));

        let alpha_scale = _mm256_set1_ps(255.0);
        let alpha_half = _mm256_set1_ps(0.5);

        // Permutation index for AoS->SoA transpose final step
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        let mut src_ptr = src as *const u16;
        let mut dst_ptr = dst as *mut u8;
        let mut remaining = pixel_count;

        // Inline helper: approximate sRGB gamma for 8 floats in [0,1]
        // Returns 8 floats in [0,255]
        macro_rules! srgb_gamma_ps {
            ($v:expr) => {{
                // Clamp to [0,1]
                let clamped = _mm256_min_ps(_mm256_max_ps($v, zero), one);

                // Linear segment: 12.92 * x
                let lin = _mm256_mul_ps(clamped, linear_scale);

                // Power segment: 1.055 * x^(1/2.4) - 0.055
                // Fast x^(1/2.4) via integer bit trick
                let xi = _mm256_castps_si256(clamped);
                let pow_i = _mm256_add_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(xi), pow_scale)),
                    _mm256_cvtps_epi32(pow_offset),
                );
                let pow_approx = _mm256_castsi256_ps(pow_i);
                let gamma = _mm256_add_ps(_mm256_mul_ps(a, pow_approx), b);

                // Select: linear for small values, gamma for the rest
                let mask = _mm256_cmp_ps(clamped, threshold, _CMP_GT_OQ);
                let result = _mm256_blendv_ps(lin, gamma, mask);

                // Scale to [0,255] and round
                _mm256_add_ps(_mm256_mul_ps(result, scale255), half)
            }};
        }

        // Process 8 pixels per iteration (8 RGBA f16 pixels = 32 u16 = 64 bytes)
        while remaining >= 8 {
            // Prefetch source data for the next iteration (64 bytes ahead)
            if remaining >= 16 {
                _mm_prefetch(src_ptr.add(32) as *const i8, _MM_HINT_T0);
            }

            // Load 32 half-floats as 4 groups of 8, convert to f32
            let h0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let h1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
            let h2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let h3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);

            // f0 = [R0 G0 B0 A0 | R1 G1 B1 A1]  (AoS layout, 128-bit lanes)
            let f0 = _mm256_cvtph_ps(h0);
            let f1 = _mm256_cvtph_ps(h1);
            let f2 = _mm256_cvtph_ps(h2);
            let f3 = _mm256_cvtph_ps(h3);

            // Transpose AoS -> SoA to get 8-wide R, G, B vectors.
            // Step 1: interleave within 128-bit lanes
            let t0 = _mm256_unpacklo_ps(f0, f1); // [R0 R2 G0 G2 | R1 R3 G1 G3]
            let t1 = _mm256_unpackhi_ps(f0, f1); // [B0 B2 A0 A2 | B1 B3 A1 A3]
            let t2 = _mm256_unpacklo_ps(f2, f3); // [R4 R6 G4 G6 | R5 R7 G5 G7]
            let t3 = _mm256_unpackhi_ps(f2, f3); // [B4 B6 A4 A6 | B5 B7 A5 A7]

            // Step 2: shuffle to collect channel pairs
            let rr = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00);
            let gg = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10);
            let bb = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
            let aa = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);

            // Step 3: final permute to sequential order
            let r_vals = _mm256_permutevar8x32_ps(rr, perm);
            let g_vals = _mm256_permutevar8x32_ps(gg, perm);
            let b_vals = _mm256_permutevar8x32_ps(bb, perm);
            let a_vals = _mm256_permutevar8x32_ps(aa, perm);

            // Apply sRGB gamma to each channel (8-wide)
            let r_srgb = srgb_gamma_ps!(r_vals);
            let g_srgb = srgb_gamma_ps!(g_vals);
            let b_srgb = srgb_gamma_ps!(b_vals);

            // Alpha: linear clamp to [0,1] then scale to [0,255]
            let a_clamped = _mm256_min_ps(_mm256_max_ps(a_vals, zero), one);
            let a_srgb = _mm256_add_ps(_mm256_mul_ps(a_clamped, alpha_scale), alpha_half);

            // Convert to i32 and pack: pixel = R | (G << 8) | (B << 16) | (A << 24)
            let r_i = _mm256_cvttps_epi32(r_srgb);
            let g_i = _mm256_cvttps_epi32(g_srgb);
            let b_i = _mm256_cvttps_epi32(b_srgb);
            let a_i = _mm256_cvttps_epi32(a_srgb);

            let g_shifted = _mm256_slli_epi32(g_i, 8);
            let b_shifted = _mm256_slli_epi32(b_i, 16);
            let a_shifted = _mm256_slli_epi32(a_i, 24);
            let rgba = _mm256_or_si256(
                _mm256_or_si256(r_i, g_shifted),
                _mm256_or_si256(b_shifted, a_shifted),
            );

            if nontemporal {
                _mm256_stream_si256(dst_ptr as *mut __m256i, rgba);
            } else {
                _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba);
            }

            src_ptr = src_ptr.add(32); // 8 pixels * 4 channels
            dst_ptr = dst_ptr.add(32); // 8 pixels * 4 bytes
            remaining -= 8;
        }

        if nontemporal && fence {
            _mm_sfence();
        }

        // Scalar tail
        if remaining > 0 {
            convert_f16_rgba_to_srgb_scalar_unchecked(src_ptr as *const u8, dst_ptr, remaining);
        }
    } // unsafe
}

// ---------------------------------------------------------------------------
// F16 HDR->sRGB via AVX2 + F16C (with PQ tonemap)
// ---------------------------------------------------------------------------
//
// SIMD version of the HDR→SDR tonemap pipeline.  F16C converts half-floats
// to f32, then we apply the same three-step algorithm as the scalar path:
//
//   1. White-point adjustment — rescale linear RGB by the ratio of SDR
//      white level to HDR paper white (see `adjust_hdr_whites` in f16.rs).
//   2. Peak-luminance limiting — encode the max channel into the SMPTE
//      ST 2084 PQ curve, clamp to `hdr_maximum_nits`, decode back to
//      linear, and scale all channels uniformly. Because PQ encode/decode
//      are monotonic inverses, this is equivalent to maxRGB clamping in
//      linear space with hue-preserving uniform scaling (channel-max
//      style, aligned with ITU-R BT.2408-7 Annex 5 guidance). BT.2408
//      also notes possible perceptual artifacts for maxRGB-style limiting.
//   3. sRGB gamma — apply the IEC 61966-2-1 transfer function.
//
// The PQ and sRGB transfer functions both use the fast-pow IEEE 754 bit
// trick (see the F16→sRGB section above for references).

#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_hdr_f16c_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    params: HdrToSdrParams,
) {
    unsafe {
        convert_f16_rgba_to_srgb_hdr_f16c_inner(src, dst, pixel_count, params);
    }
}

#[target_feature(enable = "avx2,f16c")]
unsafe fn convert_f16_rgba_to_srgb_hdr_f16c_inner(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    params: HdrToSdrParams,
) {
    use std::arch::x86_64::*;

    unsafe {
        // Pre-compute scalar constants from params
        let white_adjust = (params.sdr_white_level_nits / params.hdr_paper_white_nits).max(0.01);
        let inv_white_adjust = 1.0 / white_adjust;
        let max_nits_normalized = params.hdr_maximum_nits / 1000.0; // HDR_NITS_REFERENCE

        // Broadcast constants
        let v_inv_white = _mm256_set1_ps(inv_white_adjust);
        let v_zero = _mm256_setzero_ps();
        let v_one = _mm256_set1_ps(1.0);
        let v_epsilon = _mm256_set1_ps(1e-6);

        // PQ constants (used inside macros directly)
        let _pq_m1 = _mm256_set1_ps(0.159_301_758);
        let _pq_m2 = _mm256_set1_ps(78.843_75);
        let pq_c1 = _mm256_set1_ps(0.835_937_5);
        let pq_c2 = _mm256_set1_ps(18.851_562_5);
        let pq_c3 = _mm256_set1_ps(18.687_5);
        let _pq_inv_m2 = _mm256_set1_ps(1.0 / 78.843_75);
        let _pq_inv_m1 = _mm256_set1_ps(1.0 / 0.159_301_758);

        // sRGB constants
        let srgb_threshold = _mm256_set1_ps(0.003_130_8_f32);
        let srgb_linear_scale = _mm256_set1_ps(12.92);
        let srgb_a = _mm256_set1_ps(1.055);
        let srgb_b = _mm256_set1_ps(-0.055);
        let scale255 = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);

        // Fast pow constants
        let pow_bias_f = 0x3F80_0000u32 as f32;

        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        // Fast approximate pow for 8 floats: x^p via integer bit trick
        // Input must be > 0 for meaningful results.
        macro_rules! fast_pow_ps {
            ($x:expr, $p:expr) => {{
                let xi = _mm256_castps_si256($x);
                let p_scale = _mm256_set1_ps($p);
                let p_offset = _mm256_set1_ps(pow_bias_f * (1.0 - $p));
                let pow_i = _mm256_add_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(xi), p_scale)),
                    _mm256_cvtps_epi32(p_offset),
                );
                _mm256_castsi256_ps(pow_i)
            }};
        }

        // sRGB gamma for 8 floats in [0,1] -> [0,255]
        macro_rules! srgb_gamma_ps {
            ($v:expr) => {{
                let clamped = _mm256_min_ps(_mm256_max_ps($v, v_zero), v_one);
                let lin = _mm256_mul_ps(clamped, srgb_linear_scale);
                let pow_approx = fast_pow_ps!(clamped, 1.0 / 2.4);
                let gamma = _mm256_add_ps(_mm256_mul_ps(srgb_a, pow_approx), srgb_b);
                let mask = _mm256_cmp_ps(clamped, srgb_threshold, _CMP_GT_OQ);
                let result = _mm256_blendv_ps(lin, gamma, mask);
                _mm256_add_ps(_mm256_mul_ps(result, scale255), half)
            }};
        }

        // PQ: linear_to_st2084 for 8 floats
        // v^PQ_M1 -> p, then ((C1 + C2*p) / (1 + C3*p))^PQ_M2
        macro_rules! linear_to_pq_ps {
            ($v:expr) => {{
                let clamped = _mm256_max_ps($v, v_zero);
                let p = fast_pow_ps!(clamped, 0.159_301_758); // PQ_M1
                let num = _mm256_add_ps(pq_c1, _mm256_mul_ps(pq_c2, p));
                let den = _mm256_add_ps(v_one, _mm256_mul_ps(pq_c3, p));
                let ratio = _mm256_div_ps(num, den);
                // ratio^PQ_M2
                fast_pow_ps!(ratio, 78.843_75) // PQ_M2
            }};
        }

        // PQ: st2084_to_linear for 8 floats
        // v^(1/PQ_M2) -> p, then ((p - C1) / (C2 - C3*p))^(1/PQ_M1)
        macro_rules! pq_to_linear_ps {
            ($v:expr) => {{
                let clamped = _mm256_max_ps($v, v_zero);
                let p = fast_pow_ps!(clamped, 1.0 / 78.843_75); // 1/PQ_M2
                let num = _mm256_max_ps(_mm256_sub_ps(p, pq_c1), v_zero);
                let den = _mm256_sub_ps(pq_c2, _mm256_mul_ps(pq_c3, p));
                let ratio = _mm256_div_ps(num, den);
                fast_pow_ps!(ratio, 1.0 / 0.159_301_758) // 1/PQ_M1
            }};
        }

        // Pre-compute max_2084 (constant across all pixels)
        let v_max_nits = _mm256_set1_ps(max_nits_normalized);
        let max_2084 = linear_to_pq_ps!(v_max_nits);

        let mut src_ptr = src as *const u16;
        let mut dst_ptr = dst as *mut u8;
        let mut remaining = pixel_count;

        // Process 8 pixels per iteration
        while remaining >= 8 {
            // Prefetch source data for the next iteration (64 bytes ahead)
            if remaining >= 16 {
                _mm_prefetch(src_ptr.add(32) as *const i8, _MM_HINT_T0);
            }

            // Load and convert F16 -> F32
            let h0 = _mm_loadu_si128(src_ptr as *const __m128i);
            let h1 = _mm_loadu_si128(src_ptr.add(8) as *const __m128i);
            let h2 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
            let h3 = _mm_loadu_si128(src_ptr.add(24) as *const __m128i);

            let f0 = _mm256_cvtph_ps(h0);
            let f1 = _mm256_cvtph_ps(h1);
            let f2 = _mm256_cvtph_ps(h2);
            let f3 = _mm256_cvtph_ps(h3);

            // AoS -> SoA transpose
            let t0 = _mm256_unpacklo_ps(f0, f1);
            let t1 = _mm256_unpackhi_ps(f0, f1);
            let t2 = _mm256_unpacklo_ps(f2, f3);
            let t3 = _mm256_unpackhi_ps(f2, f3);

            let rr = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00);
            let gg = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10);
            let bb = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
            let aa_hdr = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);

            let mut r = _mm256_max_ps(_mm256_permutevar8x32_ps(rr, perm), v_zero);
            let mut g = _mm256_max_ps(_mm256_permutevar8x32_ps(gg, perm), v_zero);
            let mut b = _mm256_max_ps(_mm256_permutevar8x32_ps(bb, perm), v_zero);
            let a_vals = _mm256_permutevar8x32_ps(aa_hdr, perm);

            // White point adjustment: rgb /= white_adjust  (== rgb * inv_white_adjust)
            r = _mm256_mul_ps(r, v_inv_white);
            g = _mm256_mul_ps(g, v_inv_white);
            b = _mm256_mul_ps(b, v_inv_white);

            // Maximum nits limiting via PQ curve
            // color_max = max(r, max(g, b))
            let color_max = _mm256_max_ps(r, _mm256_max_ps(g, b));
            let needs_limit = _mm256_cmp_ps(color_max, v_epsilon, _CMP_GT_OQ);

            // Only compute PQ if any pixel needs it (branch prediction friendly
            // for mostly-dark or mostly-SDR content, but we always compute for
            // simplicity since the SIMD cost is low).
            let color_2084 = linear_to_pq_ps!(color_max);
            let limited_2084 = _mm256_min_ps(color_2084, max_2084);
            let limited_linear = pq_to_linear_ps!(limited_2084);

            // scale = limited_linear / color_max (avoid div-by-zero)
            let safe_max = _mm256_max_ps(color_max, v_epsilon);
            let scale = _mm256_div_ps(limited_linear, safe_max);

            // Apply scale only where color_max > epsilon
            let final_scale = _mm256_blendv_ps(v_one, scale, needs_limit);
            r = _mm256_mul_ps(r, final_scale);
            g = _mm256_mul_ps(g, final_scale);
            b = _mm256_mul_ps(b, final_scale);

            // sRGB gamma
            let r_srgb = srgb_gamma_ps!(r);
            let g_srgb = srgb_gamma_ps!(g);
            let b_srgb = srgb_gamma_ps!(b);

            // Alpha: linear clamp to [0,1] then scale to [0,255]
            let a_clamped = _mm256_min_ps(_mm256_max_ps(a_vals, v_zero), v_one);
            let a_srgb = _mm256_add_ps(_mm256_mul_ps(a_clamped, scale255), half);

            // Pack to RGBA u8
            let r_i = _mm256_cvttps_epi32(r_srgb);
            let g_i = _mm256_cvttps_epi32(g_srgb);
            let b_i = _mm256_cvttps_epi32(b_srgb);
            let a_i = _mm256_cvttps_epi32(a_srgb);

            let g_shifted = _mm256_slli_epi32(g_i, 8);
            let b_shifted = _mm256_slli_epi32(b_i, 16);
            let a_shifted = _mm256_slli_epi32(a_i, 24);
            let rgba = _mm256_or_si256(
                _mm256_or_si256(r_i, g_shifted),
                _mm256_or_si256(b_shifted, a_shifted),
            );

            _mm256_storeu_si256(dst_ptr as *mut __m256i, rgba);

            src_ptr = src_ptr.add(32);
            dst_ptr = dst_ptr.add(32);
            remaining -= 8;
        }

        // Scalar tail
        if remaining > 0 {
            convert_f16_rgba_to_srgb_hdr_scalar_unchecked(
                src_ptr as *const u8,
                dst_ptr,
                remaining,
                params,
            );
        }
    } // unsafe
}

// ---------------------------------------------------------------------------
// F16->sRGB via AVX-512 + F16C (16 pixels per iteration)
// ---------------------------------------------------------------------------
//
// Processes 16 RGBA f16 pixels at a time by splitting into two 8-wide
// batches (F16C's `vcvtph2ps` operates on 128→256 bit), applying the
// same fast-pow sRGB gamma approximation, then packing both halves into
// a single 512-bit store.

#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_avx512_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { convert_f16_rgba_to_srgb_avx512_inner(src, dst, pixel_count, false, false) }
}

/// Non-temporal store variant for AVX-512 F16→sRGB.
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_avx512_nt_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { convert_f16_rgba_to_srgb_avx512_inner(src, dst, pixel_count, true, true) }
}

#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_avx512_nt_nofence_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
) {
    unsafe { convert_f16_rgba_to_srgb_avx512_inner(src, dst, pixel_count, true, false) }
}

#[target_feature(enable = "avx512f,avx512bw,f16c")]
unsafe fn convert_f16_rgba_to_srgb_avx512_inner(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    nontemporal: bool,
    fence: bool,
) {
    use std::arch::x86_64::*;

    unsafe {
        // sRGB constants (256-bit for the 8-wide sub-batches)
        let threshold = _mm256_set1_ps(0.003_130_8_f32);
        let linear_scale = _mm256_set1_ps(12.92);
        let a = _mm256_set1_ps(1.055);
        let b = _mm256_set1_ps(-0.055);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let scale255 = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);

        let pow_scale = _mm256_set1_ps(1.0 / 2.4);
        let pow_bias_f = 0x3F80_0000u32 as f32;
        let pow_offset = _mm256_set1_ps(pow_bias_f * (1.0 - 1.0 / 2.4));

        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        macro_rules! srgb_gamma_ps {
            ($v:expr) => {{
                let clamped = _mm256_min_ps(_mm256_max_ps($v, zero), one);
                let lin = _mm256_mul_ps(clamped, linear_scale);
                let xi = _mm256_castps_si256(clamped);
                let pow_i = _mm256_add_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(xi), pow_scale)),
                    _mm256_cvtps_epi32(pow_offset),
                );
                let pow_approx = _mm256_castsi256_ps(pow_i);
                let gamma = _mm256_add_ps(_mm256_mul_ps(a, pow_approx), b);
                let mask = _mm256_cmp_ps(clamped, threshold, _CMP_GT_OQ);
                let result = _mm256_blendv_ps(lin, gamma, mask);
                _mm256_add_ps(_mm256_mul_ps(result, scale255), half)
            }};
        }

        // Process 8 RGBA f16 pixels → 8 RGBA u8 pixels (256-bit output)
        macro_rules! convert_8px {
            ($src_ptr:expr) => {{
                let h0 = _mm_loadu_si128($src_ptr as *const __m128i);
                let h1 = _mm_loadu_si128(($src_ptr).add(8) as *const __m128i);
                let h2 = _mm_loadu_si128(($src_ptr).add(16) as *const __m128i);
                let h3 = _mm_loadu_si128(($src_ptr).add(24) as *const __m128i);

                let f0 = _mm256_cvtph_ps(h0);
                let f1 = _mm256_cvtph_ps(h1);
                let f2 = _mm256_cvtph_ps(h2);
                let f3 = _mm256_cvtph_ps(h3);

                let t0 = _mm256_unpacklo_ps(f0, f1);
                let t1 = _mm256_unpackhi_ps(f0, f1);
                let t2 = _mm256_unpacklo_ps(f2, f3);
                let t3 = _mm256_unpackhi_ps(f2, f3);

                let rr = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00);
                let gg = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10);
                let bb = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
                let aa = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);

                let r_vals = _mm256_permutevar8x32_ps(rr, perm);
                let g_vals = _mm256_permutevar8x32_ps(gg, perm);
                let b_vals = _mm256_permutevar8x32_ps(bb, perm);
                let a_vals = _mm256_permutevar8x32_ps(aa, perm);

                let r_srgb = srgb_gamma_ps!(r_vals);
                let g_srgb = srgb_gamma_ps!(g_vals);
                let b_srgb = srgb_gamma_ps!(b_vals);

                // Alpha: linear clamp to [0,1] then scale to [0,255]
                let a_clamped = _mm256_min_ps(_mm256_max_ps(a_vals, zero), one);
                let a_srgb = _mm256_add_ps(_mm256_mul_ps(a_clamped, scale255), half);

                let r_i = _mm256_cvttps_epi32(r_srgb);
                let g_i = _mm256_cvttps_epi32(g_srgb);
                let b_i = _mm256_cvttps_epi32(b_srgb);
                let a_i = _mm256_cvttps_epi32(a_srgb);

                let g_shifted = _mm256_slli_epi32(g_i, 8);
                let b_shifted = _mm256_slli_epi32(b_i, 16);
                let a_shifted = _mm256_slli_epi32(a_i, 24);
                _mm256_or_si256(
                    _mm256_or_si256(r_i, g_shifted),
                    _mm256_or_si256(b_shifted, a_shifted),
                )
            }};
        }

        let mut src_ptr = src as *const u16;
        let mut dst_ptr = dst as *mut u8;
        let mut remaining = pixel_count;

        // Process 16 pixels per iteration (two 8-wide batches → one 512-bit store)
        while remaining >= 16 {
            // Prefetch source data for the next iteration (128 bytes ahead)
            if remaining >= 32 {
                _mm_prefetch(src_ptr.add(64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src_ptr.add(96) as *const i8, _MM_HINT_T0);
            }

            let lo = convert_8px!(src_ptr);
            let hi = convert_8px!(src_ptr.add(32));

            // Combine two 256-bit results into one 512-bit register and store
            let combined = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            if nontemporal {
                _mm512_stream_si512(dst_ptr as *mut __m512i, combined);
            } else {
                _mm512_storeu_si512(dst_ptr as *mut __m512i, combined);
            }

            src_ptr = src_ptr.add(64); // 16 pixels * 4 channels
            dst_ptr = dst_ptr.add(64); // 16 pixels * 4 bytes
            remaining -= 16;
        }

        // 8-pixel tail
        if remaining >= 8 {
            let result = convert_8px!(src_ptr);
            if nontemporal {
                _mm256_stream_si256(dst_ptr as *mut __m256i, result);
            } else {
                _mm256_storeu_si256(dst_ptr as *mut __m256i, result);
            }
            src_ptr = src_ptr.add(32);
            dst_ptr = dst_ptr.add(32);
            remaining -= 8;
        }

        if nontemporal && fence {
            _mm_sfence();
        }

        // Scalar tail
        if remaining > 0 {
            convert_f16_rgba_to_srgb_scalar_unchecked(src_ptr as *const u8, dst_ptr, remaining);
        }
    } // unsafe
}

// ---------------------------------------------------------------------------
// F16 HDR->sRGB via AVX-512 + F16C (16 pixels per iteration, with PQ tonemap)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn convert_f16_rgba_to_srgb_hdr_avx512_unchecked(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    params: HdrToSdrParams,
) {
    unsafe { convert_f16_rgba_to_srgb_hdr_avx512_inner(src, dst, pixel_count, params) }
}

#[target_feature(enable = "avx512f,avx512bw,f16c")]
unsafe fn convert_f16_rgba_to_srgb_hdr_avx512_inner(
    src: *const u8,
    dst: *mut u8,
    pixel_count: usize,
    params: HdrToSdrParams,
) {
    use std::arch::x86_64::*;

    unsafe {
        let white_adjust = (params.sdr_white_level_nits / params.hdr_paper_white_nits).max(0.01);
        let inv_white_adjust = 1.0 / white_adjust;
        let max_nits_normalized = params.hdr_maximum_nits / 1000.0;

        let v_inv_white = _mm256_set1_ps(inv_white_adjust);
        let v_zero = _mm256_setzero_ps();
        let v_one = _mm256_set1_ps(1.0);
        let v_epsilon = _mm256_set1_ps(1e-6);

        let pq_c1 = _mm256_set1_ps(0.835_937_5);
        let pq_c2 = _mm256_set1_ps(18.851_562_5);
        let pq_c3 = _mm256_set1_ps(18.687_5);

        let srgb_threshold = _mm256_set1_ps(0.003_130_8_f32);
        let srgb_linear_scale = _mm256_set1_ps(12.92);
        let srgb_a = _mm256_set1_ps(1.055);
        let srgb_b = _mm256_set1_ps(-0.055);
        let scale255 = _mm256_set1_ps(255.0);
        let half = _mm256_set1_ps(0.5);

        let pow_bias_f = 0x3F80_0000u32 as f32;

        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        macro_rules! fast_pow_ps {
            ($x:expr, $p:expr) => {{
                let xi = _mm256_castps_si256($x);
                let p_scale = _mm256_set1_ps($p);
                let p_offset = _mm256_set1_ps(pow_bias_f * (1.0 - $p));
                let pow_i = _mm256_add_epi32(
                    _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(xi), p_scale)),
                    _mm256_cvtps_epi32(p_offset),
                );
                _mm256_castsi256_ps(pow_i)
            }};
        }

        macro_rules! srgb_gamma_ps {
            ($v:expr) => {{
                let clamped = _mm256_min_ps(_mm256_max_ps($v, v_zero), v_one);
                let lin = _mm256_mul_ps(clamped, srgb_linear_scale);
                let pow_approx = fast_pow_ps!(clamped, 1.0 / 2.4);
                let gamma = _mm256_add_ps(_mm256_mul_ps(srgb_a, pow_approx), srgb_b);
                let mask = _mm256_cmp_ps(clamped, srgb_threshold, _CMP_GT_OQ);
                let result = _mm256_blendv_ps(lin, gamma, mask);
                _mm256_add_ps(_mm256_mul_ps(result, scale255), half)
            }};
        }

        macro_rules! linear_to_pq_ps {
            ($v:expr) => {{
                let clamped = _mm256_max_ps($v, v_zero);
                let p = fast_pow_ps!(clamped, 0.159_301_758);
                let num = _mm256_add_ps(pq_c1, _mm256_mul_ps(pq_c2, p));
                let den = _mm256_add_ps(v_one, _mm256_mul_ps(pq_c3, p));
                let ratio = _mm256_div_ps(num, den);
                fast_pow_ps!(ratio, 78.843_75)
            }};
        }

        macro_rules! pq_to_linear_ps {
            ($v:expr) => {{
                let clamped = _mm256_max_ps($v, v_zero);
                let p = fast_pow_ps!(clamped, 1.0 / 78.843_75);
                let num = _mm256_max_ps(_mm256_sub_ps(p, pq_c1), v_zero);
                let den = _mm256_sub_ps(pq_c2, _mm256_mul_ps(pq_c3, p));
                let ratio = _mm256_div_ps(num, den);
                fast_pow_ps!(ratio, 1.0 / 0.159_301_758)
            }};
        }

        let v_max_nits = _mm256_set1_ps(max_nits_normalized);
        let max_2084 = linear_to_pq_ps!(v_max_nits);

        // Process 8 HDR f16 pixels → 8 RGBA u8 pixels (256-bit output)
        macro_rules! convert_8px_hdr {
            ($src_ptr:expr) => {{
                let h0 = _mm_loadu_si128($src_ptr as *const __m128i);
                let h1 = _mm_loadu_si128(($src_ptr).add(8) as *const __m128i);
                let h2 = _mm_loadu_si128(($src_ptr).add(16) as *const __m128i);
                let h3 = _mm_loadu_si128(($src_ptr).add(24) as *const __m128i);

                let f0 = _mm256_cvtph_ps(h0);
                let f1 = _mm256_cvtph_ps(h1);
                let f2 = _mm256_cvtph_ps(h2);
                let f3 = _mm256_cvtph_ps(h3);

                let t0 = _mm256_unpacklo_ps(f0, f1);
                let t1 = _mm256_unpackhi_ps(f0, f1);
                let t2 = _mm256_unpacklo_ps(f2, f3);
                let t3 = _mm256_unpackhi_ps(f2, f3);

                let rr = _mm256_shuffle_ps(t0, t2, 0b01_00_01_00);
                let gg = _mm256_shuffle_ps(t0, t2, 0b11_10_11_10);
                let bb = _mm256_shuffle_ps(t1, t3, 0b01_00_01_00);
                let aa = _mm256_shuffle_ps(t1, t3, 0b11_10_11_10);

                let mut r = _mm256_max_ps(_mm256_permutevar8x32_ps(rr, perm), v_zero);
                let mut g = _mm256_max_ps(_mm256_permutevar8x32_ps(gg, perm), v_zero);
                let mut b = _mm256_max_ps(_mm256_permutevar8x32_ps(bb, perm), v_zero);
                let a_vals = _mm256_permutevar8x32_ps(aa, perm);

                r = _mm256_mul_ps(r, v_inv_white);
                g = _mm256_mul_ps(g, v_inv_white);
                b = _mm256_mul_ps(b, v_inv_white);

                let color_max = _mm256_max_ps(r, _mm256_max_ps(g, b));
                let needs_limit = _mm256_cmp_ps(color_max, v_epsilon, _CMP_GT_OQ);

                let color_2084 = linear_to_pq_ps!(color_max);
                let limited_2084 = _mm256_min_ps(color_2084, max_2084);
                let limited_linear = pq_to_linear_ps!(limited_2084);

                let safe_max = _mm256_max_ps(color_max, v_epsilon);
                let scale = _mm256_div_ps(limited_linear, safe_max);
                let final_scale = _mm256_blendv_ps(v_one, scale, needs_limit);

                r = _mm256_mul_ps(r, final_scale);
                g = _mm256_mul_ps(g, final_scale);
                b = _mm256_mul_ps(b, final_scale);

                let r_srgb = srgb_gamma_ps!(r);
                let g_srgb = srgb_gamma_ps!(g);
                let b_srgb = srgb_gamma_ps!(b);

                // Alpha: linear clamp to [0,1] then scale to [0,255]
                let a_clamped = _mm256_min_ps(_mm256_max_ps(a_vals, v_zero), v_one);
                let a_srgb = _mm256_add_ps(_mm256_mul_ps(a_clamped, scale255), half);

                let r_i = _mm256_cvttps_epi32(r_srgb);
                let g_i = _mm256_cvttps_epi32(g_srgb);
                let b_i = _mm256_cvttps_epi32(b_srgb);
                let a_i = _mm256_cvttps_epi32(a_srgb);

                let g_shifted = _mm256_slli_epi32(g_i, 8);
                let b_shifted = _mm256_slli_epi32(b_i, 16);
                let a_shifted = _mm256_slli_epi32(a_i, 24);
                _mm256_or_si256(
                    _mm256_or_si256(r_i, g_shifted),
                    _mm256_or_si256(b_shifted, a_shifted),
                )
            }};
        }

        let mut src_ptr = src as *const u16;
        let mut dst_ptr = dst as *mut u8;
        let mut remaining = pixel_count;

        while remaining >= 16 {
            // Prefetch source data for the next iteration (128 bytes ahead)
            if remaining >= 32 {
                _mm_prefetch(src_ptr.add(64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(src_ptr.add(96) as *const i8, _MM_HINT_T0);
            }

            let lo = convert_8px_hdr!(src_ptr);
            let hi = convert_8px_hdr!(src_ptr.add(32));

            let combined = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            _mm512_storeu_si512(dst_ptr as *mut __m512i, combined);

            src_ptr = src_ptr.add(64);
            dst_ptr = dst_ptr.add(64);
            remaining -= 16;
        }

        if remaining >= 8 {
            let result = convert_8px_hdr!(src_ptr);
            _mm256_storeu_si256(dst_ptr as *mut __m256i, result);
            src_ptr = src_ptr.add(32);
            dst_ptr = dst_ptr.add(32);
            remaining -= 8;
        }

        if remaining > 0 {
            convert_f16_rgba_to_srgb_hdr_scalar_unchecked(
                src_ptr as *const u8,
                dst_ptr,
                remaining,
                params,
            );
        }
    } // unsafe
}
