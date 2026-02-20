cbuffer Params : register(b0) {
    float hdr_paper_white_nits;
    float hdr_maximum_nits;
    float sdr_white_level_nits;
    float _pad0;
    uint tex_width;
    uint tex_height;
    uint _pad1;
    uint _pad2;
};

Texture2D<float4> src_tex : register(t0);
RWTexture2D<unorm float4> dst_tex : register(u0);

// ---------------------------------------------------------------------------
// SMPTE ST 2084 (PQ) constants (from SMPTE ST 2084:2014).
//
// These are the exact rational values from the specification:
//   m1 = 2610/16384       ≈ 0.159301758
//   m2 = 2523/4096 × 128  = 78.84375
//   c1 = 3424/4096         = 0.8359375   (also: c3 − c2 + 1)
//   c2 = 2413/4096 × 32   = 18.8515625
//   c3 = 2392/4096 × 32   = 18.6875
// ---------------------------------------------------------------------------
static const float PQ_M1 = 0.159301758;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

// Reference luminance for normalising PQ input.  The PQ curve's absolute
// range is 0–10 000 cd/m², but HDR content is typically mastered with a
// peak of ~1 000 nits, so we normalise linear values as L / 1000.
static const float HDR_NITS_REF = 1000.0;

// SMPTE ST 2084 EOTF⁻¹: linear luminance → PQ non-linear signal.
// See SMPTE ST 2084:2014, Section 5.1 (Equation 1).
float linear_to_st2084(float v) {
    float p = pow(max(v, 0.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2);
}

// SMPTE ST 2084 EOTF: PQ non-linear signal → linear luminance.
// See SMPTE ST 2084:2014, Section 5.2 (Equation 2).
float st2084_to_linear(float v) {
    float p = pow(max(v, 0.0), 1.0 / PQ_M2);
    float num = max(p - PQ_C1, 0.0);
    float den = max(PQ_C2 - PQ_C3 * p, 1e-6);
    return pow(max(num / den, 0.0), 1.0 / PQ_M1);
}

// sRGB EOTF⁻¹ (linear → sRGB gamma), per IEC 61966-2-1:1999, Section 4.7.
float linear_to_srgb(float c) {
    c = saturate(c);
    return (c <= 0.0031308) ? (c * 12.92) : (1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055);
}

void tonemap_pixel(uint2 coord, uint w, uint h) {
    if (coord.x >= w || coord.y >= h) {
        return;
    }

    float4 hdr = src_tex[coord];
    float3 rgb = max(hdr.rgb, 0.0);

    // Step 1: White-point adjustment.
    // Rescale linear RGB so that the HDR "paper white" reference level
    // maps to the SDR display's white level. This is the inverse of the
    // SDR white-level boost used by Windows Advanced Color composition.
    // On Windows this level comes from DISPLAYCONFIG_SDR_WHITE_LEVEL,
    // converted to nits as SDRWhiteLevel * 80 / 1000.
    float white_adjust = max(sdr_white_level_nits / hdr_paper_white_nits, 0.01);
    rgb /= white_adjust;

    // Step 2: Peak-luminance limiting via the PQ curve.
    // Map the brightest channel through ST 2084, clamp to the display's
    // reported maximum luminance, then map back to linear. Applying the
    // resulting ratio uniformly to all channels preserves hue/saturation.
    // Since PQ encode/decode are monotonic inverses, this is equivalent
    // to clamping maxRGB in linear space (channel-max style, aligned with
    // ITU-R BT.2408-7 Annex 5 guidance). BT.2408 also notes that this
    // simple maxRGB limiter can produce perceptual artifacts in some scenes.
    float color_max = max(rgb.r, max(rgb.g, rgb.b));
    if (color_max > 1e-6) {
        float color_2084 = linear_to_st2084(color_max);
        float max_2084   = linear_to_st2084(hdr_maximum_nits / HDR_NITS_REF);
        float limited     = st2084_to_linear(min(color_2084, max_2084));
        float scale       = limited / color_max;
        rgb *= scale;
    }

    // Step 3: Apply the IEC 61966-2-1 sRGB gamma curve.
    // Linear to sRGB gamma
    float3 srgb = float3(linear_to_srgb(rgb.r), linear_to_srgb(rgb.g), linear_to_srgb(rgb.b));
    dst_tex[coord] = float4(srgb, 1.0);
}

// Standard 2D dispatch for typical resolutions (16x16 = 256 threads).
[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    tonemap_pixel(dtid.xy, tex_width, tex_height);
}

// 1D dispatch for small textures where 2D dispatch has poor occupancy.
// Uses 256x1 thread groups dispatched as (groups_x, height, 1) to
// avoid wasting threads on partially-filled 16x16 tiles.
[numthreads(256, 1, 1)]
void main_1d(uint3 dtid : SV_DispatchThreadID) {
    tonemap_pixel(uint2(dtid.x, dtid.y), tex_width, tex_height);
}

// ---------------------------------------------------------------------------
// Plain F16 linear → sRGB conversion (no HDR tonemapping)
// ---------------------------------------------------------------------------
// Used when the source is RGBA16Float but no HDR-to-SDR tonemap is needed.
// Converts linear light values directly to sRGB gamma, avoiding the
// expensive CPU-side F16→sRGB SIMD path entirely.

void convert_f16_pixel(uint2 coord, uint w, uint h) {
    if (coord.x >= w || coord.y >= h) {
        return;
    }

    float4 src = src_tex[coord];
    float3 rgb = saturate(src.rgb);
    float3 srgb = float3(linear_to_srgb(rgb.r), linear_to_srgb(rgb.g), linear_to_srgb(rgb.b));
    dst_tex[coord] = float4(srgb, 1.0);
}

[numthreads(16, 16, 1)]
void main_f16(uint3 dtid : SV_DispatchThreadID) {
    convert_f16_pixel(dtid.xy, tex_width, tex_height);
}

[numthreads(256, 1, 1)]
void main_f16_1d(uint3 dtid : SV_DispatchThreadID) {
    convert_f16_pixel(uint2(dtid.x, dtid.y), tex_width, tex_height);
}
