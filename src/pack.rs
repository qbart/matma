//! GPU vector packing: normalized-integer quantization, octahedral unit-vector
//! encoding and IEEE half-float packing.
//!
//! Everything that doesn't need a `sqrt` is a `const fn`, so encoders can run at
//! compile time (`const N: short2 = oct_encode16(..)`).
//!
//! Octahedral mapping follows Meyer et al. 2010 ("On Floating-Point Normal
//! Vectors"); the decode uses the Cigolle et al. survey form. See [`oct_project`] /
//! [`oct_unproject`] and the Slang snippet on [`oct_unproject`].

use crate::vector::{Vector2, Vector3, float2, float3, short2};

// ---- const helpers --------------------------------------------------------------

#[inline]
const fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
#[inline]
const fn absf(v: f32) -> f32 {
    if v < 0.0 { -v } else { v }
}
/// signNotZero: +1 for v >= 0 (including +0), -1 for v < 0.
#[inline]
const fn oct_sign(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}
/// Round half away from zero. Matches `lround` except within 1 ulp of a
/// half-integer, where the `v + 0.5` add can double-round (e.g. 0.49999997 -> 1);
/// for the quantizers that is at most one code of difference.
#[inline]
const fn round_half_away(v: f32) -> i32 {
    (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

// ---- normalized-integer quantization ------------------------------------------
// snorm: float in [-1, 1] <-> signed int, scale 2^(n-1)-1. Encode clamps + rounds;
// decode clamps to [-1, 1] (the most-negative code, e.g. -32768, maps below -1).
// unorm: float in [0, 1] <-> unsigned int, scale 2^n-1.

/// `[-1, 1]` -> `i16` (scale 32767). Clamps the input.
#[inline]
pub const fn quantize_snorm16(v: f32) -> i16 {
    round_half_away(clampf(v, -1.0, 1.0) * 32767.0) as i16
}
/// `i16` -> `[-1, 1]` (scale 32767). Clamps the output.
#[inline]
pub const fn dequantize_snorm16(v: i16) -> f32 {
    clampf(v as f32 / 32767.0, -1.0, 1.0)
}
/// `[-1, 1]` -> `i8` (scale 127). Clamps the input.
#[inline]
pub const fn quantize_snorm8(v: f32) -> i8 {
    round_half_away(clampf(v, -1.0, 1.0) * 127.0) as i8
}
/// `i8` -> `[-1, 1]` (scale 127). Clamps the output.
#[inline]
pub const fn dequantize_snorm8(v: i8) -> f32 {
    clampf(v as f32 / 127.0, -1.0, 1.0)
}
/// 10-bit snorm: value in the low 10 bits of an `i16` (range `[-511, 511]`).
#[inline]
pub const fn quantize_snorm10(v: f32) -> i16 {
    round_half_away(clampf(v, -1.0, 1.0) * 511.0) as i16
}
/// Inverse of [`quantize_snorm10`]. Clamps the output.
#[inline]
pub const fn dequantize_snorm10(v: i16) -> f32 {
    clampf(v as f32 / 511.0, -1.0, 1.0)
}
/// `[0, 1]` -> `u16` (scale 65535). Clamps the input.
#[inline]
pub const fn quantize_unorm16(v: f32) -> u16 {
    round_half_away(clampf(v, 0.0, 1.0) * 65535.0) as u16
}
/// `u16` -> `[0, 1]` (scale 65535).
#[inline]
pub const fn dequantize_unorm16(v: u16) -> f32 {
    v as f32 / 65535.0
}
/// `[0, 1]` -> `u8` (scale 255). Clamps the input.
#[inline]
pub const fn quantize_unorm8(v: f32) -> u8 {
    round_half_away(clampf(v, 0.0, 1.0) * 255.0) as u8
}
/// `u8` -> `[0, 1]` (scale 255).
#[inline]
pub const fn dequantize_unorm8(v: u8) -> f32 {
    v as f32 / 255.0
}
/// 10-bit unorm: value in the low 10 bits of a `u16` (range `[0, 1023]`).
#[inline]
pub const fn quantize_unorm10(v: f32) -> u16 {
    round_half_away(clampf(v, 0.0, 1.0) * 1023.0) as u16
}
/// Inverse of [`quantize_unorm10`].
#[inline]
pub const fn dequantize_unorm10(v: u16) -> f32 {
    v as f32 / 1023.0
}

// ---- octahedral unit-vector encoding ------------------------------------------

/// Fold a unit vector onto the octahedron; result in `[-1, 1]^2`.
pub const fn oct_project(n: float3) -> float2 {
    let inv_l1 = 1.0 / (absf(n.x) + absf(n.y) + absf(n.z));
    let p = Vector2::new(n.x * inv_l1, n.y * inv_l1);
    if n.z < 0.0 {
        Vector2::new(
            (1.0 - absf(p.y)) * oct_sign(p.x),
            (1.0 - absf(p.x)) * oct_sign(p.y),
        )
    } else {
        p
    }
}

/// Inverse of [`oct_project`]: unfold back to a (re-normalized) unit vector.
///
/// Matching Slang/HLSL decoder (kept in sync with this function):
///
/// ```hlsl
/// float3 octDecode(float2 e) {
///     float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
///     float  t = max(-v.z, 0.0);
///     v.x += (v.x >= 0.0) ? -t : t;
///     v.y += (v.y >= 0.0) ? -t : t;
///     return normalize(v);
/// }
/// ```
///
/// Vertex formats: normal -> `R16G16_SNORM` (hardware divides by 32767 on fetch);
/// tangent -> `R16G16_SINT` (the shader needs the raw int to read the handedness
/// LSB, and divides by 32767 itself). Decode of either: `octDecode(...)`.
/// Bitangent in the shader: `cross(normal, tangent) * handedness`.
pub fn oct_unproject(e: float2) -> float3 {
    let mut v = Vector3::new(e.x, e.y, 1.0 - e.x.abs() - e.y.abs());
    let t = (-v.z).max(0.0);
    v.x += if v.x >= 0.0 { -t } else { t };
    v.y += if v.y >= 0.0 { -t } else { t };
    v.normalize()
}

/// Unit vector -> packed snorm16 octahedral pair.
#[inline]
pub const fn oct_encode16(n: float3) -> short2 {
    let e = oct_project(n);
    Vector2::new(quantize_snorm16(e.x), quantize_snorm16(e.y))
}
/// Packed snorm16 octahedral pair -> unit vector.
#[inline]
pub fn oct_decode16(e: short2) -> float3 {
    oct_unproject(Vector2::new(
        dequantize_snorm16(e.x),
        dequantize_snorm16(e.y),
    ))
}

// ---- tangent packing ----------------------------------------------------------
// Bitangent sign rides in the LSB of .x: 0 => +1, 1 => -1. The 1-LSB nudge is
// ~1/32767 ~= 3e-5 of perturbation on the tangent direction.

/// Pack a unit tangent plus bitangent `handedness` (`+1` / `-1`) into a snorm16
/// octahedral pair; the sign rides in the LSB of `.x`.
#[inline]
pub const fn oct_encode_tangent(t: float3, handedness: f32) -> short2 {
    let mut e = oct_encode16(t);
    e.x &= !1; // clear sign LSB
    if handedness < 0.0 {
        e.x |= 1;
    }
    e
}

/// Inverse of [`oct_encode_tangent`]: `(tangent, handedness)`. Reads the handedness
/// from `packed.x`'s LSB, then decodes the tangent from `packed` AS-IS (LSB not
/// cleared) so this matches the shader path bit-for-bit. Rebuild the bitangent as
/// `cross(normal, tangent) * handedness`.
#[inline]
pub fn oct_decode_tangent(packed: short2) -> (float3, f32) {
    let handedness = if packed.x & 1 != 0 { -1.0 } else { 1.0 };
    (oct_decode16(packed), handedness)
}

// ---- float16 packing (IEEE-754 half) ------------------------------------------
// Two f32 <-> a u32 of two IEEE half-floats (1 sign / 5 exp / 10 mantissa):
// .x in the low 16 bits, .y in the high 16. Round-to-nearest; subnormals/underflow
// flush to +/-0, overflow to +/-inf, NaN stays NaN. Distinct from the
// octahedral/snorm path above: this is a general number format (keeps an
// exponent), valid for any float pair, not just unit vectors. Mirrors GLSL/Slang
// pack_half2x16 / unpack_half2x16.

/// `f32` -> IEEE half bits. Underflow flushes to `±0`, overflow to `±inf`, NaN stays NaN.
pub const fn float_to_half(value: f32) -> u16 {
    let f = value.to_bits();
    let sign = (f >> 16) & 0x8000;
    let exp = ((f >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = f & 0x7F_FFFF;

    if exp <= 0 {
        return sign as u16; // underflow -> +/-0
    }
    if exp >= 31 {
        // overflow / inf / NaN
        // Source NaN (exp field 255 -> rebased 143, mantissa != 0) must stay NaN,
        // not collapse to inf: keep the top payload bits and force the quiet bit
        // so the half mantissa can't round-trip to 0 (which would read as inf).
        if exp == 143 && mant != 0 {
            return (sign | 0x7E00 | (mant >> 13)) as u16;
        }
        return (sign | 0x7C00) as u16; // +/-inf
    }

    let mut half = (sign | ((exp as u32) << 10) | (mant >> 13)) as u16;
    if mant & 0x1000 != 0 {
        half += 1; // round to nearest
    }
    half
}

/// IEEE half bits -> `f32`.
pub const fn half_to_float(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let mut exp = ((h >> 10) & 0x1F) as i32; // signed: goes negative for subnormals
    let mut mant = (h & 0x3FF) as u32;
    let f = if exp == 0 {
        if mant == 0 {
            sign // +/-0
        } else {
            exp = 1; // normalize a half-subnormal into a float-normal
            while mant & 0x400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x3FF;
            sign | (((exp + (127 - 15)) as u32) << 23) | (mant << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13) // inf / NaN
    } else {
        sign | (((exp + (127 - 15)) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f)
}

/// Two `f32` -> two IEEE halves in a `u32` (`.x` low 16 bits, `.y` high 16 bits).
#[inline]
pub const fn pack_half2x16(v: float2) -> u32 {
    float_to_half(v.x) as u32 | ((float_to_half(v.y) as u32) << 16)
}
/// Inverse of [`pack_half2x16`].
#[inline]
pub const fn unpack_half2x16(p: u32) -> float2 {
    Vector2::new(
        half_to_float((p & 0xFFFF) as u16),
        half_to_float((p >> 16) as u16),
    )
}

// ---- compile-time sanity ------------------------------------------------------
const _: () = {
    assert!(quantize_snorm16(1.0) == 32767 && quantize_snorm16(-1.0) == -32767);
    assert!(quantize_snorm16(2.0) == 32767); // clamps on encode
    assert!(dequantize_snorm16(-32768) == -1.0); // clamps on decode
    assert!(quantize_unorm8(1.0) == 255 && quantize_unorm8(0.0) == 0);
    assert!(quantize_unorm10(1.0) == 1023);
    let e = oct_encode16(Vector3::new(0.0, 0.0, 1.0));
    assert!(e.x == 0 && e.y == 0);
    assert!(oct_encode_tangent(Vector3::new(0.0, 0.0, 1.0), 1.0).x & 1 == 0); // +1 -> LSB 0
    assert!(oct_encode_tangent(Vector3::new(0.0, 0.0, 1.0), -1.0).x & 1 == 1); // -1 -> LSB 1
    assert!(pack_half2x16(Vector2::new(1.0, -2.0)) == 0xC000_3C00);
};
