//! SIMD backend for 4-wide `f32`.
//!
//! This is the only module that touches platform intrinsics. Everything above it
//! works on plain `[f32; 4]` arrays and lets this layer decide how to crunch four
//! lanes at once. Three backends, selected at compile time:
//!
//! | Backend | When                                             |
//! |---------|--------------------------------------------------|
//! | SSE2    | `x86`/`x86_64` with `sse2` (the x86-64 baseline) |
//! | NEON    | `aarch64` (Apple Silicon, ARM servers/phones)    |
//! | scalar  | otherwise, or with the `force-scalar` feature    |
//!
//! The public surface is intentionally tiny: an [`F32x4`] register type plus the
//! handful of ops the vector code needs. [`BACKEND`] names the one in use.
#![allow(unsafe_code)]

#[cfg(all(
    not(feature = "force-scalar"),
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
mod imp {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Name of the active backend.
    pub const BACKEND: &str = "sse2";

    /// Four packed `f32` lanes.
    pub type F32x4 = __m128;

    #[inline(always)]
    pub fn load4(a: [f32; 4]) -> F32x4 {
        // SAFETY: `a` is a valid, readable 16-byte array; loadu has no alignment requirement.
        unsafe { _mm_loadu_ps(a.as_ptr()) }
    }
    #[inline(always)]
    pub fn store4(v: F32x4) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        // SAFETY: `out` is a valid, writable 16-byte array; storeu has no alignment requirement.
        unsafe { _mm_storeu_ps(out.as_mut_ptr(), v) };
        out
    }
    #[inline(always)]
    pub fn splat(s: f32) -> F32x4 {
        // SAFETY: the cfg gate on this module guarantees sse2 is compiled in.
        unsafe { _mm_set1_ps(s) }
    }
    #[inline(always)]
    pub fn add(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { _mm_add_ps(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { _mm_sub_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { _mm_mul_ps(a, b) }
    }
    #[inline(always)]
    pub fn div(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { _mm_div_ps(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: F32x4) -> F32x4 {
        unsafe { _mm_sub_ps(_mm_setzero_ps(), a) }
    }
    /// Horizontal sum, SSE2-only (no SSE3 `haddps`): shuffle + add the halves down
    /// into lane 0.
    #[inline(always)]
    pub fn hsum(v: F32x4) -> f32 {
        // SAFETY: sse2 is guaranteed by the module's cfg gate.
        unsafe {
            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(v, v); // [b, a, d, c]
            let sums = _mm_add_ps(v, shuf); // a+b, b+a, c+d, d+c
            let shuf = _mm_movehl_ps(shuf, sums); // c+d into the low lane
            let sums = _mm_add_ss(sums, shuf); // (a+b) + (c+d)
            _mm_cvtss_f32(sums)
        }
    }
}

#[cfg(all(
    not(feature = "force-scalar"),
    target_arch = "aarch64",
    target_feature = "neon"
))]
mod imp {
    use core::arch::aarch64::*;

    /// Name of the active backend.
    pub const BACKEND: &str = "neon";

    /// Four packed `f32` lanes.
    pub type F32x4 = float32x4_t;

    #[inline(always)]
    pub fn load4(a: [f32; 4]) -> F32x4 {
        // SAFETY: `a` is a valid, readable 16-byte array.
        unsafe { vld1q_f32(a.as_ptr()) }
    }
    #[inline(always)]
    pub fn store4(v: F32x4) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        // SAFETY: `out` is a valid, writable 16-byte array.
        unsafe { vst1q_f32(out.as_mut_ptr(), v) };
        out
    }
    #[inline(always)]
    pub fn splat(s: f32) -> F32x4 {
        // SAFETY: the cfg gate on this module guarantees neon is compiled in.
        unsafe { vdupq_n_f32(s) }
    }
    #[inline(always)]
    pub fn add(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { vsubq_f32(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    pub fn div(a: F32x4, b: F32x4) -> F32x4 {
        unsafe { vdivq_f32(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: F32x4) -> F32x4 {
        unsafe { vnegq_f32(a) }
    }
    /// Horizontal sum across the four lanes (AArch64 has a single instruction).
    #[inline(always)]
    pub fn hsum(v: F32x4) -> f32 {
        unsafe { vaddvq_f32(v) }
    }
}

#[cfg(not(any(
    all(
        not(feature = "force-scalar"),
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ),
    all(
        not(feature = "force-scalar"),
        target_arch = "aarch64",
        target_feature = "neon"
    )
)))]
mod imp {
    /// Name of the active backend.
    pub const BACKEND: &str = "scalar";

    /// Four packed `f32` lanes (portable fallback).
    #[derive(Clone, Copy, Debug)]
    pub struct F32x4(pub [f32; 4]);

    #[inline(always)]
    pub fn load4(a: [f32; 4]) -> F32x4 {
        F32x4(a)
    }
    #[inline(always)]
    pub fn store4(v: F32x4) -> [f32; 4] {
        v.0
    }
    #[inline(always)]
    pub fn splat(s: f32) -> F32x4 {
        F32x4([s; 4])
    }
    #[inline(always)]
    pub fn add(a: F32x4, b: F32x4) -> F32x4 {
        F32x4([
            a.0[0] + b.0[0],
            a.0[1] + b.0[1],
            a.0[2] + b.0[2],
            a.0[3] + b.0[3],
        ])
    }
    #[inline(always)]
    pub fn sub(a: F32x4, b: F32x4) -> F32x4 {
        F32x4([
            a.0[0] - b.0[0],
            a.0[1] - b.0[1],
            a.0[2] - b.0[2],
            a.0[3] - b.0[3],
        ])
    }
    #[inline(always)]
    pub fn mul(a: F32x4, b: F32x4) -> F32x4 {
        F32x4([
            a.0[0] * b.0[0],
            a.0[1] * b.0[1],
            a.0[2] * b.0[2],
            a.0[3] * b.0[3],
        ])
    }
    #[inline(always)]
    pub fn div(a: F32x4, b: F32x4) -> F32x4 {
        F32x4([
            a.0[0] / b.0[0],
            a.0[1] / b.0[1],
            a.0[2] / b.0[2],
            a.0[3] / b.0[3],
        ])
    }
    #[inline(always)]
    pub fn neg(a: F32x4) -> F32x4 {
        F32x4([-a.0[0], -a.0[1], -a.0[2], -a.0[3]])
    }
    #[inline(always)]
    pub fn hsum(a: F32x4) -> f32 {
        a.0[0] + a.0[1] + a.0[2] + a.0[3]
    }
}

pub use imp::{BACKEND, F32x4};

/// Load four lanes from an array.
#[inline(always)]
pub fn load4(a: [f32; 4]) -> F32x4 {
    imp::load4(a)
}
/// Store four lanes to an array.
#[inline(always)]
pub fn store4(v: F32x4) -> [f32; 4] {
    imp::store4(v)
}
/// Broadcast `s` to all four lanes.
#[inline(always)]
pub fn splat(s: f32) -> F32x4 {
    imp::splat(s)
}
/// Lane-wise `a + b`.
#[inline(always)]
pub fn add(a: F32x4, b: F32x4) -> F32x4 {
    imp::add(a, b)
}
/// Lane-wise `a - b`.
#[inline(always)]
pub fn sub(a: F32x4, b: F32x4) -> F32x4 {
    imp::sub(a, b)
}
/// Lane-wise `a * b`.
#[inline(always)]
pub fn mul(a: F32x4, b: F32x4) -> F32x4 {
    imp::mul(a, b)
}
/// Lane-wise `a / b`.
#[inline(always)]
pub fn div(a: F32x4, b: F32x4) -> F32x4 {
    imp::div(a, b)
}
/// Lane-wise `-a`.
#[inline(always)]
pub fn neg(a: F32x4) -> F32x4 {
    imp::neg(a)
}
/// Sum of the four lanes.
#[inline(always)]
pub fn hsum(v: F32x4) -> f32 {
    imp::hsum(v)
}
