//! Swizzle + common-function tests.

mod util;
use core::f32::consts::PI;
use matma::*;
use util::*;

#[test]
fn swizzles() {
    let v = float4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.xyz(), float3::new(1.0, 2.0, 3.0)); // drop w
    assert_eq!(v.xy(), float2::new(1.0, 2.0));
    assert_eq!(v.zw(), float2::new(3.0, 4.0));
    assert_eq!(float3::new(1.0, 2.0, 3.0).yz(), float2::new(2.0, 3.0));
    assert_eq!(float2::new(1.0, 2.0).yx(), float2::new(2.0, 1.0));
    assert_eq!(v.swizzle4::<3, 2, 1, 0>(), float4::new(4.0, 3.0, 2.0, 1.0)); // reverse
    assert_eq!(v.swizzle3::<0, 0, 0>(), float3::splat(1.0)); // splat
    assert_eq!(v.swizzle2::<3, 1>(), float2::new(4.0, 2.0));
    assert_eq!(
        float2::new(1.0, 2.0).swizzle4::<1, 0, 1, 0>(),
        float4::new(2.0, 1.0, 2.0, 1.0)
    );
    assert_eq!(v.rgb(), float3::new(1.0, 2.0, 3.0));
    assert_eq!(v.rg(), float2::new(1.0, 2.0));
}

#[test]
fn algebraic() {
    assert_eq!(clamp(5.0f32, 0.0, 1.0), 1.0);
    assert_eq!(saturate(-3.0f32), 0.0);
    assert_eq!(lerp(0.0f32, 10.0, 0.25), 2.5);
    assert_eq!(
        lerp(float3::ZERO, float3::new(2.0, 4.0, 6.0), 0.5),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        lerp_vec(
            float3::ZERO,
            float3::splat(10.0),
            float3::new(0.0, 0.5, 1.0)
        ),
        float3::new(0.0, 5.0, 10.0)
    );
    assert_eq!(
        min(float3::new(1.0, 5.0, 3.0), float3::new(4.0, 2.0, 6.0)),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        max(float3::new(1.0, 5.0, 3.0), float3::splat(4.0)),
        float3::new(4.0, 5.0, 4.0)
    );
    assert_eq!(min(3, 7), 3);
    assert_eq!(
        abs(float3::new(-1.0, 2.0, -3.0)),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(abs(-2.5f32), 2.5);
    assert_eq!(
        sign(float3::new(-1.0, 0.0, 3.0)),
        float3::new(-1.0, 0.0, 1.0)
    );
    assert_eq!(step(0.5f32, 0.75f32), 1.0);
    assert_eq!(
        step(0.5, float3::new(0.0, 0.5, 1.0)),
        float3::new(0.0, 1.0, 1.0)
    );
    assert_eq!(
        reflect(float3::new(1.0, -1.0, 0.0), float3::Y),
        float3::new(1.0, 1.0, 0.0)
    ); // bounce off floor
    assert_eq!(distance2(float3::ZERO, float3::new(0.0, 3.0, 4.0)), 25.0); // 5^2, no sqrt
    assert_eq!(close_to(1.0f32, 1.05, 0.1), 1.0);
    assert_eq!(close_to(1.0f32, 1.5, 0.1), 0.0);
    assert_eq!(min_component(float3::new(3.0, -1.0, 2.0)), -1.0);
    assert_eq!(max_component(int4::new(3, -1, 9, 2)), 9);
}

#[test]
fn nan_semantics_match_hlsl() {
    // The comparison is false for NaN, so min/max return `b`.
    assert_eq!(min(f32::NAN, 1.0), 1.0);
    assert!(min(1.0, f32::NAN).is_nan());
    // clamp = max(lo, min(x, hi)): min(NaN, hi) picks hi, so clamp/saturate map NaN
    // to the UPPER bound instead of propagating it (same as vkm / HLSL).
    assert_eq!(saturate(f32::NAN), 1.0);
    assert_eq!(clamp(f32::NAN, -2.0, 5.0), 5.0);
    assert_eq!(
        saturate(float3::new(f32::NAN, -1.0, 0.5)),
        float3::new(1.0, 0.0, 0.5)
    );
}

#[test]
fn transcendental() {
    check!(near(rsqrt(4.0f32), 0.5));
    check!(near(frac(3.25f32), 0.25));
    check!(near(
        distance(float3::ZERO, float3::new(0.0, 3.0, 4.0)),
        5.0
    ));

    let v = sqrt(float3::new(4.0, 9.0, 16.0));
    check!(near3(v, float3::new(2.0, 3.0, 4.0)));

    let f = floor(float3::new(1.7, 2.2, -0.3));
    check!(near3(f, float3::new(1.0, 2.0, -1.0)));

    check!(near(smoothstep(0.0f32, 1.0, 0.5), 0.5));
    check!(near3(
        smoothstep(0.0, 1.0, float3::new(0.0, 0.5, 1.0)),
        float3::new(0.0, 0.5, 1.0)
    ));

    // Trig family.
    check!(near(sin(0.0f32), 0.0));
    check!(near(cos(0.0f32), 1.0));
    check!(near(asin(1.0f32), core::f32::consts::FRAC_PI_2));
    check!(near(acos(1.0f32), 0.0));
    check!(near(atan2(1.0f32, 1.0), core::f32::consts::FRAC_PI_4));
    let (s, c) = sincos(0.0f32);
    check!(near(s, 0.0));
    check!(near(c, 1.0));
    check!(near3(sin(float3::ZERO), float3::ZERO));
    check!(near3(cos(float3::ZERO), float3::ONE));

    // Misc.
    check!(near(fmod(5.5f32, 2.0), 1.5));
    check!(near(fmod(-5.5f32, 2.0), -1.5)); // sign follows `a`
    check!(near(exp2(3.0f32), 8.0));
    check!(near(log2(8.0f32), 3.0));
    check!(near(log(core::f32::consts::E), 1.0));
    check!(near(pow(2.0f32, 10.0), 1024.0));
    check!(near3(
        pow(float3::splat(2.0), float3::new(1.0, 2.0, 3.0)),
        float3::new(2.0, 4.0, 8.0)
    ));

    // Works for f64 too.
    assert!((sqrt(2.0f64) - core::f64::consts::SQRT_2).abs() < 1e-12);
}

#[test]
fn angle_does_not_overflow_on_large_inputs() {
    // Regression: the denominator was sqrt(length2(a) * length2(b)), which hits inf
    // at |a|*|b| ~ 1.8e19 and made every angle come out pi/2 — even for parallel vectors.
    check!(near(
        angle(float3::new(3e10, 0.0, 0.0), float3::new(0.0, 3e10, 0.0)),
        PI / 2.0
    ));
    check!(near(
        angle(float3::new(3e10, 0.0, 0.0), float3::new(3e10, 0.0, 0.0)),
        0.0
    ));
}

#[test]
fn game_loop_helpers() {
    check!(near(repeat(7.0f32, 3.0), 1.0));
    check!(near(repeat(-1.0f32, 3.0), 2.0));
    check!(near(ping_pong(4.0f32, 3.0), 2.0));
    check!(near(lerp_angle(0.0f32, 6.0, 0.5), -0.14159265)); // crosses the seam
    check!(near(rad(180.0f32), PI));
    check!(near(deg(PI), 180.0));
}
