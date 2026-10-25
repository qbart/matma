//! The float4 fast path must agree with the scalar reference, whichever backend
//! is compiled in. Run with `--features force-scalar` to exercise the fallback.

use matma::*;

fn reference(a: float4, b: float4) -> [float4; 5] {
    let f = |op: fn(f32, f32) -> f32| {
        float4::new(op(a.x, b.x), op(a.y, b.y), op(a.z, b.z), op(a.w, b.w))
    };
    [
        f(|x, y| x + y),
        f(|x, y| x - y),
        f(|x, y| x * y),
        f(|x, y| x / y),
        float4::splat(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w),
    ]
}

#[test]
fn backend_is_reported() {
    println!("matma SIMD backend: {}", simd::BACKEND);
    assert!(["sse2", "neon", "scalar"].contains(&simd::BACKEND));
    #[cfg(feature = "force-scalar")]
    assert_eq!(simd::BACKEND, "scalar");
}

#[test]
fn float4_ops_match_scalar_reference() {
    // Deterministic pseudo-random values, incl. negatives, tiny and large magnitudes.
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let unit = (seed >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
        (unit - 0.5) * 2.0 * 10f32.powi((seed % 7) as i32 - 3)
    };
    for _ in 0..2000 {
        let a = float4::new(next(), next(), next(), next());
        let b = float4::new(next(), next(), next(), next());
        let [add, sub, mul, div, dot] = reference(a, b);
        assert_eq!(a + b, add);
        assert_eq!(a - b, sub);
        assert_eq!(a * b, mul);
        assert_eq!(a / b, div);
        // The horizontal sum may associate differently ((a+b)+(c+d) vs left-to-right),
        // so allow a few ulps relative to the magnitude of the TERMS (the result
        // itself may be tiny after cancellation).
        let d = a.dot(b);
        let scale = (a * b).abs().sum().max(f32::MIN_POSITIVE);
        assert!(
            (d - dot.x).abs() <= scale * 4.0 * f32::EPSILON,
            "{a:?}·{b:?}: {d} vs {}",
            dot.x
        );
    }
}

#[test]
fn raw_backend_ops() {
    use matma::simd::*;
    let a = load4([1.0, 2.0, 3.0, 4.0]);
    let b = splat(2.0);
    assert_eq!(store4(add(a, b)), [3.0, 4.0, 5.0, 6.0]);
    assert_eq!(store4(sub(a, b)), [-1.0, 0.0, 1.0, 2.0]);
    assert_eq!(store4(mul(a, b)), [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(store4(div(a, b)), [0.5, 1.0, 1.5, 2.0]);
    assert_eq!(store4(neg(a)), [-1.0, -2.0, -3.0, -4.0]);
    assert_eq!(hsum(a), 10.0);
}
