//! GPU packing tests: snorm/unorm quantization round trips + octahedral
//! normal/tangent encode/decode + half-float packing.

mod util;
use matma::*;
use util::*;

// Compile-time round trips (the encoders are const fn).
const _: () = {
    assert!(dequantize_unorm16(quantize_unorm16(1.0)) == 1.0);
    assert!(quantize_snorm10(1.0) == 511 && quantize_unorm10(1.0) == 1023);
};
const UP_PACKED: short2 = oct_encode16(float3::new(0.0, 1.0, 0.0));

#[test]
fn snorm_round_trips() {
    // error <= 0.5 / scale
    for v in [-1.0f32, -0.5, -0.123, 0.0, 0.25, 0.5, 1.0] {
        assert!(near_eps(dequantize_snorm16(quantize_snorm16(v)), v, 4e-5));
        assert!(near_eps(dequantize_snorm8(quantize_snorm8(v)), v, 1e-2));
        assert!(near_eps(dequantize_snorm10(quantize_snorm10(v)), v, 2.5e-3));
    }
}

#[test]
fn unorm_round_trips() {
    for v in [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
        assert!(near_eps(dequantize_unorm16(quantize_unorm16(v)), v, 2e-5));
        assert!(near_eps(dequantize_unorm8(quantize_unorm8(v)), v, 4e-3));
        assert!(near_eps(dequantize_unorm10(quantize_unorm10(v)), v, 1e-3));
    }
}

#[test]
fn clamping() {
    // encode clamps the input, snorm decode clamps the output
    assert_eq!(quantize_snorm16(2.0), 32767);
    assert_eq!(quantize_snorm16(-2.0), -32767);
    assert!(near_eps(dequantize_snorm16(-32768), -1.0, 1e-6)); // most-negative code -> -1
    assert_eq!(quantize_unorm8(2.0), 255);
    assert_eq!(quantize_unorm8(-1.0), 0);
    assert_eq!(quantize_snorm8(f32::NAN), 0); // NaN -> 0, not UB
}

#[test]
fn octahedral_normal_round_trip() {
    // incl. both hemispheres + axes/poles
    for d in [
        float3::X,
        float3::Y,
        float3::Z,
        -float3::Z,
        -float3::X,
        float3::ONE,
        float3::new(-1.0, 2.0, -3.0),
        float3::new(0.3, -0.7, 0.6),
        float3::new(2.0, -1.0, -2.0),
    ] {
        let n = d.normalize();
        let r = oct_decode16(oct_encode16(n));
        assert!(near_eps(r.length(), 1.0, 1e-4)); // unproject re-normalizes
        assert!(n.dot(r) > 0.999, "{n:?} -> {r:?}"); // 16-bit oct is accurate
    }
    assert_eq!(oct_encode16(float3::Z), short2::new(0, 0));
    check!(near3(oct_decode16(UP_PACKED), float3::Y));
    // project/unproject without quantization is (nearly) exact
    let n = float3::new(0.3, -0.7, -0.6).normalize();
    check!(near3(oct_unproject(oct_project(n)), n));
}

#[test]
fn tangent_packing() {
    // handedness in the LSB, direction preserved
    let t = float3::new(1.0, 2.0, 3.0).normalize();
    for h in [1.0f32, -1.0] {
        let enc = oct_encode_tangent(t, h);
        assert_eq!(enc.x & 1, if h < 0.0 { 1 } else { 0 }); // LSB encodes the sign
        let (dt, dh) = oct_decode_tangent(enc);
        assert_eq!(dh, h);
        assert!(t.dot(dt) > 0.999);
        assert!(near_eps(dt.length(), 1.0, 1e-4));
    }
    // bitangent reconstruction (caller-side), as documented:
    let n = float3::Y;
    let (dt, dh) = oct_decode_tangent(oct_encode_tangent(t, -1.0));
    let bitangent = n.cross(dt) * dh;
    assert!(near_eps(bitangent.length(), n.cross(dt).length(), 1e-4)); // sign-flip keeps magnitude
}

#[test]
fn half_float_packing() {
    let rt = unpack_half2x16(pack_half2x16(float2::new(1.5, -0.25)));
    assert_eq!(rt, float2::new(1.5, -0.25)); // exactly representable in half

    // Overflow -> +/-inf; NaN stays NaN. (Regression: the overflow branch dropped
    // the mantissa, so a packed NaN unpacked as +inf.)
    let s = unpack_half2x16(pack_half2x16(float2::new(1e30, f32::NAN)));
    assert!(s.x.is_infinite() && s.x > 0.0);
    assert!(s.y.is_nan());
    let s2 = unpack_half2x16(pack_half2x16(float2::new(-1e30, -f32::NAN)));
    assert!(s2.x.is_infinite() && s2.x < 0.0);
    assert!(s2.y.is_nan());

    // Known bit patterns.
    assert_eq!(float_to_half(1.0), 0x3C00);
    assert_eq!(float_to_half(-2.0), 0xC000);
    assert_eq!(float_to_half(0.0), 0x0000);
    assert_eq!(float_to_half(-0.0), 0x8000);
    assert_eq!(float_to_half(65504.0), 0x7BFF); // max half
    assert_eq!(half_to_float(0x3C00), 1.0);
    assert_eq!(half_to_float(0x0001), 2f32.powi(-24)); // smallest half subnormal
    assert!(half_to_float(0x7C00).is_infinite());
    assert_eq!(float_to_half(1e-8), 0); // underflow flushes to zero
    // Round-trip a sweep with the expected relative error (10-bit mantissa).
    let mut x = 1e-4f32;
    while x < 60000.0 {
        let y = half_to_float(float_to_half(x));
        assert!(((y - x) / x).abs() < 1e-3, "{x} -> {y}");
        x *= 1.37;
    }
}
