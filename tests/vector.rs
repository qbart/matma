//! Vector smoke tests: component-wise ops, dot/cross, construction, both the
//! generic path (float3, int3) and the 4-lane hook path (float4 -> SIMD backend).

mod util;
use matma::*;
use util::*;

#[test]
fn arithmetic_float4() {
    let a = float4::new(1.0, 2.0, 3.0, 4.0);
    let b = float4::new(5.0, 6.0, 7.0, 8.0);

    assert_eq!(a + b, float4::new(6.0, 8.0, 10.0, 12.0));
    assert_eq!(b - a, float4::new(4.0, 4.0, 4.0, 4.0));
    assert_eq!(a * b, float4::new(5.0, 12.0, 21.0, 32.0));
    assert_eq!(b / a, float4::new(5.0, 3.0, 7.0 / 3.0, 2.0));
    assert_eq!(a * 2.0, float4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(2.0 * a, float4::new(2.0, 4.0, 6.0, 8.0));
    assert_eq!(a / 2.0, float4::new(0.5, 1.0, 1.5, 2.0));
    assert_eq!(a + 1.0, float4::new(2.0, 3.0, 4.0, 5.0));
    assert_eq!(-a, float4::new(-1.0, -2.0, -3.0, -4.0));
    check!(near(dot(a, b), 5.0 + 12.0 + 21.0 + 32.0));
    check!(near(a.dot(b), 70.0));
    check!(near(
        dot(float4::new(1.0, 2.0, 3.0, 4.0), float4::ONE),
        10.0
    ));
}

#[test]
fn arithmetic_float3_and_ints() {
    let a = float3::new(1.0, 2.0, 3.0);
    let b = float3::new(4.0, 5.0, 6.0);
    assert_eq!(a + b, float3::new(5.0, 7.0, 9.0));
    assert_eq!(a * 2.0, float3::new(2.0, 4.0, 6.0));
    assert_eq!(2.0 * a, float3::new(2.0, 4.0, 6.0));
    assert_eq!(dot(float3::X, float3::Y), 0.0);
    assert_eq!(cross(float3::X, float3::Y), float3::Z);
    assert_eq!(float3::X.cross(float3::Y), float3::Z);

    assert_eq!(int3::new(1, 2, 3) * int3::new(2, 2, 2), int3::new(2, 4, 6));
    assert_eq!(
        uint4::new(1, 2, 3, 4) + uint4::splat(1),
        uint4::new(2, 3, 4, 5)
    );
    assert_eq!(int2::new(-3, 4).abs(), int2::new(3, 4));
    assert_eq!(uint2::new(3, 4).abs(), uint2::new(3, 4)); // no-op for unsigned, no Neg needed
    assert_eq!(short2::new(-3, 4) * 2, short2::new(-6, 8));
}

#[test]
fn construction_and_conversion() {
    assert_eq!(
        float3::new(1.0, 2.0, 3.0).extend(4.0),
        float4::new(1.0, 2.0, 3.0, 4.0)
    );
    assert_eq!(
        float2::new(1.0, 2.0).extend(3.0),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        float4::new(1.0, 2.0, 3.0, 4.0).truncate(),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(float3::splat(7.0), float3::new(7.0, 7.0, 7.0));
    assert_eq!(float3::from([1.0, 2.0, 3.0]), float3::new(1.0, 2.0, 3.0));
    assert_eq!(
        <[f32; 3]>::from(float3::new(1.0, 2.0, 3.0)),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(float3::from((1.0, 2.0, 3.0)), float3::new(1.0, 2.0, 3.0));
    assert_eq!(
        float4::from((float3::new(1.0, 2.0, 3.0), 4.0)),
        float4::new(1.0, 2.0, 3.0, 4.0)
    );
    assert_eq!(
        float3::new(1.5, 2.5, 3.5).map(|x| x as i32),
        int3::new(1, 2, 3)
    );
    assert_eq!(float3::ZERO, float3::default());

    // Indexing.
    let v = float4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!((v[0], v[1], v[2], v[3]), (1.0, 2.0, 3.0, 4.0));
    let mut m = v;
    m[2] = 9.0;
    assert_eq!(m.z, 9.0);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic]
fn index_out_of_range_panics() {
    let v = float3::ZERO;
    let _ = v[3];
}

#[test]
fn geometry() {
    let n = normalize(float3::new(0.0, 3.0, 4.0));
    check!(near(length(n), 1.0));
    check!(near3(n, float3::new(0.0, 0.6, 0.8)));
    check!(near(length(float4::new(0.0, 0.0, 0.0, 2.0)), 2.0));
    assert_eq!(length2(float3::new(0.0, 3.0, 4.0)), 25.0);
    assert_eq!(float3::ZERO.normalize_or_zero(), float3::ZERO);
    assert!(float3::ZERO.normalize().is_nan());
}

#[test]
fn layout_is_tightly_packed_repr_c() {
    use core::mem::{align_of, size_of};
    assert_eq!(size_of::<float2>(), 8);
    assert_eq!(size_of::<float3>(), 12);
    assert_eq!(size_of::<float4>(), 16);
    assert_eq!(size_of::<double4>(), 32);
    assert_eq!(size_of::<short2>(), 4);
    assert_eq!(align_of::<short2>(), 2);
    assert_eq!(size_of::<ubyte4>(), 4);
    assert_eq!(size_of::<bool3>(), 3);
    assert_eq!(size_of::<float4x4>(), 64);
    assert_eq!(size_of::<float3x3>(), 36);
}

#[test]
fn bool_vectors_and_comparisons() {
    let a = float3::new(1.0, 5.0, 3.0);
    let b = float3::new(4.0, 2.0, 3.0);
    assert_eq!(a.cmp_lt(b), bool3::new(true, false, false));
    assert_eq!(a.cmp_le(b), bool3::new(true, false, true));
    assert_eq!(a.cmp_eq(b), bool3::new(false, false, true));
    assert!(a.cmp_lt(b).any());
    assert!(!a.cmp_lt(b).all());
    assert!((!a.cmp_lt(b)).any());
    assert_eq!(
        float3::select(a.cmp_lt(b), a, b),
        float3::new(1.0, 2.0, 3.0)
    ); // == min
    assert_eq!(float3::select(a.cmp_lt(b), a, b), min(a, b));
}

#[test]
fn direction_constants_right_handed_y_up() {
    assert_eq!(float3::UP, float3::new(0.0, 1.0, 0.0));
    assert_eq!(float3::FORWARD, float3::new(0.0, 0.0, -1.0));
    assert_eq!(float3::FORWARD, -float3::BACK);
    assert_eq!(cross(float3::RIGHT, float3::UP), float3::BACK); // RH basis: X x Y = +Z = back
    assert_eq!(int3::LEFT, int3::new(-1, 0, 0)); // works for any signed scalar
}

#[test]
fn sum_iterator() {
    let pts = [float3::X, float3::Y, float3::Z];
    assert_eq!(pts.iter().sum::<float3>(), float3::ONE);
    assert_eq!(pts.into_iter().sum::<float3>(), float3::ONE);
    assert_eq!(float3::new(1.0, 2.0, 3.0).sum(), 6.0);
    assert_eq!(float3::new(1.0, 2.0, 3.0).product(), 6.0);
}
