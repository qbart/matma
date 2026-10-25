//! Method-style API + compound operators.

mod util;
use matma::*;
use util::*;

#[test]
fn vector_methods() {
    assert_eq!(float3::new(0.0, 3.0, 4.0).length2(), 25.0);
    assert_eq!(
        float3::new(-1.0, 2.0, -3.0).abs(),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        float3::new(5.0, -1.0, 0.5).saturate(),
        float3::new(1.0, 0.0, 0.5)
    );
    assert_eq!(
        float3::new(5.0, -1.0, 3.0).clamp_scalar(0.0, 2.0),
        float3::new(2.0, 0.0, 2.0)
    );
    assert_eq!(
        float3::new(5.0, -1.0, 3.0).clamp(float3::ZERO, float3::new(2.0, 2.0, 10.0)),
        float3::new(2.0, 0.0, 3.0)
    );
    assert_eq!(
        float3::new(1.0, -1.0, 0.0).reflect(float3::Y),
        float3::new(1.0, 1.0, 0.0)
    );
    assert_eq!(
        float3::new(-2.0, 0.0, 3.0).sign(),
        float3::new(-1.0, 0.0, 1.0)
    );
    assert_eq!(
        float3::new(1.0, 5.0, 3.0).min(float3::new(4.0, 2.0, 6.0)),
        float3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        float3::new(1.0, 5.0, 3.0).max(float3::new(4.0, 2.0, 6.0)),
        float3::new(4.0, 5.0, 6.0)
    );
    assert_eq!(float3::new(1.0, 5.0, 3.0).min_component(), 1.0);
    assert_eq!(float3::new(1.0, 5.0, 3.0).max_component(), 5.0);
    assert_eq!(
        float3::ZERO.lerp(float3::new(2.0, 4.0, 6.0), 0.5),
        float3::new(1.0, 2.0, 3.0)
    );

    let v = float3::new(0.0, 3.0, 4.0);
    check!(near(v.length(), 5.0));
    check!(near3(v.normalize(), float3::new(0.0, 0.6, 0.8)));
    check!(near3(v, float3::new(0.0, 3.0, 4.0))); // normalize() did not mutate

    let p = float3::new(1.0, 2.0, 3.0);
    assert_eq!(p.as_slice(), &[1.0, 2.0, 3.0]); // contiguous
    let mut q = p;
    q.as_mut_slice()[1] = 7.0;
    assert_eq!(q.y, 7.0);

    let f = float3::new(1.7, 2.2, -0.3);
    assert_eq!(f.floor(), float3::new(1.0, 2.0, -1.0));
    assert_eq!(f.ceil(), float3::new(2.0, 3.0, -0.0));
    assert_eq!(f.round(), float3::new(2.0, 2.0, -0.0));
    assert_eq!(f.trunc(), float3::new(1.0, 2.0, -0.0));
    check!(near3(f.frac(), float3::new(0.7, 0.2, 0.7)));
    assert_eq!(
        float3::new(4.0, 9.0, 16.0).sqrt(),
        float3::new(2.0, 3.0, 4.0)
    );
}

#[test]
fn compound_assignment() {
    let mut v = float3::new(1.0, 2.0, 3.0);
    v += float3::ONE;
    assert_eq!(v, float3::new(2.0, 3.0, 4.0));
    v -= 1.0;
    assert_eq!(v, float3::new(1.0, 2.0, 3.0));

    let mut a = float4::new(1.0, 2.0, 3.0, 4.0);
    a += float4::splat(10.0);
    assert_eq!(a, float4::new(11.0, 12.0, 13.0, 14.0));
    a -= float4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(a, float4::splat(10.0));
    a *= 0.5;
    assert_eq!(a, float4::splat(5.0));
    a /= 5.0;
    assert_eq!(a, float4::ONE);
    let mut b = float4::splat(2.0);
    b *= float4::splat(3.0);
    assert_eq!(b, float4::splat(6.0));
    b /= float4::splat(2.0);
    assert_eq!(b, float4::splat(3.0));
}

#[test]
fn quaternion_methods() {
    assert_eq!(
        quat::new(1.0, 2.0, 3.0, 4.0).conjugate(),
        quat::new(-1.0, -2.0, -3.0, 4.0)
    );
    let q = quat::from_axis_angle(float3::Z, rad(90.0));
    check!(near(q.length(), 1.0));
    // q * inverse(q) is the identity rotation.
    check!(near3((q * q.inverse()).rotate(float3::X), float3::X));
    // For a unit quaternion, conjugate() == inverse().
    check!(near3(q.conjugate() * float3::Y, q.inverse() * float3::Y));
    assert_eq!(q.as_slice()[3], q.w);
    assert_eq!(q.to_array()[3], q.w);
    let mut acc = quat::IDENTITY;
    acc *= q; // in-place Hamilton product
    check!(near3(acc * float3::X, q * float3::X));
    assert_eq!(quat::default(), quat::IDENTITY);
    // v * q rotates by the inverse.
    check!(near3(float3::X * q, q.inverse() * float3::X));
    check!(near3((q * float3::X) * q, float3::X));
}

#[test]
fn matrix_methods() {
    let q = quat::from_axis_angle(float3::Z, rad(90.0));
    let m = compose_trs(float3::new(1.0, 2.0, 3.0), q, float3::splat(2.0));
    check!(near(m.determinant(), 8.0)); // det(T R S) = det(S) = 2^3
    check!(near_m4(&(m * m.inverse()), &float4x4::IDENTITY));
    assert_eq!(m.transpose().transpose(), m); // double transpose round-trips
    assert_eq!(m.row(0), m.transpose().col(0));

    // as_slice / to_cols_array are column-major: a translation's components land in the last 4 floats.
    let t = translate(float3::new(7.0, 8.0, 9.0));
    let s = t.as_slice();
    assert_eq!(&s[12..16], &[7.0, 8.0, 9.0, 1.0]);
    assert_eq!(s.len(), 16);
    let a = t.to_cols_array();
    assert_eq!(&a[12..16], &[7.0, 8.0, 9.0, 1.0]);
    assert_eq!(float4x4::from_cols_array(a), t);

    // Non-square: 3 rows x 4 columns packs 12 floats, transposes to 4x3.
    let m34 = float3x4::from_cols([float3::X, float3::Y, float3::Z, float3::splat(7.0)]);
    assert_eq!(m34.as_slice().len(), 12);
    let m43: float4x3 = m34.transpose();
    assert_eq!(m43.cols[0], float4::new(1.0, 0.0, 0.0, 7.0));
    assert_eq!(m34.row(0), float4::new(1.0, 0.0, 0.0, 7.0));
    assert_eq!(
        float3x4::from_rows([m34.row(0), m34.row(1), m34.row(2)]),
        m34
    );

    let mut i2 = float4x4::IDENTITY;
    i2 += float4x4::IDENTITY;
    check!(near_m4(&i2, &(2.0 * float4x4::IDENTITY)));
    i2 *= 0.5;
    check!(near_m4(&i2, &float4x4::IDENTITY));
    let mut m2 = m;
    m2 *= m.inverse();
    check!(near_m4(&m2, &float4x4::IDENTITY));
    assert_eq!(float4x4::IDENTITY.trace(), 4.0);
    assert_eq!(float3x3::identity(), float3x3::IDENTITY);
    assert_eq!(float2x2::identity(), float2x2::IDENTITY);
}
