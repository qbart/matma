//! Determinant / inverse / normal-matrix tests.

mod util;
use matma::*;
use util::*;

#[test]
fn determinants() {
    assert_eq!(float2x2::IDENTITY.determinant(), 1.0);
    assert_eq!(float3x3::IDENTITY.determinant(), 1.0);
    assert_eq!(float4x4::IDENTITY.determinant(), 1.0);
    // determinant of a pure scale = product of the diagonal.
    check!(near(scale(float3::new(2.0, 3.0, 4.0)).determinant(), 24.0));
}

#[test]
fn inverse_round_trips() {
    // inverse of a translation is the negated translation (exact in float).
    assert_eq!(
        translate(float3::new(1.0, 2.0, 3.0)).inverse(),
        translate(float3::new(-1.0, -2.0, -3.0))
    );

    let a = float2x2::from_cols([float2::new(4.0, 2.0), float2::new(7.0, 6.0)]);
    let prod = a.inverse() * a;
    check!(near(prod[(0, 0)], 1.0));
    check!(near(prod[(1, 1)], 1.0));
    check!(near(prod[(1, 0)], 0.0));
    check!(near(prod[(0, 1)], 0.0));

    let b = float3x3::from_cols([
        float3::new(1.0, 2.0, 3.0),
        float3::new(0.0, 1.0, 4.0),
        float3::new(5.0, 6.0, 0.0),
    ]);
    check!(near_m3(&(b.inverse() * b), &float3x3::IDENTITY));

    // 4x4 inverse of a full TRS round-trips.
    let m = compose_trs(
        float3::new(3.0, -2.0, 5.0),
        quat::from_axis_angle(float3::new(0.3, 1.0, 0.5), rad(40.0)),
        float3::new(2.0, 0.5, 1.5),
    );
    check!(near_m4(&(m.inverse() * m), &float4x4::IDENTITY));
    check!(near_m4(&(m * m.inverse()), &float4x4::IDENTITY));

    // inverse of a view matrix recovers a point (un-project the camera-space origin).
    let v = look_at(float3::new(1.0, 2.0, 3.0), float3::ZERO, float3::Y);
    check!(near4(
        v.inverse() * float4::W,
        float4::new(1.0, 2.0, 3.0, 1.0)
    )); // camera-space origin -> eye
}

#[test]
fn normal_matrix_of_rotation_is_the_rotation() {
    let r = quat::from_axis_angle(float3::Y, rad(33.0));
    let m = r.to_mat4();
    check!(near_m3(&normal_matrix(&m), &r.to_mat3()));
    check!(near_m3(&m.normal_matrix(), &r.to_mat3()));
    // With non-uniform scale it differs from the plain upper-left 3x3 ...
    let ms = m * scale(float3::new(1.0, 2.0, 1.0));
    assert!(!near_m3(&ms.normal_matrix(), &ms.upper_left()));
    // ... but still maps the normal of a transformed plane correctly: a normal
    // (0,1,0) on a plane scaled by 2 in Y stays (0,1,0) after normalization.
    let n = (ms.normal_matrix() * float3::Y).normalize();
    check!(near3(n, (m.upper_left() * float3::Y).normalize()));
}

#[test]
fn double_precision_inverse() {
    let m = dmat4::from_translation(double3::new(1.0, 2.0, 3.0))
        * dmat4::from_scale(double3::new(2.0, 4.0, 8.0));
    let p = m.inverse() * m;
    for c in 0..4 {
        for r in 0..4 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert!((p[(r, c)] - expected).abs() < 1e-12);
        }
    }
}
