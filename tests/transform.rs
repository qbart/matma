//! Transform + quaternion + Vulkan-projection tests.

mod util;
use matma::*;
use util::*;

#[test]
fn quaternion_rotation_and_interpolation() {
    // 90 deg about +Z maps +X -> +Y
    let qz = quat::from_axis_angle(float3::Z, rad(90.0));
    check!(near3(qz * float3::X, float3::Y)); // optimized rotate
    check!(near4(
        qz.to_mat4() * float4::new(1.0, 0.0, 0.0, 1.0),
        float4::new(0.0, 1.0, 0.0, 1.0)
    )); // via matrix
    check!(near4(
        rotate(rad(90.0), float3::Z) * float4::new(1.0, 0.0, 0.0, 1.0),
        float4::new(0.0, 1.0, 0.0, 1.0)
    ));
    // Composition order: q*q = 180 deg about Z maps +X -> -X.
    check!(near3((qz * qz) * float3::X, -float3::X));
    check!(near3(mul(qz, qz) * float3::X, -float3::X));

    // halfway between identity and a 90deg-about-Z rotation is a 45deg rotation.
    let half = slerp(quat::IDENTITY, qz, 0.5);
    let s = core::f32::consts::FRAC_1_SQRT_2; // sin(45) = cos(45)
    check!(near3(half * float3::X, float3::new(s, s, 0.0)));
    // endpoints are exact.
    check!(near3(slerp(quat::IDENTITY, qz, 0.0) * float3::X, float3::X));
    check!(near3(slerp(quat::IDENTITY, qz, 1.0) * float3::X, float3::Y));
    // nlerp lands near the same place at the midpoint (close orientations).
    check!(near3(
        nlerp(quat::IDENTITY, qz, 0.5) * float3::X,
        float3::new(s, s, 0.0)
    ));
    // slerp takes the short way round when the dot is negative.
    check!(near3(
        quat::IDENTITY.slerp(-qz, 0.5) * float3::X,
        float3::new(s, s, 0.0)
    ));
    // rotate_deg yaws about UP by default usage
    check!(near3(
        rotate_deg(quat::IDENTITY, 90.0, float3::UP) * float3::X,
        float3::FORWARD
    ));
}

#[test]
fn affine() {
    check!(near4(
        translate(float3::new(10.0, 20.0, 30.0)) * float4::new(1.0, 2.0, 3.0, 1.0),
        float4::new(11.0, 22.0, 33.0, 1.0)
    ));
    check!(near4(
        scale(float3::new(2.0, 3.0, 4.0)) * float4::ONE,
        float4::new(2.0, 3.0, 4.0, 1.0)
    ));
    assert_eq!(translate(float3::X), float4x4::from_translation(float3::X));
    assert_eq!(scale(float3::X), float4x4::from_scale(float3::X));
    // TRS applies scale, then rotation, then translation.
    let q = quat::from_axis_angle(float3::Z, rad(90.0));
    let m = compose_trs(float3::new(0.0, 0.0, 5.0), q, float3::splat(2.0));
    check!(near4(
        m * float4::new(1.0, 0.0, 0.0, 1.0),
        float4::new(0.0, 2.0, 5.0, 1.0)
    ));
    assert_eq!(
        m,
        float4x4::from_trs(float3::new(0.0, 0.0, 5.0), q, float3::splat(2.0))
    );
}

#[test]
fn look_at_right_handed() {
    // camera 5 units down +Z, looking at origin
    let v = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::Y);
    check!(near4(v * float4::W, float4::new(0.0, 0.0, -5.0, 1.0))); // origin -> 5 in front (-Z)
    check!(near4(v * float4::new(0.0, 0.0, 5.0, 1.0), float4::W)); // eye -> view origin
}

#[test]
fn perspective_vulkan_depth_and_y_flip() {
    let (n, f) = (0.1, 100.0);
    let p = perspective(rad(60.0), 16.0 / 9.0, n, f);
    let cnear = p * float4::new(0.0, 0.0, -n, 1.0); // point on near plane
    let cfar = p * float4::new(0.0, 0.0, -f, 1.0); // point on far plane
    check!(near(cnear.z / cnear.w, 0.0)); // near -> 0
    check!(near(cfar.z / cfar.w, 1.0)); // far  -> 1

    // Y is flipped (a point ABOVE center -> negative clip Y)
    let cup = p * float4::new(0.0, 1.0, -1.0, 1.0);
    assert!(cup.y < 0.0);
}

#[test]
fn ortho_vulkan_depth() {
    let (n, f) = (0.1, 100.0);
    let o = ortho(-1.0, 1.0, -1.0, 1.0, n, f);
    check!(near((o * float4::new(0.0, 0.0, -n, 1.0)).z, 0.0));
    check!(near((o * float4::new(0.0, 0.0, -f, 1.0)).z, 1.0));
    // Y flipped: top of the box maps to clip -1.
    check!(near((o * float4::new(0.0, 1.0, -1.0, 1.0)).y, -1.0));
}

#[test]
fn full_mvp_pipeline() {
    let p = perspective(rad(60.0), 16.0 / 9.0, 0.1, 100.0);
    let v = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::Y);
    let m = compose_trs(float3::ZERO, quat::IDENTITY, float3::ONE);
    let mvp = p * v * m;
    let clip = mvp * float4::W; // origin should be in front of camera
    assert!(clip.w > 0.0); // positive w -> in front, will divide sanely
    let ndc = clip.xyz() / clip.w;
    assert!(ndc.z > 0.0 && ndc.z < 1.0);
}

#[test]
fn decompose_trs() {
    let t = float3::new(3.0, -2.0, 5.0);
    let r = quat::from_axis_angle(float3::new(0.3, 1.0, 0.5), rad(40.0));
    let s = float3::new(2.0, 0.5, 1.5);
    let d = decompose(&compose_trs(t, r, s)).expect("invertible");
    check!(near3(d.translation, t));
    check!(near3(d.scale, s));
    check!(near3(d.skew, float3::ZERO));
    check!(near4(d.perspective, float4::W));
    for v in [float3::X, float3::Y, float3::new(1.0, 2.0, 3.0)] {
        check!(near3(d.rotation * v, r * v)); // sign-agnostic comparison
    }
    // Rebuilt matrix matches.
    check!(near_m4(
        &compose_trs(d.translation, d.rotation, d.scale),
        &compose_trs(t, r, s)
    ));

    // Degenerate: a zero-scale matrix is not decomposable.
    assert!(decompose(&scale(float3::new(1.0, 0.0, 1.0))).is_none());
    assert!(float4x4::ZERO.decompose().is_none());

    // Mirrored basis -> negative scale.
    let d = decompose(&scale(float3::new(-1.0, 1.0, 1.0))).unwrap();
    assert!(d.scale.x < 0.0 || d.scale.y < 0.0 || d.scale.z < 0.0);
    assert!(d.scale.product() < 0.0);

    // Projective matrix: the perspective row is recovered.
    let mut proj = compose_trs(t, r, s);
    proj[(3, 0)] = 0.1;
    proj[(3, 2)] = -0.3;
    let d = decompose(&proj).unwrap();
    assert!(!near4(d.perspective, float4::W));
    // The affine part is unaffected, and the perspective vector `p` is defined
    // (as in GLM / Graphics Gems "unmatrix") by  bottom_row = affineᵀ · p.
    check!(near3(d.scale, s));
    check!(near3(d.translation, t));
    check!(near4(
        compose_trs(t, r, s).transpose() * d.perspective,
        float4::new(0.1, 0.0, -0.3, 1.0)
    ));

    // A perspective projection has m[(3,3)] == 0 and is (by design) not decomposable.
    assert!(decompose(&perspective(rad(60.0), 1.0, 0.1, 100.0)).is_none());
}
