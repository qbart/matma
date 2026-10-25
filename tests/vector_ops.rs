//! Unity-style vector ops: angle / project / project_on_plane / move_towards /
//! rotate_towards (plus a reflect sanity check).

mod util;
use core::f32::consts::PI;
use matma::*;
use util::*;

#[test]
fn angle_radians() {
    check!(near(angle(float3::X, float3::Y), PI / 2.0)); // perpendicular
    check!(near(
        angle(float3::new(2.0, 0.0, 0.0), float3::new(5.0, 0.0, 0.0)),
        0.0
    )); // parallel
    check!(near(angle(float3::X, -float3::X), PI)); // opposite
    check!(near(
        angle(float3::new(3.0, 0.0, 0.0), float3::new(0.0, 0.5, 0.0)),
        PI / 2.0
    )); // non-unit
    check!(near(angle(float3::ZERO, float3::X), 0.0)); // degenerate -> 0, no NaN
    check!(near(float3::X.angle(float3::Y), PI / 2.0));
}

#[test]
fn project_and_project_on_plane() {
    check!(near3(
        project(float3::new(2.0, 3.0, 0.0), float3::X),
        float3::new(2.0, 0.0, 0.0)
    ));
    check!(near3(
        project(float3::new(1.0, 5.0, 1.0), float3::new(0.0, 2.0, 0.0)),
        float3::new(0.0, 5.0, 0.0)
    )); // non-unit normal
    check!(near3(
        project(float3::new(1.0, 2.0, 3.0), float3::ZERO),
        float3::ZERO
    )); // zero normal -> 0
    check!(near3(
        project_on_plane(float3::new(1.0, 2.0, 3.0), float3::Z),
        float3::new(1.0, 2.0, 0.0)
    )); // drop Z
    check!(near3(
        project_on_plane(float3::new(1.0, 2.0, 3.0), float3::ZERO),
        float3::new(1.0, 2.0, 3.0)
    )); // zero -> v

    // identity: projection + in-plane component reconstruct the original vector
    let (v, n) = (float3::new(1.0, 2.0, 3.0), float3::Y);
    check!(near3(project(v, n) + project_on_plane(v, n), v));
    check!(near3(v.project(n) + v.project_on_plane(n), v));
}

#[test]
fn reflect_matches_unity_form() {
    check!(near3(
        reflect(float3::new(1.0, -1.0, 0.0), float3::Y),
        float3::new(1.0, 1.0, 0.0)
    ));
}

#[test]
fn move_towards_scalar_and_vector() {
    check!(near(move_towards(0.0f32, 10.0, 3.0), 3.0)); // step
    check!(near(move_towards(0.0f32, 10.0, 100.0), 10.0)); // clamp, no overshoot
    check!(near(move_towards(10.0f32, 0.0, 3.0), 7.0)); // toward smaller
    check!(near3(
        move_towards(float3::ZERO, float3::new(10.0, 0.0, 0.0), 3.0),
        float3::new(3.0, 0.0, 0.0)
    ));
    check!(near3(
        move_towards(float3::ZERO, float3::new(10.0, 0.0, 0.0), 100.0),
        float3::new(10.0, 0.0, 0.0)
    ));
    check!(near3(
        float3::ZERO.move_towards(float3::new(10.0, 0.0, 0.0), 3.0),
        float3::new(3.0, 0.0, 0.0)
    ));
}

#[test]
fn rotate_towards_vector() {
    // enough angular budget -> snap onto target direction (lengths equal -> length stays 1)
    check!(near3(
        rotate_towards(float3::X, float3::Y, PI, 0.0),
        float3::Y
    ));
    // partial 45-degree turn within the plane, unit length preserved
    check!(near3(
        rotate_towards(float3::X, float3::Y, PI / 4.0, 0.0),
        float3::new((PI / 4.0).cos(), (PI / 4.0).sin(), 0.0)
    ));
    // magnitude steps toward |target| by at most max_magnitude_delta (dir snaps via large angle budget)
    check!(near3(
        rotate_towards(float3::X, float3::new(0.0, 5.0, 0.0), PI, 2.0),
        float3::new(0.0, 3.0, 0.0)
    ));
    // antiparallel with full budget -> lands on target (the degenerate-axis path)
    check!(near3(
        rotate_towards(float3::X, -float3::X, PI, 0.0),
        -float3::X
    ));
    // antiparallel, partial turn -> stays unit length and is perpendicular to the start
    let r = rotate_towards(float3::X, -float3::X, PI / 2.0, 0.0);
    check!(near(length(r), 1.0));
    check!(near(dot(r, float3::X), 0.0));
    // zero start: no direction, take target's direction at the stepped magnitude
    check!(near3(
        rotate_towards(float3::ZERO, float3::new(0.0, 3.0, 0.0), PI, 2.0),
        float3::new(0.0, 2.0, 0.0)
    ));
}

#[test]
fn ray_triangle() {
    let ray = Ray::new(float3::new(0.2, 0.2, 5.0), float3::new(0.0, 0.0, -1.0));
    let (a, b, c) = (float3::ZERO, float3::X, float3::Y);
    let t = ray_triangle_intersect(&ray, a, b, c).expect("hit");
    check!(near(t, 5.0));
    check!(near3(ray.at(t), float3::new(0.2, 0.2, 0.0)));
    // miss: outside the triangle
    assert!(
        ray_triangle_intersect(
            &Ray::new(float3::new(2.0, 2.0, 5.0), float3::FORWARD),
            a,
            b,
            c
        )
        .is_none()
    );
    // miss: pointing away
    assert!(
        ray_triangle_intersect(&Ray::new(float3::new(0.2, 0.2, 5.0), float3::BACK), a, b, c)
            .is_none()
    );
    // miss: parallel
    assert!(
        ray_triangle_intersect(&Ray::new(float3::new(0.2, 0.2, 5.0), float3::X), a, b, c).is_none()
    );
}
