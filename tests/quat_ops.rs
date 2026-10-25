//! Unity-style quaternion ops: angle / from_to_rotation / look_rotation /
//! rotate_towards / from_mat3, plus inverse/normalize and the direction constants.

mod util;
use core::f32::consts::PI;
use matma::*;
use util::*;

#[test]
fn angle_between_rotations() {
    let q90 = quat::from_axis_angle(float3::Z, PI / 2.0);
    check!(near(quat::IDENTITY.angle_to(q90), PI / 2.0));
    check!(near(q90.angle_to(q90), 0.0));
    check!(near(q90.angle_to(-q90), 0.0)); // same rotation, other cover
}

#[test]
fn from_mat3_inverts_to_mat3() {
    // compare via rotation effect, sign-agnostic
    for q in [
        quat::from_axis_angle(float3::Y, 0.7),
        quat::from_axis_angle(float3::new(1.0, 2.0, 3.0), 2.1),
        quat::from_axis_angle(float3::new(-1.0, 0.5, 2.0), -1.3),
        quat::from_axis_angle(float3::X, 3.0), // near-pi cases hit the other Shepperd branches
        quat::from_axis_angle(float3::Y, 3.1),
        quat::from_axis_angle(float3::Z, -3.1),
    ] {
        let rt = quat::from_mat3(q.to_mat3());
        let rt4 = quat::from_mat4(&q.to_mat4());
        let via_from: quat = q.to_mat3().into();
        for v in [float3::X, float3::Y, float3::new(1.0, 2.0, 3.0)] {
            check!(near3(rt * v, q * v));
            check!(near3(rt4 * v, q * v));
            check!(near3(via_from * v, q * v));
        }
    }
}

#[test]
fn from_to_rotation() {
    check!(near3(
        quat::from_to_rotation(float3::X, float3::Y) * float3::X,
        float3::Y
    ));
    check!(near3(
        quat::from_to_rotation(float3::new(2.0, 0.0, 0.0), float3::new(2.0, 0.0, 0.0)) * float3::X,
        float3::X
    )); // aligned -> identity
    check!(near3(
        quat::from_to_rotation(float3::X, -float3::X) * float3::X,
        -float3::X
    )); // opposite -> 180 deg
    let (a, b) = (float3::new(1.0, 2.0, 3.0), float3::new(-2.0, 1.0, 0.5));
    check!(near3(
        quat::from_to_rotation(a, b) * a.normalize(),
        b.normalize()
    ));
}

#[test]
fn look_rotation() {
    // local forward (-Z) maps to the look direction, up stays up
    let q = quat::look_rotation(float3::FORWARD, float3::UP); // already facing -Z, up +Y
    check!(near3(q * float3::FORWARD, float3::FORWARD));
    check!(near3(q * float3::UP, float3::UP));
    check!(near_q(q, quat::IDENTITY));

    let q = quat::look_rotation(float3::X, float3::UP); // face +X
    check!(near3(q * float3::FORWARD, float3::X)); // -Z -> +X
    check!(near3(q * float3::UP, float3::Y)); // up preserved

    for dir in [
        float3::Z,
        float3::new(1.0, 1.0, 0.0),
        float3::new(1.0, 2.0, 2.0),
    ] {
        let q = quat::look_rotation(dir, float3::UP);
        check!(near3(q * float3::FORWARD, dir.normalize())); // forward axis hits the target dir
    }
}

#[test]
fn rotate_towards_quat() {
    let q90 = quat::from_axis_angle(float3::Z, PI / 2.0);
    let (from, to) = (quat::IDENTITY, q90);
    check!(near(from.rotate_towards(to, PI).angle_to(to), 0.0)); // budget covers it -> to
    let half = from.rotate_towards(to, PI / 4.0); // half the pi/2 arc
    check!(near(from.angle_to(half), PI / 4.0));
    check!(near(half.angle_to(to), PI / 4.0));
    check!(near(from.rotate_towards(from, 1.0).angle_to(from), 0.0)); // already there
}

#[test]
fn inverse_and_normalize() {
    let q = quat::from_axis_angle(float3::new(0.2, 1.0, -0.5), 1.1);
    check!(near3(
        (q * q.inverse()) * float3::new(1.0, 2.0, 3.0),
        float3::new(1.0, 2.0, 3.0)
    )); // q * q^-1 = id
    check!(near(
        quat::new(1.0, 2.0, 3.0, 4.0).normalize().length(),
        1.0
    )); // non-unit -> unit copy
    check!(near(quat::new(1.0, 2.0, 3.0, 4.0).length2(), 30.0));
    // inverse of a non-unit quaternion still composes to identity
    let nq = quat::new(1.0, 2.0, 3.0, 4.0);
    check!(near_q(nq * nq.inverse(), quat::IDENTITY));
}

#[test]
fn euler() {
    // yaw 90° about Y takes +X to -Z (forward), RH
    let q = quat::from_euler(float3::new(0.0, rad(90.0), 0.0));
    check!(near3(q * float3::X, float3::FORWARD));
    let qd = quat::from_euler_deg(float3::new(0.0, 90.0, 0.0));
    check!(near_q(q, qd));
    check!(near_q(quat_angles(float3::new(0.0, 90.0, 0.0)), qd));
    // pitch about X applied first, then yaw, then roll: qz * qy * qx
    let e = float3::new(0.3, -0.7, 1.1);
    let expected = quat::from_axis_angle(float3::Z, e.z)
        * quat::from_axis_angle(float3::Y, e.y)
        * quat::from_axis_angle(float3::X, e.x);
    check!(near_q(quat::from_euler(e), expected));
}

#[test]
fn double_precision_quaternion() {
    let q = dquat::from_axis_angle(double3::Z, core::f64::consts::FRAC_PI_2);
    let v = q * double3::X;
    assert!((v.x - 0.0).abs() < 1e-15 && (v.y - 1.0).abs() < 1e-15);
}
