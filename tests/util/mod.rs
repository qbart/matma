//! Approximate-equality helpers shared by the integration tests.
#![allow(dead_code)]

use matma::*;

pub const EPS: f32 = 1e-4;

pub fn near_eps(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}
pub fn near(a: f32, b: f32) -> bool {
    near_eps(a, b, EPS)
}
pub fn near2(a: float2, b: float2) -> bool {
    near(a.x, b.x) && near(a.y, b.y)
}
pub fn near3(a: float3, b: float3) -> bool {
    near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z)
}
pub fn near4(a: float4, b: float4) -> bool {
    near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z) && near(a.w, b.w)
}
pub fn near_q(a: quat, b: quat) -> bool {
    near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z) && near(a.w, b.w)
}
pub fn near_m3(a: &float3x3, b: &float3x3) -> bool {
    (0..3).all(|c| near3(a.cols[c], b.cols[c]))
}
pub fn near_m4(a: &float4x4, b: &float4x4) -> bool {
    (0..4).all(|c| near4(a.cols[c], b.cols[c]))
}

/// `assert!` with the two values printed on failure.
#[macro_export]
macro_rules! check {
    ($pred:ident($a:expr, $b:expr $(, $rest:expr)*)) => {{
        let (a, b) = ($a, $b);
        assert!($pred(a, b $(, $rest)*), "{} failed:\n  left:  {:?}\n  right: {:?}", stringify!($pred), a, b);
    }};
}
