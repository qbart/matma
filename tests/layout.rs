//! std140/std430 layout tests.
//!
//! Every assert below is a number straight from the GLSL/Vulkan layout rules.
//! If one breaks, the encoded rule is wrong.

use core::mem::{align_of, offset_of, size_of};
use matma::*;

// ---- base alignment rules -------------------------------------------------------

const _: () = {
    assert!(std140_align::<f32>() == 4);
    assert!(std140_align::<float2>() == 8);
    assert!(std140_align::<float3>() == 16); // the classic gotcha
    assert!(std140_align::<float4>() == 16);
    assert!(std140_align::<double2>() == 16);
    assert!(std140_align::<double3>() == 32);
    assert!(std430_align::<float3>() == 16); // vec3 is 16 in std430 too
    assert!(std430_align::<float2x2>() == 8); // column = vec2 in std430
    assert!(std140_align::<float2x2>() == 16); // column slot rounds up to 16
    assert!(std140_slot_align::<f32>() == 16);
    assert!(std140_slot_align::<double4>() == 32);
    // Double matrices: dvec3/dvec4 columns round up to 32-byte slots, not 16.
    assert!(std140_align::<double2x2>() == 16);
    assert!(std140_align::<double3x3>() == 32);
    assert!(std140_align::<double4x4>() == 32);
};

#[test]
fn scalar_and_vector_wrappers() {
    assert_eq!(
        (size_of::<Std140<f32>>(), align_of::<Std140<f32>>()),
        (4, 4)
    );
    assert_eq!(
        (size_of::<Std140<float2>>(), align_of::<Std140<float2>>()),
        (8, 8)
    );
    assert_eq!(
        (size_of::<Std140<float3>>(), align_of::<Std140<float3>>()),
        (16, 16)
    );
    assert_eq!(
        (size_of::<Std140<float4>>(), align_of::<Std140<float4>>()),
        (16, 16)
    );
    assert_eq!(
        (size_of::<Std140<double3>>(), align_of::<Std140<double3>>()),
        (32, 32)
    );
    assert_eq!(
        (size_of::<Std430<float3>>(), align_of::<Std430<float3>>()),
        (16, 16)
    );
    assert_eq!(
        (size_of::<Std430<uint2>>(), align_of::<Std430<uint2>>()),
        (8, 8)
    );
}

#[test]
fn matrix_wrappers_pad_columns_to_slots() {
    assert_eq!(
        (
            size_of::<Std140<float4x4>>(),
            align_of::<Std140<float4x4>>()
        ),
        (64, 16)
    );
    assert_eq!(
        (
            size_of::<Std140<float3x3>>(),
            align_of::<Std140<float3x3>>()
        ),
        (48, 16)
    ); // 3 cols * 16
    assert_eq!(
        (
            size_of::<Std140<float2x2>>(),
            align_of::<Std140<float2x2>>()
        ),
        (32, 16)
    ); // 2 cols * 16
    assert_eq!(size_of::<Std430<float4x4>>(), 64);
    assert_eq!(size_of::<Std430<float3x3>>(), 48); // vec3 columns -> 16 each
    assert_eq!(size_of::<Std430<float2x2>>(), 16); // vec2 columns -> 8 each (no round)
    assert_eq!(
        (
            size_of::<Std140<double4x4>>(),
            align_of::<Std140<double4x4>>()
        ),
        (128, 32)
    );
    assert_eq!(
        (
            size_of::<Std140<double3x3>>(),
            align_of::<Std140<double3x3>>()
        ),
        (96, 32)
    ); // 3 cols * 32
    assert_eq!(
        (
            size_of::<Std140<double2x2>>(),
            align_of::<Std140<double2x2>>()
        ),
        (32, 16)
    ); // 2 cols * 16
}

#[test]
fn arrays() {
    assert_eq!(size_of::<Std140Array<f32, 4>>(), 64); // stride 16 each
    assert_eq!(size_of::<Std140Array<float4, 2>>(), 32);
    assert_eq!(
        (
            size_of::<Std140Array<double4, 2>>(),
            align_of::<Std140Array<double4, 2>>()
        ),
        (64, 32)
    ); // dvec4 slots stay 32-aligned
    assert_eq!(size_of::<Std430Array<f32, 4>>(), 16); // stride 4 each
    assert_eq!(size_of::<Std430Array<float2, 3>>(), 24); // stride 8 each
    assert_eq!(size_of::<Std430Array<float3, 2>>(), 32); // vec3 stride still 16
    assert_eq!(size_of::<Std140Array<float4x4, 2>>(), 128);
}

// ---- a real UBO block, wrapper path -------------------------------------------
// Matches a Slang/GLSL std140 block declared field-for-field the same way
// (note: uses float4 for the eye position, sidestepping the vec3+scalar tuck).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CameraData {
    view: Std140<float4x4>, // offset 0
    proj: Std140<float4x4>, // offset 64
    eye: Std140<float4>,    // offset 128
    jitter: Std140<float2>, // offset 144
    near: Std140<f32>,      // offset 152
    far: Std140<f32>,       // offset 156
}

#[test]
fn ubo_block_offsets() {
    assert_eq!(offset_of!(CameraData, view), 0);
    assert_eq!(offset_of!(CameraData, proj), 64);
    assert_eq!(offset_of!(CameraData, eye), 128);
    assert_eq!(offset_of!(CameraData, jitter), 144);
    assert_eq!(offset_of!(CameraData, near), 152);
    assert_eq!(offset_of!(CameraData, far), 156);
    assert_eq!(
        (size_of::<CameraData>(), align_of::<CameraData>()),
        (160, 16)
    );

    let cam = CameraData {
        view: translate(float3::new(7.0, 8.0, 9.0)).into(),
        eye: float4::new(1.0, 2.0, 3.0, 1.0).into(),
        near: 0.1,
        far: 100.0,
        ..Default::default()
    };
    assert_eq!(cam.view.into_inner(), translate(float3::new(7.0, 8.0, 9.0)));
    assert_eq!(cam.eye.into_inner().xyz(), float3::new(1.0, 2.0, 3.0));
    assert_eq!(cam.far, 100.0);
}

// ---- byte-exact "tuck-in", hand-aligned path ---------------------------------
// A Slang block { vec3 c; float d; } packs d into the vec3's 4th lane (offset
// 12). The wrapper path can't express that; a hand-written `repr(C, align(16))`
// can — and the alignment constant verifies the number you wrote.
#[repr(C, align(16))]
struct Tuck {
    c: float3, // natural size 12; the struct's align(16) stands in for the vec3 rule
    d: f32,    // tucks in at offset 12
}
const _: () = assert!(align_of::<Tuck>() == std140_align::<float3>());

#[test]
fn tuck_in() {
    assert_eq!(offset_of!(Tuck, c), 0);
    assert_eq!(offset_of!(Tuck, d), 12); // <-- the tuck-in
    assert_eq!(size_of::<Tuck>(), 16);
}

#[test]
fn round_trips() {
    // Round-trip a matrix through the std140 wrapper (memcpy-able to a UBO).
    let m = float4x4::from_cols([
        float4::X,
        float4::Y,
        float4::Z,
        float4::new(7.0, 8.0, 9.0, 1.0),
    ]);
    let packed: Std140<float4x4> = m.into();
    let back: float4x4 = packed.into_inner();
    assert_eq!(back, m);

    let m3 = float3x3::from_rows([
        float3::new(1.0, 2.0, 3.0),
        float3::new(4.0, 5.0, 6.0),
        float3::new(7.0, 8.0, 9.0),
    ]);
    let mut p3: Std140<float3x3> = m3.into();
    assert_eq!(p3.into_inner(), m3);
    p3.set(float3x3::IDENTITY);
    assert_eq!(p3.into_inner(), float3x3::IDENTITY);
    assert_eq!(p3.cols[1].0, float3::Y); // padded column is an Align16<float3>

    let mut arr: Std140Array<float4, 3> = Default::default();
    arr[1] = float4::new(1.0, 2.0, 3.0, 4.0).into();
    let read: float4 = arr[1].into_inner();
    assert_eq!(read, float4::new(1.0, 2.0, 3.0, 4.0));

    let mut farr: Std140Array<f32, 4> = Default::default();
    farr[2] = 5.0.into();
    assert_eq!(farr[2].0, 5.0);
    assert_eq!(size_of_val(&farr), 64);
}

fn size_of_val<T>(_: &T) -> usize {
    size_of::<T>()
}
