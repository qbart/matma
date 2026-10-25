//! `#[derive(Std140)]` / `#[derive(Std430)]` tests.
//!
//! Every expected offset is what the GLSL spec (OpenGL 4.6 §7.6.2.2) / Vulkan
//! gives for the same block, so a failure means the derive disagrees with the
//! shader compiler.

#![cfg(feature = "derive")]

use core::mem::{align_of, offset_of, size_of};
use matma::*;

// layout(std140) uniform Camera { mat4 view; mat4 proj; vec3 eye; float near;
//                                 float far; vec2 jitter; float time; };
#[derive(Clone, Copy, Debug, Default, PartialEq, Std140)]
struct Camera {
    view: float4x4, // 0
    proj: float4x4, // 64
    eye: float3,    // 128 (align 16)
    near: f32,      // 140 <- tucked into the vec3's 4th lane
    far: f32,       // 144
    jitter: float2, // 152 (align 8)
    time: f32,      // 160
} // size 176 (164 rounded to 16)

#[test]
fn camera_offsets_match_glsl() {
    assert_eq!(offset_of!(CameraStd140, view), 0);
    assert_eq!(offset_of!(CameraStd140, proj), 64);
    assert_eq!(offset_of!(CameraStd140, eye), 128);
    assert_eq!(offset_of!(CameraStd140, near), 140);
    assert_eq!(offset_of!(CameraStd140, far), 144);
    assert_eq!(offset_of!(CameraStd140, jitter), 152);
    assert_eq!(offset_of!(CameraStd140, time), 160);
    assert_eq!(size_of::<CameraStd140>(), 176);
    assert_eq!(CameraStd140::SIZE, 176);
    assert_eq!(CameraStd140::ALIGN, 16);
    assert_eq!(align_of::<CameraStd140>(), 16);
    assert_eq!(std140_align::<Camera>(), 16);
}

#[test]
fn camera_round_trip_and_bytes() {
    let cam = Camera {
        view: translate(float3::new(1.0, 2.0, 3.0)),
        proj: perspective(rad(60.0), 1.5, 0.1, 100.0),
        eye: float3::new(7.0, 8.0, 9.0),
        near: 0.1,
        far: 100.0,
        jitter: float2::new(0.25, -0.25),
        time: 42.0,
    };
    let gpu: Std140<Camera> = cam.as_std140(); // alias resolves to CameraStd140
    let gpu2: CameraStd140 = cam.into();
    assert_eq!(gpu.eye, cam.eye); // fields keep their natural type
    assert_eq!(gpu.view.into_inner(), cam.view);
    assert_eq!(Camera::from(gpu), cam);
    assert_eq!(gpu2.into_inner(), cam);

    let bytes = gpu.as_bytes();
    assert_eq!(bytes.len(), 176);
    // `near` sits at byte 140, little-endian f32.
    let near = f32::from_ne_bytes(bytes[140..144].try_into().unwrap());
    assert_eq!(near, 0.1);
    // Translation column of `view` at 48..64.
    let tx = f32::from_ne_bytes(bytes[48..52].try_into().unwrap());
    assert_eq!(tx, 1.0);
    // Padding bytes (164..176) are zero.
    assert!(bytes[164..176].iter().all(|&b| b == 0));

    let zero = CameraStd140::default();
    assert!(zero.as_bytes().iter().all(|&b| b == 0));
    assert_eq!(zero.into_inner(), Camera::default());

    let dbg = format!("{gpu:?}");
    assert!(dbg.starts_with("CameraStd140 {"));
    assert!(dbg.contains("time: 42.0"));
    assert!(!dbg.contains("_pad"));
}

// Arrays: std140 rounds every element up to a 16-byte slot.
// layout(std140) uniform Lights { uint count; vec3 positions[4]; vec4 colors[4];
//                                 float intensities[4]; };
#[derive(Clone, Copy, Std140)]
struct Lights {
    count: u32,             // 0
    positions: [float3; 4], // 16, stride 16 -> 80
    colors: [float4; 4],    // 80 -> 144
    intensities: [f32; 4],  // 144, stride 16 -> 208
} // size 208

#[test]
fn std140_arrays_use_16_byte_slots() {
    assert_eq!(offset_of!(LightsStd140, count), 0);
    assert_eq!(offset_of!(LightsStd140, positions), 16);
    assert_eq!(offset_of!(LightsStd140, colors), 80);
    assert_eq!(offset_of!(LightsStd140, intensities), 144);
    assert_eq!(LightsStd140::SIZE, 208);

    let l = Lights {
        count: 2,
        positions: [float3::X, float3::Y, float3::Z, float3::ONE],
        colors: [float4::ONE; 4],
        intensities: [1.0, 2.0, 3.0, 4.0],
    };
    let gpu = l.as_std140();
    assert_eq!(gpu.positions[1].value, float3::Y);
    assert_eq!(gpu.intensities[3].value, 4.0);
    let bytes = gpu.as_bytes();
    // intensities[1] lives at 144 + 16.
    let i1 = f32::from_ne_bytes(bytes[160..164].try_into().unwrap());
    assert_eq!(i1, 2.0);
    assert_eq!(gpu.into_inner().intensities, l.intensities);
}

// The same struct in both layouts: std430 drops the slot rounding.
#[derive(Clone, Copy, Debug, PartialEq, Std140, Std430)]
struct Weights {
    w: [f32; 8],
    scale: float2,
}

#[test]
fn std430_keeps_natural_array_stride() {
    // std140: w = 8 slots of 16 -> 128; scale at 128; 136 -> 144.
    assert_eq!(offset_of!(WeightsStd140, scale), 128);
    assert_eq!(WeightsStd140::SIZE, 144);
    assert_eq!(WeightsStd140::ALIGN, 16);
    // std430: w = 32 bytes; scale at 32; struct align = max(4, 8) = 8 -> 40.
    assert_eq!(offset_of!(WeightsStd430, scale), 32);
    assert_eq!(WeightsStd430::SIZE, 40);
    assert_eq!(WeightsStd430::ALIGN, 8);
    assert_eq!(std430_align::<Weights>(), 8);

    let v = Weights {
        w: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        scale: float2::new(0.5, 2.0),
    };
    let a: Std430<Weights> = v.as_std430();
    assert_eq!(a.w, v.w); // std430 field is the plain array
    assert_eq!(a.as_bytes().len(), 40);
    assert_eq!(a.into_inner(), v);
    assert_eq!(v.as_std140().into_inner(), v);
}

// vec3 + scalar tuck-in exists in std430 too; small matrices are not rounded.
#[derive(Clone, Copy, Std430)]
struct Particle {
    pos: float3,     // 0
    life: f32,       // 12
    vel: float2,     // 16
    seed: u32,       // 24
    basis: float2x2, // 32 (align 8; two vec2 columns -> 16 bytes)
} // 48, align 16

#[test]
fn std430_particle() {
    assert_eq!(offset_of!(ParticleStd430, life), 12);
    assert_eq!(offset_of!(ParticleStd430, vel), 16);
    assert_eq!(offset_of!(ParticleStd430, seed), 24);
    assert_eq!(offset_of!(ParticleStd430, basis), 32);
    assert_eq!(ParticleStd430::SIZE, 48);
    assert_eq!(ParticleStd430::ALIGN, 16);
    // An SSBO array of particles has stride SIZE.
    assert_eq!(size_of::<Std430Array<Particle, 3>>(), 144);
}

// Nesting: a derived struct is itself a layout type, usable as a field or array.
#[derive(Clone, Copy, Std140)]
struct Sun {
    dir: float3,   // 0
    color: float3, // 16
} // 32

#[derive(Clone, Copy, Std140)]
struct Scene {
    sun: Sun,             // 0
    ambient: f32,         // 32
    cameras: [Camera; 2], // 48 (struct align 16), stride 176 -> 400
    exposure: f32,        // 400
} // 416

#[test]
fn nested_structs_and_struct_arrays() {
    assert_eq!(SunStd140::SIZE, 32);
    assert_eq!(offset_of!(SceneStd140, sun), 0);
    assert_eq!(offset_of!(SceneStd140, ambient), 32);
    assert_eq!(offset_of!(SceneStd140, cameras), 48);
    assert_eq!(offset_of!(SceneStd140, exposure), 400);
    assert_eq!(SceneStd140::SIZE, 416);
    assert_eq!(size_of::<Std140Array<Sun, 3>>(), 96);

    let scene = Scene {
        sun: Sun {
            dir: float3::UP,
            color: float3::ONE,
        },
        ambient: 0.1,
        cameras: [Camera::default(); 2],
        exposure: 1.5,
    };
    let gpu = scene.as_std140();
    assert_eq!(gpu.sun.color, float3::ONE); // nested field is the nested Std140 struct
    assert_eq!(gpu.cameras[1].time, 0.0);
    let bytes = gpu.as_bytes();
    let exposure = f32::from_ne_bytes(bytes[400..404].try_into().unwrap());
    assert_eq!(exposure, 1.5);
}

// Doubles: dvec3 has base alignment 32 and makes the struct 32-aligned.
#[derive(Clone, Copy, Std140)]
struct Precise {
    scale: f32,      // 0
    origin: double3, // 32
    t: f64,          // 56 (tucked into the dvec3's 4th lane)
} // 64

#[test]
fn double_vectors_align_to_32() {
    assert_eq!(offset_of!(PreciseStd140, origin), 32);
    assert_eq!(offset_of!(PreciseStd140, t), 56);
    assert_eq!(PreciseStd140::SIZE, 64);
    assert_eq!(PreciseStd140::ALIGN, 32);
}

// Visibility and `pub` fields are carried over; a public derived struct must be
// constructible from another module through `From`.
mod api {
    use matma::*;

    #[derive(Clone, Copy, Std140)]
    pub struct Material {
        pub albedo: float4,
        pub roughness: f32,
        pub(crate) metallic: f32,
    }
}

#[test]
fn visibility_is_preserved() {
    let m = api::Material {
        albedo: float4::ONE,
        roughness: 0.5,
        metallic: 1.0,
    };
    let gpu: api::MaterialStd140 = m.into();
    assert_eq!(gpu.roughness, 0.5);
    assert_eq!(gpu.metallic, 1.0);
    assert_eq!(api::MaterialStd140::SIZE, 32);
}
