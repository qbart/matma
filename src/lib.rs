//! # matma
//!
//! SIMD-first math for games and Vulkan, with Slang/HLSL naming. Pure Rust, zero
//! dependencies. A port of the C++23 library [vkm](https://github.com/qbart/vkm).
//!
//! Think "GLM, but Vulkan conventions by default and Slang-style names so Rust and
//! shader code mirror each other."
//!
//! ```
//! use matma::*;
//!
//! let q = quat::from_axis_angle(float3::Z, rad(30.0));
//! let model = compose_trs(float3::new(0.0, 1.0, 0.0), q, float3::ONE);
//! let view = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::UP);
//! let proj = perspective(rad(60.0), 16.0 / 9.0, 0.1, 100.0); // Vulkan clip: depth [0,1], Y down
//!
//! let mvp = proj * view * model;                 // compose right-to-left ...
//! let clip = mvp * float4::new(0.0, 0.0, 0.0, 1.0); // ... then M * v
//! let ndc = clip.xyz() / clip.w;
//! assert!(ndc.z > 0.0 && ndc.z < 1.0);
//! ```
//!
//! ## Conventions
//!
//! * **Names**: Slang/HLSL first — `float3`, `float4x4`, `uint4`, `quat` — as
//!   aliases of the generic [`Vector3<T>`], [`Matrix<V, C>`], [`Quaternion<T>`].
//!   GLSL aliases `vec3`/`mat4`/... are provided too (square matrices only).
//! * **`*` on vectors is component-wise** (as in Slang). `Matrix * Vector` and
//!   `Matrix * Matrix` are the linear-algebra products; [`mul`] mirrors Slang's `mul`.
//! * **Column-major storage, column-vector math**: `v' = M * v`, compose
//!   right-to-left `M = P * V * Model`. A `float4x4` memcpy's into a UBO `mat4`.
//! * **Vulkan clip space baked in**: projections produce depth in `[0, 1]` with Y down.
//! * **World space** is right-handed, Y-up, forward = `-Z` ([`Vector3::FORWARD`]).
//! * **Angles are radians** everywhere; [`rad`] / [`deg`] convert.
//!
//! ## Modules
//!
//! Everything is re-exported at the crate root (one flat namespace, like the C++
//! `vkm::`), and also reachable through its module:
//!
//! | module          | contents                                                        |
//! |-----------------|-----------------------------------------------------------------|
//! | [`vector`]      | `Vector2/3/4<T>`, Slang/GLSL aliases, swizzles, direction consts |
//! | [`matrix`]      | `Matrix<V, C>`, `mul`, transpose/inverse/determinant             |
//! | [`quaternion`]  | `Quaternion<T>`, slerp/nlerp, look rotation, from/to matrix      |
//! | [`transform`]   | translate/scale/rotate, TRS, look_at, perspective/ortho, decompose |
//! | [`common`]      | min/max/clamp/saturate/lerp/step/smoothstep, trig, game helpers  |
//! | [`curves`]      | Hermite / Catmull-Rom / cardinal splines, arc length, `AnimationCurve` |
//! | [`pack`]        | snorm/unorm quantization, octahedral normals, half-float packing |
//! | [`layout`]      | std140/std430 rules, `#[derive(Std140)]`, padded wrappers         |
//! | [`scalar`]      | the `Scalar`/`Float`/`Componentwise` traits generic code is built on |
//! | [`simd`]        | the 4-wide `f32` backend (SSE2 / NEON / scalar)                  |

#![deny(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod common;
pub mod curves;
pub mod layout;
pub mod matrix;
pub mod pack;
pub mod quaternion;
pub mod scalar;
pub mod simd;
pub mod transform;
pub mod vector;

/// `#[derive(Std140)]` / `#[derive(Std430)]`: generate the GPU layout of a struct.
/// See [`layout`] and [`Std140Layout`].
#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use matma_derive::{Std140, Std430};

pub use common::*;
pub use curves::*;
pub use layout::*;
pub use matrix::*;
pub use pack::*;
pub use quaternion::*;
pub use scalar::*;
pub use transform::*;
pub use vector::*;
