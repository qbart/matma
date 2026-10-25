# matma

SIMD-first math for games and Vulkan, with Slang/HLSL naming. Pure Rust, zero
dependencies, `#![deny(unsafe_code)]` outside the SIMD seam.

Think "GLM, but Vulkan conventions by default and Slang-style names so Rust and
shader code mirror each other." A port of the C++23 library
[vkm](https://github.com/qbart/vkm).

```rust
use matma::*;

let q = quat::from_axis_angle(float3::Z, rad(30.0));
let model = compose_trs(float3::new(0.0, 1.0, 0.0), q, float3::ONE);
let view = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::UP);
let proj = perspective(rad(60.0), 16.0 / 9.0, 0.1, 100.0); // Vulkan clip: depth [0,1], Y down

let mvp = proj * view * model;                      // compose right-to-left ...
let clip = mvp * float4::new(0.0, 0.0, 0.0, 1.0);   // ... then M * v
let ndc = clip.xyz() / clip.w;

let normal_packed: short2 = oct_encode16(float3::UP); // -> R16G16_SNORM vertex attribute
```

## Goals

- **Slang/HLSL naming.** `float3`, `float4x4`, `uint4`, `quat` — not `Vec3`/`Mat4`.
  GLSL aliases (`vec3`, `mat4`, `ivec2`, ...) are there too.
- **SIMD from the start.** SSE2 / NEON / scalar backends behind one tiny seam
  (`simd` module); `float4` arithmetic and dot products go through it.
- **Vulkan conventions baked in.** `[0, 1]` depth + flipped Y in the projection
  builders; `std140` / `std430` layout rules and wrappers for UBO/SSBO interop;
  GPU packing (snorm/unorm, octahedral normals, half floats).
- **Generic underneath.** `Vector3<T>`, `Matrix<V, C>`, `Quaternion<T>` work for
  `f32`, `f64` and the integer types; `f32` is just the one with SIMD hooks.
- **Readable over clever.** A learning vehicle that is still production-usable.

## Conventions

- **`*` on vectors is component-wise** (as in Slang). `Matrix * Vector`,
  `Matrix * Matrix` and `quat * quat` are the algebraic products; `mul(a, b)`
  mirrors Slang's `mul`.
- **Column-major storage + column-vector math.** `v' = M * v`, compose
  right-to-left `M = P * V * Model`. `float4x4` is 64 contiguous bytes with the
  translation in the last column, so `m.as_slice()` / `m.to_cols_array()` memcpy
  straight into a UBO `mat4`. `m[c]` is a column, `m[(r, c)]` an element.
- **Matrix shape is in the type.** `float3x4 = Matrix<float3, 4>` is 3 rows x 4
  columns (Slang's `floatRxC`). Products check shapes at compile time:
  `float2x3 * float3x2 = float2x2`.
- **World space** is right-handed, Y-up, forward = `-Z` (`float3::FORWARD`), the
  direction a `look_at` camera faces. Vulkan's Y-down lives only in clip space.
- **Angles are radians** everywhere; `rad(90.0)` / `deg(x)` convert. Everything
  Unity-flavoured (`angle`, `move_towards`, `rotate_towards`, `look_rotation`,
  `lerp_angle`, `repeat`, `ping_pong`) is in radians too.
- **NaN** in `min`/`max`/`clamp`/`saturate` behaves like HLSL, not IEEE: the
  comparison is false, so the second operand / the upper bound wins.

## Module map

Everything is re-exported at the crate root (one flat namespace, like `vkm::`):

| module       | contents |
|--------------|----------|
| `vector`     | `Vector2/3/4<T>`; aliases `float3`, `int2`, `uint4`, `bool3`, `double4`, `short2`, `ubyte4`, `vec3`, ...; swizzles (`xyz()`, `swizzle3::<2,1,0>()`); `X/Y/Z/UP/FORWARD/...` consts; comparisons to `boolN` + `select` |
| `matrix`     | `Matrix<V, C>`, `floatRxC` / `mat{2,3,4}` aliases, `mul`, `transpose`, `inverse`, `determinant`, `normal_matrix`, `from_rows` / `to_cols_array` |
| `quaternion` | `Quaternion<T>` (`quat`, `dquat`): axis-angle, Euler, from/to matrix, `slerp`/`nlerp`, `from_to_rotation`, `look_rotation`, `rotate_towards`, `angle_to` |
| `transform`  | `translate`, `scale`, `rotate`, `compose_trs`, `look_at`, `perspective`, `ortho` (Vulkan), `decompose`, `rotate_towards` for vectors, `rad`/`deg` |
| `common`     | `min/max/clamp/saturate/lerp/step/smoothstep/abs/sign/reflect/distance/angle/project/move_towards`, `floor/sqrt/sin/.../atan2/fmod/pow/frac/rsqrt`, `repeat/ping_pong/lerp_angle`, `Ray` + `ray_triangle_intersect` |
| `pack`       | `quantize_snorm16` & friends (8/10/16-bit, snorm/unorm), `oct_encode16`/`oct_decode16`, tangent + handedness packing, `pack_half2x16` — all `const fn` where no `sqrt` is involved |
| `layout`     | `std140_align::<T>()`, `Std140<T>` / `Std430<T>` / `Std140Array<T, N>` wrappers that resolve to correctly aligned storage, column-padded matrices |
| `scalar`     | `Scalar`, `Signed`, `Float`, `Componentwise`, `Vector`, `Dim` — the traits generic code is written against |
| `simd`       | the 4-wide `f32` register and ops; `simd::BACKEND` names the one compiled in |

Every function in `common` is generic over `Componentwise`, so one definition
serves scalars and vectors alike, exactly like the overload sets in a shader:
`saturate(x)` works for `f32`, `f64`, `float3`, `int4`, ... Most also exist as
methods (`v.saturate()`, `v.normalize()`, `m.inverse()`).

## SIMD

The `simd` module is the only place with intrinsics. It exposes `F32x4` plus
`load4/store4/splat/add/sub/mul/div/neg/hsum`, picked at compile time:

| backend | when |
|---------|------|
| SSE2    | `x86`/`x86_64` (sse2 is the x86-64 baseline) |
| NEON    | `aarch64` |
| scalar  | anything else, or the `force-scalar` cargo feature (for debugging / diffing results) |

`Vector4<T>` arithmetic calls 4-lane hooks on the `Scalar` trait (`add4`, `mul4`,
`dot4`, ...). Their defaults are plain loops; `f32` overrides them with the SIMD
backend. Nothing above `simd` knows which backend is in use.

## Layout helpers

```rust
use matma::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CameraData {
    view: Std140<float4x4>, // offset 0
    proj: Std140<float4x4>, // offset 64
    eye: Std140<float4>,    // offset 128
    jitter: Std140<float2>, // offset 144   (Align8<float2>)
    near: Std140<f32>,      // offset 152   (plain f32)
    far: Std140<f32>,       // offset 156
}
assert_eq!(core::mem::size_of::<CameraData>(), 160);

let mut cam = CameraData::default();
cam.view = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::UP).into();
let back: float4x4 = cam.view.into_inner();
```

`Std140<T>` resolves to `T` when the natural alignment already matches
(`f32`, `f64`), to an `Align8/16/32<T>` wrapper otherwise, and to column-padded
storage for matrices (`Std140<float3x3>` is 48 bytes: three 16-byte columns).
Rust cannot compute `#[repr(align)]` from a type, so for the byte-exact
`vec3 + float` "tuck-in" write `#[repr(C, align(16))]` by hand and verify it with
`std140_align::<float3>()`. `bool` is deliberately unsupported (4 bytes in GLSL,
1 in Rust): use `u32`.

## Differences from vkm (C++)

| vkm (C++)                                   | matma (Rust) |
|---------------------------------------------|--------------|
| `vector<T, N>` with specialisations         | `Vector2<T>` / `Vector3<T>` / `Vector4<T>` (named fields, `#[repr(C)]`) |
| `matrix<T, R, C>`                           | `Matrix<V, C>` where `V` is the column type; `Matrix4x4<T>` etc. are aliases |
| `m[r, c]`                                   | `m[(r, c)]` |
| `float4x4::identity()`                      | `float4x4::IDENTITY` (also `identity()` for generic code) |
| `v.normalize()` in place / `v.normalized()` | `v.normalize()` returns the result (vectors are `Copy`) |
| `rotate(q, v)`                              | `q.rotate(v)` or `q * v` (`rotate(angle, axis)` builds a matrix, as in vkm) |
| `angle(q1, q2)`                             | `q1.angle_to(q2)` (`angle(a, b)` is the vector version) |
| `to_float3x3(q)` / `to_quat(m)`             | `q.to_mat3()` / `quat::from_mat3(m)` (also `From`/`Into`) |
| `decompose(m, s, q, t, skew, p) -> bool`    | `decompose(&m) -> Option<Decomposed>` |
| `ray_triangle_intersect(r, a, b, c, &t)`    | `-> Option<t>` |
| `oct_decode_tangent(p, &h)`                 | `-> (tangent, handedness)` |
| `90.0_deg` literals                         | `rad(90.0)` |
| `alignas(16) float4`                        | `float4` has natural alignment (4); use `Std140<float4>` / `Align16` where 16 is required |
| `std140<T>` wrapper struct                  | `Std140<T>` type alias resolving to the right storage; read with `.into_inner()` |

## Building

```sh
cargo test                           # SSE2 / NEON backend
cargo test --features force-scalar   # portable fallback, diff against SIMD results
cargo run --example usage
```

MSRV 1.87 (edition 2024).

## License

MIT, see [LICENSE.md](LICENSE.md).
