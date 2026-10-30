# matma — design

A SIMD-first math library for games and Vulkan, with Slang/HLSL naming, in pure
Rust with no third-party dependencies. A port of [vkm](https://github.com/qbart/vkm) (C++23);
this file records what the port kept, what Rust forced us to change, and why.

## Goals (unchanged from vkm)

- **Slang/HLSL naming.** `float3`, `float4x4`, `uint4` — not `Vec3`/`Mat4`.
- **SIMD from the start.** SSE2 / NEON / scalar backends behind one tiny seam.
- **Vulkan conventions baked in.** `[0,1]` depth + flipped Y for projections;
  `std140`/`std430` layout helpers for UBO/SSBO interop.
- **Production-usable, but a learning vehicle.** Readable over clever.

## Layout

```
src/
  lib.rs          umbrella; re-exports every module at the root
  scalar.rs       Scalar / Signed / Float / Componentwise / Vector / Dim traits
  simd.rs         ISA backend (SSE2 / NEON / scalar). Only file with intrinsics.
  vector.rs       Vector2/3/4<T> + floatN/intN/uintN/boolN/doubleN/short2 aliases
  matrix.rs       Matrix<V, C> + floatRxC aliases, mul/transpose/inverse/determinant
  quaternion.rs   Quaternion<T> (quat / dquat)
  transform.rs    translate/rotate/scale, look_at, perspective/ortho (Vulkan), decompose
  common.rs       min/max/clamp/lerp/... generic over Componentwise
  curves.rs       Hermite/Catmull-Rom/cardinal splines, arc-length table, AnimationCurve, MonotoneCurve (needs alloc)
  pack.rs         snorm/unorm quantization, octahedral encoding, half floats
  layout.rs       std140 / std430 alignment rules and padded storage
matma-derive/     proc-macro crate: #[derive(Std140)] / #[derive(Std430)] (feature `derive`)
tests/            one file per vkm test, plus simd.rs (backend vs scalar reference)
                  and derive.rs (generated offsets vs the GLSL rules)
```

## Conventions

- **Names:** Slang/HLSL first, GLSL aliases second (square matrices only — GLSL
  names non-square matrices columns-first, opposite to Slang's rows-first
  `floatRxC`). Lowercase type aliases carry `#[allow(non_camel_case_types)]`;
  the generic types underneath are CamelCase (`Vector3<T>`, `Matrix<V, C>`).
- **`*` on vectors is component-wise** (matches Slang). Matrix product is
  `Matrix * Matrix`, `Matrix * Vector`, and the generic `mul(a, b)`.
- **Storage:** named fields `x, y, z, w` in `#[repr(C)]` structs, so `&v.x` is a
  contiguous array and the type drops straight into GPU buffers. `as_slice()`
  exposes that view (one line of `unsafe`, justified by `repr(C)`).

### Matrix convention — column-major + M*v

**Column-major storage + column-vector math** (`v' = M * v`, compose
right-to-left `M = P * V * Model`). `float4x4` is 64 bytes with the translation
in the last column, so it memcpy's into a UBO `mat4` directly. Indexing: `m[c]`
is a column; `m[(r, c)]` is an element.

- Matches Vulkan/SPIR-V's default column-major layout → upload with a plain copy.
- Matches GLM / textbook convention → tutorials and ported code just work.
- The Slang *name* (`float4x4`) is independent of CPU storage; `mul()` is provided
  alongside `*`. If Slang is compiled with its row-major default, pass
  `slangc -matrix-layout-column-major` or transpose on upload.

### Why `Matrix<V, C>` and not `Matrix<T, R, C>`

vkm's `matrix<T, R, C>` stores `vector<T, R> cols[C]`. Rust has no way to map a
const `R` to a vector type with named fields (`Vector3<T>` for `R = 3`) without
unstable generic-const-expressions, so the column *type* is the parameter:
`float4x4 = Matrix<float4, 4>`, `float3x4 = Matrix<float3, 4>` (3 rows, 4 cols).
The row count is still checked at compile time through the `Dim<const N>` marker
trait: `Matrix<VR, K> * Matrix<VK, C>` requires `VK: Dim<K>`, so
`float2x3 * float3x2` type-checks and `float2x3 * float2x3` does not.

Per-shape operations that change the type (`transpose`, `row`, `from_rows`,
`to_cols_array`) are generated for the 9 shapes 2..4 x 2..4 by a macro, because an
inherent impl cannot introduce an "output row count" that is not determined by
`Self`.

### Why traits instead of overloads

C++ `common.hpp` has scalar and `vector<T,N>` overloads of every function. Rust has
no overloading, so there is one `Componentwise` trait implemented by every scalar
(one lane) and every vector (N lanes) with `splat / map / zip / zip3 / fold / dot`.
`saturate<V: Componentwise>(x: V)` is then both the scalar and the component-wise
function — the same shape as a shader's overload set. `Scalar` has `Componentwise`
as a supertrait, so generic code can `clamp` a `T::Scalar` as freely as a vector.

Quaternions intentionally do **not** implement `Componentwise`: `angle(q1, q2)`
would silently compute the 4-D vector angle (half the rotation angle). The
quaternion operations are methods (`q.angle_to`, `q.slerp`, `q.rotate`) plus the
free `slerp`/`nlerp`.

## SIMD

`simd.rs` is the only file that includes intrinsics. It exposes a 4-wide float
register `F32x4` and a minimal op set (`load4/store4/splat/add/sub/mul/div/neg/
hsum`). Backends, chosen at compile time:

| Backend | When                                  | Notes |
|---------|---------------------------------------|-------|
| SSE2    | `x86`/`x86_64` with `sse2` (baseline) | hsum via shuffles, no SSE3 `haddps` |
| NEON    | `aarch64`                             | `vaddvq_f32` for the horizontal sum |
| scalar  | otherwise, or `--features force-scalar` | always builds; SIMD/scalar diff testing |

C++ vkm dispatches with non-template `float4` overloads and `if consteval`. Rust
has neither specialization nor a way to put intrinsics in `const fn`, so the seam
is a set of 4-lane hooks on the `Scalar` trait (`add4`, `sub4`, `mul4`, `div4`,
`dot4`) with scalar-loop defaults; `impl Scalar for f32` overrides them with the
backend. `Vector4<T>`'s operators call the hooks, so `Vector4<f32>` is SIMD and
`Vector4<i32>` is a loop, from one generic impl. Higher layers stay generic.

Not ported: `constexpr` arithmetic. Rust operator traits cannot be `const`, so
only constructors, constants and the `pack` encoders are `const fn`.

## Layout helpers

vkm's `std140<T>` is `alignas(std140_align_v<T>)` — a computed alignment, which
Rust's `#[repr(align(N))]` cannot express. The port has two paths.

**Derive path** (`#[derive(Std140)]`, `#[derive(Std430)]`, crate `matma-derive`).
The macro sees the field list, so it can do what `alignas` cannot: compute every
offset. For `struct Name { f0: T0, f1: T1, .. }` it emits

- a hidden helper type whose associated consts walk the fields:
  `OFF_i = align_up(END_{i-1}, <T_i as Std140Layout>::ALIGN)`,
  `END_i = OFF_i + size_of::<<T_i>::Field>()`, struct `ALIGN` = max member
  alignment (rounded to 16 for std140), `SIZE = align_up(END_last, ALIGN)`;
- `#[repr(C)] struct NameStd140 { _pad0: [u8; PAD0], f0: <T0>::Field, _pad1: .., f1, .., _tail }`
  — explicit zero-byte padding, so the struct has no uninitialised bytes and
  `as_bytes()` is sound;
- `Padded<Name>` (`wrap` / `into_inner` / `zeroed`), `From` both ways, `Default`,
  `Debug` (hides the pads), `Name::as_std140()`, `SIZE` / `ALIGN`;
- `impl Std140Layout for Name` so derived structs nest and form arrays;
- `const` assertions that `offset_of!` of every field equals the computed offset
  and `size_of` equals `SIZE`. If Rust ever placed a field elsewhere, the crate
  would not compile.

`Field` is the third storage type on the layout traits and the key to the
tuck-in: for scalars and vectors it is the type itself (a `float3` field is 12
bytes, so a following `f32` lands at offset 12 like GLSL), for matrices and arrays
it is the slot-padded storage. The macro is a hand-rolled token walker over
`proc_macro` (fixed input shape: non-generic struct, named fields), so the
repository still has no third-party dependencies. The trait is `unsafe trait`:
implementors promise that `Padded`/`Slot`/`Field` are fully initialised plain
data, which is what `as_bytes()` relies on.

Limits: std140 structs get `#[repr(C, align(16))]` (always correct: a 32-aligned
struct still has a 32-multiple size); std430 structs keep Rust's natural
alignment, so a derived std430 struct placed in a *hand-written* parent must be
aligned by hand (inside a derived parent the pads take care of it).

**Wrapper path** for hand-written `#[repr(C)]` structs:

- `Std140Layout` / `Std430Layout` traits carrying `ALIGN` and the storage types
  `Padded` (lone member), `Slot` (array element, std140 only) and `Field`.
- `Padded8/16/32<T, PAD>` wrappers (`#[repr(C, align(N))]` with `PAD` explicit zero
  bytes, so no implicit padding) and `PaddedCols<S, C>` for column-padded
  matrices; the `Padded<T>` trait (`wrap`, `into_inner`, `set`, `zeroed`) unifies
  them, with identity impls for scalars and vectors and an impl for arrays.
- `Std140<T> = <T as Std140Layout>::Padded` resolves to `f32` for `f32`, to
  `Padded16<float3, 4>` for `float3`, to `PaddedCols<Padded16<float3, 4>, 3>` for
  `float3x3`, and to `NameStd140` for a derived struct.

The alignment constants are still there (`std140_align::<T>()`) to check a
hand-written `#[repr(C, align(16))]` struct, same as vkm's alignas-trait path.

`float4` is **not** over-aligned to 16 (vkm's `alignas(4*sizeof(T))`): a generic
`Vector4<T>` cannot take a type-dependent alignment, and forcing 16 would make
`Vector4<u8>` 16 bytes and break RGBA8 vertex formats. SIMD uses unaligned
loads, and the derive / wrappers supply the 16 where the GPU needs it.

## Roadmap

1. ✅ Project skeleton, SIMD backend, `Vector2/3/4<T>` (+ `float4` SIMD), tests.
2. ✅ `Matrix<V, C>` (column-major, M*v), mul/transpose/identity, shape-checked products.
3. ✅ `Quaternion<T>` + `transform`: translate/scale/rotate/look_at/compose_trs,
   `perspective`/`ortho` with Vulkan depth [0,1] + Y-flip, `decompose`.
4. ✅ `layout`: std140/std430 rules, `Std140<T>`/`Std430<T>` wrappers and arrays,
   `#[derive(Std140)]` / `#[derive(Std430)]`.
5. ✅ Swizzles, `common` (incl. Unity-style helpers), `pack` (snorm/unorm/oct/half).
6. ✅ determinant/inverse (2x2/3x3/4x4) + `normal_matrix`.
6b. ✅ `curves`: splines + arc length + `AnimationCurve`, ported from the engine's
   `spline.cpp` / `animation_curve.cpp` (generic over the vector type; the
   `Timer` overload stayed behind with the engine). `MonotoneCurve` adds a
   Fritsch-Carlson monotone cubic Hermite `y = f(x)` (new, not from vkm).
7. ⏳ Wider SIMD (`float3` padded load, `f64` via AVX), `bytemuck` feature, `no_std`
   via `libm`, benchmarks vs glam.
