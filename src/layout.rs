//! std140 / std430 layout helpers for UBO/SSBO interop.
//!
//! GPU buffer blocks follow strict alignment rules that differ from Rust's. The
//! infamous one: a 3-component vector has a base alignment of 16 (not 12), in
//! BOTH std140 and std430. Get this wrong and every field past the first `vec3` is
//! silently misaligned against the shader.
//!
//! This module gives you two things:
//!
//! 1. **Alignment rules as constants** — [`Std140Layout::ALIGN`] /
//!    [`Std430Layout::ALIGN`] (or [`std140_align::<T>()`](std140_align)). Rust has
//!    no computed `align(..)`, so use them to *check* a hand-written
//!    `#[repr(C, align(16))]` struct, which is the only way to get the
//!    `vec3 + float` "tuck-in" the spec allows.
//!
//! 2. **Wrapper types** — [`Std140<T>`] / [`Std430<T>`] and [`Std140Array`] /
//!    [`Std430Array`] — that resolve to a type carrying the right alignment (and,
//!    for matrices, pad each column to its slot). Ergonomic and composable; the
//!    tradeoff is each wrapped member gets its own slot, so a wrapped `vec3`
//!    occupies 16 bytes (no tuck-in). Pick the wrapper path and declare your
//!    Slang/GLSL struct the same way, OR use the align-constant path for literal
//!    byte-for-byte parity.
//!
//! Rules encoded (base alignment):
//!
//! | type                           | alignment                                   |
//! |--------------------------------|---------------------------------------------|
//! | scalar (`f32`/`i32`/`u32`)     | 4 (`f64`: 8)                                |
//! | vec2                           | 2 * sizeof(component)                       |
//! | vec3, vec4                     | 4 * sizeof(component)                       |
//! | matrix                         | array of columns; each column is a vecR     |
//! | array element / matrix column  | std140 rounds the slot up to 16; std430 keeps the natural alignment |
//!
//! `bool` is intentionally unsupported — GLSL `bool` is 4 bytes, Rust's is 1. Use
//! `u32` for GPU-facing booleans.

use crate::matrix::Matrix;
use crate::scalar::Vector;
use crate::vector::{Vector2, Vector3, Vector4};

// ---- padded storage -----------------------------------------------------------

/// A storage type holding a `T` at some (larger) alignment: `From<T>` plus
/// [`into_inner`](Self::into_inner) / [`set`](Self::set). Implemented by the
/// [`Align8`]/[`Align16`]/[`Align32`] wrappers, [`PaddedCols`], and — trivially —
/// the scalars whose natural alignment already satisfies the rules.
pub trait Padded<T>: Copy + Default + From<T> {
    /// The wrapped value.
    fn into_inner(self) -> T;
    /// Replace the wrapped value.
    fn set(&mut self, value: T);
}

macro_rules! padded_identity {
    ($($t:ty),*) => {$(
        impl Padded<$t> for $t {
            #[inline]
            fn into_inner(self) -> $t { self }
            #[inline]
            fn set(&mut self, value: $t) { *self = value; }
        }
    )*};
}
padded_identity!(f32, f64, i32, u32);

macro_rules! align_wrapper {
    ($(#[$doc:meta] $name:ident = $n:literal;)*) => {$(
        #[$doc]
        #[repr(C, align($n))]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name<T>(pub T);

        impl<T> From<T> for $name<T> {
            #[inline]
            fn from(v: T) -> Self { Self(v) }
        }
        impl<T: Copy + Default> Padded<T> for $name<T> {
            #[inline]
            fn into_inner(self) -> T { self.0 }
            #[inline]
            fn set(&mut self, value: T) { self.0 = value; }
        }
    )*};
}
align_wrapper! {
    /// `T` stored at 8-byte alignment (padded to a multiple of 8).
    Align8 = 8;
    /// `T` stored at 16-byte alignment (padded to a multiple of 16).
    Align16 = 16;
    /// `T` stored at 32-byte alignment (padded to a multiple of 32).
    Align32 = 32;
}

/// A matrix stored as `C` padded columns of type `S` (a [`Padded`] column vector).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaddedCols<S, const C: usize> {
    /// The padded columns.
    pub cols: [S; C],
}

impl<S: Copy + Default, const C: usize> Default for PaddedCols<S, C> {
    #[inline]
    fn default() -> Self {
        Self {
            cols: [S::default(); C],
        }
    }
}
impl<V: Vector, S: Padded<V>, const C: usize> From<Matrix<V, C>> for PaddedCols<S, C> {
    #[inline]
    fn from(m: Matrix<V, C>) -> Self {
        Self {
            cols: m.cols.map(S::from),
        }
    }
}
impl<V: Vector, S: Padded<V>, const C: usize> Padded<Matrix<V, C>> for PaddedCols<S, C> {
    #[inline]
    fn into_inner(self) -> Matrix<V, C> {
        Matrix::from_cols(self.cols.map(S::into_inner))
    }
    #[inline]
    fn set(&mut self, m: Matrix<V, C>) {
        self.cols = m.cols.map(S::from);
    }
}

// ---- layout traits ------------------------------------------------------------

/// std140 rules for a type: its base alignment and the storage types that carry it.
pub trait Std140Layout: Copy {
    /// Base alignment of a lone member.
    const ALIGN: usize;
    /// Alignment of an array element (or matrix column): `ALIGN` rounded up to 16.
    /// Not 16 flat — `double3`/`double4` elements need 32.
    const SLOT_ALIGN: usize = if Self::ALIGN < 16 { 16 } else { Self::ALIGN };
    /// Storage for a lone member.
    type Padded: Padded<Self>;
    /// Storage for an array element.
    type Slot: Padded<Self>;
}

/// std430 rules for a type: its base alignment and the storage type that carries it.
/// Array elements keep the natural alignment, so there is no separate slot type.
pub trait Std430Layout: Copy {
    /// Base alignment of a lone member (also the array stride alignment).
    const ALIGN: usize;
    /// Storage for a member or array element.
    type Padded: Padded<Self>;
}

/// `T` stored with std140 alignment. `Std140<float3>` is 16 bytes; `Std140<f32>` is
/// just `f32`.
pub type Std140<T> = <T as Std140Layout>::Padded;
/// `T` stored as a std140 array element (slot rounded up to 16 bytes).
pub type Std140Slot<T> = <T as Std140Layout>::Slot;
/// `[T; N]` with std140 array stride: every element slot is rounded up to 16.
pub type Std140Array<T, const N: usize> = [<T as Std140Layout>::Slot; N];
/// `T` stored with std430 alignment.
pub type Std430<T> = <T as Std430Layout>::Padded;
/// `[T; N]` with std430 array stride (natural element alignment).
pub type Std430Array<T, const N: usize> = [<T as Std430Layout>::Padded; N];

/// std140 base alignment of `T`.
#[inline]
pub const fn std140_align<T: Std140Layout>() -> usize {
    T::ALIGN
}
/// std140 array-element (and matrix-column) slot alignment of `T`.
#[inline]
pub const fn std140_slot_align<T: Std140Layout>() -> usize {
    T::SLOT_ALIGN
}
/// std430 base alignment of `T`.
#[inline]
pub const fn std430_align<T: Std430Layout>() -> usize {
    T::ALIGN
}

// ---- scalars and vectors ------------------------------------------------------

macro_rules! layout_impl {
    ($t:ty => $align:literal, $padded:ty, $slot:ty) => {
        impl Std140Layout for $t {
            const ALIGN: usize = $align;
            type Padded = $padded;
            type Slot = $slot;
        }
        impl Std430Layout for $t {
            const ALIGN: usize = $align;
            type Padded = $padded;
        }
    };
}

macro_rules! layout_scalar4 {
    ($($t:ty),*) => {$(
        layout_impl!($t => 4, $t, Align16<$t>);
        layout_impl!(Vector2<$t> => 8, Align8<Vector2<$t>>, Align16<Vector2<$t>>);
        layout_impl!(Vector3<$t> => 16, Align16<Vector3<$t>>, Align16<Vector3<$t>>);
        layout_impl!(Vector4<$t> => 16, Align16<Vector4<$t>>, Align16<Vector4<$t>>);
    )*};
}
layout_scalar4!(f32, i32, u32);

layout_impl!(f64 => 8, f64, Align16<f64>);
layout_impl!(Vector2<f64> => 16, Align16<Vector2<f64>>, Align16<Vector2<f64>>);
layout_impl!(Vector3<f64> => 32, Align32<Vector3<f64>>, Align32<Vector3<f64>>);
layout_impl!(Vector4<f64> => 32, Align32<Vector4<f64>>, Align32<Vector4<f64>>);

// ---- matrices (array of columns; each column padded to its slot) --------------

impl<V: Vector + Std140Layout, const C: usize> Std140Layout for Matrix<V, C> {
    /// The column slot's alignment: 16 for every f32 matrix and dmat2, 32 for
    /// dmat3/dmat4 (dvec4 slots).
    const ALIGN: usize = V::SLOT_ALIGN;
    type Padded = PaddedCols<V::Slot, C>;
    type Slot = PaddedCols<V::Slot, C>;
}
impl<V: Vector + Std430Layout, const C: usize> Std430Layout for Matrix<V, C> {
    /// The column's natural alignment, not rounded.
    const ALIGN: usize = V::ALIGN;
    type Padded = PaddedCols<V::Padded, C>;
}
