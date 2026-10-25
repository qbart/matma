//! Vector types.
//!
//! [`Vector2`], [`Vector3`] and [`Vector4`] are the generic structs; `float3`,
//! `uint4`, `bool2`, ... are aliases, so the names mirror Slang/HLSL shader code.
//! Storage is the named components themselves (`x, y, z, w`) in a `#[repr(C)]`
//! struct, so a vector drops straight into a GPU buffer.
//!
//! `operator*` on vectors is **component-wise** (as in Slang); matrix products
//! use `Matrix * Vector`.
//!
//! Arithmetic on [`Vector4`] goes through the 4-lane hooks on [`Scalar`], which
//! `f32` routes to the SIMD backend in [`crate::simd`].
#![allow(non_camel_case_types)]

use core::iter::Sum;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Sub, SubAssign,
};

use crate::common;
use crate::scalar::{Componentwise, Dim, Float, Scalar, Signed, Vector};

/// A 2-component vector.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector2<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
}

/// A 3-component vector.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector3<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
    /// Third component.
    pub z: T,
}

/// A 4-component vector.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector4<T> {
    /// First component.
    pub x: T,
    /// Second component.
    pub y: T,
    /// Third component.
    pub z: T,
    /// Fourth component.
    pub w: T,
}

// ---- helpers used by the macros -----------------------------------------------

#[inline]
fn dot_fold<V: Componentwise>(a: V, b: V) -> V::Scalar {
    a.zip(b, |x, y| x * y)
        .fold(V::Scalar::ZERO, |acc, x| acc + x)
}

#[inline]
fn dot_hook4<T: Scalar>(a: Vector4<T>, b: Vector4<T>) -> T {
    T::dot4(a.to_array(), b.to_array())
}

// ---- shared implementation ----------------------------------------------------

macro_rules! vector_common {
    ($V:ident, $n:literal, $dot:path, [$($i:tt => $f:ident),+]) => {
        impl<T: Copy> $V<T> {
            /// Build from components.
            #[inline]
            pub const fn new($($f: T),+) -> Self {
                Self { $($f),+ }
            }
            /// Every component set to `s`.
            #[inline]
            pub const fn splat(s: T) -> Self {
                Self { $($f: s),+ }
            }
            /// Build from an array.
            #[inline]
            pub const fn from_array(a: [T; $n]) -> Self {
                Self { $($f: a[$i]),+ }
            }
            /// The components as an array.
            #[inline]
            pub const fn to_array(self) -> [T; $n] {
                [$(self.$f),+]
            }
            /// The components as a contiguous slice (for GPU upload / C APIs).
            #[inline]
            #[allow(unsafe_code)]
            pub fn as_slice(&self) -> &[T] {
                // SAFETY: `#[repr(C)]` with all fields of type `T` means the
                // components are laid out contiguously, starting at `x`.
                unsafe { core::slice::from_raw_parts(&self.x as *const T, $n) }
            }
            /// The components as a contiguous mutable slice.
            #[inline]
            #[allow(unsafe_code)]
            pub fn as_mut_slice(&mut self) -> &mut [T] {
                // SAFETY: as in `as_slice`; `&mut self` grants exclusive access.
                unsafe { core::slice::from_raw_parts_mut(&mut self.x as *mut T, $n) }
            }
            /// Pointer to the first component.
            #[inline]
            pub const fn as_ptr(&self) -> *const T {
                &self.x as *const T
            }
            /// Mutable pointer to the first component.
            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut T {
                &mut self.x as *mut T
            }
            /// Apply `f` to every component (may change the component type:
            /// `v.map(|x| x as f32)`).
            #[inline]
            pub fn map<U: Copy>(self, mut f: impl FnMut(T) -> U) -> $V<U> {
                $V { $($f: f(self.$f)),+ }
            }
            /// Generic swizzle to two components: `v.swizzle2::<1, 0>()` is `v.yx()`.
            #[inline]
            pub fn swizzle2<const A: usize, const B: usize>(self) -> Vector2<T> {
                const { assert!(A < $n && B < $n, "swizzle index out of range") }
                Vector2::new(self[A], self[B])
            }
            /// Generic swizzle to three components: `v.swizzle3::<2, 1, 0>()`.
            #[inline]
            pub fn swizzle3<const A: usize, const B: usize, const C: usize>(self) -> Vector3<T> {
                const { assert!(A < $n && B < $n && C < $n, "swizzle index out of range") }
                Vector3::new(self[A], self[B], self[C])
            }
            /// Generic swizzle to four components: `v.swizzle4::<0, 0, 0, 0>()` splats `x`.
            #[inline]
            pub fn swizzle4<const A: usize, const B: usize, const C: usize, const D: usize>(
                self,
            ) -> Vector4<T> {
                const { assert!(A < $n && B < $n && C < $n && D < $n, "swizzle index out of range") }
                Vector4::new(self[A], self[B], self[C], self[D])
            }
            /// `a` where `mask` is true, `b` elsewhere (HLSL `select`).
            #[inline]
            pub fn select(mask: $V<bool>, a: Self, b: Self) -> Self {
                $V { $($f: if mask.$f { a.$f } else { b.$f }),+ }
            }
        }

        impl<T: Copy + PartialOrd> $V<T> {
            /// Component-wise `self < other`.
            #[inline]
            pub fn cmp_lt(self, other: Self) -> $V<bool> { $V { $($f: self.$f < other.$f),+ } }
            /// Component-wise `self <= other`.
            #[inline]
            pub fn cmp_le(self, other: Self) -> $V<bool> { $V { $($f: self.$f <= other.$f),+ } }
            /// Component-wise `self > other`.
            #[inline]
            pub fn cmp_gt(self, other: Self) -> $V<bool> { $V { $($f: self.$f > other.$f),+ } }
            /// Component-wise `self >= other`.
            #[inline]
            pub fn cmp_ge(self, other: Self) -> $V<bool> { $V { $($f: self.$f >= other.$f),+ } }
            /// Component-wise `self == other`.
            #[inline]
            pub fn cmp_eq(self, other: Self) -> $V<bool> { $V { $($f: self.$f == other.$f),+ } }
            /// Component-wise `self != other`.
            #[inline]
            pub fn cmp_ne(self, other: Self) -> $V<bool> { $V { $($f: self.$f != other.$f),+ } }
        }

        impl $V<bool> {
            /// `true` if any component is `true`.
            #[inline]
            pub const fn any(self) -> bool { $(self.$f)||+ }
            /// `true` if every component is `true`.
            #[inline]
            pub const fn all(self) -> bool { $(self.$f)&&+ }
        }

        impl Not for $V<bool> {
            type Output = Self;
            #[inline]
            fn not(self) -> Self { Self { $($f: !self.$f),+ } }
        }

        impl<T> Index<usize> for $V<T> {
            type Output = T;
            #[inline]
            fn index(&self, i: usize) -> &T {
                match i {
                    $($i => &self.$f,)+
                    _ => panic!(concat!(stringify!($V), " index out of range")),
                }
            }
        }
        impl<T> IndexMut<usize> for $V<T> {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut T {
                match i {
                    $($i => &mut self.$f,)+
                    _ => panic!(concat!(stringify!($V), " index out of range")),
                }
            }
        }

        impl<T: Copy> From<[T; $n]> for $V<T> {
            #[inline]
            fn from(a: [T; $n]) -> Self { Self::from_array(a) }
        }
        impl<T: Copy> From<$V<T>> for [T; $n] {
            #[inline]
            fn from(v: $V<T>) -> Self { v.to_array() }
        }
        impl<T: Copy> AsRef<[T]> for $V<T> {
            #[inline]
            fn as_ref(&self) -> &[T] { self.as_slice() }
        }
        impl<T: Copy> AsMut<[T]> for $V<T> {
            #[inline]
            fn as_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
        }

        // ---- arithmetic-dependent API ----

        impl<T: Scalar> $V<T> {
            /// All zeros.
            pub const ZERO: Self = Self::splat(T::ZERO);
            /// All ones.
            pub const ONE: Self = Self::splat(T::ONE);

            /// Dot product.
            #[inline]
            pub fn dot(self, other: Self) -> T { common::dot(self, other) }
            /// Squared magnitude (no sqrt; cheaper for comparisons).
            #[inline]
            pub fn length2(self) -> T { common::length2(self) }
            /// Squared distance to `other`.
            #[inline]
            pub fn distance2(self, other: Self) -> T { common::distance2(self, other) }
            /// Component-wise minimum.
            #[inline]
            pub fn min(self, other: Self) -> Self { common::min(self, other) }
            /// Component-wise maximum.
            #[inline]
            pub fn max(self, other: Self) -> Self { common::max(self, other) }
            /// Smallest component.
            #[inline]
            pub fn min_component(self) -> T { common::min_component(self) }
            /// Largest component.
            #[inline]
            pub fn max_component(self) -> T { common::max_component(self) }
            /// Component-wise clamp between two vectors.
            #[inline]
            pub fn clamp(self, lo: Self, hi: Self) -> Self { common::clamp(self, lo, hi) }
            /// Clamp every component to `[lo, hi]`.
            #[inline]
            pub fn clamp_scalar(self, lo: T, hi: T) -> Self {
                common::clamp(self, Self::splat(lo), Self::splat(hi))
            }
            /// Clamp every component to `[0, 1]`.
            #[inline]
            pub fn saturate(self) -> Self { common::saturate(self) }
            /// Component-wise absolute value.
            #[inline]
            pub fn abs(self) -> Self { common::abs(self) }
            /// Component-wise sign: `-1`, `0` or `1`.
            #[inline]
            pub fn sign(self) -> Self { common::sign(self) }
            /// Linear interpolation toward `other` by `t`.
            #[inline]
            pub fn lerp(self, other: Self, t: T) -> Self { common::lerp(self, other, t) }
            /// Reflect about normal `n`: `self - 2 * dot(self, n) * n`.
            #[inline]
            pub fn reflect(self, n: Self) -> Self { common::reflect(self, n) }
            /// Component-wise `0` where `self < edge`, else `1`.
            #[inline]
            pub fn step(self, edge: T) -> Self { common::step(edge, self) }
            /// Sum of the components.
            #[inline]
            pub fn sum(self) -> T { Componentwise::fold(self, T::ZERO, |a, x| a + x) }
            /// Product of the components.
            #[inline]
            pub fn product(self) -> T { Componentwise::fold(self, T::ONE, |a, x| a * x) }
        }

        impl<T: Float> $V<T> {
            /// Magnitude.
            #[inline]
            pub fn length(self) -> T { common::length(self) }
            /// Unit vector in the same direction. A zero vector yields NaN components
            /// (as in GLM); see [`Self::normalize_or_zero`].
            #[inline]
            pub fn normalize(self) -> Self { common::normalize(self) }
            /// Unit vector in the same direction, or zero if the length is ~zero.
            #[inline]
            pub fn normalize_or_zero(self) -> Self {
                let l2 = self.length2();
                if l2 < T::MIN_POSITIVE { Self::ZERO } else { self * (T::ONE / l2.sqrt()) }
            }
            /// Distance to `other`.
            #[inline]
            pub fn distance(self, other: Self) -> T { common::distance(self, other) }
            /// Unsigned angle to `other`, in radians, `[0, pi]`.
            #[inline]
            pub fn angle(self, other: Self) -> T { common::angle(self, other) }
            /// Projection onto `on_normal` (need not be unit).
            #[inline]
            pub fn project(self, on_normal: Self) -> Self { common::project(self, on_normal) }
            /// Component in the plane with the given normal.
            #[inline]
            pub fn project_on_plane(self, plane_normal: Self) -> Self {
                common::project_on_plane(self, plane_normal)
            }
            /// Move toward `target` by at most `max_delta`, never overshooting.
            #[inline]
            pub fn move_towards(self, target: Self, max_delta: T) -> Self {
                common::move_towards(self, target, max_delta)
            }
            /// Component-wise floor.
            #[inline]
            pub fn floor(self) -> Self { common::floor(self) }
            /// Component-wise ceil.
            #[inline]
            pub fn ceil(self) -> Self { common::ceil(self) }
            /// Component-wise round (half away from zero).
            #[inline]
            pub fn round(self) -> Self { common::round(self) }
            /// Component-wise truncation.
            #[inline]
            pub fn trunc(self) -> Self { common::trunc(self) }
            /// Component-wise fractional part `x - floor(x)`.
            #[inline]
            pub fn frac(self) -> Self { common::frac(self) }
            /// Component-wise square root.
            #[inline]
            pub fn sqrt(self) -> Self { common::sqrt(self) }
            /// Component-wise reciprocal square root.
            #[inline]
            pub fn rsqrt(self) -> Self { common::rsqrt(self) }
            /// Component-wise smoothstep of `self` between `e0` and `e1`.
            #[inline]
            pub fn smoothstep(self, e0: T, e1: T) -> Self { common::smoothstep(e0, e1, self) }
            /// `true` if any component is NaN.
            #[inline]
            pub fn is_nan(self) -> bool { Componentwise::fold(self, false, |a, x| a || x.is_nan()) }
            /// `true` if every component is finite.
            #[inline]
            pub fn is_finite(self) -> bool { Componentwise::fold(self, true, |a, x| a && x.is_finite()) }
        }

        // ---- traits ----

        impl<T: Scalar> Componentwise for $V<T> {
            type Scalar = T;
            const N: usize = $n;
            #[inline]
            fn splat(s: T) -> Self { Self::splat(s) }
            #[inline]
            fn map(self, mut f: impl FnMut(T) -> T) -> Self { Self { $($f: f(self.$f)),+ } }
            #[inline]
            fn zip(self, o: Self, mut f: impl FnMut(T, T) -> T) -> Self {
                Self { $($f: f(self.$f, o.$f)),+ }
            }
            #[inline]
            fn zip3(self, b: Self, c: Self, mut f: impl FnMut(T, T, T) -> T) -> Self {
                Self { $($f: f(self.$f, b.$f, c.$f)),+ }
            }
            #[inline]
            fn fold<A>(self, init: A, mut f: impl FnMut(A, T) -> A) -> A {
                let mut acc = init;
                $(acc = f(acc, self.$f);)+
                acc
            }
            #[inline]
            fn dot(self, o: Self) -> T { $dot(self, o) }
        }

        impl<T> Dim<$n> for $V<T> {}

        impl<T: Scalar> Vector for $V<T> {
            const ZERO: Self = Self::ZERO;
        }

        impl<T: Signed> Neg for $V<T> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self { Self { $($f: -self.$f),+ } }
        }

        impl<T: Scalar> AddAssign for $V<T> {
            #[inline]
            fn add_assign(&mut self, o: Self) { *self = *self + o; }
        }
        impl<T: Scalar> SubAssign for $V<T> {
            #[inline]
            fn sub_assign(&mut self, o: Self) { *self = *self - o; }
        }
        impl<T: Scalar> MulAssign for $V<T> {
            #[inline]
            fn mul_assign(&mut self, o: Self) { *self = *self * o; }
        }
        impl<T: Scalar> DivAssign for $V<T> {
            #[inline]
            fn div_assign(&mut self, o: Self) { *self = *self / o; }
        }
        impl<T: Scalar> AddAssign<T> for $V<T> {
            #[inline]
            fn add_assign(&mut self, s: T) { *self = *self + s; }
        }
        impl<T: Scalar> SubAssign<T> for $V<T> {
            #[inline]
            fn sub_assign(&mut self, s: T) { *self = *self - s; }
        }
        impl<T: Scalar> MulAssign<T> for $V<T> {
            #[inline]
            fn mul_assign(&mut self, s: T) { *self = *self * s; }
        }
        impl<T: Scalar> DivAssign<T> for $V<T> {
            #[inline]
            fn div_assign(&mut self, s: T) { *self = *self / s; }
        }

        impl<T: Scalar> Sum for $V<T> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, b| a + b)
            }
        }
        impl<'a, T: Scalar> Sum<&'a $V<T>> for $V<T> {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, b| a + *b)
            }
        }
    };
}

vector_common!(Vector2, 2, dot_fold, [0 => x, 1 => y]);
vector_common!(Vector3, 3, dot_fold, [0 => x, 1 => y, 2 => z]);
vector_common!(Vector4, 4, dot_hook4, [0 => x, 1 => y, 2 => z, 3 => w]);

// ---- binary operators: direct component-wise (2 and 3 components) ------------

macro_rules! vector_binop_direct {
    ($V:ident, $Trait:ident, $method:ident, $op:tt, [$($f:ident),+]) => {
        impl<T: Scalar> $Trait for $V<T> {
            type Output = Self;
            #[inline]
            fn $method(self, o: Self) -> Self { Self { $($f: self.$f $op o.$f),+ } }
        }
        impl<T: Scalar> $Trait<T> for $V<T> {
            type Output = Self;
            #[inline]
            fn $method(self, s: T) -> Self { Self { $($f: self.$f $op s),+ } }
        }
    };
}
macro_rules! vector_ops_direct {
    ($V:ident, [$($f:ident),+]) => {
        vector_binop_direct!($V, Add, add, +, [$($f),+]);
        vector_binop_direct!($V, Sub, sub, -, [$($f),+]);
        vector_binop_direct!($V, Mul, mul, *, [$($f),+]);
        vector_binop_direct!($V, Div, div, /, [$($f),+]);
    };
}
vector_ops_direct!(Vector2, [x, y]);
vector_ops_direct!(Vector3, [x, y, z]);

// ---- binary operators: Vector4 through the 4-lane hooks (SIMD for f32) -------

macro_rules! vector4_binop {
    ($Trait:ident, $method:ident, $hook:ident) => {
        impl<T: Scalar> $Trait for Vector4<T> {
            type Output = Self;
            #[inline]
            fn $method(self, o: Self) -> Self {
                Self::from_array(T::$hook(self.to_array(), o.to_array()))
            }
        }
        impl<T: Scalar> $Trait<T> for Vector4<T> {
            type Output = Self;
            #[inline]
            fn $method(self, s: T) -> Self {
                Self::from_array(T::$hook(self.to_array(), [s; 4]))
            }
        }
    };
}
vector4_binop!(Add, add, add4);
vector4_binop!(Sub, sub, sub4);
vector4_binop!(Mul, mul, mul4);
vector4_binop!(Div, div, div4);

// ---- scalar on the left: `2.0 * v` ---------------------------------------------
// Coherence forbids `impl<T> Mul<Vector3<T>> for T`, so these are spelled out per
// primitive type.

macro_rules! scalar_lhs_ops {
    ($($s:ty),*) => {$(
        scalar_lhs_ops!(@one $s, Vector2);
        scalar_lhs_ops!(@one $s, Vector3);
        scalar_lhs_ops!(@one $s, Vector4);
    )*};
    (@one $s:ty, $V:ident) => {
        impl Add<$V<$s>> for $s {
            type Output = $V<$s>;
            #[inline]
            fn add(self, v: $V<$s>) -> $V<$s> { $V::splat(self) + v }
        }
        impl Sub<$V<$s>> for $s {
            type Output = $V<$s>;
            #[inline]
            fn sub(self, v: $V<$s>) -> $V<$s> { $V::splat(self) - v }
        }
        impl Mul<$V<$s>> for $s {
            type Output = $V<$s>;
            #[inline]
            fn mul(self, v: $V<$s>) -> $V<$s> { v * self }
        }
        impl Div<$V<$s>> for $s {
            type Output = $V<$s>;
            #[inline]
            fn div(self, v: $V<$s>) -> $V<$s> { $V::splat(self) / v }
        }
    };
}
scalar_lhs_ops!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// ---- per-type extras: swizzles, axes, conversions --------------------------------

impl<T: Copy> Vector2<T> {
    /// `(y, x)`.
    #[inline]
    pub const fn yx(self) -> Vector2<T> {
        Vector2::new(self.y, self.x)
    }
    /// Append a third component.
    #[inline]
    pub const fn extend(self, z: T) -> Vector3<T> {
        Vector3::new(self.x, self.y, z)
    }
}

impl<T: Copy> Vector3<T> {
    /// `(x, y)`.
    #[inline]
    pub const fn xy(self) -> Vector2<T> {
        Vector2::new(self.x, self.y)
    }
    /// `(x, z)`.
    #[inline]
    pub const fn xz(self) -> Vector2<T> {
        Vector2::new(self.x, self.z)
    }
    /// `(y, z)`.
    #[inline]
    pub const fn yz(self) -> Vector2<T> {
        Vector2::new(self.y, self.z)
    }
    /// Append a fourth component (`v.extend(1.0)` makes a point, `v.extend(0.0)` a direction).
    #[inline]
    pub const fn extend(self, w: T) -> Vector4<T> {
        Vector4::new(self.x, self.y, self.z, w)
    }
    /// Drop `z`.
    #[inline]
    pub const fn truncate(self) -> Vector2<T> {
        self.xy()
    }
}

impl<T: Copy> Vector4<T> {
    /// `(x, y, z)` — drop `w`.
    #[inline]
    pub const fn xyz(self) -> Vector3<T> {
        Vector3::new(self.x, self.y, self.z)
    }
    /// `(x, y)`.
    #[inline]
    pub const fn xy(self) -> Vector2<T> {
        Vector2::new(self.x, self.y)
    }
    /// `(x, z)`.
    #[inline]
    pub const fn xz(self) -> Vector2<T> {
        Vector2::new(self.x, self.z)
    }
    /// `(y, z)`.
    #[inline]
    pub const fn yz(self) -> Vector2<T> {
        Vector2::new(self.y, self.z)
    }
    /// `(z, w)`.
    #[inline]
    pub const fn zw(self) -> Vector2<T> {
        Vector2::new(self.z, self.w)
    }
    /// Color alias for `xyz`.
    #[inline]
    pub const fn rgb(self) -> Vector3<T> {
        self.xyz()
    }
    /// Color alias for `xy`.
    #[inline]
    pub const fn rg(self) -> Vector2<T> {
        self.xy()
    }
    /// Drop `w`.
    #[inline]
    pub const fn truncate(self) -> Vector3<T> {
        self.xyz()
    }
}

impl<T: Scalar> Vector2<T> {
    /// `(1, 0)`.
    pub const X: Self = Self::new(T::ONE, T::ZERO);
    /// `(0, 1)`.
    pub const Y: Self = Self::new(T::ZERO, T::ONE);
}

impl<T: Scalar> Vector3<T> {
    /// `(1, 0, 0)`.
    pub const X: Self = Self::new(T::ONE, T::ZERO, T::ZERO);
    /// `(0, 1, 0)`.
    pub const Y: Self = Self::new(T::ZERO, T::ONE, T::ZERO);
    /// `(0, 0, 1)`.
    pub const Z: Self = Self::new(T::ZERO, T::ZERO, T::ONE);

    /// Cross product (right-handed).
    #[inline]
    pub fn cross(self, b: Self) -> Self {
        common::cross(self, b)
    }
}

impl<T: Signed> Vector3<T> {
    // World space is right-handed and Y-up, with forward = -Z: that is the
    // direction a camera built by `look_at` faces, and matches GLM/OpenGL world
    // conventions. Vulkan's Y-down only lives in clip space (baked into the
    // projection builders), not here. NB: this differs from Unity, which is
    // left-handed with forward = +Z.

    /// World right, `+X`.
    pub const RIGHT: Self = Self::new(T::ONE, T::ZERO, T::ZERO);
    /// World left, `-X`.
    pub const LEFT: Self = Self::new(T::NEG_ONE, T::ZERO, T::ZERO);
    /// World up, `+Y`.
    pub const UP: Self = Self::new(T::ZERO, T::ONE, T::ZERO);
    /// World down, `-Y`.
    pub const DOWN: Self = Self::new(T::ZERO, T::NEG_ONE, T::ZERO);
    /// World forward, `-Z` (right-handed, the way a `look_at` camera faces).
    pub const FORWARD: Self = Self::new(T::ZERO, T::ZERO, T::NEG_ONE);
    /// World back, `+Z`.
    pub const BACK: Self = Self::new(T::ZERO, T::ZERO, T::ONE);
}

impl<T: Scalar> Vector4<T> {
    /// `(1, 0, 0, 0)`.
    pub const X: Self = Self::new(T::ONE, T::ZERO, T::ZERO, T::ZERO);
    /// `(0, 1, 0, 0)`.
    pub const Y: Self = Self::new(T::ZERO, T::ONE, T::ZERO, T::ZERO);
    /// `(0, 0, 1, 0)`.
    pub const Z: Self = Self::new(T::ZERO, T::ZERO, T::ONE, T::ZERO);
    /// `(0, 0, 0, 1)`.
    pub const W: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ONE);
}

impl<T: Copy> From<(T, T)> for Vector2<T> {
    #[inline]
    fn from((x, y): (T, T)) -> Self {
        Self::new(x, y)
    }
}
impl<T: Copy> From<(T, T, T)> for Vector3<T> {
    #[inline]
    fn from((x, y, z): (T, T, T)) -> Self {
        Self::new(x, y, z)
    }
}
impl<T: Copy> From<(T, T, T, T)> for Vector4<T> {
    #[inline]
    fn from((x, y, z, w): (T, T, T, T)) -> Self {
        Self::new(x, y, z, w)
    }
}
impl<T: Copy> From<(Vector2<T>, T)> for Vector3<T> {
    #[inline]
    fn from((xy, z): (Vector2<T>, T)) -> Self {
        xy.extend(z)
    }
}
impl<T: Copy> From<(Vector3<T>, T)> for Vector4<T> {
    #[inline]
    fn from((xyz, w): (Vector3<T>, T)) -> Self {
        xyz.extend(w)
    }
}
impl<T: Copy> From<(Vector2<T>, T, T)> for Vector4<T> {
    #[inline]
    fn from((xy, z, w): (Vector2<T>, T, T)) -> Self {
        Vector4::new(xy.x, xy.y, z, w)
    }
}

// ---- aliases (Slang/HLSL names) -------------------------------------------------

/// Slang's `uint`.
pub type uint = u32;

/// `Vector2<f32>`.
pub type float2 = Vector2<f32>;
/// `Vector3<f32>`.
pub type float3 = Vector3<f32>;
/// `Vector4<f32>`.
pub type float4 = Vector4<f32>;
/// `Vector2<f64>`.
pub type double2 = Vector2<f64>;
/// `Vector3<f64>`.
pub type double3 = Vector3<f64>;
/// `Vector4<f64>`.
pub type double4 = Vector4<f64>;
/// `Vector2<i32>`.
pub type int2 = Vector2<i32>;
/// `Vector3<i32>`.
pub type int3 = Vector3<i32>;
/// `Vector4<i32>`.
pub type int4 = Vector4<i32>;
/// `Vector2<u32>`.
pub type uint2 = Vector2<u32>;
/// `Vector3<u32>`.
pub type uint3 = Vector3<u32>;
/// `Vector4<u32>`.
pub type uint4 = Vector4<u32>;
/// `Vector2<bool>`.
pub type bool2 = Vector2<bool>;
/// `Vector3<bool>`.
pub type bool3 = Vector3<bool>;
/// `Vector4<bool>`.
pub type bool4 = Vector4<bool>;
/// `Vector2<i16>`: 4 bytes, memcpy's straight into a `R16G16_SNORM` / `_SINT`
/// vertex attribute (see [`crate::pack`]).
pub type short2 = Vector2<i16>;
/// `Vector4<u8>`: an RGBA8 color.
pub type ubyte4 = Vector4<u8>;

// ---- GLSL-style aliases ---------------------------------------------------------

/// GLSL name for [`float2`].
pub type vec2 = float2;
/// GLSL name for [`float3`].
pub type vec3 = float3;
/// GLSL name for [`float4`].
pub type vec4 = float4;
/// GLSL name for [`double2`].
pub type dvec2 = double2;
/// GLSL name for [`double3`].
pub type dvec3 = double3;
/// GLSL name for [`double4`].
pub type dvec4 = double4;
/// GLSL name for [`int2`].
pub type ivec2 = int2;
/// GLSL name for [`int3`].
pub type ivec3 = int3;
/// GLSL name for [`int4`].
pub type ivec4 = int4;
/// GLSL name for [`uint2`].
pub type uvec2 = uint2;
/// GLSL name for [`uint3`].
pub type uvec3 = uint3;
/// GLSL name for [`uint4`].
pub type uvec4 = uint4;
/// GLSL name for [`bool2`].
pub type bvec2 = bool2;
/// GLSL name for [`bool3`].
pub type bvec3 = bool3;
/// GLSL name for [`bool4`].
pub type bvec4 = bool4;
