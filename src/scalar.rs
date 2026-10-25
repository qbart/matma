//! Scalar traits: the numeric building blocks every vector and matrix is generic over.
//!
//! - [`Scalar`]: anything you can store in a vector and do arithmetic on
//!   (`f32`, `f64`, `i32`, `u32`, `i16`, ...).
//! - [`Signed`]: adds negation (`f32`, `i32`, ...; not `u32`).
//! - [`Float`]: adds the `<cmath>`-style functions (`sqrt`, `sin`, `acos`, ...).
//! - [`Componentwise`]: the "component-wise" abstraction shared by scalars (one lane)
//!   and vectors (N lanes). The functions in [`crate::common`] are written against it
//!   once and work for both, like Slang's scalar/vector overloads.
//! - [`Vector`] and [`Dim`]: what the matrix code needs from a column type.

use core::fmt::Debug;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::simd;

// ---- Scalar -------------------------------------------------------------------

/// A number that can live in a vector: `Copy`, comparable, and closed under `+ - * /`.
///
/// Every scalar is also a one-lane [`Componentwise`], so the functions in
/// [`crate::common`] accept scalars and vectors alike.
///
/// The `*4` hooks are the SIMD seam for 4-component vectors: the generic
/// `Vector4<T>` operators call them, the defaults are plain scalar loops, and the
/// `f32` implementation overrides them with the backend in [`crate::simd`].
pub trait Scalar:
    Componentwise<Scalar = Self>
    + Copy
    + Default
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;

    /// Lane-wise `a + b` for four lanes.
    #[inline]
    fn add4(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
    /// Lane-wise `a - b` for four lanes.
    #[inline]
    fn sub4(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }
    /// Lane-wise `a * b` for four lanes.
    #[inline]
    fn mul4(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }
    /// Lane-wise `a / b` for four lanes.
    #[inline]
    fn div4(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        [a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]]
    }
    /// Four-lane dot product.
    #[inline]
    fn dot4(a: [Self; 4], b: [Self; 4]) -> Self {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
}

/// A [`Scalar`] with a sign: supports unary minus.
pub trait Signed: Scalar + Neg<Output = Self> {
    /// `-1`.
    const NEG_ONE: Self;
}

/// A floating-point [`Scalar`] (`f32` or `f64`) with the transcendental functions.
///
/// Method names follow Rust's inherent float methods (`ln`, `powf`, `atan2`);
/// the Slang-spelled free functions live in [`crate::common`].
pub trait Float: Signed {
    /// Archimedes' constant.
    const PI: Self;
    /// `2 * PI`.
    const TAU: Self;
    /// Machine epsilon.
    const EPSILON: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Largest finite value.
    const MAX: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Quiet NaN.
    const NAN: Self;

    /// Lossy conversion from `f64` (for literals in generic code).
    fn from_f64(v: f64) -> Self;
    /// Widening conversion to `f64`.
    fn to_f64(self) -> f64;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Largest integer `<= self`.
    fn floor(self) -> Self;
    /// Smallest integer `>= self`.
    fn ceil(self) -> Self;
    /// Nearest integer, half away from zero.
    fn round(self) -> Self;
    /// Integer part (toward zero).
    fn trunc(self) -> Self;
    /// `e^self`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// `2^self`.
    fn exp2(self) -> Self;
    /// Base-2 logarithm.
    fn log2(self) -> Self;
    /// Sine (radians).
    fn sin(self) -> Self;
    /// Cosine (radians).
    fn cos(self) -> Self;
    /// Tangent (radians).
    fn tan(self) -> Self;
    /// Arcsine, in `[-pi/2, pi/2]`.
    fn asin(self) -> Self;
    /// Arccosine, in `[0, pi]`.
    fn acos(self) -> Self;
    /// Arctangent, in `(-pi/2, pi/2)`.
    fn atan(self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// `self^exp`.
    fn powf(self, exp: Self) -> Self;
    /// Four-quadrant arctangent of `self / x`, in `(-pi, pi]`.
    fn atan2(self, x: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// `true` for NaN.
    fn is_nan(self) -> bool;
    /// `true` unless infinite or NaN.
    fn is_finite(self) -> bool;
}

// ---- implementations for the primitive types ----------------------------------

macro_rules! impl_scalar {
    ($($t:ty => $zero:expr, $one:expr;)*) => {$(
        impl Scalar for $t {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
        }
    )*};
}

impl_scalar! {
    f64 => 0.0, 1.0;
    i8 => 0, 1;
    i16 => 0, 1;
    i32 => 0, 1;
    i64 => 0, 1;
    isize => 0, 1;
    u8 => 0, 1;
    u16 => 0, 1;
    u32 => 0, 1;
    u64 => 0, 1;
    usize => 0, 1;
}

// f32 routes the 4-wide hooks through the SIMD backend.
impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        simd::store4(simd::add(simd::load4(a), simd::load4(b)))
    }
    #[inline]
    fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        simd::store4(simd::sub(simd::load4(a), simd::load4(b)))
    }
    #[inline]
    fn mul4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        simd::store4(simd::mul(simd::load4(a), simd::load4(b)))
    }
    #[inline]
    fn div4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        simd::store4(simd::div(simd::load4(a), simd::load4(b)))
    }
    #[inline]
    fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        simd::hsum(simd::mul(simd::load4(a), simd::load4(b)))
    }
}

macro_rules! impl_signed {
    ($($t:ty),*) => {$(
        impl Signed for $t {
            const NEG_ONE: Self = -Self::ONE;
        }
    )*};
}
impl_signed!(f32, f64, i8, i16, i32, i64, isize);

macro_rules! impl_float {
    ($t:ident) => {
        impl Float for $t {
            const PI: Self = core::$t::consts::PI;
            const TAU: Self = core::$t::consts::TAU;
            const EPSILON: Self = <$t>::EPSILON;
            const MIN_POSITIVE: Self = <$t>::MIN_POSITIVE;
            const MAX: Self = <$t>::MAX;
            const INFINITY: Self = <$t>::INFINITY;
            const NAN: Self = <$t>::NAN;

            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            #[inline]
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            #[inline]
            fn round(self) -> Self {
                <$t>::round(self)
            }
            #[inline]
            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline]
            fn exp2(self) -> Self {
                <$t>::exp2(self)
            }
            #[inline]
            fn log2(self) -> Self {
                <$t>::log2(self)
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline]
            fn tan(self) -> Self {
                <$t>::tan(self)
            }
            #[inline]
            fn asin(self) -> Self {
                <$t>::asin(self)
            }
            #[inline]
            fn acos(self) -> Self {
                <$t>::acos(self)
            }
            #[inline]
            fn atan(self) -> Self {
                <$t>::atan(self)
            }
            #[inline]
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }
            #[inline]
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            #[inline]
            fn powf(self, exp: Self) -> Self {
                <$t>::powf(self, exp)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                <$t>::atan2(self, x)
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline]
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
            #[inline]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

// ---- Componentwise --------------------------------------------------------------

/// Something made of one or more [`Scalar`] components that can be processed lane
/// by lane: a scalar (one lane) or a `Vector2/3/4` (N lanes).
///
/// This is how [`crate::common`] gets Slang-style overloading: `saturate(0.5)` and
/// `saturate(float3 { .. })` are the same generic function.
pub trait Componentwise: Copy {
    /// The component type.
    type Scalar: Scalar;
    /// Number of components.
    const N: usize;

    /// Every component set to `s`.
    fn splat(s: Self::Scalar) -> Self;
    /// Apply `f` to each component.
    fn map(self, f: impl FnMut(Self::Scalar) -> Self::Scalar) -> Self;
    /// Combine matching components of `self` and `other` with `f`.
    fn zip(self, other: Self, f: impl FnMut(Self::Scalar, Self::Scalar) -> Self::Scalar) -> Self;
    /// Combine matching components of three values with `f`.
    fn zip3(
        self,
        b: Self,
        c: Self,
        f: impl FnMut(Self::Scalar, Self::Scalar, Self::Scalar) -> Self::Scalar,
    ) -> Self;
    /// Left fold over the components.
    fn fold<A>(self, init: A, f: impl FnMut(A, Self::Scalar) -> A) -> A;

    /// Sum of component products (for a scalar this is plain multiplication).
    #[inline]
    fn dot(self, other: Self) -> Self::Scalar {
        self.zip(other, |a, b| a * b)
            .fold(Self::Scalar::ZERO, |acc, x| acc + x)
    }
}

macro_rules! impl_componentwise_scalar {
    ($($t:ty),*) => {$(
        impl Componentwise for $t {
            type Scalar = $t;
            const N: usize = 1;
            #[inline] fn splat(s: Self) -> Self { s }
            #[inline] fn map(self, mut f: impl FnMut(Self) -> Self) -> Self { f(self) }
            #[inline] fn zip(self, o: Self, mut f: impl FnMut(Self, Self) -> Self) -> Self { f(self, o) }
            #[inline] fn zip3(self, b: Self, c: Self, mut f: impl FnMut(Self, Self, Self) -> Self) -> Self { f(self, b, c) }
            #[inline] fn fold<A>(self, init: A, mut f: impl FnMut(A, Self) -> A) -> A { f(init, self) }
            #[inline] fn dot(self, o: Self) -> Self { self * o }
        }
    )*};
}
impl_componentwise_scalar!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// ---- Vector / Dim ---------------------------------------------------------------

/// Marker: this type has exactly `N` components. Lets the matrix code pin the
/// column count of one operand to the row count of the other at compile time.
pub trait Dim<const N: usize> {}

/// A fixed-size vector usable as a matrix column: indexable, with the arithmetic
/// operators defined. Implemented by `Vector2/3/4<T>` for every [`Scalar`] `T`.
pub trait Vector:
    Componentwise
    + Default
    + PartialEq
    + Debug
    + Index<usize, Output = <Self as Componentwise>::Scalar>
    + IndexMut<usize>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Mul<<Self as Componentwise>::Scalar, Output = Self>
    + Div<<Self as Componentwise>::Scalar, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + MulAssign<<Self as Componentwise>::Scalar>
    + DivAssign<<Self as Componentwise>::Scalar>
{
    /// All components zero.
    const ZERO: Self;
}
