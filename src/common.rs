//! Common scalar/vector functions, Slang/HLSL spelling.
//!
//! Names follow Slang: `lerp` (not mix), `frac` (not fract), `saturate`, `rsqrt`,
//! `log` (natural). Every function is generic over [`Componentwise`], so one
//! definition serves scalars and vectors alike: `saturate(1.5)` and
//! `saturate(float3::new(..))` are the same function, exactly like the scalar and
//! component-wise overloads in a shader.
//!
//! The vector types also expose most of these as methods (`v.saturate()`).

use crate::scalar::{Componentwise, Float, Scalar};
use crate::vector::Vector3;

// ---- min / max ----------------------------------------------------------------
// NaN behaves like HLSL/GLSL, not IEEE minNum: the comparison is false for a NaN
// operand, so min/max return `b` — a NaN in `a` is dropped, a NaN in `b` wins.
// Consequently clamp/saturate quietly map NaN to a bound instead of propagating it.

/// Component-wise minimum (`a < b ? a : b`).
#[inline]
pub fn min<V: Componentwise>(a: V, b: V) -> V {
    a.zip(b, |a, b| if a < b { a } else { b })
}
/// Component-wise maximum (`a > b ? a : b`).
#[inline]
pub fn max<V: Componentwise>(a: V, b: V) -> V {
    a.zip(b, |a, b| if a > b { a } else { b })
}
/// Smallest component.
#[inline]
pub fn min_component<V: Componentwise>(v: V) -> V::Scalar {
    v.fold(None, |acc, x| match acc {
        Some(a) if x < a => Some(x),
        Some(a) => Some(a),
        None => Some(x),
    })
    .unwrap_or(V::Scalar::ZERO)
}
/// Largest component.
#[inline]
pub fn max_component<V: Componentwise>(v: V) -> V::Scalar {
    v.fold(None, |acc, x| match acc {
        Some(a) if x > a => Some(x),
        Some(a) => Some(a),
        None => Some(x),
    })
    .unwrap_or(V::Scalar::ZERO)
}

// ---- clamp / saturate ---------------------------------------------------------

/// Component-wise `max(lo, min(x, hi))`.
#[inline]
pub fn clamp<V: Componentwise>(x: V, lo: V, hi: V) -> V {
    x.zip3(lo, hi, |x, lo, hi| {
        let m = if x < hi { x } else { hi };
        if lo > m { lo } else { m }
    })
}
/// Clamp to `[0, 1]`.
#[inline]
pub fn saturate<V: Componentwise>(x: V) -> V {
    clamp(x, V::splat(V::Scalar::ZERO), V::splat(V::Scalar::ONE))
}

// ---- lerp ---------------------------------------------------------------------

/// `a + (b - a) * t`, with a scalar `t`.
#[inline]
pub fn lerp<V: Componentwise>(a: V, b: V, t: V::Scalar) -> V {
    a.zip(b, |a, b| a + (b - a) * t)
}
/// `a + (b - a) * t`, with a per-component `t`.
#[inline]
pub fn lerp_vec<V: Componentwise>(a: V, b: V, t: V) -> V {
    a.zip3(b, t, |a, b, t| a + (b - a) * t)
}

// ---- step / smoothstep --------------------------------------------------------

/// `0` where `x < edge`, else `1`.
#[inline]
pub fn step<V: Componentwise>(edge: V::Scalar, x: V) -> V {
    x.map(|x| {
        if x < edge {
            V::Scalar::ZERO
        } else {
            V::Scalar::ONE
        }
    })
}
/// Hermite smoothstep of `x` between `e0` and `e1`.
#[inline]
pub fn smoothstep<V: Componentwise>(e0: V::Scalar, e1: V::Scalar, x: V) -> V {
    let one = V::Scalar::ONE;
    let two = one + one;
    let three = two + one;
    x.map(|x| {
        let t = saturate((x - e0) / (e1 - e0));
        t * t * (three - two * t)
    })
}

// ---- abs / sign ---------------------------------------------------------------

/// Component-wise absolute value.
#[inline]
pub fn abs<V: Componentwise>(v: V) -> V {
    v.map(|x| {
        if x < V::Scalar::ZERO {
            V::Scalar::ZERO - x
        } else {
            x
        }
    })
}
/// Component-wise sign: `-1`, `0` or `1`.
#[inline]
pub fn sign<V: Componentwise>(v: V) -> V {
    v.map(|x| {
        if x > V::Scalar::ZERO {
            V::Scalar::ONE
        } else if x < V::Scalar::ZERO {
            V::Scalar::ZERO - V::Scalar::ONE
        } else {
            V::Scalar::ZERO
        }
    })
}
/// `1` where `a` and `b` are within `epsilon` of each other, else `0` (HLSL-style mask).
#[inline]
pub fn close_to<V: Componentwise>(a: V, b: V, epsilon: V::Scalar) -> V {
    a.zip(b, |a, b| {
        let d = if a > b { a - b } else { b - a };
        if d <= epsilon {
            V::Scalar::ONE
        } else {
            V::Scalar::ZERO
        }
    })
}

// ---- geometric ----------------------------------------------------------------

/// Dot product (for a scalar, plain multiplication).
#[inline]
pub fn dot<V: Componentwise>(a: V, b: V) -> V::Scalar {
    a.dot(b)
}
/// Cross product (right-handed).
#[inline]
pub fn cross<T: Scalar>(a: Vector3<T>, b: Vector3<T>) -> Vector3<T> {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}
/// Squared magnitude (no sqrt).
#[inline]
pub fn length2<V: Componentwise>(v: V) -> V::Scalar {
    v.dot(v)
}
/// Magnitude.
#[inline]
pub fn length<V: Componentwise>(v: V) -> V::Scalar
where
    V::Scalar: Float,
{
    v.dot(v).sqrt()
}
/// Unit vector in the same direction. NB: a zero vector divides by zero and yields
/// NaN components (as in GLM); guard upstream when the input can be zero.
#[inline]
pub fn normalize<V: Componentwise>(v: V) -> V
where
    V::Scalar: Float,
{
    let inv = V::Scalar::ONE / length(v);
    v.map(|x| x * inv)
}
/// HLSL reflect: incident `i` about normal `n` -> `i - 2 * dot(i, n) * n`.
#[inline]
pub fn reflect<V: Componentwise>(i: V, n: V) -> V {
    let k = (V::Scalar::ONE + V::Scalar::ONE) * dot(i, n);
    i.zip(n, |i, n| i - n * k)
}
/// Distance between two points.
#[inline]
pub fn distance<V: Componentwise>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    length(a.zip(b, |a, b| a - b))
}
/// Squared distance: no sqrt, cheaper for comparisons.
#[inline]
pub fn distance2<V: Componentwise>(a: V, b: V) -> V::Scalar {
    length2(a.zip(b, |a, b| a - b))
}

// ---- projection / angle / move-towards ----------------------------------------

/// Unsigned angle between `a` and `b`, in **radians**, range `[0, pi]`. Inputs need
/// not be unit length. Zero if either is ~zero. (Unity `Vector3.Angle`, but radians.)
#[inline]
pub fn angle<V: Componentwise>(a: V, b: V) -> V::Scalar
where
    V::Scalar: Float,
{
    // sqrt each length separately: length2(a) * length2(b) squares the magnitudes
    // and overflows f32 already at |a|*|b| ~ 1.8e19 (making every angle pi/2).
    let denom = length(a) * length(b);
    if denom < V::Scalar::from_f64(1e-20) {
        return V::Scalar::ZERO;
    }
    clamp(dot(a, b) / denom, -V::Scalar::ONE, V::Scalar::ONE).acos()
}

/// Projection of `v` onto `on_normal` (need not be unit). Zero if `on_normal` is ~zero.
#[inline]
pub fn project<V: Componentwise>(v: V, on_normal: V) -> V
where
    V::Scalar: Float,
{
    let sq = dot(on_normal, on_normal);
    if sq < V::Scalar::from_f64(1e-20) {
        return V::splat(V::Scalar::ZERO);
    }
    let k = dot(v, on_normal) / sq;
    on_normal.map(|n| n * k)
}

/// Component of `v` in the plane with the given normal: `v` minus its projection
/// onto the normal. Returns `v` unchanged when `plane_normal` is ~zero.
/// (Unity `ProjectOnPlane`.)
#[inline]
pub fn project_on_plane<V: Componentwise>(v: V, plane_normal: V) -> V
where
    V::Scalar: Float,
{
    v.zip(project(v, plane_normal), |v, p| v - p)
}

/// Move `current` toward `target` by at most `max_delta`; never overshoots, a
/// negative delta moves away. Works for scalars and vectors (Unity `MoveTowards`).
#[inline]
pub fn move_towards<V: Componentwise>(current: V, target: V, max_delta: V::Scalar) -> V
where
    V::Scalar: Float,
{
    let d = target.zip(current, |t, c| t - c);
    let dist = length(d);
    if dist <= max_delta || dist < V::Scalar::from_f64(1e-20) {
        return target;
    }
    let k = max_delta / dist;
    current.zip(d, |c, d| c + d * k)
}

// ---- transcendental wrappers --------------------------------------------------

macro_rules! unary_float_fn {
    ($(#[$doc:meta] $name:ident => $method:ident;)*) => {$(
        #[$doc]
        #[inline]
        pub fn $name<V: Componentwise>(v: V) -> V
        where
            V::Scalar: Float,
        {
            v.map(Float::$method)
        }
    )*};
}
unary_float_fn! {
    /// Component-wise floor.
    floor => floor;
    /// Component-wise ceil.
    ceil => ceil;
    /// Component-wise round (half away from zero).
    round => round;
    /// Component-wise truncation toward zero.
    trunc => trunc;
    /// Component-wise square root.
    sqrt => sqrt;
    /// Component-wise `e^x`.
    exp => exp;
    /// Component-wise natural logarithm (Slang `log`).
    log => ln;
    /// Component-wise `2^x`.
    exp2 => exp2;
    /// Component-wise base-2 logarithm.
    log2 => log2;
    /// Component-wise sine (radians).
    sin => sin;
    /// Component-wise cosine (radians).
    cos => cos;
    /// Component-wise tangent (radians).
    tan => tan;
    /// Component-wise arcsine.
    asin => asin;
    /// Component-wise arccosine.
    acos => acos;
    /// Component-wise arctangent.
    atan => atan;
    /// Component-wise hyperbolic sine.
    sinh => sinh;
    /// Component-wise hyperbolic cosine.
    cosh => cosh;
    /// Component-wise hyperbolic tangent.
    tanh => tanh;
}

/// Component-wise `atan2(y, x)`: full-circle angle in radians, range `(-pi, pi]`.
#[inline]
pub fn atan2<V: Componentwise>(y: V, x: V) -> V
where
    V::Scalar: Float,
{
    y.zip(x, Float::atan2)
}
/// Component-wise floating-point remainder; the sign follows `a` (like HLSL `fmod`).
#[inline]
pub fn fmod<V: Componentwise>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    a.zip(b, |a, b| a - (a / b).trunc() * b)
}
/// Component-wise `a^b`.
#[inline]
pub fn pow<V: Componentwise>(a: V, b: V) -> V
where
    V::Scalar: Float,
{
    a.zip(b, Float::powf)
}
/// Both sine and cosine of `x` in one call (handy for building rotations).
#[inline]
pub fn sincos<T: Float>(x: T) -> (T, T) {
    (x.sin(), x.cos())
}
/// Component-wise fractional part `x - floor(x)`.
#[inline]
pub fn frac<V: Componentwise>(v: V) -> V
where
    V::Scalar: Float,
{
    v.map(|x| x - x.floor())
}
/// Component-wise `1 / sqrt(x)`.
#[inline]
pub fn rsqrt<V: Componentwise>(v: V) -> V
where
    V::Scalar: Float,
{
    v.map(|x| V::Scalar::ONE / x.sqrt())
}

// ---- game-loop helpers --------------------------------------------------------

/// Wrap `x` into `[0, length]` — like a positive fmod (Unity `Mathf.Repeat`). Useful
/// for looping animation time or wrapping a value around a range. NB: the upper
/// bound is closed: rounding can land exactly on `length` (e.g. for a tiny
/// negative `x`), so don't rely on the result being strictly less than `length`.
#[inline]
pub fn repeat<T: Float>(x: T, length: T) -> T {
    clamp(x - (x / length).floor() * length, T::ZERO, length)
}

/// Bounce `x` back and forth in `[0, length]` at unit speed (Unity `Mathf.PingPong`).
#[inline]
pub fn ping_pong<T: Float>(x: T, length: T) -> T {
    let t = repeat(x, length + length);
    length - (t - length).abs()
}

/// Shortest-path angular lerp, in **radians**. Interpolates the wrapped difference
/// so it crosses the `±pi` seam correctly (Unity `Mathf.LerpAngle`, but radians).
#[inline]
pub fn lerp_angle<T: Float>(a: T, b: T, t: T) -> T {
    let mut delta = repeat(b - a, T::TAU);
    if delta > T::PI {
        delta -= T::TAU;
    }
    a + delta * t
}

// ---- rays ---------------------------------------------------------------------

/// A ray as origin + direction. `direction` need not be unit length; the distance
/// reported by [`ray_triangle_intersect`] is then measured in units of `|direction|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray<T> {
    /// Start point.
    pub origin: Vector3<T>,
    /// Direction (any length).
    pub direction: Vector3<T>,
}

impl<T: Copy> Ray<T> {
    /// Build from origin and direction.
    #[inline]
    pub const fn new(origin: Vector3<T>, direction: Vector3<T>) -> Self {
        Self { origin, direction }
    }
}

impl<T: Scalar> Ray<T> {
    /// `origin + t * direction`.
    #[inline]
    pub fn at(&self, t: T) -> Vector3<T> {
        self.origin + self.direction * t
    }
}

/// Möller–Trumbore ray/triangle intersection. On a front- or back-facing hit,
/// returns the ray parameter `t`; otherwise `None` (ray parallel to, or pointing
/// away from, the triangle). The hit point is `ray.at(t)`.
pub fn ray_triangle_intersect<T: Float>(
    r: &Ray<T>,
    a: Vector3<T>,
    b: Vector3<T>,
    c: Vector3<T>,
) -> Option<T> {
    let epsilon = T::from_f64(1e-7);
    let edge1 = b - a;
    let edge2 = c - a;
    let h = cross(r.direction, edge2);
    let det = dot(edge1, h);

    if det > -epsilon && det < epsilon {
        return None; // ray parallel to triangle
    }

    let inv_det = T::ONE / det;
    let s = r.origin - a;
    let u = dot(s, h) * inv_det;
    if u < T::ZERO || u > T::ONE {
        return None;
    }

    let q = cross(s, edge1);
    let v = dot(r.direction, q) * inv_det;
    if v < T::ZERO || u + v > T::ONE {
        return None;
    }

    let t = dot(edge2, q) * inv_det;
    if t > epsilon { Some(t) } else { None } // else: line intersection, but behind the origin
}
