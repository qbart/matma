//! Quaternion.
//!
//! Stored as `(x, y, z, w)` with `w` the scalar part; identity is `(0, 0, 0, 1)`.
//! Used for rotations: composes without gimbal lock and converts cheaply to a
//! rotation matrix. Rotations are right-handed and act as `v' = q * v`
//! (column-vector convention, matching the matrices).
#![allow(non_camel_case_types)]

use core::ops::{Mul, MulAssign, Neg};

use crate::matrix::{Matrix3x3, Matrix4x4};
use crate::scalar::Float;
use crate::vector::{Vector3, Vector4};

/// A rotation quaternion `(x, y, z, w)`, `w` scalar.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion<T> {
    /// Vector part, x.
    pub x: T,
    /// Vector part, y.
    pub y: T,
    /// Vector part, z.
    pub z: T,
    /// Scalar part.
    pub w: T,
}

/// `Quaternion<f32>`.
pub type quat = Quaternion<f32>;
/// `Quaternion<f64>`.
pub type dquat = Quaternion<f64>;

impl<T: Copy> Quaternion<T> {
    /// Build from components; `w` is the scalar part.
    #[inline]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self { x, y, z, w }
    }
    /// Build from an `[x, y, z, w]` array.
    #[inline]
    pub const fn from_array(a: [T; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
    /// The components as `[x, y, z, w]`.
    #[inline]
    pub const fn to_array(self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }
    /// The vector part.
    #[inline]
    pub const fn xyz(self) -> Vector3<T> {
        Vector3::new(self.x, self.y, self.z)
    }
    /// The components as a `Vector4` `(x, y, z, w)`.
    #[inline]
    pub const fn to_vector4(self) -> Vector4<T> {
        Vector4::new(self.x, self.y, self.z, self.w)
    }
    /// Reinterpret a `Vector4` `(x, y, z, w)` as a quaternion.
    #[inline]
    pub const fn from_vector4(v: Vector4<T>) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
    /// The components as a contiguous slice.
    #[inline]
    #[allow(unsafe_code)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `#[repr(C)]` with four fields of type `T` -> contiguous from `x`.
        unsafe { core::slice::from_raw_parts(&self.x as *const T, 4) }
    }
    /// Pointer to the first component.
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        &self.x as *const T
    }
}

impl<T: Float> Default for Quaternion<T> {
    /// The identity rotation.
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<T: Float> Quaternion<T> {
    /// The identity rotation `(0, 0, 0, 1)`.
    pub const IDENTITY: Self = Self::new(T::ZERO, T::ZERO, T::ZERO, T::ONE);

    // ---- construction ----

    /// Rotation of `angle` radians about `axis` (need not be normalized, but a zero
    /// axis normalizes to NaN — guard upstream if that can happen).
    pub fn from_axis_angle(axis: Vector3<T>, angle: T) -> Self {
        let n = axis.normalize();
        let h = angle * T::from_f64(0.5);
        let s = h.sin();
        Self::new(n.x * s, n.y * s, n.z * s, h.cos())
    }

    /// Intrinsic Tait-Bryan angles (**radians**) -> quaternion. `euler` =
    /// (pitch about X, yaw about Y, roll about Z); the product is `qz * qy * qx`
    /// (X applied first).
    pub fn from_euler(euler: Vector3<T>) -> Self {
        let half = T::from_f64(0.5);
        let c = Vector3::new(
            (euler.x * half).cos(),
            (euler.y * half).cos(),
            (euler.z * half).cos(),
        );
        let s = Vector3::new(
            (euler.x * half).sin(),
            (euler.y * half).sin(),
            (euler.z * half).sin(),
        );
        Self::new(
            s.x * c.y * c.z - c.x * s.y * s.z,
            c.x * s.y * c.z + s.x * c.y * s.z,
            c.x * c.y * s.z - s.x * s.y * c.z,
            c.x * c.y * c.z + s.x * s.y * s.z,
        )
    }

    /// Like [`Self::from_euler`], but the angles are in **degrees**.
    #[inline]
    pub fn from_euler_deg(angles: Vector3<T>) -> Self {
        Self::from_euler(angles.map(crate::transform::rad))
    }

    /// Rotation matrix -> unit quaternion (inverse of [`Self::to_mat3`]). Shepperd's
    /// method: build from whichever diagonal term is largest, for numerical stability.
    /// The basis is assumed orthonormal.
    pub fn from_mat3(m: Matrix3x3<T>) -> Self {
        // Column-major: m[c][r] is the element at row r, column c.
        let m = &m.cols;
        let (m00, m11, m22) = (m[0][0], m[1][1], m[2][2]);
        let trace = m00 + m11 + m22;
        let two = T::from_f64(2.0);
        let quarter = T::from_f64(0.25);
        if trace > T::ZERO {
            let s = (trace + T::ONE).sqrt() * two; // 4*w
            Self::new(
                (m[1][2] - m[2][1]) / s,
                (m[2][0] - m[0][2]) / s,
                (m[0][1] - m[1][0]) / s,
                quarter * s,
            )
        } else if m00 > m11 && m00 > m22 {
            let s = (T::ONE + m00 - m11 - m22).sqrt() * two; // 4*x
            Self::new(
                quarter * s,
                (m[1][0] + m[0][1]) / s,
                (m[2][0] + m[0][2]) / s,
                (m[1][2] - m[2][1]) / s,
            )
        } else if m11 > m22 {
            let s = (T::ONE + m11 - m00 - m22).sqrt() * two; // 4*y
            Self::new(
                (m[1][0] + m[0][1]) / s,
                quarter * s,
                (m[2][1] + m[1][2]) / s,
                (m[2][0] - m[0][2]) / s,
            )
        } else {
            let s = (T::ONE + m22 - m00 - m11).sqrt() * two; // 4*z
            Self::new(
                (m[2][0] + m[0][2]) / s,
                (m[2][1] + m[1][2]) / s,
                quarter * s,
                (m[0][1] - m[1][0]) / s,
            )
        }
    }

    /// Rotation part of a 4x4 (upper-left 3x3) -> unit quaternion. Ignores
    /// translation, scale and the projective row; the basis is assumed orthonormal.
    #[inline]
    pub fn from_mat4(m: &Matrix4x4<T>) -> Self {
        Self::from_mat3(m.upper_left())
    }

    /// Shortest-arc rotation taking direction `from` onto direction `to` (neither
    /// need be unit). (Unity `Quaternion.FromToRotation`.)
    pub fn from_to_rotation(from: Vector3<T>, to: Vector3<T>) -> Self {
        let f = from.normalize();
        let t = to.normalize();
        let d = f.dot(t);
        let eps = T::from_f64(1e-6);
        if d >= T::ONE - eps {
            return Self::IDENTITY; // already aligned
        }
        if d <= -T::ONE + eps {
            // opposite: 180° about any perpendicular axis
            let mut axis = Vector3::RIGHT.cross(f);
            if axis.length2() < eps {
                axis = Vector3::UP.cross(f);
            }
            let axis = axis.normalize();
            return Self::new(axis.x, axis.y, axis.z, T::ZERO); // 180° -> w = cos(90°) = 0
        }
        let c = f.cross(t);
        let s = ((T::ONE + d) * T::from_f64(2.0)).sqrt();
        let inv = T::ONE / s;
        Self::new(c.x * inv, c.y * inv, c.z * inv, s * T::from_f64(0.5)).normalize()
    }

    /// Rotation whose forward (local `-Z`, [`Vector3::FORWARD`]) points along
    /// `forward`, with local up aligned as closely as possible to `up_hint`.
    /// Right-handed / Y-up. (Unity `Quaternion.LookRotation`; Unity is left-handed
    /// with `+Z` forward, so its result differs by that convention.)
    ///
    /// NB: a zero `forward`, or `forward` parallel to `up_hint`, yields NaN (the
    /// right vector degenerates) — pick a different `up_hint` for straight up/down.
    pub fn look_rotation(forward: Vector3<T>, up_hint: Vector3<T>) -> Self {
        let f = forward.normalize();
        let r = f.cross(up_hint).normalize(); // right
        let u = r.cross(f); // re-orthogonalized up
        // Columns map local +X,+Y,+Z to world. Local -Z is forward, so column 2
        // (image of local +Z) is -f.
        Self::from_mat3(Matrix3x3::from_cols([r, u, -f]))
    }

    // ---- queries ----

    /// `|q|^2`.
    #[inline]
    pub fn length2(self) -> T {
        self.dot(self)
    }
    /// `|q|`.
    #[inline]
    pub fn length(self) -> T {
        self.length2().sqrt()
    }
    /// Four-component dot product.
    #[inline]
    pub fn dot(self, o: Self) -> T {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
    /// Unit quaternion in the same direction.
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = T::ONE / self.length();
        Self::new(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
    }
    /// `(-x, -y, -z, w)`; equals the inverse for a unit quaternion.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }
    /// `conjugate / |q|^2`.
    #[inline]
    pub fn inverse(self) -> Self {
        let inv = T::ONE / self.length2();
        Self::new(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)
    }

    /// Angular difference to `other`, in **radians**, range `[0, pi]`. Both are
    /// assumed unit. (Unity `Quaternion.Angle`, but radians.) Measured from the
    /// relative rotation via `atan2`, which stays accurate near 0 and pi where
    /// `2 * acos(|dot|)` loses precision.
    pub fn angle_to(self, other: Self) -> T {
        let r = self.conjugate() * other; // relative rotation
        let im = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt(); // |sin(theta/2)|
        T::from_f64(2.0) * im.atan2(r.w.abs())
    }

    // ---- interpolation ----
    // Raw lerp does NOT preserve unit length, so it's rarely what you want for a
    // rotation on its own. Reach for nlerp (cheap; great for blending close
    // orientations) or slerp (constant angular velocity; correct for wide arcs).

    /// Component-wise linear interpolation (unnormalized).
    #[inline]
    pub fn lerp(self, b: Self, t: T) -> Self {
        Self::new(
            self.x + (b.x - self.x) * t,
            self.y + (b.y - self.y) * t,
            self.z + (b.z - self.z) * t,
            self.w + (b.w - self.w) * t,
        )
    }
    /// Normalized lerp. Picks the shortest arc (flips `b` if the dot is negative).
    #[inline]
    pub fn nlerp(self, mut b: Self, t: T) -> Self {
        if self.dot(b) < T::ZERO {
            b = -b;
        }
        self.lerp(b, t).normalize()
    }
    /// Spherical linear interpolation: constant angular velocity along the shortest
    /// arc. Falls back to nlerp when the inputs are nearly parallel (`sin θ -> 0`).
    pub fn slerp(self, mut b: Self, t: T) -> Self {
        let mut cos_theta = self.dot(b);
        if cos_theta < T::ZERO {
            // shortest path
            b = -b;
            cos_theta = -cos_theta;
        }
        if cos_theta > T::from_f64(0.9995) {
            return self.nlerp(b, t); // avoid division by ~0
        }
        let theta = cos_theta.acos();
        let inv_sin = T::ONE / theta.sin();
        let wa = ((T::ONE - t) * theta).sin() * inv_sin;
        let wb = (t * theta).sin() * inv_sin;
        Self::new(
            wa * self.x + wb * b.x,
            wa * self.y + wb * b.y,
            wa * self.z + wb * b.z,
            wa * self.w + wb * b.w,
        )
    }

    /// Rotate toward `to` by at most `max_radians_delta`; snaps to `to` once within
    /// reach. (Unity `Quaternion.RotateTowards`, but radians.)
    pub fn rotate_towards(self, to: Self, max_radians_delta: T) -> Self {
        let a = self.angle_to(to);
        if a < T::from_f64(1e-6) {
            return to;
        }
        let t = max_radians_delta / a;
        if t >= T::ONE { to } else { self.slerp(to, t) }
    }

    // ---- application / conversion ----

    /// Rotate a vector by this (unit) quaternion. Optimized, no matrix needed:
    /// `t = 2 * cross(q.xyz, v);  v' = v + q.w * t + cross(q.xyz, t)`.
    #[inline]
    pub fn rotate(self, v: Vector3<T>) -> Vector3<T> {
        let u = self.xyz();
        let t = u.cross(v) * T::from_f64(2.0);
        v + t * self.w + u.cross(t)
    }

    /// Unit quaternion -> 3x3 rotation matrix (column-major, `v' = R * v`).
    pub fn to_mat3(self) -> Matrix3x3<T> {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        let one = T::ONE;
        let two = T::from_f64(2.0);
        Matrix3x3::from_cols([
            Vector3::new(one - two * (yy + zz), two * (xy + wz), two * (xz - wy)), // column 0
            Vector3::new(two * (xy - wz), one - two * (xx + zz), two * (yz + wx)), // column 1
            Vector3::new(two * (xz + wy), two * (yz - wx), one - two * (xx + yy)), // column 2
        ])
    }

    /// Unit quaternion -> 4x4 (rotation in the upper-left 3x3).
    pub fn to_mat4(self) -> Matrix4x4<T> {
        let r = self.to_mat3();
        Matrix4x4::from_cols([
            r.cols[0].extend(T::ZERO),
            r.cols[1].extend(T::ZERO),
            r.cols[2].extend(T::ZERO),
            Vector4::W,
        ])
    }

    /// `true` if any component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }
}

/// Hamilton product: applying `a * b` rotates by `b`, then `a`.
impl<T: Float> Mul for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn mul(self, b: Self) -> Self {
        let a = self;
        Self::new(
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )
    }
}
impl<T: Float> MulAssign for Quaternion<T> {
    #[inline]
    fn mul_assign(&mut self, b: Self) {
        *self = *self * b;
    }
}

/// `q * v` rotates `v` by `q` (column-vector convention, `v' = q * v`).
impl<T: Float> Mul<Vector3<T>> for Quaternion<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, v: Vector3<T>) -> Vector3<T> {
        self.rotate(v)
    }
}
/// `v * q` rotates `v` by `inverse(q)`.
impl<T: Float> Mul<Quaternion<T>> for Vector3<T> {
    type Output = Vector3<T>;
    #[inline]
    fn mul(self, q: Quaternion<T>) -> Vector3<T> {
        q.inverse().rotate(self)
    }
}

/// Negate every component (the same rotation, other cover).
impl<T: Float> Neg for Quaternion<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl<T: Float> From<Quaternion<T>> for Matrix3x3<T> {
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        q.to_mat3()
    }
}
impl<T: Float> From<Quaternion<T>> for Matrix4x4<T> {
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        q.to_mat4()
    }
}
impl<T: Float> From<Matrix3x3<T>> for Quaternion<T> {
    #[inline]
    fn from(m: Matrix3x3<T>) -> Self {
        Self::from_mat3(m)
    }
}
impl<T: Copy> From<[T; 4]> for Quaternion<T> {
    #[inline]
    fn from(a: [T; 4]) -> Self {
        Self::from_array(a)
    }
}
impl<T: Copy> From<Quaternion<T>> for [T; 4] {
    #[inline]
    fn from(q: Quaternion<T>) -> Self {
        q.to_array()
    }
}

/// Normalized lerp along the shortest arc. See [`Quaternion::nlerp`].
#[inline]
pub fn nlerp<T: Float>(a: Quaternion<T>, b: Quaternion<T>, t: T) -> Quaternion<T> {
    a.nlerp(b, t)
}
/// Spherical linear interpolation along the shortest arc. See [`Quaternion::slerp`].
#[inline]
pub fn slerp<T: Float>(a: Quaternion<T>, b: Quaternion<T>, t: T) -> Quaternion<T> {
    a.slerp(b, t)
}
