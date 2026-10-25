//! Affine transforms and Vulkan projections.
//!
//! Conventions:
//! * column-major matrices, column-vector math: `v' = M * v`
//! * view space is right-handed, camera looks down `-Z`
//! * **Vulkan** clip space: depth in `[0, 1]` (not GL's `[-1, 1]`) and Y points DOWN.
//!   The Y-down flip is baked into the projection builders (`proj[(1,1)] *= -1`),
//!   so a projection from here is upload-and-go for a default Vulkan viewport.
//!
//! Every builder is both a free function (`translate(t)`, as in vkm/GLM) and an
//! associated function on `Matrix4x4` (`Matrix4x4::from_translation(t)`).

use crate::common;
use crate::matrix::{Matrix3x3, Matrix4x4};
use crate::quaternion::Quaternion;
use crate::scalar::{Float, Scalar};
use crate::vector::{Vector3, Vector4};

/// Degrees -> radians.
#[inline]
pub fn rad<T: Float>(degrees: T) -> T {
    degrees * (T::PI / T::from_f64(180.0))
}
/// Radians -> degrees.
#[inline]
pub fn deg<T: Float>(radians: T) -> T {
    radians * (T::from_f64(180.0) / T::PI)
}

/// Euler angles in **degrees** -> quaternion; `angles` = (pitch about X, yaw about Y,
/// roll about Z). Convenience over [`Quaternion::from_euler`], which takes radians.
#[inline]
pub fn quat_angles<T: Float>(angles: Vector3<T>) -> Quaternion<T> {
    Quaternion::from_euler_deg(angles)
}

// ---- affine builders ----------------------------------------------------------

/// Translation matrix (translation lives in the last column, column-major).
#[inline]
pub fn translate<T: Scalar>(t: Vector3<T>) -> Matrix4x4<T> {
    Matrix4x4::from_translation(t)
}
/// Non-uniform scale matrix.
#[inline]
pub fn scale<T: Scalar>(s: Vector3<T>) -> Matrix4x4<T> {
    Matrix4x4::from_scale(s)
}
/// Rotation matrix of `angle` radians about `axis`.
#[inline]
pub fn rotate<T: Float>(angle: T, axis: Vector3<T>) -> Matrix4x4<T> {
    Quaternion::from_axis_angle(axis, angle).to_mat4()
}
/// Post-rotate `q` by `degrees` about `axis` (pass [`Vector3::UP`] for a yaw).
#[inline]
pub fn rotate_deg<T: Float>(q: Quaternion<T>, degrees: T, axis: Vector3<T>) -> Quaternion<T> {
    q * Quaternion::from_axis_angle(axis, rad(degrees))
}
/// `T * R * S`: scale, then rotate, then translate.
#[inline]
pub fn compose_trs<T: Float>(t: Vector3<T>, r: Quaternion<T>, s: Vector3<T>) -> Matrix4x4<T> {
    Matrix4x4::from_trs(t, r, s)
}

// ---- vector rotation helper ---------------------------------------------------

/// Rotate `current` toward `target`: its direction turns by at most
/// `max_radians_delta`, and its length moves toward `|target|` by at most
/// `max_magnitude_delta`. Both are clamped against overshoot. (Unity `RotateTowards`.)
pub fn rotate_towards<T: Float>(
    current: Vector3<T>,
    target: Vector3<T>,
    max_radians_delta: T,
    max_magnitude_delta: T,
) -> Vector3<T> {
    let eps = T::from_f64(1e-20);
    let len_c = current.length();
    let len_t = target.length();
    let new_len = common::move_towards(len_c, len_t, max_magnitude_delta); // step the magnitude

    // A (near-)zero endpoint has no direction to rotate; keep whatever axis exists.
    if len_c < eps || len_t < eps {
        let dir = if len_c >= eps {
            current / len_c
        } else if len_t >= eps {
            target / len_t
        } else {
            Vector3::ZERO
        };
        return dir * new_len;
    }

    let dc = current / len_c;
    let dt = target / len_t;
    let theta = common::clamp(dc.dot(dt), -T::ONE, T::ONE).acos(); // angle between dirs, [0, pi]
    if theta < eps {
        return dc * new_len; // already aligned
    }

    let turn = if max_radians_delta < theta {
        max_radians_delta
    } else {
        theta
    }; // don't overshoot
    let axis = dc.cross(dt);
    let axis_len = axis.length();
    let axis = if axis_len < eps {
        // antiparallel (theta ~ pi): cross is degenerate, pick any perpendicular axis.
        let r = if dc.x.abs() < T::from_f64(0.9) {
            Vector3::X
        } else {
            Vector3::Y
        };
        dc.cross(r).normalize()
    } else {
        axis / axis_len
    };
    Quaternion::from_axis_angle(axis, turn).rotate(dc) * new_len
}

// ---- view / projection --------------------------------------------------------

/// Right-handed look-at: camera at `eye` looking toward `center`, world up `up`.
#[inline]
pub fn look_at<T: Float>(eye: Vector3<T>, center: Vector3<T>, up: Vector3<T>) -> Matrix4x4<T> {
    Matrix4x4::look_at(eye, center, up)
}
/// Right-handed perspective for Vulkan (depth `[0, 1]`, Y down). `fovy` is the
/// vertical field of view in radians.
#[inline]
pub fn perspective<T: Float>(fovy: T, aspect: T, z_near: T, z_far: T) -> Matrix4x4<T> {
    Matrix4x4::perspective(fovy, aspect, z_near, z_far)
}
/// Right-handed orthographic for Vulkan (depth `[0, 1]`, Y down).
#[inline]
pub fn ortho<T: Float>(left: T, right: T, bottom: T, top: T, z_near: T, z_far: T) -> Matrix4x4<T> {
    Matrix4x4::ortho(left, right, bottom, top, z_near, z_far)
}

/// The parts of an affine/projective 4x4, as returned by [`decompose`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decomposed<T> {
    /// Per-axis scale (negative when the basis was left-handed).
    pub scale: Vector3<T>,
    /// Orientation.
    pub rotation: Quaternion<T>,
    /// Translation.
    pub translation: Vector3<T>,
    /// Shear: `(yz, xz, xy)`.
    pub skew: Vector3<T>,
    /// Perspective row; `(0, 0, 0, 1)` for an affine matrix.
    pub perspective: Vector4<T>,
}

/// Split an affine/projective 4x4 into translation, rotation, scale (+ skew and a
/// perspective row). Orientation is recovered via [`Quaternion::from_mat3`] once the
/// basis is made orthonormal (Gram-Schmidt). `None` on a degenerate matrix.
pub fn decompose<T: Float>(model: &Matrix4x4<T>) -> Option<Decomposed<T>> {
    model.decompose()
}

// ---- Matrix4x4 constructors ---------------------------------------------------

impl<T: Scalar> Matrix4x4<T> {
    /// Translation matrix (translation in the last column).
    #[inline]
    pub fn from_translation(t: Vector3<T>) -> Self {
        Self::from_cols([Vector4::X, Vector4::Y, Vector4::Z, t.extend(T::ONE)])
    }
    /// Non-uniform scale matrix.
    #[inline]
    pub fn from_scale(s: Vector3<T>) -> Self {
        Self::from_cols([
            Vector4::new(s.x, T::ZERO, T::ZERO, T::ZERO),
            Vector4::new(T::ZERO, s.y, T::ZERO, T::ZERO),
            Vector4::new(T::ZERO, T::ZERO, s.z, T::ZERO),
            Vector4::W,
        ])
    }
}

impl<T: Float> Matrix4x4<T> {
    /// Rotation matrix of `angle` radians about `axis`.
    #[inline]
    pub fn from_axis_angle(axis: Vector3<T>, angle: T) -> Self {
        Quaternion::from_axis_angle(axis, angle).to_mat4()
    }
    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(q: Quaternion<T>) -> Self {
        q.to_mat4()
    }
    /// `T * R * S`: scale, then rotate, then translate.
    #[inline]
    pub fn from_trs(t: Vector3<T>, r: Quaternion<T>, s: Vector3<T>) -> Self {
        Self::from_translation(t) * r.to_mat4() * Self::from_scale(s)
    }

    /// Right-handed look-at: camera at `eye` looking toward `center`, world up `up`.
    pub fn look_at(eye: Vector3<T>, center: Vector3<T>, up: Vector3<T>) -> Self {
        let f = (center - eye).normalize(); // forward
        let s = f.cross(up).normalize(); // right
        let u = s.cross(f); // recomputed up
        Self::from_cols([
            Vector4::new(s.x, u.x, -f.x, T::ZERO),
            Vector4::new(s.y, u.y, -f.y, T::ZERO),
            Vector4::new(s.z, u.z, -f.z, T::ZERO),
            Vector4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), T::ONE),
        ])
    }

    /// Right-handed perspective for Vulkan: depth `[0, 1]`, Y down. `fovy` is the
    /// vertical field of view in radians.
    pub fn perspective(fovy: T, aspect: T, z_near: T, z_far: T) -> Self {
        let t = (fovy * T::from_f64(0.5)).tan();
        let mut m = Self::ZERO;
        m[(0, 0)] = T::ONE / (aspect * t);
        m[(1, 1)] = -T::ONE / t; // Y-flip for Vulkan
        m[(2, 2)] = z_far / (z_near - z_far);
        m[(3, 2)] = -T::ONE;
        m[(2, 3)] = (z_far * z_near) / (z_near - z_far);
        m
    }

    /// Right-handed orthographic for Vulkan: depth `[0, 1]`, Y down.
    pub fn ortho(left: T, right: T, bottom: T, top: T, z_near: T, z_far: T) -> Self {
        let two = T::from_f64(2.0);
        let mut m = Self::ZERO;
        m[(0, 0)] = two / (right - left);
        m[(1, 1)] = -two / (top - bottom); // Y-flip for Vulkan
        m[(2, 2)] = -T::ONE / (z_far - z_near);
        m[(0, 3)] = -(right + left) / (right - left);
        m[(1, 3)] = (top + bottom) / (top - bottom); // Y-flip: row negated
        m[(2, 3)] = -z_near / (z_far - z_near);
        m[(3, 3)] = T::ONE;
        m
    }

    /// Split into translation, rotation, scale (+ skew and a perspective row).
    /// `None` on a degenerate matrix. See [`decompose`].
    pub fn decompose(&self) -> Option<Decomposed<T>> {
        let eps = T::from_f64(1e-6);
        let mut local = *self;

        let w = local[(3, 3)];
        if w.abs() < eps {
            return None;
        }
        for c in 0..4 {
            local.cols[c] /= w;
        }

        // Singularity test on the upper-left 3x3 (perspective partition cleared).
        let mut persp = local;
        for c in 0..3 {
            persp.cols[c].w = T::ZERO;
        }
        persp.cols[3].w = T::ONE;
        if persp.determinant().abs() < eps {
            return None;
        }

        // Isolate the perspective row (last row, column-major).
        let perspective = if local.cols[0].w.abs() > eps
            || local.cols[1].w.abs() > eps
            || local.cols[2].w.abs() > eps
        {
            let rhs = Vector4::new(
                local.cols[0].w,
                local.cols[1].w,
                local.cols[2].w,
                local.cols[3].w,
            );
            let p = persp.inverse().transpose() * rhs;
            local.cols[0].w = T::ZERO;
            local.cols[1].w = T::ZERO;
            local.cols[2].w = T::ZERO;
            local.cols[3].w = T::ONE;
            p
        } else {
            Vector4::W
        };

        let translation = local.cols[3].xyz();
        local.cols[3] = Vector4::new(T::ZERO, T::ZERO, T::ZERO, local.cols[3].w);

        let mut basis = [
            local.cols[0].xyz(),
            local.cols[1].xyz(),
            local.cols[2].xyz(),
        ];
        let mut scale = Vector3::ZERO;
        let mut skew = Vector3::ZERO;

        scale.x = basis[0].length();
        basis[0] = basis[0].normalize();
        skew.z = basis[0].dot(basis[1]);
        basis[1] -= basis[0] * skew.z;

        scale.y = basis[1].length();
        basis[1] = basis[1].normalize();
        skew.z /= scale.y;

        skew.y = basis[0].dot(basis[2]);
        basis[2] -= basis[0] * skew.y;
        skew.x = basis[1].dot(basis[2]);
        basis[2] -= basis[1] * skew.x;

        scale.z = basis[2].length();
        basis[2] = basis[2].normalize();
        skew.y /= scale.z;
        skew.x /= scale.z;

        // Coordinate-system flip: if the basis is left-handed, negate it and the scale.
        if basis[0].dot(basis[1].cross(basis[2])) < T::ZERO {
            scale = -scale;
            for b in basis.iter_mut() {
                *b = -*b;
            }
        }

        let rotation = Quaternion::from_mat3(Matrix3x3::from_cols(basis));
        Some(Decomposed {
            scale,
            rotation,
            translation,
            skew,
            perspective,
        })
    }
}
