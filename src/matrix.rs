//! Matrix types.
//!
//! [`Matrix<V, C>`] has `C` columns, each a vector `V` (so `V`'s component count is
//! the row count): `float4x4 = Matrix<float4, 4>`, `float3x4 = Matrix<float3, 4>`
//! (3 rows, 4 columns — Slang's `floatRxC` reading). Storage is **column-major**:
//! an array of `C` column vectors. That choice means:
//!
//! * a matrix uploads to a Vulkan UBO/SSBO with a plain memcpy (SPIR-V's default
//!   matrix layout is column-major) — no transpose, and
//! * `M * v` is a linear combination of the columns, so it falls straight onto
//!   the `float4` SIMD operators with no extra code.
//!
//! Math convention is column-vector: `v' = M * v`, and transforms compose
//! right-to-left, `M = P * V * Model`.
//!
//! Indexing:
//! * `m[c]` -> column `c`, a vector (GLM-like, **not** HLSL's row)
//! * `m[(r, c)]` -> the element at row `r`, column `c`
#![allow(non_camel_case_types)]

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::scalar::{Componentwise, Dim, Float, Scalar, Signed, Vector};
use crate::vector::{Vector2, Vector3, Vector4};

/// A column-major matrix with `C` columns of type `V`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Matrix<V, const C: usize> {
    /// `cols[c]` is the c-th column.
    pub cols: [V; C],
}

// ---- generic API ----------------------------------------------------------------

impl<V: Vector, const C: usize> Matrix<V, C> {
    /// All zeros.
    pub const ZERO: Self = Self { cols: [V::ZERO; C] };

    /// Build from columns.
    #[inline]
    pub const fn from_cols(cols: [V; C]) -> Self {
        Self { cols }
    }
    /// Column `c`.
    #[inline]
    pub fn col(&self, c: usize) -> V {
        self.cols[c]
    }
    /// Element at row `r`, column `c`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> V::Scalar {
        self.cols[c][r]
    }
    /// Set the element at row `r`, column `c`.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, value: V::Scalar) {
        self.cols[c][r] = value;
    }
    /// Apply `f` to every element.
    #[inline]
    pub fn map(self, mut f: impl FnMut(V::Scalar) -> V::Scalar) -> Self {
        Self {
            cols: self.cols.map(|c| Componentwise::map(c, &mut f)),
        }
    }
    /// All elements in column-major order as a contiguous slice: ready to memcpy
    /// into a UBO for 4-row matrices. NB: 2/3-row matrices pack their columns
    /// tightly here, but std140 (and, for 3-row, std430) pads each column to a
    /// bigger slot; route those through [`crate::layout`] instead of a raw copy.
    #[inline]
    #[allow(unsafe_code)]
    pub fn as_slice(&self) -> &[V::Scalar] {
        // SAFETY: `#[repr(C)]` array of `#[repr(C)]` vectors whose fields are all
        // `V::Scalar`, so the `C * V::N` scalars are contiguous from `cols[0].x`.
        unsafe { core::slice::from_raw_parts(self.cols.as_ptr() as *const V::Scalar, C * V::N) }
    }
    /// Pointer to the first element (column-major).
    #[inline]
    pub fn as_ptr(&self) -> *const V::Scalar {
        self.cols.as_ptr() as *const V::Scalar
    }
}

impl<V: Vector + Dim<C>, const C: usize> Matrix<V, C> {
    /// Identity (square only). The concrete sizes also have an `IDENTITY` const.
    #[inline]
    pub fn identity() -> Self {
        let mut m = Self::ZERO;
        for i in 0..C {
            m.cols[i][i] = V::Scalar::ONE;
        }
        m
    }
    /// Sum of the diagonal.
    #[inline]
    pub fn trace(&self) -> V::Scalar {
        let mut t = V::Scalar::ZERO;
        for i in 0..C {
            t += self.cols[i][i];
        }
        t
    }
}

impl<V: Vector, const C: usize> Default for Matrix<V, C> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<V, const C: usize> Index<usize> for Matrix<V, C> {
    type Output = V;
    /// `m[c]` -> column `c`.
    #[inline]
    fn index(&self, c: usize) -> &V {
        &self.cols[c]
    }
}
impl<V, const C: usize> IndexMut<usize> for Matrix<V, C> {
    #[inline]
    fn index_mut(&mut self, c: usize) -> &mut V {
        &mut self.cols[c]
    }
}
impl<V: Vector, const C: usize> Index<(usize, usize)> for Matrix<V, C> {
    type Output = V::Scalar;
    /// `m[(r, c)]` -> element at row `r`, column `c`.
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &V::Scalar {
        &self.cols[c][r]
    }
}
impl<V: Vector, const C: usize> IndexMut<(usize, usize)> for Matrix<V, C> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut V::Scalar {
        &mut self.cols[c][r]
    }
}

impl<V: Vector, const C: usize> From<[V; C]> for Matrix<V, C> {
    #[inline]
    fn from(cols: [V; C]) -> Self {
        Self { cols }
    }
}
impl<V: Vector, const C: usize> From<Matrix<V, C>> for [V; C] {
    #[inline]
    fn from(m: Matrix<V, C>) -> Self {
        m.cols
    }
}

// ---- linear algebra -------------------------------------------------------------

/// `M * v`: linear combination of the columns. For `float4x4 * float4` this is
/// `float4` scalar-multiply + add, i.e. the SIMD path, with no special-casing.
impl<VR, VK, const K: usize> Mul<VK> for Matrix<VR, K>
where
    VR: Vector,
    VK: Vector<Scalar = VR::Scalar> + Dim<K>,
{
    type Output = VR;
    #[inline]
    fn mul(self, v: VK) -> VR {
        let mut r = self.cols[0] * v[0];
        for k in 1..K {
            r += self.cols[k] * v[k];
        }
        r
    }
}

/// `M(RxK) * N(KxC) = (RxC)`. Each result column is `M * (column of N)`.
impl<VR, VK, const K: usize, const C: usize> Mul<Matrix<VK, C>> for Matrix<VR, K>
where
    VR: Vector,
    VK: Vector<Scalar = VR::Scalar> + Dim<K>,
{
    type Output = Matrix<VR, C>;
    #[inline]
    fn mul(self, n: Matrix<VK, C>) -> Matrix<VR, C> {
        Matrix {
            cols: n.cols.map(|col| self * col),
        }
    }
}

impl<V: Vector + Dim<C>, const C: usize> MulAssign for Matrix<V, C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<V: Vector, const C: usize> Add for Matrix<V, C> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        let mut out = self;
        for c in 0..C {
            out.cols[c] += o.cols[c];
        }
        out
    }
}
impl<V: Vector, const C: usize> Sub for Matrix<V, C> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        let mut out = self;
        for c in 0..C {
            out.cols[c] -= o.cols[c];
        }
        out
    }
}
impl<V: Vector, const C: usize> AddAssign for Matrix<V, C> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<V: Vector, const C: usize> SubAssign for Matrix<V, C> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

// Scalar multiply, per primitive (coherence forbids `Mul<V::Scalar>` alongside `Mul<VK>`).
macro_rules! matrix_scalar_ops {
    ($($s:ty),*) => {$(
        impl<V: Vector<Scalar = $s>, const C: usize> Mul<$s> for Matrix<V, C> {
            type Output = Self;
            #[inline]
            fn mul(self, s: $s) -> Self {
                Matrix { cols: self.cols.map(|c| c * s) }
            }
        }
        impl<V: Vector<Scalar = $s>, const C: usize> Mul<Matrix<V, C>> for $s {
            type Output = Matrix<V, C>;
            #[inline]
            fn mul(self, m: Matrix<V, C>) -> Matrix<V, C> {
                m * self
            }
        }
        impl<V: Vector<Scalar = $s>, const C: usize> MulAssign<$s> for Matrix<V, C> {
            #[inline]
            fn mul_assign(&mut self, s: $s) {
                *self = *self * s;
            }
        }
    )*};
}
matrix_scalar_ops!(f32, f64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Generic product: `mul(a, b)` is `a * b` for any pair with a `*` defined
/// (matrix·matrix, matrix·vector, quaternion·quaternion, ...). Mirrors Slang's `mul`.
#[inline]
pub fn mul<A: Mul<B>, B>(a: A, b: B) -> A::Output {
    a * b
}

// ---- per-shape API: transpose / rows / flat arrays -------------------------------

macro_rules! matrix_shape {
    ($VR:ident, $R:literal, $VC:ident, $C:literal, $RC:literal) => {
        impl<T: Scalar> Matrix<$VR<T>, $C> {
            /// Number of rows.
            pub const ROWS: usize = $R;
            /// Number of columns.
            pub const COLUMNS: usize = $C;

            /// Build from rows (transposes into column-major storage).
            #[inline]
            pub fn from_rows(rows: [$VC<T>; $R]) -> Self {
                Matrix::<$VC<T>, $R>::from_cols(rows).transpose()
            }
            /// Row `r`.
            #[inline]
            pub fn row(&self, r: usize) -> $VC<T> {
                let mut out = $VC::<T>::ZERO;
                for c in 0..$C {
                    out[c] = self.cols[c][r];
                }
                out
            }
            /// Set row `r`.
            #[inline]
            pub fn set_row(&mut self, r: usize, row: $VC<T>) {
                for c in 0..$C {
                    self.cols[c][r] = row[c];
                }
            }
            /// The transpose.
            #[inline]
            pub fn transpose(&self) -> Matrix<$VC<T>, $R> {
                let mut out = Matrix::<$VC<T>, $R>::ZERO;
                for c in 0..$C {
                    for r in 0..$R {
                        out.cols[r][c] = self.cols[c][r];
                    }
                }
                out
            }
            /// Build from a flat column-major array.
            #[inline]
            pub fn from_cols_array(a: [T; $RC]) -> Self {
                let mut m = Self::ZERO;
                for c in 0..$C {
                    for r in 0..$R {
                        m.cols[c][r] = a[c * $R + r];
                    }
                }
                m
            }
            /// All elements as a flat column-major array (upload-ready for 4-row matrices).
            #[inline]
            pub fn to_cols_array(&self) -> [T; $RC] {
                let mut a = [T::ZERO; $RC];
                for c in 0..$C {
                    for r in 0..$R {
                        a[c * $R + r] = self.cols[c][r];
                    }
                }
                a
            }
        }
    };
}
matrix_shape!(Vector2, 2, Vector2, 2, 4);
matrix_shape!(Vector2, 2, Vector3, 3, 6);
matrix_shape!(Vector2, 2, Vector4, 4, 8);
matrix_shape!(Vector3, 3, Vector2, 2, 6);
matrix_shape!(Vector3, 3, Vector3, 3, 9);
matrix_shape!(Vector3, 3, Vector4, 4, 12);
matrix_shape!(Vector4, 4, Vector2, 2, 8);
matrix_shape!(Vector4, 4, Vector3, 3, 12);
matrix_shape!(Vector4, 4, Vector4, 4, 16);

// ---- square matrices: identity consts, determinant, inverse ----------------------

impl<T: Scalar> Matrix2x2<T> {
    /// The identity.
    pub const IDENTITY: Self = Self::from_cols([Vector2::X, Vector2::Y]);
}
impl<T: Scalar> Matrix3x3<T> {
    /// The identity.
    pub const IDENTITY: Self = Self::from_cols([Vector3::X, Vector3::Y, Vector3::Z]);
}
impl<T: Scalar> Matrix4x4<T> {
    /// The identity.
    pub const IDENTITY: Self = Self::from_cols([Vector4::X, Vector4::Y, Vector4::Z, Vector4::W]);
}

impl<T: Signed> Matrix2x2<T> {
    /// Determinant.
    #[inline]
    pub fn determinant(&self) -> T {
        let m = &self.cols;
        m[0][0] * m[1][1] - m[1][0] * m[0][1]
    }
}
impl<T: Float> Matrix2x2<T> {
    /// Inverse. Assumes the matrix is invertible; a singular matrix yields inf/NaN
    /// (as in GLM), not an error.
    #[inline]
    pub fn inverse(&self) -> Self {
        let m = &self.cols;
        let (a, c, b, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);
        let inv = T::ONE / (a * d - b * c);
        Self::from_cols([
            Vector2::new(d * inv, -c * inv),
            Vector2::new(-b * inv, a * inv),
        ])
    }
}

impl<T: Signed> Matrix3x3<T> {
    /// Determinant.
    #[inline]
    pub fn determinant(&self) -> T {
        let e = |r: usize, c: usize| self.cols[c][r];
        e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
            - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
            + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0))
    }
}
impl<T: Float> Matrix3x3<T> {
    /// Inverse. Assumes the matrix is invertible; a singular matrix yields inf/NaN.
    #[inline]
    pub fn inverse(&self) -> Self {
        let e = |r: usize, c: usize| self.cols[c][r];
        let (a, b, c) = (e(0, 0), e(0, 1), e(0, 2));
        let (d, f, g) = (e(1, 0), e(1, 1), e(1, 2));
        let (h, i, j) = (e(2, 0), e(2, 1), e(2, 2));
        let inv = T::ONE / self.determinant();
        // Columns of the result (adjugate / det).
        Self::from_cols([
            Vector3::new(
                (f * j - g * i) * inv,
                (g * h - d * j) * inv,
                (d * i - f * h) * inv,
            ),
            Vector3::new(
                (c * i - b * j) * inv,
                (a * j - c * h) * inv,
                (b * h - a * i) * inv,
            ),
            Vector3::new(
                (b * g - c * f) * inv,
                (c * d - a * g) * inv,
                (a * f - b * d) * inv,
            ),
        ])
    }
}

impl<T: Signed> Matrix4x4<T> {
    /// Determinant.
    pub fn determinant(&self) -> T {
        let m = self.to_cols_array(); // m[c*4 + r]
        let c0 = m[5] * (m[10] * m[15] - m[11] * m[14]) - m[9] * (m[6] * m[15] - m[7] * m[14])
            + m[13] * (m[6] * m[11] - m[7] * m[10]);
        let c1 = -m[4] * (m[10] * m[15] - m[11] * m[14]) + m[8] * (m[6] * m[15] - m[7] * m[14])
            - m[12] * (m[6] * m[11] - m[7] * m[10]);
        let c2 = m[4] * (m[9] * m[15] - m[11] * m[13]) - m[8] * (m[5] * m[15] - m[7] * m[13])
            + m[12] * (m[5] * m[11] - m[7] * m[9]);
        let c3 = -m[4] * (m[9] * m[14] - m[10] * m[13]) + m[8] * (m[5] * m[14] - m[6] * m[13])
            - m[12] * (m[5] * m[10] - m[6] * m[9]);
        m[0] * c0 + m[1] * c1 + m[2] * c2 + m[3] * c3
    }
}
impl<T: Float> Matrix4x4<T> {
    /// Inverse. Assumes the matrix is invertible; a singular matrix yields inf/NaN
    /// (as in GLM), not an error.
    #[rustfmt::skip]
    pub fn inverse(&self) -> Self {
        let m = self.to_cols_array(); // m[c*4 + r] == element at row r, column c
        let mut inv = [T::ZERO; 16];
        inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
        inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
        inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
        inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
        inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
        inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
        inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
        inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
        inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
        inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
        inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
        inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
        inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

        let det = T::ONE / (m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]);
        for v in inv.iter_mut() { *v *= det; }
        Self::from_cols_array(inv)
    }

    /// Normal matrix: `transpose(inverse(upper-left 3x3))` of a model/model-view
    /// matrix. Use it to transform normals when the model has non-uniform scale.
    #[inline]
    pub fn normal_matrix(&self) -> Matrix3x3<T> {
        self.upper_left().inverse().transpose()
    }

    /// The upper-left 3x3 (rotation/scale part).
    #[inline]
    pub fn upper_left(&self) -> Matrix3x3<T> {
        Matrix3x3::from_cols([self.cols[0].xyz(), self.cols[1].xyz(), self.cols[2].xyz()])
    }
}

/// Normal matrix: `transpose(inverse(upper-left 3x3))`. See [`Matrix4x4::normal_matrix`].
#[inline]
pub fn normal_matrix<T: Float>(m: &Matrix4x4<T>) -> Matrix3x3<T> {
    m.normal_matrix()
}

// ---- aliases ----------------------------------------------------------------------

/// 2 rows x 2 columns.
pub type Matrix2x2<T> = Matrix<Vector2<T>, 2>;
/// 2 rows x 3 columns.
pub type Matrix2x3<T> = Matrix<Vector2<T>, 3>;
/// 2 rows x 4 columns.
pub type Matrix2x4<T> = Matrix<Vector2<T>, 4>;
/// 3 rows x 2 columns.
pub type Matrix3x2<T> = Matrix<Vector3<T>, 2>;
/// 3 rows x 3 columns.
pub type Matrix3x3<T> = Matrix<Vector3<T>, 3>;
/// 3 rows x 4 columns.
pub type Matrix3x4<T> = Matrix<Vector3<T>, 4>;
/// 4 rows x 2 columns.
pub type Matrix4x2<T> = Matrix<Vector4<T>, 2>;
/// 4 rows x 3 columns.
pub type Matrix4x3<T> = Matrix<Vector4<T>, 3>;
/// 4 rows x 4 columns.
pub type Matrix4x4<T> = Matrix<Vector4<T>, 4>;

// Slang/HLSL names: floatRxC = R rows x C cols.

/// `Matrix2x2<f32>`.
pub type float2x2 = Matrix2x2<f32>;
/// `Matrix2x3<f32>` (2 rows, 3 columns).
pub type float2x3 = Matrix2x3<f32>;
/// `Matrix2x4<f32>` (2 rows, 4 columns).
pub type float2x4 = Matrix2x4<f32>;
/// `Matrix3x2<f32>` (3 rows, 2 columns).
pub type float3x2 = Matrix3x2<f32>;
/// `Matrix3x3<f32>`.
pub type float3x3 = Matrix3x3<f32>;
/// `Matrix3x4<f32>` (3 rows, 4 columns).
pub type float3x4 = Matrix3x4<f32>;
/// `Matrix4x2<f32>` (4 rows, 2 columns).
pub type float4x2 = Matrix4x2<f32>;
/// `Matrix4x3<f32>` (4 rows, 3 columns).
pub type float4x3 = Matrix4x3<f32>;
/// `Matrix4x4<f32>`.
pub type float4x4 = Matrix4x4<f32>;
/// `Matrix2x2<f64>`.
pub type double2x2 = Matrix2x2<f64>;
/// `Matrix3x3<f64>`.
pub type double3x3 = Matrix3x3<f64>;
/// `Matrix4x4<f64>`.
pub type double4x4 = Matrix4x4<f64>;

// GLSL-style aliases. SQUARE matrices only: GLSL names non-square matrices
// columns-first (GLSL mat3x4 = 3 cols x 4 rows) — the opposite of Slang's
// rows-first floatRxC — so aliasing those would invite mistakes.

/// GLSL name for [`float2x2`].
pub type mat2 = float2x2;
/// GLSL name for [`float3x3`].
pub type mat3 = float3x3;
/// GLSL name for [`float4x4`].
pub type mat4 = float4x4;
/// GLSL name for [`double2x2`].
pub type dmat2 = double2x2;
/// GLSL name for [`double3x3`].
pub type dmat3 = double3x3;
/// GLSL name for [`double4x4`].
pub type dmat4 = double4x4;
