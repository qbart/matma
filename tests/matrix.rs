//! Matrix smoke tests.
//!
//! Pins down the column-major + M*v convention: a translation lives in the last
//! COLUMN, M*v is a linear combination of columns, and composition is
//! right-to-left.

mod util;
use matma::*;
use util::*;

// Column-major translation: translation components sit in the last column.
fn translation(x: f32, y: f32, z: f32) -> float4x4 {
    float4x4::from_cols([float4::X, float4::Y, float4::Z, float4::new(x, y, z, 1.0)])
}
fn scaling(x: f32, y: f32, z: f32) -> float4x4 {
    float4x4::from_cols([
        float4::new(x, 0.0, 0.0, 0.0),
        float4::new(0.0, y, 0.0, 0.0),
        float4::new(0.0, 0.0, z, 0.0),
        float4::W,
    ])
}

#[test]
fn matrix_vector_product() {
    assert_eq!(
        float4x4::IDENTITY * float4::new(1.0, 2.0, 3.0, 4.0),
        float4::new(1.0, 2.0, 3.0, 4.0)
    );
    assert_eq!(
        translation(10.0, 20.0, 30.0) * float4::new(1.0, 2.0, 3.0, 1.0),
        float4::new(11.0, 22.0, 33.0, 1.0)
    );
    assert_eq!(
        scaling(2.0, 3.0, 4.0) * float4::ONE,
        float4::new(2.0, 3.0, 4.0, 1.0)
    );
    assert_eq!(
        mul(scaling(2.0, 3.0, 4.0), float4::ONE),
        float4::new(2.0, 3.0, 4.0, 1.0)
    );
    // Right-to-left composition: scale THEN translate == (T * S).
    assert_eq!(
        (translation(10.0, 0.0, 0.0) * scaling(2.0, 1.0, 1.0)) * float4::new(1.0, 0.0, 0.0, 1.0),
        float4::new(12.0, 0.0, 0.0, 1.0)
    );
    // Element access: translation x sits at row 0, last column.
    assert_eq!(translation(7.0, 8.0, 9.0)[(0, 3)], 7.0);
    assert_eq!(translation(7.0, 8.0, 9.0).get(0, 3), 7.0);
    assert_eq!(
        translation(7.0, 8.0, 9.0)[3],
        float4::new(7.0, 8.0, 9.0, 1.0)
    ); // m[c] is a column
}

#[test]
fn composition() {
    let t = translation(10.0, 20.0, 30.0);
    let s = scaling(2.0, 3.0, 4.0);

    let ts = t * s;
    let p = float4::ONE;
    assert_eq!(ts * p, t * (s * p));
    assert_eq!(ts * p, float4::new(12.0, 23.0, 34.0, 1.0));

    // transpose moves the translation row -> column.
    let tt = t.transpose();
    assert_eq!(tt[(0, 3)], 0.0);
    assert_eq!(tt[(3, 0)], 10.0);

    // Identity is a left/right multiply no-op.
    assert_eq!(float4x4::IDENTITY * t, t);
    assert_eq!(t * float4x4::IDENTITY, t);

    // Addition / subtraction / scalar multiply.
    assert_eq!(t - t, float4x4::ZERO);
    assert_eq!(t + float4x4::ZERO, t);
    assert_eq!((float4x4::IDENTITY * 3.0)[(2, 2)], 3.0);
    assert_eq!((3.0 * float4x4::IDENTITY)[(2, 2)], 3.0);
}

#[test]
fn non_square_products() {
    // (2x3) * (3x2) = (2x2), (3x2) * (2x3) = (3x3): shapes are checked at compile time.
    let a = float2x3::from_cols([
        float2::new(1.0, 2.0),
        float2::new(3.0, 4.0),
        float2::new(5.0, 6.0),
    ]);
    let b = float3x2::from_cols([float3::new(1.0, 0.0, 1.0), float3::new(0.0, 1.0, 1.0)]);
    let ab: float2x2 = a * b;
    assert_eq!(
        ab,
        float2x2::from_cols([float2::new(6.0, 8.0), float2::new(8.0, 10.0)])
    );
    let ba: float3x3 = b * a;
    assert_eq!(ba.row(2), float3::new(3.0, 7.0, 11.0));
    // (2x3) * float3 = float2
    assert_eq!(a * float3::new(1.0, 1.0, 1.0), float2::new(9.0, 12.0));
    // Integer and f64 matrices work too.
    let im = Matrix2x2::<i32>::from_cols([int2::new(1, 2), int2::new(3, 4)]);
    assert_eq!(im * int2::new(1, 1), int2::new(4, 6));
    assert_eq!(im.determinant(), -2);
    let dm = dmat4::IDENTITY * 2.0;
    assert_eq!(dm.determinant(), 16.0);
}

#[test]
fn from_rows_and_map() {
    let m = float3x3::from_rows([
        float3::new(1.0, 2.0, 3.0),
        float3::new(4.0, 5.0, 6.0),
        float3::new(7.0, 8.0, 9.0),
    ]);
    assert_eq!(m.cols[0], float3::new(1.0, 4.0, 7.0));
    assert_eq!(m[(1, 2)], 6.0);
    assert_eq!(m.map(|x| x * 2.0)[(2, 2)], 18.0);
    let mut m2 = m;
    m2.set(0, 0, 100.0);
    m2.set_row(2, float3::ZERO);
    assert_eq!(m2[(0, 0)], 100.0);
    assert_eq!(m2.row(2), float3::ZERO);
    check!(near3(m.row(1), float3::new(4.0, 5.0, 6.0)));
}
