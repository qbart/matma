//! matma — sample usage. Run with `cargo run --example usage`.

use matma::*;

fn print(label: &str, v: impl core::fmt::Debug) {
    println!("  {label:<28} {v:?}");
}

fn main() {
    println!("[simd backend: {}]", simd::BACKEND);

    // ---- 1. Vectors -----------------------------------------------------------
    println!("[vectors]");
    let a = float3::new(1.0, 2.0, 3.0);
    let b = float3::new(4.0, 5.0, 6.0);
    print("a + b =", a + b); // component-wise (SIMD for float4; scalar here)
    print("a * 2 =", a * 2.0); // vector * scalar
    print("cross(a, b) =", cross(a, b));
    print("dot(a, b) =", dot(a, b));
    print("length(a)   (magnitude)", length(a));
    print("length2(a)  (mag^2)", a.length2()); // no sqrt — cheaper for comparisons
    print("normalize(a) =", a.normalize());
    print("saturate(a - 2) =", saturate(a - 2.0)); // same fn for scalars and vectors
    print("a.xy().yx() =", a.xy().yx()); // swizzles

    // ---- 2. Rotate a vector by a 30-degree quaternion about +Z --------------
    println!("\n[quaternion rotation]");
    let q = quat::from_axis_angle(float3::Z, rad(30.0)); // axis, angle (radians)
    let v = float3::X;
    print("q * v        [fast path]", q * v); // expect (cos30, sin30, 0)
    // The same rotation applied via the matrix form (w=0 -> direction, not point):
    print("to_mat4(q) * (v, 0)", (q.to_mat4() * v.extend(0.0)).xyz());
    print("slerp(id, q, 0.5) * v", quat::IDENTITY.slerp(q, 0.5) * v); // 15 degrees

    // ---- 3. MVP matrix multiplication ---------------------------------------
    println!("\n[MVP]");
    // model: scale 1, rotate q (30 deg / Z), translate to (0,1,0)  -> T * R * S
    let model = compose_trs(float3::new(0.0, 1.0, 0.0), q, float3::ONE);
    let view = look_at(float3::new(0.0, 0.0, 5.0), float3::ZERO, float3::UP);
    let proj = perspective(rad(60.0), 16.0 / 9.0, 0.1, 100.0); // Vulkan clip

    let mvp = proj * view * model; // compose right-to-left, then M * v below

    let model_point = float4::new(0.0, 0.0, 0.0, 1.0); // a point in model space (w = 1)
    let clip = mvp * model_point; // -> clip space
    let ndc = clip.xyz() / clip.w; // perspective divide -> NDC, depth in [0, 1]
    print("clip = MVP * point", clip);
    print("ndc  = clip.xyz / clip.w", ndc);
    print("model translation (col 3)", model[3].xyz());
    print("decompose(model).scale", model.decompose().unwrap().scale);

    // ---- 4. GPU interop -------------------------------------------------------
    println!("\n[gpu]");
    print("mvp.as_slice()[12..16]", &mvp.as_slice()[12..16]); // column-major, upload-ready
    print(
        "oct_encode16(normal)",
        oct_encode16(float3::new(0.0, 0.6, 0.8)),
    ); // R16G16_SNORM
    print(
        "pack_half2x16(uv)",
        format_args!("{:#010x}", pack_half2x16(float2::new(0.25, 0.75))),
    );
    print("std140 align of float3", std140_align::<float3>()); // the classic 16
    print(
        "size of Std140<float3x3>",
        core::mem::size_of::<Std140<float3x3>>(),
    ); // 48
}
