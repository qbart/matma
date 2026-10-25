//! matma — curves sample. Run with `cargo run --example curves`.

use matma::*;

fn print(label: &str, v: impl core::fmt::Debug) {
    println!("  {label:<34} {v:?}");
}

fn main() {
    // ---- 1. AnimationCurve: a 0..1 easing curve with keyframes -----------------
    println!("[animation curve]");
    let mut fade = AnimationCurve::<f32>::from_preset(Preset::EaseInOut);
    print("ease_in_out(0.25)", fade.evaluate(0.25)); // 0.15625 (smoothstep)
    print("ease_in_out(0.5)", fade.evaluate(0.5)); // 0.5

    // Overshoot: add a key above 1 in the middle, then flatten its tangents.
    let i = fade.add_key(0.6, 1.2).unwrap();
    fade.set_out_tangent_value(i, 0.0); // locked by default -> in tangent follows
    print("after add_key(0.6, 1.2)", fade.evaluate(0.6)); // 1.2, passes through the key
    print("keys", fade.keys().len()); // 3
    print("evaluate(0.3)", fade.evaluate(0.3));
    print("evaluate(2.0) holds last value", fade.evaluate(2.0)); // 1.0

    // Editor-style tangent drag: handle moved to (0.7, 1.5) -> slope 3.
    fade.set_out_tangent(i, 0.7, 1.5);
    print("key[1].out_tangent", fade[i].out_tangent);
    print("out_tangent_handle(1)", fade.out_tangent_handle(i)); // where to draw it
    // A vertical-ish tangent becomes a step in linear mode.
    fade.set_out_tangent_value(0, AnimationCurve::<f32>::TANGENT_LIMIT as f32);
    print("step: evaluate(0.3)", fade.evaluate(0.3)); // 1.2 (holds the next key)

    // Drive something with it: alpha over a 2-second fade.
    let elapsed = 0.5f32;
    let alpha = fade.evaluate(elapsed / 2.0);
    print("alpha at t=0.5s of 2s", alpha);

    // ---- 2. Spline: a camera path through waypoints ---------------------------
    println!("\n[spline]");
    let mut path = Spline::<float3>::new(); // tension 0 (Catmull-Rom), alpha 0.5 (centripetal)
    path.add(float3::new(0.0, 1.0, 5.0));
    path.add(float3::new(3.0, 2.0, 2.0));
    path.add(float3::new(3.0, 1.5, -2.0));
    path.add(float3::new(-1.0, 1.0, -4.0));
    print("segments", path.segment_count()); // 3; u runs over 0..=3
    print("evaluate(1.0) == knot 1", path.evaluate(1.0));
    print("evaluate(1.5) mid-segment", path.evaluate(1.5));
    let dir = path.tangent(1.5).normalize();
    print("direction at 1.5", dir);
    let polyline = path.sample(8); // 3 * 8 + 1 points for drawing
    print("sample(8).len()", polyline.len());

    // Build a camera from a point on the path looking along it.
    let eye = path.evaluate(1.5);
    let view = look_at(eye, eye + dir, float3::UP);
    print("view translation col", view[3].xyz());

    // A straighter variant of the same path.
    let mut tense = path.clone();
    tense.tension = 0.7;
    print("tension 0.7 evaluate(1.5)", tense.evaluate(1.5));

    // ---- 3. SplineArcLength: constant speed and even spacing -----------------
    println!("\n[spline arc length]");
    let arc = SplineArcLength::build(&path, 32);
    print("length()", arc.length());

    // Move at 2 units/second along the path, regardless of knot spacing.
    let speed = 2.0f32;
    for t in [0.0f32, 1.0, 2.0] {
        let dist = speed * t;
        let u = arc.distance_to_u(dist);
        let p = arc.evaluate_by_distance(&path, dist);
        print(&format!("t={t}s  dist={dist}  u={u:.3}"), p);
    }

    // Evenly spaced markers along the curve (compare with sample(), which is even in u).
    let markers: Vec<float3> = (0..=4)
        .map(|i| arc.evaluate_uniform(&path, i as f32 / 4.0))
        .collect();
    print("4 even markers", markers);

    // Closest point on the path to the player, and how far along it is.
    let player = float3::new(2.0, 1.0, 0.0);
    let u = arc.nearest_u(&path, player);
    let closest = arc.nearest_point(&path, player);
    print("nearest_u(player)", u);
    print("nearest_point(player)", closest);
    print("distance to path", length(closest - player));

    // Edit the path -> rebuild the table (it holds no reference to the spline).
    path.points_mut()[3] = float3::new(-3.0, 1.0, -6.0);
    let arc = SplineArcLength::build(&path, 32);
    print("length() after edit", arc.length());
}
