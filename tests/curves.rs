//! Spline / arc-length / animation-curve tests.

mod util;

use matma::*;
use util::*;

fn pts() -> [float3; 4] {
    [
        float3::new(0.0, 0.0, 0.0),
        float3::new(1.0, 2.0, 0.0),
        float3::new(3.0, 2.0, 1.0),
        float3::new(4.0, 0.0, 1.0),
    ]
}

// ---- basis functions ------------------------------------------------------------

#[test]
fn hermite_hits_endpoints_and_tangents() {
    let (p0, m0, p1, m1) = (1.0f32, 3.0, 5.0, -2.0);
    check!(near(hermite(p0, m0, p1, m1, 0.0), p0));
    check!(near(hermite(p0, m0, p1, m1, 1.0), p1));
    check!(near(hermite_derivative(p0, m0, p1, m1, 0.0), m0));
    check!(near(hermite_derivative(p0, m0, p1, m1, 1.0), m1));
    // Works for vectors too.
    let [a, b, c, d] = pts();
    check!(near3(hermite(a, b, c, d, 0.0), a));
    check!(near3(hermite(a, b, c, d, 1.0), c));
}

#[test]
fn catmull_rom_passes_through_middle_points() {
    let [p0, p1, p2, p3] = pts();
    check!(near3(catmull_rom(p0, p1, p2, p3, 0.0), p1));
    check!(near3(catmull_rom(p0, p1, p2, p3, 1.0), p2));
    // Tangent at the ends is half the chord to the neighbours.
    check!(near3(
        catmull_rom_tangent(p0, p1, p2, p3, 0.0),
        (p2 - p0) * 0.5
    ));
    check!(near3(
        catmull_rom_tangent(p0, p1, p2, p3, 1.0),
        (p3 - p1) * 0.5
    ));
}

#[test]
fn tangents_match_finite_differences() {
    let [p0, p1, p2, p3] = pts();
    let h = 1e-3f32;
    for &t in &[0.1f32, 0.5, 0.9] {
        let fd =
            (catmull_rom(p0, p1, p2, p3, t + h) - catmull_rom(p0, p1, p2, p3, t - h)) / (2.0 * h);
        check!(near3_eps(catmull_rom_tangent(p0, p1, p2, p3, t), fd, 1e-2));
        let fd = (cardinal(p0, p1, p2, p3, t + h, 0.3, 0.5)
            - cardinal(p0, p1, p2, p3, t - h, 0.3, 0.5))
            / (2.0 * h);
        check!(near3_eps(
            cardinal_tangent(p0, p1, p2, p3, t, 0.3, 0.5),
            fd,
            1e-2
        ));
    }
}

#[test]
fn cardinal_reduces_to_catmull_rom_and_to_lines() {
    let [p0, p1, p2, p3] = pts();
    for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        // tension 0, uniform spacing == Catmull-Rom
        check!(near3(
            cardinal(p0, p1, p2, p3, t, 0.0, 0.0),
            catmull_rom(p0, p1, p2, p3, t)
        ));
        // tension 1 == straight segment (eased parameterization: zero tangents)
        check!(near3(
            cardinal(p0, p1, p2, p3, t, 1.0, 0.5),
            p1.lerp(p2, smoothstep(0.0, 1.0, t))
        ));
    }
    // Coincident knots don't produce NaN.
    let c = cardinal(p1, p1, p2, p2, 0.5, 0.0, 0.5);
    assert!(c.is_finite());
}

// ---- Spline ---------------------------------------------------------------------

#[test]
fn spline_interpolates_knots() {
    let s = Spline::from_points(pts());
    assert_eq!(s.len(), 4);
    assert_eq!(s.segment_count(), 3);
    for (i, &p) in pts().iter().enumerate() {
        check!(near3(s.evaluate(i as f32), p));
        assert_eq!(s[i], p);
    }
    // Out of range clamps.
    check!(near3(s.evaluate(-1.0), pts()[0]));
    check!(near3(s.evaluate(7.5), pts()[3]));
    check!(near3(s.evaluate(f32::NAN), pts()[0]));
    // Phantom knots mirror the end.
    check!(near3(s.knot(-1), pts()[0] * 2.0 - pts()[1]));
    check!(near3(s.knot(4), pts()[3] * 2.0 - pts()[2]));
    // Continuous across a knot.
    check!(near3_eps(s.evaluate(0.999), s.evaluate(1.001), 0.02));
}

#[test]
fn spline_degenerate_sizes() {
    let mut s = Spline::<float3>::new();
    assert!(s.is_empty());
    assert_eq!(s.evaluate(0.5), float3::ZERO);
    assert_eq!(s.tangent(0.5), float3::ZERO);
    assert!(s.sample(8).is_empty());
    s.add(float3::X);
    assert_eq!(s.evaluate(3.0), float3::X);
    assert_eq!(s.segment_count(), 0);
    s.add(float3::Y);
    check!(near3(s.evaluate(0.5), float3::new(0.5, 0.5, 0.0))); // a 2-knot spline is a line
}

#[test]
fn spline_sampling_and_tangent() {
    let s = Spline::from_points(pts());
    let samples = s.sample(10);
    assert_eq!(samples.len(), 3 * 10 + 1);
    check!(near3(samples[0], pts()[0]));
    check!(near3(samples[30], pts()[3]));
    check!(near3(samples[10], pts()[1]));
    // Tangent agrees with a finite difference of evaluate.
    let h = 1e-3;
    let fd = (s.evaluate(1.5 + h) - s.evaluate(1.5 - h)) / (2.0 * h);
    check!(near3_eps(s.tangent(1.5), fd, 1e-2));
    // tension 1 -> polyline; mid-segment speed is 1.5x the chord (smoothstep slope).
    let mut line = Spline::from_points(pts());
    line.tension = 1.0;
    check!(near3(line.evaluate(1.5), pts()[1].lerp(pts()[2], 0.5)));
    check!(near3(line.tangent(1.5), (pts()[2] - pts()[1]) * 1.5));
}

// ---- SplineArcLength -------------------------------------------------------------

#[test]
fn arc_length_of_a_straight_path() {
    // Collinear knots: the curve is a straight line regardless of tension/alpha,
    // so the length is exact and evaluate_uniform is linear in distance.
    let s = Spline::from_points([
        float3::ZERO,
        float3::new(1.0, 0.0, 0.0),
        float3::new(3.0, 0.0, 0.0),
        float3::new(6.0, 0.0, 0.0),
    ]);
    let arc = SplineArcLength::build(&s, 64);
    assert!(!arc.is_empty());
    check!(near_eps(arc.length(), 6.0, 1e-3));
    check!(near3_eps(
        arc.evaluate_uniform(&s, 0.5),
        float3::new(3.0, 0.0, 0.0),
        1e-2
    ));
    check!(near3_eps(
        arc.evaluate_by_distance(&s, 1.5),
        float3::new(1.5, 0.0, 0.0),
        1e-2
    ));
    // Clamping.
    check!(near3(arc.evaluate_by_distance(&s, -1.0), float3::ZERO));
    check!(near3(
        arc.evaluate_by_distance(&s, 100.0),
        float3::new(6.0, 0.0, 0.0)
    ));
    // distance_to_u is monotonic.
    let mut prev = -1.0;
    for i in 0..=20 {
        let u = arc.distance_to_u(6.0 * i as f32 / 20.0);
        assert!(u >= prev, "u not monotonic at {i}");
        prev = u;
    }
    assert_eq!(arc.distance_to_u(6.0), 3.0);
}

#[test]
fn arc_length_uniform_steps_are_even() {
    let s = Spline::from_points(pts());
    let arc = SplineArcLength::build(&s, 64);
    let n = 16;
    let step = arc.length() / n as f32;
    let mut prev = arc.evaluate_uniform(&s, 0.0);
    for i in 1..=n {
        let p = arc.evaluate_uniform(&s, i as f32 / n as f32);
        // Each step covers ~1/16 of the length along the curve; the chord is a bit
        // shorter than the arc, so allow some slack.
        let chord = length(p - prev);
        assert!(
            (chord - step).abs() < step * 0.15,
            "step {i}: chord {chord} vs {step}"
        );
        prev = p;
    }
}

#[test]
fn nearest_point_queries() {
    let s = Spline::from_points(pts());
    let arc = SplineArcLength::build(&s, 32);
    // A point on the curve is its own nearest point.
    let on = s.evaluate(1.37);
    check!(near_eps(arc.nearest_u(&s, on), 1.37, 1e-3));
    check!(near3_eps(arc.nearest_point(&s, on), on, 1e-3));
    // An off-curve query lands on a point no farther than any sample.
    let q = float3::new(2.0, 5.0, 0.5);
    let best = arc.nearest_point(&s, q);
    for p in s.sample(64) {
        assert!(length2(best - q) <= length2(p - q) + 1e-4);
    }
    // Beyond the ends clamps to the end knots.
    check!(near3(
        arc.nearest_point(&s, float3::new(-10.0, 0.0, 0.0)),
        pts()[0]
    ));
    // Empty table.
    let empty = SplineArcLength::<float3>::new();
    assert_eq!(empty.length(), 0.0);
    assert_eq!(empty.nearest_u(&s, q), 0.0);
    assert_eq!(empty.distance_to_u(1.0), 0.0);
}

// ---- AnimationCurve -------------------------------------------------------------

#[test]
fn presets() {
    let lin = AnimationCurve::<f32>::new();
    assert_eq!(lin.len(), 2);
    assert_eq!(lin.segments(), 1);
    assert_eq!(lin.duration(), 1.0);
    for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        check!(near(lin.evaluate(t), t));
    }
    // Held outside [0, 1].
    assert_eq!(lin.evaluate(-5.0), 0.0);
    assert_eq!(lin.evaluate(5.0), 1.0);

    assert_eq!(
        AnimationCurve::<f32>::from_preset(Preset::Zero).evaluate(0.5),
        0.0
    );
    assert_eq!(
        AnimationCurve::<f32>::from_preset(Preset::One).evaluate(0.5),
        1.0
    );

    let ease = AnimationCurve::<f32>::from_preset(Preset::EaseInOut);
    check!(near(ease.evaluate(0.5), 0.5));
    check!(near(ease.evaluate(0.25), smoothstep(0.0, 1.0, 0.25))); // flat tangents == smoothstep
    check!(near(ease.evaluate(0.25) + ease.evaluate(0.75), 1.0)); // symmetric

    let ease_in = AnimationCurve::<f32>::from_preset(Preset::EaseIn);
    assert!(ease_in.evaluate(0.25) < 0.25 && ease_in.evaluate(0.75) < 0.75);
    let ease_out = AnimationCurve::<f32>::from_preset(Preset::EaseOut);
    assert!(ease_out.evaluate(0.25) > 0.25 && ease_out.evaluate(0.75) > 0.75);
}

#[test]
fn add_remove_and_move_keys() {
    let mut c = AnimationCurve::<f32>::new();
    assert_eq!(c.add_key(0.5, 2.0), Some(1));
    assert_eq!(c.add_key(0.5, 3.0), None); // duplicate time ignored
    assert_eq!(c.add_key(7.0, 3.0), None); // clamps onto the end anchor -> ignored
    assert_eq!(c.len(), 3);
    assert_eq!(c[1], Key::new(0.5, 2.0, 0.0, 0.0));
    check!(near(c.evaluate(0.5), 2.0)); // passes through the key
    assert!(c.is_locked(1));
    assert!(c.has_in_tangent(1) && c.has_out_tangent(1));
    assert!(!c.has_in_tangent(0) && !c.has_out_tangent(2));

    // Time clamps between neighbours with the gap.
    c.set_key(1, 0.0, 1.0);
    check!(near(c[1].time, 0.001));
    c.set_key(1, 9.0, 1.0);
    check!(near(c[1].time, 0.999));
    assert_eq!(c[1].value, 1.0);
    // End anchors move between 0/1 and their neighbour.
    c.set_key(0, -1.0, 0.5);
    assert_eq!(c[0].time, 0.0);
    assert_eq!(c[0].value, 0.5);
    assert_eq!(c.start_time(), 0.0);

    // Anchors are never removed; middle keys are.
    c.remove_key(0);
    c.remove_key(2);
    assert_eq!(c.len(), 3);
    c.remove_key(1);
    assert_eq!(c.len(), 2);
    assert_eq!(c.keys().len(), 2);
}

#[test]
fn tangents_and_locks() {
    let mut c = AnimationCurve::<f32>::new();
    c.add_key(0.5, 0.5);

    // Dragging the out handle up-right sets both tangents while locked.
    c.set_out_tangent(1, 0.6, 0.7); // slope 2
    check!(near(c[1].out_tangent, 2.0));
    check!(near(c[1].in_tangent, 2.0));
    // Handle positions.
    check!(near2(c.anchor(1), float2::new(0.5, 0.5)));
    let out = c.out_tangent_handle(1);
    let dir = float2::new(1.0, 2.0).normalize();
    check!(near2(out, float2::new(0.5, 0.5) + dir * 0.12));
    check!(near2(
        c.in_tangent_handle(1),
        float2::new(0.5, 0.5) - dir * 0.12
    ));

    // Split: tangents move independently.
    c.toggle_tangent_split_join(1, Tangent::Out);
    assert!(!c.is_locked(1));
    c.set_in_tangent_value(1, -1.0);
    assert_eq!(c[1].in_tangent, -1.0);
    check!(near(c[1].out_tangent, 2.0));
    // Join with In dominant: out snaps to in.
    c.toggle_tangent_split_join(1, Tangent::In);
    assert!(c.is_locked(1));
    assert_eq!(c[1].out_tangent, -1.0);

    // Degenerate handle (dt == 0) and out-of-range indices are ignored.
    c.set_out_tangent(1, 0.5, 9.0);
    assert_eq!(c[1].out_tangent, -1.0);
    c.set_out_tangent(2, 0.9, 0.9); // last key has no out tangent
    c.set_in_tangent(0, 0.1, 0.1); // first key has no in tangent
    c.toggle_tangent_split_join(0, Tangent::Out);
    assert!(c.is_locked(0));

    // In linear mode a dragged tangent is clamped to twice the limit.
    c.set_out_tangent(0, 0.001, 1.0); // slope 1000
    check!(near(c[0].out_tangent, 50.0));
    c.set_linear(false);
    assert!(!c.is_linear());
    c.set_out_tangent(0, 0.001, 1.0);
    check!(near_eps(c[0].out_tangent, 1000.0, 0.1));
}

#[test]
fn linear_mode_turns_steep_tangents_into_steps() {
    let mut c = AnimationCurve::<f32>::new();
    c.add_key(0.5, 1.0);
    c.set_out_tangent_value(0, 25.0); // at the limit: hold the next value
    assert_eq!(c.evaluate(0.1), 1.0);
    assert_eq!(c.evaluate(0.49), 1.0);
    c.set_out_tangent_value(0, -25.0); // hold the previous value
    assert_eq!(c.evaluate(0.49), 0.0);
    // Second segment unaffected.
    check!(near(
        c.evaluate(0.75),
        hermite(1.0, 0.0, 1.0, 1.0 * 0.5, 0.5)
    ));
    // Without linear mode the same tangent is a real cubic.
    c.set_linear(false);
    let v = c.evaluate(0.25);
    assert!(v != 0.0 && v != 1.0);
    check!(near(v, hermite(0.0, -25.0 * 0.5, 1.0, 0.0, 0.5)));
}

#[test]
fn f64_curves() {
    let c = AnimationCurve::<f64>::from_preset(Preset::EaseInOut);
    assert!((c.evaluate(0.5) - 0.5).abs() < 1e-12);
    let s = Spline::from_points([double3::ZERO, double3::X, double3::Y]);
    let arc = SplineArcLength::build(&s, 16);
    assert!(arc.length() > 1.0);
}
