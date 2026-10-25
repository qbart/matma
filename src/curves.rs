//! Curves: cubic Hermite / Catmull-Rom / cardinal splines, arc-length
//! reparameterization, and keyframed animation curves.
//!
//! * [`hermite`] / [`hermite_derivative`]: the cubic Hermite basis, for scalars
//!   and vectors alike.
//! * [`catmull_rom`] / [`catmull_rom_tangent`]: uniform Catmull-Rom through four
//!   control points (segment `p1 -> p2`).
//! * [`cardinal`] / [`cardinal_tangent`]: Catmull-Rom generalised with `tension`
//!   and non-uniform knot spacing `alpha` (0 uniform, 0.5 centripetal, 1 chordal).
//! * [`Spline`]: a list of knots evaluated by a segment parameter `u` in
//!   `[0, segments]`, with phantom end knots so the curve passes through every
//!   point including the first and last.
//! * [`SplineArcLength`]: a distance -> `u` table for constant-speed travel,
//!   evenly spaced samples, and closest-point queries.
//! * [`AnimationCurve`]: Unity-style keyframes on `[0, 1]` with in/out tangents,
//!   presets (linear, ease in/out, ...) and the editor helpers (locked tangents,
//!   tangent handles) that go with them.
//!
//! Everything is generic over the vector type (`float3`, `float2`, `double3`, or
//! a plain `f32` for 1-D curves) and uses `Vec`, so this module needs `alloc`.
//!
//! ```
//! use matma::*;
//!
//! let mut path = Spline::<float3>::new();
//! for p in [float3::ZERO, float3::X, float3::new(1.0, 0.0, 1.0), float3::Z] {
//!     path.add(p);
//! }
//! assert_eq!(path.evaluate(1.0), float3::X); // integer u lands on a knot
//!
//! let arc = SplineArcLength::build(&path, 32);
//! let halfway = arc.evaluate_uniform(&path, 0.5); // by distance, not by u
//! let dir = path.tangent(arc.distance_to_u(arc.length() * 0.5)).normalize();
//! assert!(halfway.is_finite() && dir.is_finite());
//!
//! let ease = AnimationCurve::<f32>::from_preset(Preset::EaseInOut);
//! assert_eq!(ease.evaluate(0.5), 0.5);
//! assert!(ease.evaluate(0.25) < 0.25);
//! ```

use core::ops::{Add, Index, Mul};

use crate::common::{clamp, dot, length, length2};
use crate::scalar::{Componentwise, Float, Scalar, Vector};
use crate::vector::Vector2;

// ---- basis functions ----------------------------------------------------------

/// Cubic Hermite interpolation: `p0` at `t = 0` with tangent `m0`, `p1` at `t = 1`
/// with tangent `m1`. Tangents are per unit of `t`; scale them by the segment's
/// duration when `t` was normalised from a longer interval.
#[inline]
pub fn hermite<V>(p0: V, m0: V, p1: V, m1: V, t: V::Scalar) -> V
where
    V: Componentwise + Mul<<V as Componentwise>::Scalar, Output = V> + Add<Output = V>,
    V::Scalar: Float,
{
    let two = V::Scalar::from_f64(2.0);
    let three = V::Scalar::from_f64(3.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = two * t3 - three * t2 + V::Scalar::ONE;
    let h10 = t3 - two * t2 + t;
    let h01 = three * t2 - two * t3;
    let h11 = t3 - t2;
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Derivative of [`hermite`] with respect to `t` (velocity along the curve).
#[inline]
pub fn hermite_derivative<V>(p0: V, m0: V, p1: V, m1: V, t: V::Scalar) -> V
where
    V: Componentwise + Mul<<V as Componentwise>::Scalar, Output = V> + Add<Output = V>,
    V::Scalar: Float,
{
    let two = V::Scalar::from_f64(2.0);
    let three = V::Scalar::from_f64(3.0);
    let four = V::Scalar::from_f64(4.0);
    let six = V::Scalar::from_f64(6.0);
    let t2 = t * t;
    let h00 = six * t2 - six * t;
    let h10 = three * t2 - four * t + V::Scalar::ONE;
    let h01 = six * t - six * t2;
    let h11 = three * t2 - two * t;
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Uniform Catmull-Rom: the segment from `p1` (`t = 0`) to `p2` (`t = 1`), with
/// `p0` and `p3` shaping the tangents.
///
/// `p(t) = 0.5 * (2 p1 + (p2 - p0) t + (2 p0 - 5 p1 + 4 p2 - p3) t² + (3 p1 - p0 - 3 p2 + p3) t³)`
#[inline]
pub fn catmull_rom<V: Vector>(p0: V, p1: V, p2: V, p3: V, t: V::Scalar) -> V
where
    V::Scalar: Float,
{
    let half = V::Scalar::from_f64(0.5);
    let two = V::Scalar::from_f64(2.0);
    let three = V::Scalar::from_f64(3.0);
    let four = V::Scalar::from_f64(4.0);
    let five = V::Scalar::from_f64(5.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let c0 = p1 * two;
    let c1 = p2 - p0;
    let c2 = p0 * two - p1 * five + p2 * four - p3;
    let c3 = p1 * three - p0 - p2 * three + p3;
    (c0 + c1 * t + c2 * t2 + c3 * t3) * half
}

/// Derivative of [`catmull_rom`] with respect to `t`: the un-normalised velocity
/// along the segment. Normalise it for a direction.
#[inline]
pub fn catmull_rom_tangent<V: Vector>(p0: V, p1: V, p2: V, p3: V, t: V::Scalar) -> V
where
    V::Scalar: Float,
{
    let half = V::Scalar::from_f64(0.5);
    let two = V::Scalar::from_f64(2.0);
    let three = V::Scalar::from_f64(3.0);
    let four = V::Scalar::from_f64(4.0);
    let five = V::Scalar::from_f64(5.0);
    let t2 = t * t;
    let c1 = p2 - p0;
    let c2 = p0 * two - p1 * five + p2 * four - p3;
    let c3 = p1 * three - p0 - p2 * three + p3;
    (c1 + c2 * (two * t) + c3 * (three * t2)) * half
}

/// Non-uniform Catmull-Rom tangents at `p1` and `p2`, scaled to the local `[0, 1]`
/// segment parameter.
///
/// `alpha` parameterises knot spacing: 0 uniform, 0.5 centripetal (cusp-free), 1
/// chordal. `tension` scales the tangents: 0 full Catmull-Rom, 1 zero (straight
/// segments). At `alpha = 0` this reduces exactly to `m1 = s (p2 - p0)`,
/// `m2 = s (p3 - p1)` with `s = (1 - tension) / 2`.
fn cardinal_tangents<V: Vector>(
    p0: V,
    p1: V,
    p2: V,
    p3: V,
    alpha: V::Scalar,
    tension: V::Scalar,
) -> (V, V)
where
    V::Scalar: Float,
{
    let eps = V::Scalar::from_f64(1e-5);
    // Knot-interval lengths^alpha, guarded against coincident knots. (`alpha = 0`
    // gives pow(0, 0) = 1, which is fine.)
    let guard = |d: V::Scalar| if d < eps { eps } else { d };
    let d01 = guard(length(p1 - p0).powf(alpha));
    let d12 = guard(length(p2 - p1).powf(alpha));
    let d23 = guard(length(p3 - p2).powf(alpha));

    // Tangents in knot-parameter space, multiplied by d12 to map onto local t.
    let m1 = ((p1 - p0) / d01 - (p2 - p0) / (d01 + d12) + (p2 - p1) / d12) * d12;
    let m2 = ((p2 - p1) / d12 - (p3 - p1) / (d12 + d23) + (p3 - p2) / d23) * d12;

    let w = V::Scalar::ONE - tension;
    (m1 * w, m2 * w)
}

/// Cardinal spline segment `p1 -> p2`: Catmull-Rom generalised with `tension`
/// (0 reproduces [`catmull_rom`], 1 zeroes the tangents so the segment is the
/// straight line `p1 -> p2` traversed with smoothstep easing) and knot spacing
/// `alpha` (0 uniform, 0.5 centripetal, 1 chordal).
#[inline]
pub fn cardinal<V: Vector>(
    p0: V,
    p1: V,
    p2: V,
    p3: V,
    t: V::Scalar,
    tension: V::Scalar,
    alpha: V::Scalar,
) -> V
where
    V::Scalar: Float,
{
    let (m1, m2) = cardinal_tangents(p0, p1, p2, p3, alpha, tension);
    hermite(p1, m1, p2, m2, t)
}

/// Derivative of [`cardinal`] with respect to `t` (un-normalised velocity).
#[inline]
pub fn cardinal_tangent<V: Vector>(
    p0: V,
    p1: V,
    p2: V,
    p3: V,
    t: V::Scalar,
    tension: V::Scalar,
    alpha: V::Scalar,
) -> V
where
    V::Scalar: Float,
{
    let (m1, m2) = cardinal_tangents(p0, p1, p2, p3, alpha, tension);
    hermite_derivative(p1, m1, p2, m2, t)
}

// ---- Spline -------------------------------------------------------------------

/// A cardinal spline through a list of knots.
///
/// The curve parameter `u` runs over `[0, segments]`: the integer part selects the
/// segment, the fraction is the position inside it, so `u = 2.0` is exactly the
/// third knot. Out-of-range `u` clamps to the ends. Phantom knots mirrored past
/// each end make the curve pass through the first and last points.
///
/// `u` is not proportional to distance; see [`SplineArcLength`] for that.
#[derive(Clone, Debug, PartialEq)]
pub struct Spline<V: Vector> {
    /// 0 = Catmull-Rom (default), 1 = straight lines between knots.
    pub tension: V::Scalar,
    /// Knot spacing: 0 uniform, 0.5 centripetal (default, cusp-free), 1 chordal.
    pub alpha: V::Scalar,
    points: Vec<V>,
}

impl<V: Vector> Default for Spline<V>
where
    V::Scalar: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Vector> Spline<V>
where
    V::Scalar: Float,
{
    /// An empty spline with Catmull-Rom tension and centripetal spacing.
    pub fn new() -> Self {
        Self {
            tension: V::Scalar::ZERO,
            alpha: V::Scalar::from_f64(0.5),
            points: Vec::with_capacity(4),
        }
    }

    /// A spline through `points`, with default tension and spacing.
    pub fn from_points(points: impl Into<Vec<V>>) -> Self {
        Self {
            points: points.into(),
            ..Self::new()
        }
    }

    /// Append a knot.
    pub fn add(&mut self, p: V) {
        self.points.push(p);
    }

    /// Remove every knot.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// The knots.
    #[inline]
    pub fn points(&self) -> &[V] {
        &self.points
    }

    /// Mutable access to the knots (rebuild any [`SplineArcLength`] afterwards).
    #[inline]
    pub fn points_mut(&mut self) -> &mut Vec<V> {
        &mut self.points
    }

    /// Number of knots.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` with no knots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of segments: knots minus one (0 for fewer than two knots).
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.len().saturating_sub(1)
    }

    /// Largest meaningful `u`: `segment_count()` as a scalar.
    #[inline]
    pub fn max_u(&self) -> V::Scalar {
        V::Scalar::from_f64(self.segment_count() as f64)
    }

    /// Knot `i`, extrapolated linearly past either end (`i < 0` or
    /// `i >= len()`), which is what gives the end segments their tangents.
    /// Needs at least two knots.
    pub fn knot(&self, i: isize) -> V {
        let n = self.points.len();
        let two = V::Scalar::from_f64(2.0);
        if i < 0 {
            self.points[0] * two - self.points[1]
        } else if i as usize >= n {
            self.points[n - 1] * two - self.points[n - 2]
        } else {
            self.points[i as usize]
        }
    }

    /// Segment index and local `t` for `u`, clamped to the curve.
    fn locate(&self, u: V::Scalar) -> (usize, V::Scalar) {
        let segs = self.segment_count();
        if u.is_nan() || u <= V::Scalar::ZERO {
            return (0, V::Scalar::ZERO);
        }
        let whole = u.trunc();
        let seg = whole.to_f64() as usize;
        if seg >= segs {
            return (segs - 1, V::Scalar::ONE);
        }
        (seg, u - whole)
    }

    /// Point on the curve at `u`. With fewer than two knots returns the only knot
    /// or zero.
    pub fn evaluate(&self, u: V::Scalar) -> V {
        if self.segment_count() < 1 {
            return self.points.first().copied().unwrap_or(V::ZERO);
        }
        let (seg, t) = self.locate(u);
        let s = seg as isize;
        cardinal(
            self.knot(s - 1),
            self.knot(s),
            self.knot(s + 1),
            self.knot(s + 2),
            t,
            self.tension,
            self.alpha,
        )
    }

    /// Velocity along the curve at `u` (per unit of `u`); normalise for a
    /// direction. Zero with fewer than two knots.
    pub fn tangent(&self, u: V::Scalar) -> V {
        if self.segment_count() < 1 {
            return V::ZERO;
        }
        let (seg, t) = self.locate(u);
        let s = seg as isize;
        cardinal_tangent(
            self.knot(s - 1),
            self.knot(s),
            self.knot(s + 1),
            self.knot(s + 2),
            t,
            self.tension,
            self.alpha,
        )
    }

    /// Append `steps_per_segment` points per segment plus the final knot to `out`
    /// (`segment_count() * steps_per_segment + 1` points, evenly spaced in `u`).
    pub fn sample_into(&self, out: &mut Vec<V>, steps_per_segment: usize) {
        let segs = self.segment_count();
        if segs < 1 {
            return;
        }
        out.reserve(segs * steps_per_segment + 1);
        for s in 0..segs {
            for i in 0..steps_per_segment {
                let t = V::Scalar::from_f64(i as f64 / steps_per_segment as f64);
                out.push(self.evaluate(V::Scalar::from_f64(s as f64) + t));
            }
        }
        out.push(self.evaluate(self.max_u())); // close on the exact final knot
    }

    /// [`sample_into`](Self::sample_into) into a fresh `Vec`.
    pub fn sample(&self, steps_per_segment: usize) -> Vec<V> {
        let mut out = Vec::new();
        self.sample_into(&mut out, steps_per_segment);
        out
    }
}

impl<V: Vector> Index<usize> for Spline<V> {
    type Output = V;
    #[inline]
    fn index(&self, i: usize) -> &V {
        &self.points[i]
    }
}

// ---- SplineArcLength ----------------------------------------------------------

/// One table entry: cumulative arc length `len` reached at spline parameter `u`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ArcEntry<T> {
    u: T,
    len: T,
}

/// Arc-length reparameterisation of a [`Spline`]: a distance -> `u` lookup table
/// so the curve can be evaluated by world-space distance (constant-speed travel,
/// even spacing) instead of the non-uniform spline parameter.
///
/// The table is built from one spline and the query methods take that spline
/// again, so the table never holds a reference. Rebuild after the knots,
/// `tension` or `alpha` change.
#[derive(Clone, Debug, PartialEq)]
pub struct SplineArcLength<V: Vector> {
    table: Vec<ArcEntry<V::Scalar>>,
}

impl<V: Vector> Default for SplineArcLength<V> {
    fn default() -> Self {
        Self { table: Vec::new() }
    }
}

impl<V: Vector> SplineArcLength<V>
where
    V::Scalar: Float,
{
    /// An empty table (`length() == 0`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the table from `spline`, sampling `steps_per_segment` straight-line
    /// pieces per segment (32 is plenty for most paths).
    pub fn build(spline: &Spline<V>, steps_per_segment: usize) -> Self {
        let mut this = Self::new();
        this.rebuild(spline, steps_per_segment);
        this
    }

    /// Rebuild in place (see [`build`](Self::build)).
    pub fn rebuild(&mut self, spline: &Spline<V>, steps_per_segment: usize) {
        self.table.clear();
        let segs = spline.segment_count();
        if segs < 1 || steps_per_segment == 0 {
            return;
        }
        let total = segs * steps_per_segment;
        self.table.reserve(total + 1);

        let mut prev = spline.evaluate(V::Scalar::ZERO);
        let mut acc = V::Scalar::ZERO;
        self.table.push(ArcEntry {
            u: V::Scalar::ZERO,
            len: V::Scalar::ZERO,
        });
        for i in 1..=total {
            // u spans [0, segs].
            let u = V::Scalar::from_f64(segs as f64 * i as f64 / total as f64);
            let p = spline.evaluate(u);
            acc += length(p - prev); // straight-line approximation of this tiny arc
            self.table.push(ArcEntry { u, len: acc });
            prev = p;
        }
    }

    /// `true` until [`build`](Self::build) has run on a spline with a segment.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Total arc length (0 until built).
    pub fn length(&self) -> V::Scalar {
        self.table.last().map_or(V::Scalar::ZERO, |e| e.len)
    }

    /// Spline parameter `u` at arc length `distance`, clamped to the curve.
    pub fn distance_to_u(&self, distance: V::Scalar) -> V::Scalar {
        let Some(last) = self.table.last() else {
            return V::Scalar::ZERO;
        };
        if distance.is_nan() || distance <= V::Scalar::ZERO {
            return self.table[0].u;
        }
        if distance >= last.len {
            return last.u;
        }
        // First entry whose cumulative length >= distance, then lerp u within it.
        let (mut lo, mut hi) = (0usize, self.table.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.table[mid].len < distance {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let a = self.table[lo - 1];
        let b = self.table[lo];
        let span = b.len - a.len;
        let f = if span > V::Scalar::from_f64(1e-6) {
            (distance - a.len) / span
        } else {
            V::Scalar::ZERO
        };
        a.u + (b.u - a.u) * f
    }

    /// Point at arc length `distance` along `spline`.
    pub fn evaluate_by_distance(&self, spline: &Spline<V>, distance: V::Scalar) -> V {
        spline.evaluate(self.distance_to_u(distance))
    }

    /// Point at fraction `s` in `[0, 1]` of the total length.
    pub fn evaluate_uniform(&self, spline: &Spline<V>, s: V::Scalar) -> V {
        self.evaluate_by_distance(spline, s * self.length())
    }

    /// Spline parameter of the point closest to `query`: a coarse scan over the
    /// table seeds a few Gauss-Newton steps on `f(u) = ½|P(u) - Q|²`.
    pub fn nearest_u(&self, spline: &Spline<V>, query: V) -> V::Scalar {
        if self.table.is_empty() {
            return V::Scalar::ZERO;
        }
        // Coarse pass. No closed form exists for "closest point on a cubic", so
        // the closest sample seeds a local polish.
        let mut best_u = self.table[0].u;
        let mut best_d2 = length2(spline.evaluate(best_u) - query);
        for e in &self.table[1..] {
            let d2 = length2(spline.evaluate(e.u) - query);
            if d2 < best_d2 {
                best_d2 = d2;
                best_u = e.u;
            }
        }
        // Refine: the minimum is where f'(u) = dot(P - Q, P') = 0; step
        // u -= f' / f'' with f'' ~= dot(P', P').
        let max_u = spline.max_u();
        for _ in 0..8 {
            let p = spline.evaluate(best_u);
            let d = spline.tangent(best_u);
            let den = dot(d, d);
            if den < V::Scalar::from_f64(1e-8) {
                break; // stationary tangent: cannot improve
            }
            let step = dot(p - query, d) / den;
            best_u = clamp(best_u - step, V::Scalar::ZERO, max_u);
            if step.abs() < V::Scalar::from_f64(1e-5) {
                break;
            }
        }
        best_u
    }

    /// The point on `spline` closest to `query`.
    pub fn nearest_point(&self, spline: &Spline<V>, query: V) -> V {
        spline.evaluate(self.nearest_u(spline, query))
    }
}

// ---- AnimationCurve -----------------------------------------------------------

/// One keyframe of an [`AnimationCurve`]: a `(time, value)` anchor with the slope
/// of the curve arriving (`in_tangent`) and leaving (`out_tangent`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Key<T> {
    /// Position on the time axis, in `[0, 1]`.
    pub time: T,
    /// Curve value at `time`.
    pub value: T,
    /// Slope (`dvalue / dtime`) of the segment ending at this key.
    pub in_tangent: T,
    /// Slope of the segment starting at this key.
    pub out_tangent: T,
}

impl<T> Key<T> {
    /// A key with explicit tangents.
    pub const fn new(time: T, value: T, in_tangent: T, out_tangent: T) -> Self {
        Self {
            time,
            value,
            in_tangent,
            out_tangent,
        }
    }
}

/// Which tangent of a key is meant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tangent {
    /// The slope arriving at the key.
    In,
    /// The slope leaving the key.
    #[default]
    Out,
}

/// Two-key starting shapes for an [`AnimationCurve`] on `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Constant 0.
    Zero,
    /// Constant 1.
    One,
    /// `y = t`.
    #[default]
    Linear,
    /// Flat start, linear slope at the end.
    EaseIn,
    /// Linear slope at the start, flat end.
    EaseOut,
    /// Flat at both ends (smoothstep).
    EaseInOut,
}

/// A keyframed curve on `[0, 1]`: cubic Hermite segments between keys, each with
/// its own in/out tangents, as in Unity's `AnimationCurve`.
///
/// There are always at least two keys (the end anchors, which cannot be removed).
/// The `linear` mode treats tangents beyond [`TANGENT_LIMIT`](Self::TANGENT_LIMIT)
/// as steps, which is how an editor lets the user drag a handle vertical to get a
/// constant segment.
///
/// Per-key `locked` flags (joined in/out tangents) and the tangent *handle*
/// positions are editor state carried alongside the keys.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimationCurve<T = f32> {
    keys: Vec<Key<T>>,
    /// Editor-only: whether a key's in/out tangents move together, parallel to `keys`.
    locked: Vec<bool>,
    linear: bool,
}

impl<T: Float> Default for AnimationCurve<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> AnimationCurve<T> {
    /// Slope magnitude at which a segment becomes a step in `linear` mode.
    pub const TANGENT_LIMIT: f64 = 25.0;
    /// How far from the anchor a tangent handle sits, in curve units.
    pub const HANDLE_LENGTH: f64 = 0.12;
    /// Minimum time gap kept between neighbouring keys by [`set_key`](Self::set_key).
    pub const KEY_GAP: f64 = 0.001;

    #[inline]
    fn limit() -> T {
        T::from_f64(Self::TANGENT_LIMIT)
    }

    /// The [`Preset::Linear`] curve, in linear mode.
    pub fn new() -> Self {
        Self::from_preset(Preset::Linear)
    }

    /// A two-key curve with the given shape, in linear mode.
    pub fn from_preset(preset: Preset) -> Self {
        let mut c = Self {
            keys: Vec::with_capacity(2),
            locked: Vec::with_capacity(2),
            linear: true,
        };
        c.apply_preset(preset);
        c
    }

    /// Replace all keys with the two anchors of `preset`.
    pub fn apply_preset(&mut self, preset: Preset) {
        let (zero, one) = (T::ZERO, T::ONE);
        let mut p0 = Key::new(zero, zero, zero, one);
        let mut p1 = Key::new(one, one, one, one);
        match preset {
            Preset::Zero => {
                p0.out_tangent = zero;
                p0.value = zero;
                p1.in_tangent = zero;
                p1.value = zero;
            }
            Preset::One => {
                p0.out_tangent = zero;
                p0.value = one;
                p1.in_tangent = zero;
                p1.value = one;
            }
            Preset::Linear => {
                p0.out_tangent = one;
                p1.in_tangent = one;
            }
            Preset::EaseInOut => {
                p0.out_tangent = zero;
                p1.in_tangent = zero;
            }
            Preset::EaseIn => {
                p0.out_tangent = zero;
                p1.in_tangent = one;
            }
            Preset::EaseOut => {
                p0.out_tangent = one;
                p1.in_tangent = zero;
            }
        }
        self.keys.clear();
        self.keys.push(p0);
        self.keys.push(p1);
        self.locked.clear();
        self.locked.resize(2, true);
    }

    /// Enable step behaviour for tangents beyond [`TANGENT_LIMIT`](Self::TANGENT_LIMIT)
    /// (on by default).
    pub fn set_linear(&mut self, value: bool) {
        self.linear = value;
    }

    /// Whether step behaviour is enabled.
    #[inline]
    pub fn is_linear(&self) -> bool {
        self.linear
    }

    /// Insert a key with flat tangents at `time` (clamped to `[0, 1]`). Does
    /// nothing if `time` coincides with an existing key. Returns the new index.
    // NOTE (from the original): better continuity without distorting the curve
    // would be nice; flat tangents work but could be smarter.
    pub fn add_key(&mut self, time: T, value: T) -> Option<usize> {
        let time = clamp(time, T::ZERO, T::ONE);
        for i in 0..self.keys.len() - 1 {
            if time > self.keys[i].time && time < self.keys[i + 1].time {
                self.keys
                    .insert(i + 1, Key::new(time, value, T::ZERO, T::ZERO));
                self.locked.insert(i + 1, true);
                return Some(i + 1);
            }
        }
        None
    }

    /// Remove key `i`. The first and last anchors are never removed.
    pub fn remove_key(&mut self, i: usize) {
        if i == 0 || i >= self.keys.len() - 1 {
            return;
        }
        self.keys.remove(i);
        self.locked.remove(i);
    }

    /// Move key `i` to `(time, value)`. Time is clamped between the neighbouring
    /// keys (keeping [`KEY_GAP`](Self::KEY_GAP)); the end anchors stay at their
    /// neighbour's bound.
    pub fn set_key(&mut self, i: usize, time: T, value: T) {
        let last = self.keys.len() - 1;
        let gap = if i == 0 || i == last {
            T::ZERO
        } else {
            T::from_f64(Self::KEY_GAP)
        };
        let lower = if i == 0 {
            T::ZERO
        } else {
            self.keys[i - 1].time
        };
        let upper = if i == last {
            T::ONE
        } else {
            self.keys[i + 1].time
        };
        self.keys[i].time = clamp(time, lower + gap, upper - gap);
        self.keys[i].value = value;
    }

    /// Set key `i`'s out tangent from a handle dragged to `(time, value)`. Ignored
    /// for the last key or when the handle sits on the anchor's time.
    pub fn set_out_tangent(&mut self, i: usize, time: T, value: T) {
        if i >= self.keys.len() - 1 {
            return;
        }
        let dt = time - self.keys[i].time;
        let dv = value - self.keys[i].value;
        if dt == T::ZERO {
            return;
        }
        let tangent = self.clamp_tangent(dv / dt);
        self.keys[i].out_tangent = tangent;
        if self.locked[i] {
            self.keys[i].in_tangent = tangent;
        }
    }

    /// Set key `i`'s in tangent from a handle dragged to `(time, value)`. Ignored
    /// for the first key or when the handle sits on the anchor's time.
    pub fn set_in_tangent(&mut self, i: usize, time: T, value: T) {
        if i == 0 {
            return;
        }
        let dt = time - self.keys[i].time;
        let dv = value - self.keys[i].value;
        if dt == T::ZERO {
            return;
        }
        let tangent = self.clamp_tangent(dv / dt);
        self.keys[i].in_tangent = tangent;
        if self.locked[i] {
            self.keys[i].out_tangent = tangent;
        }
    }

    /// Set key `i`'s out tangent (slope) directly; follows the lock.
    pub fn set_out_tangent_value(&mut self, i: usize, slope: T) {
        self.keys[i].out_tangent = slope;
        if self.locked[i] {
            self.keys[i].in_tangent = slope;
        }
    }

    /// Set key `i`'s in tangent (slope) directly; follows the lock.
    pub fn set_in_tangent_value(&mut self, i: usize, slope: T) {
        self.keys[i].in_tangent = slope;
        if self.locked[i] {
            self.keys[i].out_tangent = slope;
        }
    }

    /// Toggle whether key `i`'s tangents are joined. When joining, `dominant`
    /// says which tangent the other one snaps to. End anchors are ignored.
    pub fn toggle_tangent_split_join(&mut self, i: usize, dominant: Tangent) {
        if i == 0 || i >= self.keys.len() - 1 {
            return;
        }
        self.locked[i] = !self.locked[i];
        match dominant {
            Tangent::Out => self.keys[i].in_tangent = self.keys[i].out_tangent,
            Tangent::In => self.keys[i].out_tangent = self.keys[i].in_tangent,
        }
    }

    /// Key `i`'s anchor as `(time, value)`.
    #[inline]
    pub fn anchor(&self, i: usize) -> Vector2<T> {
        Vector2::new(self.keys[i].time, self.keys[i].value)
    }

    /// Where to draw key `i`'s out-tangent handle: [`HANDLE_LENGTH`](Self::HANDLE_LENGTH)
    /// along the slope, to the right of the anchor.
    pub fn out_tangent_handle(&self, i: usize) -> Vector2<T> {
        let dir = Vector2::new(T::ONE, self.keys[i].out_tangent).normalize();
        self.anchor(i) + dir * T::from_f64(Self::HANDLE_LENGTH)
    }

    /// Where to draw key `i`'s in-tangent handle: to the left of the anchor.
    pub fn in_tangent_handle(&self, i: usize) -> Vector2<T> {
        let dir = Vector2::new(T::ONE, self.keys[i].in_tangent).normalize();
        self.anchor(i) - dir * T::from_f64(Self::HANDLE_LENGTH)
    }

    /// In linear mode, keep a dragged tangent within twice the limit so it can
    /// still cross the step threshold without running away.
    fn clamp_tangent(&self, tangent: T) -> T {
        if self.linear {
            let limit = Self::limit() + Self::limit();
            clamp(tangent, -limit, limit)
        } else {
            tangent
        }
    }

    /// The keys, in time order.
    #[inline]
    pub fn keys(&self) -> &[Key<T>] {
        &self.keys
    }

    /// Number of keys (always at least two).
    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Always `false`: a curve has at least its two anchors.
    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether key `i`'s tangents are joined.
    #[inline]
    pub fn is_locked(&self, i: usize) -> bool {
        self.locked[i]
    }

    /// Every key but the last has an out tangent.
    #[inline]
    pub fn has_out_tangent(&self, i: usize) -> bool {
        i + 1 < self.keys.len()
    }

    /// Every key but the first has an in tangent.
    #[inline]
    pub fn has_in_tangent(&self, i: usize) -> bool {
        i >= 1
    }

    /// Time of the first key.
    #[inline]
    pub fn start_time(&self) -> T {
        self.keys[0].time
    }

    /// Time of the last key.
    #[inline]
    pub fn end_time(&self) -> T {
        self.keys[self.keys.len() - 1].time
    }

    /// `end_time - start_time`.
    #[inline]
    pub fn duration(&self) -> T {
        self.end_time() - self.start_time()
    }

    /// Number of segments: keys minus one.
    #[inline]
    pub fn segments(&self) -> usize {
        self.keys.len() - 1
    }

    /// The curve's value at `t`, held constant before the first and after the
    /// last key.
    pub fn evaluate(&self, t: T) -> T {
        let first = self.keys[0];
        let last = self.keys[self.keys.len() - 1];
        if t <= first.time {
            return first.value;
        }
        if t >= last.time {
            return last.value;
        }
        let limit = Self::limit();
        for w in self.keys.windows(2) {
            let (a, b) = (w[0], w[1]);
            if t <= b.time {
                let span = b.time - a.time;
                let u = (t - a.time) / span;
                if self.linear {
                    // Tangents at or beyond the limit turn the segment into a step:
                    // hold the next value (slope up) or the previous one (slope down).
                    if a.out_tangent >= limit || b.in_tangent >= limit {
                        return b.value;
                    }
                    if a.out_tangent <= -limit || b.in_tangent <= -limit {
                        return a.value;
                    }
                }
                // Tangents are slopes per unit time; scale to the segment's span.
                return hermite(
                    a.value,
                    a.out_tangent * span,
                    b.value,
                    b.in_tangent * span,
                    u,
                );
            }
        }
        last.value
    }
}

impl<T> Index<usize> for AnimationCurve<T> {
    type Output = Key<T>;
    #[inline]
    fn index(&self, i: usize) -> &Key<T> {
        &self.keys[i]
    }
}
