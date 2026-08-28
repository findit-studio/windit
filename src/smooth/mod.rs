//! Smoothing: stateful, span-preserving value rewriters.
//!
//! A [`Smoother`] rewrites the value of each
//! [`Windowed<V>`](crate::windowed::Windowed) one window in, one window out,
//! leaving its [`Span`](crate::plan::Span) untouched so the output stays aligned
//! with the input windows. A [`SmoothPolicy`] is the configuration that names a
//! strategy and constructs its streaming [`Smoother`]; it also drives that state
//! over a whole slice as a batch convenience. The shipped built-ins:
//!
//! - [`Identity`] passes values through unchanged — the no-rewrite baseline,
//!   generic over any `V`.
//! - [`Ema`] is an exponential moving average (temporal low-pass) over `f32`.
//! - [`CadenceEma`] is an exponential moving average whose time constant is
//!   denominated in input elements rather than in pushes, so one configuration
//!   yields the same smoothing at any cadence — regular or irregular — over
//!   `f32`.
#![cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "- [`VectorEma`] is that same exponential moving average over an *embedding*\n  — any [`Vector`] — run component-wise and\n  L2-renormalized at every window: the streaming, span-preserving sibling of\n  [`EmaRenormalized`](crate::aggregate::EmaRenormalized), which folds the same\n  recency weighting to one point instead."
)]
#![cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "- `VectorEma` — the same exponential moving average over an *embedding*, run\n  component-wise and L2-renormalized at every window — needs the heap for its\n  `dim`-sized state and so appears only under `alloc`."
)]
//!
//! The scalar states allocate nothing and live in the featureless core tier
//! alongside the traits; the `Vec`-returning batch driver and the vector
//! smoother — whose state is `dim`-sized rather than O(1) — gate on `alloc`.
//!
#![cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`SmoothPolicy::smooth`] restarts from fresh state on every call — a batch"
)]
#![cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "`SmoothPolicy::smooth` restarts from fresh state on every call — a batch"
)]
//! convenience, not an incremental decoder: smoothing a sequence chunk by chunk
//! through separate calls is not equivalent to one whole-sequence call, because a
//! running average does not carry across calls. To smooth incrementally, drive a
//! single [`Smoother`] across the chunks instead.

use crate::{error::WinditError, windowed::Windowed};

#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

// The vector smoother's arithmetic is the aggregation half's, reached rather
// than re-derived: `VectorEma` gates and renormalizes through the very routines
// `weighted_sum_renorm` ends with, so the streaming and folding siblings cannot
// drift apart.
#[cfg(any(feature = "std", feature = "alloc"))]
use crate::{
  aggregate::{l2_norm, l2_renorm},
  scalar::Real,
  windowed::{ComputeOf, Vector},
};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// Stateful, span-preserving value rewriter: one window in, one window out.
///
/// The decision plane's value stage — a running filter that carries whatever
/// state it needs (a seeded average, say) in O(1) space and allocates nothing, so
/// it lives in the featureless core tier. The configuration that constructs one is
/// a [`SmoothPolicy`].
///
/// A 1-in/1-out filter has no pending output, so it needs no terminal
/// `finish`; [`discontinuity`](Smoother::discontinuity) defaults to
/// [`reset`](Smoother::reset). `Box<dyn Smoother<f32>>` is a valid object, so a
/// smoother can be selected at run time.
///
/// # Span contract
///
/// Spans arrive in ascending [`Span::start`](crate::plan::Span::start) order,
/// equal starts admitted — and that is the only ordering guaranteed. **Ends are
/// not monotone:** nested and overlapping spans are legal, so a later span may end
/// *before* one already seen. A stage that keeps a temporal horizon must
/// therefore fold it by maximum (`horizon = max(horizon, span.end())`) and
/// measure against that fold; reading the current span's end alone would let
/// the horizon move backward. A strictly backward start is a contract
/// violation, reported as [`WinditError::NonMonotonicSpan`].
pub trait Smoother<V> {
  /// Advance by one window. The returned value keeps the input [`Span`].
  ///
  /// [`Span`]: crate::plan::Span
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::NonMonotonicSpan`] from a span-reading stage fed a
  /// start before the previous one: [`CadenceEma`] derives its coefficient from
  /// the span distance, so it reads spans and reports out-of-order input here.
  /// [`Identity`] and [`Ema`] read no spans and are infallible, always returning
  /// `Ok` and carrying the `Result` only for uniformity with the composable
  /// stages.
  ///
  /// Reading no spans does not by itself make a stage infallible, though: the
  /// vector EMA reads none either and is fallible for reasons of its own — a
  /// width that changed mid-epoch, a non-finite component, an accumulator with
  /// no determinate direction, a refused allocation. Its own documentation
  /// enumerates them.
  fn push(&mut self, w: Windowed<V>) -> Result<Windowed<V>, WinditError>;

  /// Return to the freshly-constructed state.
  fn reset(&mut self);

  /// Declare a timeline break; for a 1-in/1-out filter, which holds no pending
  /// output, this is [`reset`](Smoother::reset).
  fn discontinuity(&mut self) {
    self.reset();
  }
}

/// A smoothing configuration: names the strategy, constructs its streaming
/// [`Smoother`], and drives it as a batch convenience.
///
/// Generic over the value type `V`; the shipped built-ins implement it for
/// `V = f32` ([`Ema`], [`CadenceEma`]), for any `V` ([`Identity`]), and — under
/// `alloc` — for any embedding (the vector EMA). Implement the factory
/// [`smoother`](SmoothPolicy::smoother) to add a strategy — the batch
#[cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`smooth`](SmoothPolicy::smooth) method is provided over it."
)]
#[cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "`smooth` method is provided over it."
)]
pub trait SmoothPolicy<V> {
  /// The streaming state this configuration constructs.
  type Smoother: Smoother<V>;

  /// Fresh streaming state for this configuration.
  fn smoother(&self) -> Self::Smoother;

  /// Batch convenience: drive a fresh [`Smoother`] over `seq`, returning a
  /// smoothed sequence the same length as `seq`, each element keeping its input
  /// [`Span`](crate::plan::Span).
  ///
  /// Fresh state per call, exactly as the 0.1.x policies documented: smoothing a
  /// sequence chunk by chunk through separate calls is not equivalent to one
  /// whole-sequence call.
  ///
  /// # Errors
  ///
  /// - [`WinditError::AllocFailed`] if the output cannot be allocated.
  /// - Any error the underlying [`Smoother::push`] surfaces. Of the three scalar
  ///   built-ins only [`CadenceEma`] can raise one, and only
  ///   [`WinditError::NonMonotonicSpan`]; [`VectorEma`] has an error set of its
  ///   own, enumerated on its own documentation.
  #[cfg(any(feature = "std", feature = "alloc"))]
  #[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
  fn smooth(&self, seq: &[Windowed<V>]) -> Result<Vec<Windowed<V>>, WinditError>
  where
    V: Clone,
  {
    let mut smoother = self.smoother();
    let mut out: Vec<Windowed<V>> = Vec::new();
    out
      .try_reserve_exact(seq.len())
      .map_err(|_| WinditError::AllocFailed {
        elements: seq.len(),
      })?;
    for w in seq {
      out.push(smoother.push(w.clone())?);
    }
    Ok(out)
  }
}

/// Pass-through smoothing: every value is carried through unchanged, span
/// intact.
///
/// The semantic no-rewrite baseline — a deliberate absence of smoothing, not a
/// quality claim. Generic over any `V`, so it is the identity stage for
/// embeddings, probabilities, or logits alike.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Identity;

impl Identity {
  /// The pass-through smoother.
  #[must_use]
  pub const fn new() -> Self {
    Self
  }
}

/// The streaming state of [`Identity`]: a zero-sized, stateless pass-through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IdentityState;

impl<V> Smoother<V> for IdentityState {
  fn push(&mut self, w: Windowed<V>) -> Result<Windowed<V>, WinditError> {
    Ok(w)
  }

  fn reset(&mut self) {}
}

impl<V> SmoothPolicy<V> for Identity {
  type Smoother = IdentityState;

  fn smoother(&self) -> IdentityState {
    IdentityState
  }
}

/// Exponential moving average: `s_t = alpha * x_t + (1 - alpha) * s_{t-1}`.
///
/// Seeded with `s_0 = x_0`. A larger `alpha` tracks the input more closely; a
/// smaller one smooths harder — down to the point where `f32` runs out. State
/// and arithmetic are both `f32` here, so an `alpha` of `2^-25` (~3e-8) or below
/// makes `1 - alpha` round to exactly `1.0`. What is left is not a hold: the
/// decay term is gone but the `alpha * x` injection is not, so the recurrence
/// degenerates from a weighted average into the biased accumulator
/// `s <- s + alpha * x`. Concretely, at `alpha = 2^-25` the state
///
/// - never relaxes *toward* the input. It moves only in the direction of
///   `sign(x)`, by `alpha * |x|` per push, whatever the input's distance from it
///   — so an input below the state can never pull it down.
/// - climbs from a seed of `0.0` under a constant `x` in exact steps of
///   `alpha * x` — one push of `1.0` puts it at exactly `2^-25`, two at `2^-24`
///   — and stalls at `alpha * x * 2^24`, which is `x / 2` here, reached after
///   exactly `2^24` pushes. It never arrives at `x` at all.
/// - does genuinely hold, but only from that stalling magnitude upward, where a
///   step of `alpha * |x|` is no more than half an ulp of `|s|`. Seeding
///   `s_0 = x_0` on a steady signal starts the state there, which is why a
///   constant stream still looks like a clean hold: a state of `1.0` is a fixed
///   point against `0.0`, `1.0` and `-1.0` alike.
///
/// That is the honest resolution of an `f32` filter at a coefficient that small,
/// and `Ema` claims nothing more — its `alpha` is per-push and carries no
/// cadence, so it has no sampling-invariance property to violate.
/// [`CadenceEma`], which does claim one, carries its state in `f64` to push the
/// same degeneracy 29 binary orders further out — there `1 - alpha` collapses to
/// exactly `1.0` at `2^-54` rather than at `2^-25`, `2^-54` being the tie that
/// rounds to even and `2^-53` still being exactly representable — and bounds its
/// time constant so the collapse cannot be configured at all. That relocates the
/// boundary rather than removing it, and its own *Fine cadences* bullet states
/// what survives.
///
/// This policy is infallible, so [`Ema::new`] clamps `alpha` into `[0, 1]`
/// deterministically: a non-finite (NaN) `alpha` clamps to `0.0` (hold the
/// seed). With a clamped alpha and finite inputs, the recurrence introduces no
/// NaN.
///
/// `Ema` does not sanitize inputs: a non-finite input (`NaN` or `+inf`/`-inf`)
/// enters the recurrence and poisons the state — every output from that index on
/// is non-finite. A `NaN` stays `NaN`; an infinity propagates as that infinity
/// until a zero coefficient multiplies it (`0.0 * inf = NaN`, so `alpha = 1`
/// degrades an infinite state to `NaN` one step later, while `alpha = 0` degrades
/// an infinite input to `NaN` at its own index) or opposite infinities meet
/// (`inf - inf = NaN`). In particular, `alpha = 0` holds the seed only against
/// finite inputs. In a stream the poisoning persists across pushes until
/// [`discontinuity`](Smoother::discontinuity) or [`reset`](Smoother::reset)
/// re-seeds the state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ema {
  alpha: f32,
}

impl Ema {
  /// An exponential moving average with the given smoothing factor, clamped
  /// into `[0, 1]`.
  ///
  /// Clamping at construction is what keeps the infallible smoothing path total:
  /// above `1.0` clamps to `1.0`, below `0.0` clamps to `0.0`, and a NaN becomes
  /// `0.0` (hold the seed). [`alpha`](Ema::alpha) reports the clamped value the
  /// recurrence actually uses.
  #[must_use]
  pub const fn new(alpha: f32) -> Self {
    // A NaN fails both comparisons and falls through to `0.0`; `f32::clamp`
    // would propagate it instead, and is not const.
    let alpha = if alpha > 1.0 {
      1.0
    } else if alpha >= 0.0 {
      alpha
    } else {
      0.0
    };
    Self { alpha }
  }

  /// The smoothing factor, always in `[0, 1]`.
  #[must_use]
  pub const fn alpha(&self) -> f32 {
    self.alpha
  }
}

/// The streaming state of an [`Ema`]: the clamped coefficient and the running
/// average, unseeded until the first push.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaState {
  alpha: f32,
  seed: Option<f32>,
}

impl Smoother<f32> for EmaState {
  fn push(&mut self, w: Windowed<f32>) -> Result<Windowed<f32>, WinditError> {
    let (x, span) = w.into_parts();
    // First push seeds the state (`s_0 = x_0`); every later push runs the
    // recurrence against the retained previous state.
    let state = match self.seed {
      None => x,
      Some(prev) => self.alpha * x + (1.0 - self.alpha) * prev,
    };
    self.seed = Some(state);
    Ok(Windowed::new(state, span))
  }

  fn reset(&mut self) {
    self.seed = None;
  }
}

impl SmoothPolicy<f32> for Ema {
  type Smoother = EmaState;

  fn smoother(&self) -> EmaState {
    // `Ema::new` is the only way to set `alpha` and clamps it there, so this
    // re-clamp is currently unreachable. It is kept as the last line of defence
    // for the recurrence's coefficient invariant — `alpha` in `[0, 1]` and never
    // NaN — because that invariant must survive any future construction path that
    // bypasses `new`, a serde derive on this type being the obvious one. It
    // guards the coefficient only: a non-finite input value still propagates, as
    // the type docs specify. NaN is handled explicitly since `f32::clamp` would
    // propagate it.
    let alpha = if self.alpha.is_nan() {
      0.0
    } else {
      self.alpha.clamp(0.0, 1.0)
    };
    EmaState { alpha, seed: None }
  }
}

/// The per-step EMA coefficient for a span distance of `delta` elements under an
/// element-denominated time constant `tau`: `1 - exp(-delta / tau)`.
///
/// The exact zero-order-hold discretization of a first-order low-pass, and the
/// single definition shared by [`CadenceEma::alpha_for`] and
/// [`CadenceEmaState`]'s push — so the inspectable coefficient and the streaming
/// state can never drift.
fn cadence_alpha(tau: f32, delta: usize) -> f32 {
  // Spelled `-expm1(-x)` rather than the literal `1 - exp(-x)`: the literal
  // form loses every bit of `x` once `exp(-x)` rounds to exactly 1.0 — for f32
  // any `x` below 2^-25 — which would return a zero coefficient and freeze the
  // filter whenever the cadence is fine relative to `tau`, destroying the
  // cadence invariance this type exists for. `expm1` is exact to full precision
  // there and saturates to exactly the same 1.0 at the large end, since both
  // forms reach it as soon as `exp(-x)` falls below half an ulp of 1.
  //
  // The ratio is formed in `f64` and narrowed once. Spelling it
  // `(delta as f32) / tau` instead rounded the *count* before the division ran
  // — `delta as f32` cannot represent an integer above 2^24 exactly — and that
  // avoidable rounding composed with the division's own to put the coefficient
  // 2.25 ulps from the exact one at `tau = MAX_TAU, delta = 16_812_203`, with
  // 29_711 further breaches of the two-ulp figure below among the `delta`s that
  // one `tau` admits. `delta as f64` is exact to 2^53 and `f64::from` widens
  // `tau` exactly, so the only rounding left on the ratio is the single
  // narrowing to `f32`: worst case 1.49 coefficient ulps over the enumerated
  // boundary sweep in
  // `cadence_ema_coefficient_stays_within_two_ulps_of_the_exact_one`, and 1.51
  // over a wider offline sweep of the same structure (~2e8 probes, none above
  // two ulps, against 29_711 breaches for the cast). A `delta`
  // past 2^53 rounds in the `f64` cast too, harmlessly this time: `delta / tau`
  // is then past 2^27 for every accepted `tau`, deep inside the region where
  // the coefficient is exactly 1.0.
  //
  // Narrowing the ratio rather than carrying `f64` through `expm1` is
  // deliberate. The coefficient IS an `f32` — `alpha_for` returns it and the
  // state applies exactly that value — and every published figure is quantified
  // over it, the `2^-26` floor `MAX_TAU` is derived from included. Rounding
  // only at the end would be about an ulp sharper and would retire that
  // derivation with it: the correctly rounded coefficient at `MAX_TAU` is
  // exactly `2^-26`, the same value the first rejected `tau` yields, so the
  // accepted domain would no longer hold the applied coefficient strictly above
  // the `4 * ulp(s)` absorption bar. Sharpening the ratio, which is where the
  // defect was, costs nothing here: below 2^24 the two spellings agree bit for
  // bit at every accepted `tau` — by construction, not by measurement. `delta`
  // is exactly representable there, so both are rounding the same quotient of
  // two `f32`s, and `f64` carries more than twice `f32`'s significand
  // (53 >= 2 * 24 + 2), which is the condition under which routing a quotient
  // through the wider format cannot land it on the wrong side of an `f32`
  // midpoint.
  //
  // `expm1f` returns no NaN for the non-positive arguments produced here —
  // including the `-inf` an overflowing ratio narrows to, which returns exactly
  // `-1.0` — so with a constructor-validated `tau` the coefficient stays in
  // `[0, 1]`.
  -libm::expm1f((-(delta as f64) / f64::from(tau)) as f32)
}

/// Cadence-portable exponential moving average: an EMA whose time constant is
/// denominated in input elements, not in pushes.
///
/// Each push derives its own coefficient from the *actual* span distance,
/// `alpha = 1 - exp(-delta / tau)` where `delta` is the gap between this span's
/// start and the previous one — the exact zero-order-hold discretization of a
/// first-order low-pass. The recurrence is then the ordinary EMA
/// `s = alpha * x + (1 - alpha) * s_prev`, seeded `s_0 = x_0` on the first push.
/// The state `s` is retained in `f64` and each output is that state rounded to
/// `f32` — a fine cadence's coefficient is otherwise smaller than the state can
/// record (see *Fine cadences* below) — so re-feeding outputs into a fresh
/// smoother is not the same as carrying one across the stream.
/// Because the coefficient tracks the cadence, one `tau` yields the same
/// smoothing at any hop — regular or irregular — where a bare per-step [`Ema`]
/// `alpha` does not: the configuration carries no cadence, the data does.
/// That portability is a floating-point property with a floating-point limit.
/// Over differences the emitted `f32` can express it holds at every accepted
/// configuration; below that resolution it stays signal-dependent, and the
/// *Fine cadences* bullet below states exactly where the line falls. (Lineage:
/// the LiveKit EMA, made cadence-portable.)
///
/// `tau` is an element count in `(0, MAX_TAU]` — positive, finite, and no
/// greater than [`MAX_TAU`](CadenceEma::MAX_TAU) (`2^26 - 4` elements);
/// [`new`](CadenceEma::new) and [`try_new`](CadenceEma::try_new) reject anything
/// else, because there is no sane clamp target at either end. `tau = 0` would
/// make an equal-start step compute `0/0`; the ceiling is where the accuracy
/// contract stops holding rather than where the filter stops working. Past it
/// the unit coefficient no longer clears `2^-26`, so the absorption and
/// one-`tau` figures below stop being provable of it — but the filter still
/// runs: the first rejected `tau`, `2^26`, applies exactly `2^-26` per unit
/// step, which moves a state seeded at `0.0` to exactly `2^-26`. A `tau` that
/// genuinely cannot move a state of its own magnitude is 28 binary orders
/// further out, and even there the freeze is a property of the state as much as
/// of `tau`; [`MAX_TAU`](CadenceEma::MAX_TAU) says where. Bounding the domain is
/// what makes the figures below true of *everything this type admits* rather
/// than of a measured subrange: see [`MAX_TAU`](CadenceEma::MAX_TAU) for the
/// derivation. A caller
/// working in another unit converts at the boundary
/// (`tau_elements = tau_other * elements_per_unit`); the unit never enters this
/// API.
///
/// # Cadence edges
///
/// - **Equal starts** (`delta = 0`, admitted): `alpha = 1 - exp(0) = 0` exactly,
///   so the duplicate observation is ignored *arithmetically*, through the one
///   recurrence and not a branch. A non-finite value pushed at `delta = 0` still
///   poisons the state, though, since `0.0 * NaN` and `0.0 * inf` are both `NaN`
///   — mirroring [`Ema`] at `alpha = 0`.
/// - **Fine cadences keep their coefficient; whether they keep its *effect*
///   depends on the signal.** The coefficient is derived in a form that stays
///   exact as `delta / tau` shrinks, so it never rounds to zero — and the
///   accepted domain floors it at `2^-26` for `delta = 1`, the smallest
///   coefficient any accepted `tau` can produce — while the state is carried in
///   `f64` rather than `f32`, which moves the point where a step is lost 29
///   binary orders further out. Neither makes cadence invariance unconditional,
///   and in finite precision nothing can: the step a push contributes is
///   `alpha * (x - s)` and it is added to a state of magnitude `|s|`, so it
///   survives only while
///
///   ```text
///   alpha * |x - s|  >  4 * ulp(s)
///   ```
///
///   Below that the push may leave the state bit-identical, and a cadence
///   absorbed once is absorbed for good — the state is then a fixed point of
///   its own map — however far `alpha` sits above any flat threshold on
///   `delta / tau` alone.
///
///   The `4` is not the half-ulp a single correctly-rounded step would give.
///   The recurrence evaluates two *separately rounded* products and adds them,
///   so a step is measured against three roundings rather than one: `1 - alpha`,
///   the product `(1 - alpha) * s`, and the final sum each cost up to half an
///   ulp of `|s|`, which bounds absorption at `(1.5 + alpha) * ulp(s)`. The
///   published `4` is deliberately looser than that derivation *and* than
///   measurement: an adversarial search over the accepted domain — ~1.4e8 probes
///   above the bar, every accepted `tau` binade and
///   [`MAX_TAU`](CadenceEma::MAX_TAU) itself, non-dyadic retained states, binade
///   edges, both signs — found nothing absorbed above one `ulp(s)`, and
///   `0.95 * ulp(s)` is reachable from two ordinary pushes. The three bounds
///   this one replaces were each derived and published without such a search,
///   and each was falsified by a case the derivation did not model.
///
///   Dividing by `|s|` and writing the *contrast* `rho = |x - s| / |s|`, an
///   `f64` state makes the condition `alpha * rho > 2^-50` (`2^-51` at the top
///   of a binade) — a bound on the product, not on `alpha` alone. That is what
///   [`MAX_TAU`](CadenceEma::MAX_TAU) is for, and what it buys: it bounds
///   `alpha` from *below*, at `2^-26` for `delta = 1` and higher for every
///   larger `delta`, and an `f32` half-ulp is `2^28` `f64` ulps at the same
///   magnitude, so on the accepted domain
///
///   ```text
///   alpha * |x - s|  >  2^-26 * 2^28 * ulp(s)  =  4 * ulp(s)
///   ```
///
///   for every difference the emitted `f32` can express. **Every representable
///   difference therefore moves the state, at every accepted configuration** —
///   unconditional over the domain, not conditional on the caller having picked
///   a small enough `tau`. What stays absorbed is exactly what the output could
///   never have shown: a contrast finer than `f32` resolution, which can be
///   absorbed at any accepted `tau`. Two edges bound the picture: a state of
///   exactly `0.0` has no relative resolution to lose, so `s = alpha * x`
///   survives whatever `alpha` is; and `|alpha * x|` cannot fall below about
///   `2^-175` for a nonzero `f32` input, so the state cannot be pushed into
///   `f64`'s subnormals in one step and the relative reading of `ulp(s)` is the
///   operative one until a long decay run drives the state to a value the
///   emitted `f32` already reports as zero.
/// - **Cadences agree to about an ulp of the *swing*, not of the result:** each
///   output is the `f64` state rounded to `f32`, and the coefficient is an
///   `f32`, so the retained fraction `1 - alpha` is resolved only to an absolute
///   `2^-25`. That error multiplies the distance between the state and the
///   input, so two cadences covering the same elapsed distance differ by a small
///   multiple of `2^-25 * |x - s_0|` — measured at no more than four times that,
///   over `tau` from 3 to 10007 at distances from `tau/4` to `12 tau` and over
///   the `tau` ladder to `2^20` at distances from `tau/4` to `4 tau`. Where the
///   result is a healthy fraction of that swing this is a *few* ulps of the
///   result, not one: over one `tau` of elapsed distance, for any accepted
///   `tau >= 1`, a unit cadence lands within `4` ulps of the nearest `f32` to
///   `exp(-1)`, and within `4` ulps of a single `tau`-sized step. Measurement
///   puts both at `2` — the worst case over every integer `tau` from 1 to 1024,
///   sampled fractional `tau`, and a ladder of powers of two through
///   [`MAX_TAU`](CadenceEma::MAX_TAU) itself, which is the whole domain this
///   type accepts — so `4` is again the conservative published figure. `tau = 14`
///   and `tau = 238` both land exactly two ulps below `exp(-1)`, which is why
///   the claim is no longer "within one". This one is quantified over the
///   accepted domain rather than over a swept subrange of it: that distinction
///   is why the ceiling exists, since the same statement is *false* at
///   `tau = 2^55`, where a unit cadence cannot move a state of order `1` at all
///   and the two cadences end millions of ulps apart. Where the state has
///   instead decayed by many `tau`, the residual
///   is exponentially smaller than the error, and the same absolute agreement is
///   many ulps of that residual — about `2^-25 * exp(delta / tau)` in relative
///   terms, which measures at ~10^4 ulps by `delta / tau = 12`. Invariance is to
///   within the resolution of the emitted `f32`, never bit for bit.
/// - **Large gaps forget:** once `delta / tau` passes `ln(2^25)` (about 17.33),
///   `exp(-delta / tau)` is below half an ulp of 1 and `alpha` is exactly
///   `1.0`, so the state tracks the input exactly. Then `1 - alpha` is exactly
///   `0.0`, and `0.0 * inf = NaN` washes an *infinite* prior state to `NaN` at
///   that step; a `NaN` prior state stays `NaN` regardless. Non-finite state
///   never washes out arithmetically — only
///   [`discontinuity`](Smoother::discontinuity) or [`reset`](Smoother::reset)
///   clears it.
///
/// # Non-finite inputs
///
/// `CadenceEma` does not sanitize inputs. A non-finite value (`NaN` or an
/// infinity) enters the recurrence and poisons the state for the rest of the
/// epoch: a `NaN` stays `NaN`, and an infinity propagates as that infinity while
/// both coefficients are nonzero, degrading to `NaN` only when a zero
/// coefficient multiplies it or opposite infinities meet. In a stream the
/// poisoning persists across pushes until a
/// [`discontinuity`](Smoother::discontinuity) or [`reset`](Smoother::reset)
/// re-seeds the state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadenceEma {
  tau: f32,
}

impl CadenceEma {
  /// The largest accepted time constant: `2^26 - 4` elements — the largest `f32`
  /// strictly below `2^26`, about 6.7e7.
  ///
  /// Derived from the guarantee rather than chosen for roundness. Every
  /// unconditional statement this type makes rests on the per-step coefficient
  /// staying above `2^-26`: half an `f32` ulp is `2^28` `f64` ulps at the same
  /// magnitude, so `alpha > 2^-26` is exactly what lifts every difference the
  /// emitted `f32` can express above the `4 * ulp(s)` absorption bar (the *Fine
  /// cadences* bullet derives it). [`alpha_for(1)`](CadenceEma::alpha_for) falls
  /// as `tau` grows, and this is the largest `f32` at which it is still strictly
  /// above `2^-26`: it yields `2^-26 + 2^-49`, while one `f32` step further out,
  /// at `tau = 2^26`, the coefficient is exactly `2^-26` and the product lands
  /// *on* the bar instead of above it.
  ///
  /// No usable configuration is lost — `2^26` elements is over a week of audio
  /// at a 10 ms hop.
  ///
  /// That makes this an accuracy boundary, not a liveness one. The first
  /// rejected `tau` still filters: at `2^26` one unit push moves a state seeded
  /// at `0.0` to exactly `2^-26`, and one unit push decays a state of `1.0` by
  /// the same amount. Landing on the bar rather than above it is the whole of
  /// what it loses, and what would turn the unconditional statements above into
  /// claims this crate cannot prove of it.
  ///
  /// The regime where a filter really does stop filtering is far further out and
  /// depends on the state, not on `tau` alone: from `tau = 2^54` the `f64`
  /// `1 - alpha` rounds to exactly `1.0` at `delta = 1`, leaving the recurrence
  /// no decay term at all — it degenerates to `s <- s + alpha * x`, which drifts
  /// in `sign(x)` and cannot relax toward the input, exactly as [`Ema`]'s doc
  /// walks it at `2^-25` for an `f32` state. Against a state of order `1` that
  /// increment is under half an ulp, so the state is bit-identical forever while
  /// a single `tau`-sized step still decays it to `exp(-1)`; against a state of
  /// `0.0` it still moves, by exactly `alpha * x`. `2^54` is 28 binary orders
  /// above this ceiling, which is the margin the accuracy bar buys and the
  /// reason freeze language belongs there and not here.
  pub const MAX_TAU: f32 = 67_108_860.0;

  /// A cadence-portable EMA with the given element-denominated time constant.
  ///
  /// # Panics
  ///
  /// Panics, in every build, unless `tau` is strictly positive and no greater
  /// than [`MAX_TAU`](CadenceEma::MAX_TAU). Use
  /// [`try_new`](CadenceEma::try_new) to handle an untrusted `tau` instead.
  #[must_use]
  pub const fn new(tau: f32) -> Self {
    match Self::try_new(tau) {
      Ok(cfg) => cfg,
      Err(_) => panic!("a cadence time constant tau must be in (0, CadenceEma::MAX_TAU] elements"),
    }
  }

  /// The checked counterpart of [`new`](CadenceEma::new): validate `tau` rather
  /// than panic on it.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::TimeConstantOutOfRange`] unless `tau` lies in
  /// `(0, MAX_TAU]` — so `NaN`, either infinity, zero, a negative `tau`, and any
  /// `tau` above [`MAX_TAU`](CadenceEma::MAX_TAU) are all rejected. Neither end
  /// has a sane clamp target, so rejection is the only honest total answer: a
  /// non-positive `tau` has no meaning at all, and one past the ceiling names a
  /// filter this crate can still run but can no longer make its accuracy
  /// statements about — its unit coefficient stops clearing `2^-26` — while no
  /// substitute value could be said to approximate the `tau` the caller asked
  /// for.
  pub const fn try_new(tau: f32) -> Result<Self, WinditError> {
    // No `is_finite` test: `NaN` fails both comparisons, `+inf` fails the upper
    // bound, and `-inf` the lower, so the interval check is already total.
    if tau > 0.0 && tau <= Self::MAX_TAU {
      Ok(Self { tau })
    } else {
      Err(WinditError::TimeConstantOutOfRange)
    }
  }

  /// The element-denominated time constant, always in
  /// `(0, `[`MAX_TAU`](CadenceEma::MAX_TAU)`]`.
  #[must_use]
  pub const fn tau(&self) -> f32 {
    self.tau
  }

  /// The per-step coefficient this configuration derives for a span distance of
  /// `delta` elements: `1 - exp(-delta / tau)`.
  ///
  /// The exact function the streaming state applies, exposed for tests and
  /// downstream calibration. It is `0.0` at `delta = 0`, monotonically
  /// non-decreasing in `delta`, and saturates to `1.0` once the gap dwarfs
  /// `tau`. For every `delta >= 1` it is strictly above `2^-26`, whatever the
  /// accepted `tau` — the floor [`MAX_TAU`](CadenceEma::MAX_TAU) is chosen to
  /// hold, and the one the accuracy guarantees rest on.
  #[must_use]
  pub fn alpha_for(&self, delta: usize) -> f32 {
    cadence_alpha(self.tau, delta)
  }
}

/// The streaming state of a [`CadenceEma`]: the time constant and the previous
/// `(span start, smoothed value)`, unseeded until the first push.
///
/// The retained value is an `f64` while the emitted one is an `f32`, so that a
/// step below the resolution an `f32` state would have had is still recorded —
/// 29 binary orders further out, not unconditionally; the *Fine cadences* bullet
/// on [`CadenceEma`] states the condition that survives. The widening is
/// confined to this accumulator — `tau` and the coefficient stay `f32`, so
/// [`CadenceEma::alpha_for`] remains the exact coefficient applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadenceEmaState {
  tau: f32,
  prev: Option<(usize, f64)>,
}

impl Smoother<f32> for CadenceEmaState {
  fn push(&mut self, w: Windowed<f32>) -> Result<Windowed<f32>, WinditError> {
    let (x, span) = w.into_parts();
    let start = span.start();
    match self.prev {
      // First push (fresh, or after `reset`/`discontinuity`) seeds `s_0 = x_0`
      // and arms the timeline. No monotonicity check on the seeding push — the
      // timeline is (re-)armed here, matching `Segmenter`.
      None => {
        self.prev = Some((start, f64::from(x)));
        Ok(Windowed::new(x, span))
      }
      Some((prev_start, prev_val)) => {
        // Monotonicity, checked before any state mutation, so an out-of-order
        // push is a no-op that reports the violation: a retry with an in-order
        // span behaves as if the bad push never happened.
        if start < prev_start {
          return Err(WinditError::NonMonotonicSpan { prev_start, start });
        }
        let delta = start - prev_start; // cannot underflow after the check

        // The `f32` coefficient `alpha_for` reports, applied in `f64`. The
        // widening is the accumulator's, not the coefficient's: both paths call
        // the one `cadence_alpha` and both get the same `f32` back, so the
        // inspectable coefficient and the streaming state cannot drift.
        //
        // The *state* must be the wider type, not merely the arithmetic — an
        // f32 result would re-round to the same fixed point every step. In the
        // deepest regime the loss is inside the coefficient, where `1.0 - alpha`
        // rounds to exactly `1.0` (f32 at any `alpha` of `2^-25` or below, f64
        // at `2^-54` or below — `2^-53` is still exactly representable there,
        // and `2^-54` is the tie that rounds to even), so both addends are
        // already exact and a carried compensation term would have nothing to
        // hold. Above that regime the
        // loss is in the final addition, and a compensated or double-double
        // accumulator *would* recover it — deliberately not carried: each such
        // widening only relocates the boundary (coefficient -> f32 accumulator
        // -> f64 accumulator was three rounds of exactly that), never removes
        // it, since invariance is unachievable in finite precision. The contract
        // is documented instead: see the *Fine cadences* bullet on `CadenceEma`
        // for the contrast-dependent condition that actually holds.
        //
        // Rearranging to `prev + alpha * (x - prev)` puts the loss back into a
        // single addend and IS sharper — measured absorption ceilings of
        // `0.50 * ulp(prev)` against this form's `1.48`, since one rounding
        // replaces three. It is still declined, because it forfeits exact
        // tracking at `alpha == 1`: `prev + (x - prev)` returns `0.0`, not `x`,
        // once `prev` is large enough that `x - prev` rounds to `-prev`
        // (`cadence_ema_large_gap_forgets_and_washes_infinity_to_nan` pins that
        // at `prev = 1e30`). Trading a documented three-rounding bound for a
        // silent loss of the exact-forget edge is the worse deal; the bound is
        // published conservatively instead.
        //
        // Keeping the algebra means every coefficient edge still falls out of
        // the one expression: `alpha == 1` gives `1 * x + 0 * prev`, which
        // tracks `x` exactly and still yields `0 * inf = NaN` over an infinite
        // prior state; `alpha == 0` gives `0 * x + 1 * prev`, the
        // duplicate-ignoring `delta == 0` rule, which still poisons on a
        // non-finite `x`. Each holds in `f64` exactly as it did in `f32`.
        let alpha = f64::from(cadence_alpha(self.tau, delta));
        let s = alpha * f64::from(x) + (1.0 - alpha) * prev_val;
        self.prev = Some((start, s));
        // The retained state keeps full precision; only the emitted value
        // rounds. A convex combination of two f32-representable values cannot
        // leave f32's range, so this narrowing cannot overflow.
        Ok(Windowed::new(s as f32, span))
      }
    }
  }

  fn reset(&mut self) {
    // One field re-seeds `s_0 = x_0` on the next push and re-arms the
    // monotonicity check. `discontinuity` is the trait default (= `reset`): a
    // 1-in/1-out filter holds no pending output, so the two coincide.
    self.prev = None;
  }
}

impl SmoothPolicy<f32> for CadenceEma {
  type Smoother = CadenceEmaState;

  fn smoother(&self) -> CadenceEmaState {
    // Unlike `EmaState`, this performs no coefficient re-clamp. `EmaState` can
    // restore its `alpha` invariant because `[0, 1]` has a valid clamp target;
    // `tau` has none — there is no in-range substitute for a non-positive, a
    // non-finite, or an over-ceiling time constant — so a `tau` that bypassed
    // `new`/`try_new` (a future serde derive, say) is carried as-is and degrades
    // to `NaN` outputs or to a frozen state at worst, never a panic and never
    // UB, because `alpha` is derived per push and the recurrence is total for
    // any stored `tau`. The accuracy guarantees on `CadenceEma` are stated over
    // the *accepted* domain and so would not cover such a state; keeping the
    // only construction paths validating is what makes that domain the real one.
    CadenceEmaState {
      tau: self.tau,
      prev: None,
    }
  }
}

/// Component-wise exponential moving average over an embedding, L2-renormalized
/// at every window: the streaming, span-preserving sibling of
/// [`EmaRenormalized`](crate::aggregate::EmaRenormalized).
///
/// One window in, one window out. The accumulator advances
/// `s_t = alpha * x_t + (1 - alpha) * s_{t-1}` component-wise from `s_0 = x_0`,
/// exactly as [`Ema`] does per scalar, and each window emits the *direction* of
/// that accumulator — `s_t / ||s_t||` — reconstructed through
/// [`Vector::from_unnormalized`]. Nothing is collapsed: every pushed window
/// still produces one output, carrying its input
/// [`Span`](crate::plan::Span) unchanged. That is the shape difference from
/// [`aggregate`](crate::aggregate), which folds a finished slice to one point.
///
/// # Renormalized, not merely averaged
///
/// The renormalization is applied to an emitted *copy*; the accumulator itself
/// stays raw. That is what makes this the streaming sibling of
/// [`EmaRenormalized`](crate::aggregate::EmaRenormalized) rather than a
/// spherical filter of its own: the recency
/// weights `w_0 = (1 - alpha)^t`, `w_i = alpha * (1 - alpha)^(t - i)` that the
/// aggregate builds explicitly are exactly the ones this recurrence carries, so
/// the window at index `i` emits the direction the aggregate folds over the
/// prefix `[0..=i]`. Renormalizing the accumulator in place instead would keep
/// the state on the unit sphere and change every later weight — a different
/// filter, and not one this crate's aggregation half has a fold for.
///
/// A component-wise EMA of unit vectors is *not* a unit vector: mix `[1, 0]`
/// and `[0, 1]` at `alpha = 0.5` and the accumulator is `[0.5, 0.5]`, of norm
/// `2^-0.5`. The renormalization is what turns that back into an embedding, and
/// it runs through the same scale-aware routine the aggregation fold ends with,
/// so a direction whose norm is not representable is still normalized rather
/// than rejected.
///
/// # A recurrence, not a fold
///
/// That equivalence is exact in exact arithmetic, and neither side computes in
/// exact arithmetic. **The two are not bit-identical and cannot be made so.**
/// The aggregate materializes each weight by iterated multiplication and folds
/// the whole prefix with Neumaier compensation; this carries a two-term
/// recurrence and no compensation at all. Two different roundings of one exact
/// quantity, each with its own error bound, and no amount of compensation here
/// would close the gap: the aggregate's own weights are rounded, so even an
/// exactly-evaluated recurrence would disagree with it. What holds is narrower
/// and is what the tests pin:
///
/// - **Determinate prefixes agree.** Where the exact combination clears both
///   thresholds the two emit the same direction, to within the sum of their
///   error bounds — `1e-12` over a twelve-window sweep at five smoothing
///   factors and both storage widths, against directions of order one.
/// - **Indeterminate prefixes are refused by both.** Where the exact
///   combination is zero, each side's result is inside its own error bound of
///   zero and therefore at or below its own threshold. Neither fabricates a
///   direction out of cancellation.
/// - **Between the two there is a band where the verdicts may differ, and only
///   in one direction: this side is the more conservative.** From the second
///   window on, its threshold is never below the aggregate's (the
///   [determinacy gate](#determinacy-gate) below), because a recurrence carries
///   rounding the fold does not; over the first two windows the two thresholds
///   are equal. A prefix whose residue lands between them is emitted by the
///   aggregate and refused here. The band is narrow — the ratio of thresholds tends to
///   `1 / alpha`, over residues already at the `EPSILON` scale of the mass — and
///   an aggregate result inside it is itself within a small multiple of its own
///   error bound.
///
/// Only the first of those is an accuracy claim; the other two are what
/// "sibling" is allowed to mean across a fold and a recurrence.
///
/// # Determinacy gate
///
/// Before each renormalization the accumulator is measured against its own
/// rounding floor, `16 * `[`EPSILON`](crate::scalar::Real::EPSILON)` * ||M|| + `[`MIN_GATE_THRESHOLD`](crate::scalar::Real::MIN_GATE_THRESHOLD).
/// A result at or below it is reported as [`WinditError::NonFinite`] — "no
/// direction determined at working precision" — rather than handed to a
/// renormalization that would amplify rounding noise into a fabricated unit
/// direction. The shape, the constant and the absolute floor are the
/// aggregate's, unchanged. The gate fires on exact cancellation (`[1, 0]` then
/// `[-1, 0]` at `alpha = 0.5`) and on the near-miss residues an exact-zero test
/// would let through.
///
/// `M` is where a recurrence differs from a fold, and it is not this step's two
/// term magnitudes. Those measure the step just taken; what the accumulator
/// actually carries is the rounding of *every* step, damped by the same
/// `(1 - alpha)` the value is, because each step's error is multiplied into the
/// next:
///
/// ```text
/// M_t = |alpha * x_t| + |(1 - alpha) * s_{t-1}| + (1 - alpha) * M_{t-1},   M_0 = 0
/// ```
///
/// dominates the accumulator's distance from the exact combination
/// (`2 * u * M_t` componentwise, `u = EPSILON / 2`), which the two-term
/// magnitude does not once earlier windows have cancelled — a collapsed
/// `|s_{t-1}|` no longer records the mass it collapsed from. `M_0` is zero
/// because the seed is a copy: `s_0 = x_0` commits no rounding, so it needs no
/// allowance, and an all-zero seed is caught by the threshold's absolute floor
/// alone. From `M_1 = |alpha * x_1| + |(1 - alpha) * x_0|` — which is exactly
/// the one-window fold's mass carried forward, so the two thresholds are
/// identical over the first two windows — `M_t` dominates the mass the aggregate
/// accumulates over the same prefix
/// (`alpha * |x_t| + (1 - alpha) * M^agg_{t-1}`) at every later window, so from
/// there on this threshold is never the smaller of the two: see
/// *A recurrence, not a fold* above for what that buys and what it costs.
///
/// # Dimension
///
/// The first push after construction (or after a
/// [`reset`](Smoother::reset)/[`discontinuity`](Smoother::discontinuity)) fixes
/// the epoch's dimension. A later window of a different width is rejected with
/// [`WinditError::DimMismatch`]; there is no defensible alternative, since a
/// 512-wide state has no component to mix a 384-wide window into. The width is
/// read off the *projected* slice
/// ([`compute_components`](Vector::compute_components)) rather than
/// [`dim`](Vector::dim), because that projection is the value surface the
/// recurrence actually reads.
///
/// # Inputs are validated, not absorbed
///
/// Unlike the scalar [`Ema`], which documents a non-finite input poisoning its
/// state until a reset, this smoother refuses one: a window carrying a `NaN` or
/// an infinity is rejected with [`WinditError::NonFinite`] **before any
/// component of the accumulator is written**, so the stream continues from the
/// state the previous window left. Every rejection this type can raise is
/// checked ahead of the recurrence, so a refused push is a no-op and a retry
/// behaves as if it never happened — with one deliberate exception: a window
/// whose *output* is rejected, by the determinacy gate or by the embedding's own
/// reconstruction, has still advanced the accumulator. It was a real
/// observation; only the direction read off it was unavailable, and the next
/// window mixes against the state it produced. That matches the prefix the
/// aggregate would fold, which is the property this type is defined by.
///
/// Rejecting rather than absorbing is also what the aggregate does — its
/// input-domain check refuses a non-finite component with the same
/// [`NonFinite`](WinditError::NonFinite) — so the two halves agree.
///
/// They agree on the *magnitude* domain too: a component that is neither zero
/// nor between [`MIN_AGG_MAGNITUDE`](crate::scalar::Real::MIN_AGG_MAGNITUDE)
/// and [`MAX_AGG_MAGNITUDE`](crate::scalar::Real::MAX_AGG_MAGNITUDE) is
/// refused with [`MagnitudeOutOfRange`](WinditError::MagnitudeOutOfRange),
/// the aggregation
/// [input domain](crate::aggregate#input-domain) verbatim, before the
/// accumulator is written. It would be tempting to argue that domain away —
/// the recurrence itself is a two-term convex step, `alpha * x + (1 - alpha) * s`
/// is bounded by `max(|x|, |s|)`, and the scale-aware renormalization handles a
/// norm that is not representable — but the recurrence is not the only thing
/// running. The determinacy gate below carries a *mass* that is an `n`-term
/// geometric fold over the whole epoch, which is precisely the shape the
/// aggregation domain exists to keep inside `f64`; and the gate reads that mass
/// through an L2 norm that overflows for a diagonal of `f64::MAX` long
/// before the renormalization it guards would have. Both are bounded by the
/// domain and by nothing else, so it is this type's precondition as much as the
/// fold's. Every value an `f32`-storage embedding can produce is more than 250
/// binary orders inside it, so only an `f64`-storage embedding can reach the
/// boundary at all.
///
/// # Alpha
///
/// [`new`](VectorEma::new) clamps `alpha` into `[0, 1]` exactly as
/// [`Ema::new`] does, NaN included (it clamps to `0.0`, holding the seed
/// direction). The smoothing path is therefore total in its coefficient, and
/// `alpha` never reaches the error channel — the smoother idiom, not the
/// aggregate's deferred [`AlphaOutOfRange`](WinditError::AlphaOutOfRange)
/// check.
///
/// # Allocation
///
/// Three `dim`-length buffers — the accumulator, the gate's running mass vector,
/// and the unit copy handed to
/// [`from_unnormalized`](Vector::from_unnormalized) — are grown on the first
/// push of an epoch and reused by every push after it, and
/// [`reset`](Smoother::reset) keeps their capacity, so a discontinuity costs no
/// allocation either. The filter itself therefore allocates nothing per window.
/// Two allocations per window remain, and both belong to the embedding rather
/// than to the filter: the reconstruction
/// [`from_unnormalized`](Vector::from_unnormalized) performs, and — for storage
/// narrower than its compute domain, which is every `f32` embedding — the
/// widened buffer [`compute_components`](Vector::compute_components) returns.
/// The second is the price of reading the embedding through its declared value
/// surface instead of its raw storage, which is what keeps quantized storage
/// from being smoothed as codes; `f64` storage borrows and pays neither.
///
/// Because the state is `dim`-sized rather than O(1), this is the one smoother
/// that needs the heap, which is why it gates on `alloc` while [`Identity`],
/// [`Ema`], and [`CadenceEma`] do not.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorEma {
  alpha: f32,
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl VectorEma {
  /// A renormalizing vector EMA with the given smoothing factor, clamped into
  /// `[0, 1]`.
  ///
  /// Clamping at construction is what keeps the coefficient out of the error
  /// channel: above `1.0` clamps to `1.0`, below `0.0` clamps to `0.0`, and a
  /// NaN becomes `0.0` (hold the seed direction). [`alpha`](VectorEma::alpha)
  /// reports the clamped value the recurrence actually uses. This is
  /// [`Ema::new`]'s rule verbatim, not the aggregate
  /// [`EmaRenormalized`](crate::aggregate::EmaRenormalized)'s deferred
  /// rejection: a smoother's `push` must stay total in its configuration.
  #[must_use]
  pub const fn new(alpha: f32) -> Self {
    // A NaN fails both comparisons and falls through to `0.0`; `f32::clamp`
    // would propagate it instead, and is not const.
    let alpha = if alpha > 1.0 {
      1.0
    } else if alpha >= 0.0 {
      alpha
    } else {
      0.0
    };
    Self { alpha }
  }

  /// The smoothing factor, always in `[0, 1]`.
  #[must_use]
  pub const fn alpha(&self) -> f32 {
    self.alpha
  }
}

/// The streaming state of a [`VectorEma`]: the clamped coefficient and the raw
/// component-wise accumulator, unseeded until the first push.
///
/// The accumulator is carried in the embedding's compute domain
/// ([`ComputeOf<E>`], `f64` for every shipped scalar) rather than in its
/// storage scalar, the same widening [`CadenceEmaState`] applies to its scalar
/// state and [`aggregate`](crate::aggregate) applies to its fold. Alongside it
/// the state keeps two `dim`-length scratch buffers: the running mass vector the
/// determinacy gate measures against, and the unit copy handed to
/// [`Vector::from_unnormalized`] — so the accumulator itself is never
/// renormalized.
///
/// An empty accumulator *is* the unseeded state: [`reset`](Smoother::reset)
/// clears the buffers without releasing their capacity, so a new epoch re-seeds
/// without allocating.
///
/// The `E: Vector` bound is structural, not behavioural: the buffers are
/// `Vec<ComputeOf<E>>` — a projection through `E`'s associated `Scalar` — so the
/// field types cannot be *named* without it, and it therefore cannot be narrowed
/// onto the impls that call `E`'s methods. Nothing here stores an `E`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub struct VectorEmaState<E: Vector> {
  alpha: f32,
  /// The raw EMA accumulator, `s_t`. Empty until the first push seeds it.
  state: Vec<ComputeOf<E>>,
  /// `M`: the gate's running mass — this step's two term magnitudes plus the
  /// damped mass of every step before it. See [`ema_step`] for why it
  /// accumulates rather than being overwritten.
  mag: Vec<ComputeOf<E>>,
  /// The renormalized copy of `state` that each window emits.
  unit: Vec<ComputeOf<E>>,
}

/// Hand-written rather than derived: a derive would demand `E: Clone` for a
/// parameter this state never stores a value of.
#[cfg(any(feature = "std", feature = "alloc"))]
impl<E: Vector> Clone for VectorEmaState<E> {
  fn clone(&self) -> Self {
    Self {
      alpha: self.alpha,
      state: self.state.clone(),
      mag: self.mag.clone(),
      unit: self.unit.clone(),
    }
  }
}

/// Hand-written for the same reason as [`Clone`], and reporting the state's
/// *shape* rather than its components: [`Real`] carries no [`Debug`] bound, so
/// the accumulator's values cannot be formatted from here at all.
///
/// [`Debug`]: core::fmt::Debug
#[cfg(any(feature = "std", feature = "alloc"))]
impl<E: Vector> core::fmt::Debug for VectorEmaState<E> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("VectorEmaState")
      .field("alpha", &self.alpha)
      .field("seeded", &!self.state.is_empty())
      .field("dim", &self.state.len())
      .finish()
  }
}

// `C: Real` on this helper and the two below it is the narrowest bound the crate
// offers, not a convenience reach for the whole numeric surface: `ZERO`, `ONE`,
// `EPSILON`, `MIN_GATE_THRESHOLD`, `abs`, and `from_f32` are declared on `Real`
// and on nothing narrower, and `l2_norm`/`l2_renorm` demand it in turn. The type
// parameter is load-bearing too — `ComputeOf<E>` reaches these as an
// unnormalized projection, so a monomorphic `f64` signature would not accept the
// buffers even though `Real` is sealed to `f64` alone.

/// Fit `buf` to exactly `dim` zeroed components, reusing whatever capacity a
/// previous epoch left.
///
/// `try_reserve_exact` rather than `resize` alone: the dimension comes from a
/// caller's embedding and need not correspond to memory that exists, so a
/// refused allocation must surface as [`WinditError::AllocFailed`] rather than
/// abort the process — the same typed-OOM discipline every `Result`-returning
/// path in this crate keeps.
#[cfg(any(feature = "std", feature = "alloc"))]
fn refit<C: Real>(buf: &mut Vec<C>, dim: usize) -> Result<(), WinditError> {
  buf.clear();
  buf
    .try_reserve_exact(dim)
    .map_err(|_| WinditError::AllocFailed { elements: dim })?;
  buf.resize(dim, C::ZERO);
  Ok(())
}

/// Advance a seeded accumulator by one window, advancing alongside it the mass
/// the determinacy gate measures the result against.
///
/// The two products are formed once and used twice — summed into the
/// accumulator and, by magnitude, into `mag` — so the mass charged for this step
/// is exactly the mass the step folded, not a re-derivation that could disagree
/// with it. `mag` then *accumulates* rather than being overwritten:
///
/// ```text
/// M_t = |alpha * x_t| + |(1 - alpha) * s_{t-1}| + (1 - alpha) * M_{t-1}
/// ```
///
/// — the same recurrence, damped by the same coefficient, as the accumulator
/// itself. That third term is the whole difference between a mass that measures
/// one *step* and one that measures a *recurrence*. Writing `e_t` for the
/// accumulator's distance from the exact combination of the prefix, one step
/// gives
///
/// ```text
/// e_t = (1 - alpha) * e_{t-1} + eta_t,  |eta_t| <= 2u * (|alpha * x_t| + |(1 - alpha) * s_{t-1}|)
/// ```
///
/// (`u = EPSILON / 2`, over the two products and their sum). The rounding a step
/// commits is therefore *carried forward* under the same decay the value is, and
/// `M` is exactly the accumulator that dominates it: `|e_t| <= 2u * M_t`
/// follows by induction, componentwise, because the same `(1 - alpha)` damps
/// both. An overwrite — `|alpha * x_t| + |(1 - alpha) * s_{t-1}|` alone — takes
/// the measure of the step just taken, and when earlier windows cancelled that
/// is far below the rounding they left behind: the collapsed `|s_{t-1}|` no
/// longer knows what it collapsed from.
///
/// `M_0` is zero — [`Smoother::push`] seeds `s_0 = x_0` by copy, which commits no
/// rounding — and from `M_1` on, `M_t` also dominates componentwise the mass
/// [`EmaRenormalized`](crate::aggregate::EmaRenormalized) accumulates over the
/// same prefix (`alpha * |x_t| + (1 - alpha) * M^agg_{t-1}`), by the same
/// induction from an `M_1` the two share exactly — so of the two siblings'
/// thresholds this one is never the smaller.
#[cfg(any(feature = "std", feature = "alloc"))]
fn ema_step<C: Real>(state: &mut [C], mag: &mut [C], x: &[C], alpha: C) {
  let complement = C::ONE - alpha;
  for ((s, m), &xj) in state.iter_mut().zip(mag.iter_mut()).zip(x.iter()) {
    let recent = alpha * xj;
    let carried = complement * *s;
    *s = recent + carried;
    *m = recent.abs() + carried.abs() + complement * *m;
  }
}

/// Gate `state` against its own rounding floor and write its unit direction
/// into `unit`.
///
/// The threshold's shape and constant are the aggregation half's, and so are the
/// [`l2_norm`]/[`l2_renorm`] it is computed and applied with, so "renormalized"
/// means the same arithmetic on both sides of the shape boundary. What differs
/// is the mass: `M` here is [`ema_step`]'s propagated one, which dominates the
/// fold's — see there for why a recurrence needs it.
///
/// Neither norm can overflow, because `push` confines every input component to
/// the aggregation magnitude domain before the accumulator is written: `state`
/// is a convex combination of in-domain components and so is bounded by
/// [`MAX_AGG_MAGNITUDE`](Real::MAX_AGG_MAGNITUDE), and `M` by that over
/// `alpha` — at most `2^400 / 2^-149` for any nonzero `f32` `alpha`, and
/// `(t + 1) * 2^400` when `alpha` is zero. Both are far inside `f64`.
///
/// # Errors
///
/// [`WinditError::NonFinite`] when the accumulator is at or below
/// `16 * EPSILON * ||M|| + MIN_GATE_THRESHOLD` — no direction determined at
/// working precision — or when it cannot be normalized to a finite unit vector.
#[cfg(any(feature = "std", feature = "alloc"))]
fn gate_and_renorm<C: Real>(state: &[C], mag: &[C], unit: &mut [C]) -> Result<(), WinditError> {
  let tau = C::from_f32(16.0) * C::EPSILON * l2_norm(mag) + C::MIN_GATE_THRESHOLD;
  if l2_norm(state) <= tau {
    return Err(WinditError::NonFinite);
  }
  unit.copy_from_slice(state);
  l2_renorm(unit)
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<E: Vector> Smoother<E> for VectorEmaState<E> {
  fn push(&mut self, w: Windowed<E>) -> Result<Windowed<E>, WinditError> {
    let span = w.span();
    // Projected through the embedding's own value surface, exactly as
    // `aggregate` does: a zero-copy borrow for `f64` storage, an exact widening
    // for `f32`/`f16`/`bf16`, the implementor's dequantization for quantized
    // storage — and a refusal (`MissingDequantization`) for raw codes. Reading
    // `as_slice` directly would be one line shorter and would silently fold
    // quantization codes as if they were values.
    let projected = w.value().compute_components()?;
    let x = projected.as_ref();

    // Every rejection below runs before a single component is written, so a
    // refused push leaves the epoch exactly as it found it.
    if x.is_empty() {
      return Err(WinditError::Empty);
    }
    if !self.state.is_empty() && x.len() != self.state.len() {
      return Err(WinditError::DimMismatch {
        got: x.len(),
        expected: self.state.len(),
      });
    }
    if x.iter().any(|&c| !c.is_finite()) {
      return Err(WinditError::NonFinite);
    }
    // The aggregation input domain, enforced here for the same reason it exists
    // there: the determinacy gate's mass is an `n`-term geometric fold over the
    // whole epoch, and an unbounded one leaves `f64` (module note *Input
    // domain*). `window` is always `0` — a smoother's push carries exactly one
    // window — which is also the index the one-window fold reports.
    for (component, &c) in x.iter().enumerate() {
      if c != <ComputeOf<E> as Real>::ZERO
        && (c.abs() < <ComputeOf<E> as Real>::MIN_AGG_MAGNITUDE
          || c.abs() > <ComputeOf<E> as Real>::MAX_AGG_MAGNITUDE)
      {
        return Err(WinditError::MagnitudeOutOfRange {
          window: 0,
          component,
        });
      }
    }

    if self.state.is_empty() {
      // First push of an epoch seeds `s_0 = x_0` — a copy, not arithmetic, so it
      // rounds by nothing and the gate's mass starts at zero rather than at
      // `|x_0|`. `refit` zeroes, so that seed is the absence of a write. Being
      // exact is also what makes the first two windows' thresholds identical to
      // the aggregate's: `M_1` works out to `|alpha * x_1| + |(1 - alpha) * x_0|`,
      // which is the one-window fold's mass carried forward. The seed is still
      // gated — an all-zero window has no direction — through the absolute
      // `MIN_GATE_THRESHOLD` floor alone, which is the whole of a threshold with
      // no rounding to allow for.
      //
      // Scratch buffers first and the accumulator last: an empty accumulator is
      // what "unseeded" means, so a refused allocation leaves the smoother
      // unseeded rather than half-armed.
      refit(&mut self.mag, x.len())?;
      refit(&mut self.unit, x.len())?;
      refit(&mut self.state, x.len())?;
      self.state.copy_from_slice(x);
    } else {
      ema_step(
        &mut self.state,
        &mut self.mag,
        x,
        <ComputeOf<E> as Real>::from_f32(self.alpha),
      );
    }

    gate_and_renorm(&self.state, &self.mag, &mut self.unit)?;
    Ok(Windowed::new(E::from_unnormalized(&self.unit)?, span))
  }

  fn reset(&mut self) {
    // Clearing rather than dropping restores the unseeded state — an empty
    // accumulator is exactly that — while keeping the buffers a previous epoch
    // grew, so re-seeding after a `discontinuity` allocates nothing. Capacity
    // is not observable through the public API, so this is the
    // freshly-constructed state. `discontinuity` is the trait default (=
    // `reset`): a 1-in/1-out filter holds no pending output.
    self.state.clear();
    self.mag.clear();
    self.unit.clear();
  }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<E: Vector> SmoothPolicy<E> for VectorEma {
  type Smoother = VectorEmaState<E>;

  fn smoother(&self) -> VectorEmaState<E> {
    // `VectorEma::new` is the only way to set `alpha` and clamps it there, so
    // this re-clamp is currently unreachable — kept, exactly as `Ema`'s is, as
    // the last line of defence for the recurrence's coefficient invariant
    // (`alpha` in `[0, 1]` and never NaN) against any future construction path
    // that bypasses `new`. NaN is handled explicitly since `f32::clamp` would
    // propagate it.
    let alpha = if self.alpha.is_nan() {
      0.0
    } else {
      self.alpha.clamp(0.0, 1.0)
    };
    // No allocation here: the trait's factory is infallible, so the buffers are
    // grown on the first push, which is not.
    VectorEmaState {
      alpha,
      state: Vec::new(),
      mag: Vec::new(),
      unit: Vec::new(),
    }
  }
}

/// Forwarding so a boxed smoother is itself a [`Smoother`], letting a
/// run-time-selected `Box<dyn Smoother<V>>` be *held* as a stage — not merely
/// called through auto-deref.
///
/// `?Sized` covers `Box<dyn Smoother<V>>` and `Box<Concrete>` alike; the
/// coverage is conventional, mirroring std's `Box<impl Iterator>`. Policies stay
/// non-boxed — configs are `Copy`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
impl<V, T: Smoother<V> + ?Sized> Smoother<V> for std::boxed::Box<T> {
  fn push(&mut self, w: Windowed<V>) -> Result<Windowed<V>, WinditError> {
    (**self).push(w)
  }

  fn reset(&mut self) {
    (**self).reset();
  }

  // Forwarded explicitly, not left to the trait default: the default would route
  // this box's `discontinuity` to the box's own `reset`, silently erasing any
  // `discontinuity` override the concrete stage carries.
  fn discontinuity(&mut self) {
    (**self).discontinuity();
  }
}
