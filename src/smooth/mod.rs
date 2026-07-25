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
//!
//! The state traits and states allocate nothing and live in the featureless core
//! tier; only the `Vec`-returning batch driver gates on `alloc`.
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
/// `V = f32` ([`Ema`]) or any `V` ([`Identity`]). Implement the factory
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
  /// - Any error the underlying [`Smoother::push`] surfaces (none for the shipped
  ///   built-ins).
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
/// smaller one smooths harder. This policy is infallible, so [`Ema::new`] clamps
/// `alpha` into `[0, 1]` deterministically: a non-finite (NaN) `alpha` clamps to
/// `0.0` (hold the seed). With a clamped alpha and finite inputs, the recurrence
/// introduces no NaN.
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
  // `delta as f32` rounds above 2^24, but harmlessly: the cast is correct to a
  // relative 2^-24, so the ratio — and with it the coefficient — lands within
  // an ulp of the exact one at every `tau`. `expm1f` returns no NaN for the
  // non-positive finite arguments produced here, so with a
  // constructor-validated `tau` the coefficient stays in `[0, 1]`.
  -libm::expm1f(-(delta as f32) / tau)
}

/// Cadence-portable exponential moving average: an EMA whose time constant is
/// denominated in input elements, not in pushes.
///
/// Each push derives its own coefficient from the *actual* span distance,
/// `alpha = 1 - exp(-delta / tau)` where `delta` is the gap between this span's
/// start and the previous one — the exact zero-order-hold discretization of a
/// first-order low-pass. The recurrence is then the ordinary EMA
/// `s = alpha * x + (1 - alpha) * s_prev`, seeded `s_0 = x_0` on the first push.
/// Because the coefficient tracks the cadence, one `tau` yields the same
/// smoothing at any hop — regular or irregular — where a bare per-step [`Ema`]
/// `alpha` does not: the configuration carries no cadence, the data does.
/// (Lineage: the LiveKit EMA, made cadence-portable.)
///
/// `tau` is a positive, finite element count; [`new`](CadenceEma::new) and
/// [`try_new`](CadenceEma::try_new) reject anything else, because there is no
/// sane clamp target — `tau = 0` would make an equal-start step compute `0/0`. A
/// caller working in another unit converts at the boundary
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
/// - **Fine cadences keep their coefficient:** the coefficient is derived in a
///   form that stays exact as `delta / tau` shrinks, so it never rounds to zero
///   and the smoothed result is invariant to how finely the same signal is
///   sampled — down to ratios far below f32's epsilon.
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
  /// A cadence-portable EMA with the given element-denominated time constant.
  ///
  /// # Panics
  ///
  /// Panics, in every build, unless `tau` is finite and strictly positive. Use
  /// [`try_new`](CadenceEma::try_new) to handle an untrusted `tau` instead.
  #[must_use]
  pub const fn new(tau: f32) -> Self {
    match Self::try_new(tau) {
      Ok(cfg) => cfg,
      Err(_) => panic!("a cadence time constant tau must be finite and positive"),
    }
  }

  /// The checked counterpart of [`new`](CadenceEma::new): validate `tau` rather
  /// than panic on it.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::TimeConstantOutOfRange`] if `tau` is `NaN`, infinite,
  /// zero, or negative. A non-positive or non-finite time constant has no sane
  /// clamp target, so rejection is the only honest total answer.
  pub const fn try_new(tau: f32) -> Result<Self, WinditError> {
    if tau.is_finite() && tau > 0.0 {
      Ok(Self { tau })
    } else {
      Err(WinditError::TimeConstantOutOfRange)
    }
  }

  /// The element-denominated time constant, always finite and positive.
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
  /// `tau`.
  #[must_use]
  pub fn alpha_for(&self, delta: usize) -> f32 {
    cadence_alpha(self.tau, delta)
  }
}

/// The streaming state of a [`CadenceEma`]: the time constant and the previous
/// `(span start, smoothed value)`, unseeded until the first push.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadenceEmaState {
  tau: f32,
  prev: Option<(usize, f32)>,
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
        self.prev = Some((start, x));
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
        let alpha = cadence_alpha(self.tau, delta);
        let s = alpha * x + (1.0 - alpha) * prev_val;
        self.prev = Some((start, s));
        Ok(Windowed::new(s, span))
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
    // `tau` has none — there is no in-range substitute for a non-positive or
    // non-finite time constant — so a `tau` that bypassed `new`/`try_new` (a
    // future serde derive, say) is carried as-is and degrades to `NaN` outputs
    // at worst, never a panic and never UB, because `alpha` is derived per push
    // and the recurrence is total for any stored `tau`.
    CadenceEmaState {
      tau: self.tau,
      prev: None,
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
