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
//!   generic over any `V: Clone`.
//! - [`Ema`] is an exponential moving average (temporal low-pass) over `f32`.
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
  /// Returns a [`WinditError`] for a stage that reads spans out of order; the
  /// shipped smoothers ([`Identity`], [`Ema`]) are infallible and always return
  /// `Ok`, returning `Result` only for uniformity with the composable stages.
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
/// `V = f32` ([`Ema`]) or any `V: Clone` ([`Identity`]). Implement the factory
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
/// quality claim. Generic over any `V: Clone`, so it is the identity stage for
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

impl<V: Clone> Smoother<V> for IdentityState {
  fn push(&mut self, w: Windowed<V>) -> Result<Windowed<V>, WinditError> {
    Ok(w)
  }

  fn reset(&mut self) {}
}

impl<V: Clone> SmoothPolicy<V> for Identity {
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
