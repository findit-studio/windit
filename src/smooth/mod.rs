//! Smoothing policies: map a windowed sequence to a smoothed sequence.
//!
//! A `SmoothPolicy` rewrites the value of each `Windowed<V>` while leaving
//! its [`Span`](crate::plan::Span) untouched, so the output stays aligned with
//! the input windows. The built-ins operate on `V = f32`:
//!
//! - `Ema` is an exponential moving average (temporal low-pass).
//! - `Hysteresis` is the latching two-threshold gate used for binary VAD
//!   smoothing, generalized to any f32 score sequence.
//!
//! Each policy restarts its state on every call — these are batch conveniences,
//! not incremental decoders.

use std::vec::Vec;

use crate::windowed::Windowed;

#[cfg(test)]
mod tests;

/// A policy that smooths a windowed value sequence, preserving each span.
///
/// Generic over the value type `V`; the shipped built-ins implement it for
/// `V = f32`. An implementor that carries values through unchanged (rather
/// than computing new ones, as [`Ema`] and [`Hysteresis`] do) declares its own
/// `V: Clone` bound on its `impl`.
///
/// Each call starts from fresh policy state: smoothing a sequence chunk by chunk
/// is not equivalent to one whole-sequence call, because a running average or
/// latch does not carry across calls. These policies are batch conveniences, not
/// incremental decoders.
pub trait SmoothPolicy<V> {
  /// Return a smoothed sequence the same length as `seq`, each element keeping
  /// its input [`Span`](crate::plan::Span).
  fn smooth(&self, seq: &[Windowed<V>]) -> Vec<Windowed<V>>;
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
/// enters the recurrence and poisons the state for the remainder of the call —
/// every output from that index on is non-finite. A `NaN` stays `NaN`; an
/// infinity propagates as that infinity until a zero coefficient multiplies it
/// (`0.0 * inf = NaN`, so `alpha = 1` degrades an infinite state to `NaN` one
/// step later, while `alpha = 0` degrades an infinite input to `NaN` at its own
/// index) or opposite infinities meet (`inf - inf = NaN`). In particular,
/// `alpha = 0` holds the seed only against finite inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ema {
  alpha: f32,
}

impl Ema {
  /// An exponential moving average with the given smoothing factor, clamped
  /// into `[0, 1]`.
  ///
  /// Clamping at construction is what keeps the infallible [`SmoothPolicy`] path
  /// total: above `1.0` clamps to `1.0`, below `0.0` clamps to `0.0`, and a NaN
  /// becomes `0.0` (hold the seed). [`alpha`](Ema::alpha) reports the clamped
  /// value the recurrence actually uses.
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

/// Latching two-threshold gate producing a binary `0.0` / `1.0` sequence.
///
/// The gate turns on when a value rises to `on` or above, turns off when a value
/// falls strictly below `off`, and otherwise holds its previous state — so a
/// value exactly at `off` holds rather than turning off. It starts off. Configure
/// `on >= off`; the half-open band `off <= value < on` is the hold region that
/// suppresses chatter. If misconfigured with `on < off`, the turn-on test is
/// evaluated first and wins, so the gate degrades to a single threshold at `on`.
/// This is the binary VAD smoothing generalized to any f32 score.
///
/// Comparisons are IEEE, and every comparison with `NaN` is false, so: a `NaN`
/// score holds the previous state (including the initial off state); `+inf`
/// activates whenever `on` is not `NaN`; `-inf` releases unless `on` is `-inf`
/// (activation wins) or `off` is `NaN` or `-inf` (then it holds — `off = -inf`
/// can never release, since no value is `< -inf`); a `NaN` `on` can never
/// activate, so the output is all `0.0`; and a `NaN` `off` never releases once
/// on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hysteresis {
  on: f32,
  off: f32,
}

impl Hysteresis {
  /// A gate that latches on at `>= on` and off at `< off`.
  ///
  /// Configure `on >= off`; the band `off <= value < on` is the hold region, so
  /// a value exactly at `off` holds rather than turning off. An `on < off` pair
  /// is deliberately not rejected: the type documents the single-threshold
  /// behaviour it degrades to, which is defined and deterministic rather than
  /// invalid.
  #[must_use]
  pub const fn new(on: f32, off: f32) -> Self {
    Self { on, off }
  }

  /// The turn-on threshold: a value `>= on` latches the gate on.
  #[must_use]
  pub const fn on(&self) -> f32 {
    self.on
  }

  /// The turn-off threshold: a value `< off` latches the gate off; a value
  /// exactly at `off` holds instead.
  #[must_use]
  pub const fn off(&self) -> f32 {
    self.off
  }

  /// Advance the latch by one value and return the resulting state: on at
  /// `>= on`, off strictly below `off`, hold otherwise (so a value exactly at
  /// `off`, and any NaN, holds the previous state). The turn-on test is
  /// evaluated first and wins.
  ///
  /// The single definition of the gate transition: [`Hysteresis::smooth`] and
  /// the fused `segment::HysteresisSegment` both step through it, so the two
  /// paths cannot drift apart.
  pub(crate) const fn step(&self, active: bool, value: f32) -> bool {
    if value >= self.on {
      true
    } else if value < self.off {
      false
    } else {
      active
    }
  }
}

impl SmoothPolicy<f32> for Ema {
  fn smooth(&self, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> {
    // `Ema::new` is the only way to set `alpha` and clamps it there, so this
    // re-clamp is currently unreachable. It is kept as the last line of defence
    // for the recurrence's coefficient invariant — `alpha` in `[0, 1]` and never
    // NaN — because that invariant must survive any future construction path that
    // bypasses `new`, a serde derive on this type being the obvious one. It
    // guards the coefficients only: a non-finite input value still propagates, as
    // the type docs specify. NaN is handled explicitly since `f32::clamp` would
    // propagate it.
    let alpha = if self.alpha.is_nan() {
      0.0
    } else {
      self.alpha.clamp(0.0, 1.0)
    };
    let mut out = Vec::with_capacity(seq.len());
    let mut state = 0.0f32;
    for (i, w) in seq.iter().enumerate() {
      state = if i == 0 {
        w.value
      } else {
        alpha * w.value + (1.0 - alpha) * state
      };
      out.push(Windowed::new(state, w.span));
    }
    out
  }
}

impl SmoothPolicy<f32> for Hysteresis {
  fn smooth(&self, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> {
    let mut out = Vec::with_capacity(seq.len());
    let mut on = false;
    for w in seq {
      on = self.step(on, w.value);
      out.push(Windowed::new(if on { 1.0 } else { 0.0 }, w.span));
    }
    out
  }
}
