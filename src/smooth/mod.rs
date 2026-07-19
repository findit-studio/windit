//! Smoothing policies: map a windowed sequence to a smoothed sequence.
//!
//! A `SmoothPolicy` rewrites the value of each `Windowed<V>` while leaving
//! its [`Span`](crate::plan::Span) untouched, so the output stays aligned with
//! the input windows. The built-ins operate on `V = f32`:
//!
//! - `Ema` is an exponential moving average (temporal low-pass).
//! - `Hysteresis` is the latching two-threshold gate used for binary VAD
//!   smoothing, generalized to any f32 score sequence.

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
pub trait SmoothPolicy<V> {
  /// Return a smoothed sequence the same length as `seq`, each element keeping
  /// its input [`Span`](crate::plan::Span).
  fn smooth(&self, seq: &[Windowed<V>]) -> Vec<Windowed<V>>;
}

/// Exponential moving average: `s_t = alpha * x_t + (1 - alpha) * s_{t-1}`.
///
/// Seeded with `s_0 = x_0`. A larger `alpha` tracks the input more closely; a
/// smaller one smooths harder. This policy is infallible, so `alpha` is clamped
/// into `[0, 1]` deterministically: a non-finite (NaN) `alpha` clamps to `0.0`
/// (hold the seed). With a clamped alpha and finite inputs, the recurrence
/// introduces no NaN. `Ema` does not sanitize inputs, though: a non-finite
/// (`NaN`/infinite) input value still propagates through the recurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ema {
  /// The smoothing factor; clamped into `[0, 1]` (a NaN clamps to `0.0`).
  pub alpha: f32,
}

/// Latching two-threshold gate producing a binary `0.0` / `1.0` sequence.
///
/// The gate turns on when a value rises to `on` or above, turns off when a value
/// falls to `off` or below, and otherwise holds its previous state. It starts
/// off. Configure `on >= off`; the band between them is the hold region that
/// suppresses chatter. If misconfigured with `on < off`, the turn-on test is
/// evaluated first and wins, so the gate degrades to a single threshold at `on`.
/// This is the binary VAD smoothing generalized to any f32 score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hysteresis {
  /// The turn-on threshold: a value `>= on` latches the gate on.
  pub on: f32,
  /// The turn-off threshold: a value `<= off` latches the gate off.
  pub off: f32,
}

impl SmoothPolicy<f32> for Ema {
  fn smooth(&self, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> {
    // Clamp alpha into [0, 1] deterministically. NaN is handled explicitly
    // (mapped to 0.0, "hold the seed") because `f32::clamp` would propagate it;
    // this infallible policy must never leak a NaN into the output.
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
      if w.value >= self.on {
        on = true;
      } else if w.value <= self.off {
        on = false;
      }
      out.push(Windowed::new(if on { 1.0 } else { 0.0 }, w.span));
    }
    out
  }
}
