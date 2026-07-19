//! Smoothing policies: map a windowed sequence to a smoothed sequence.
//!
//! A [`SmoothPolicy`] rewrites the value of each [`Windowed<V>`] while leaving
//! its [`Span`](crate::plan::Span) untouched, so the output stays aligned with
//! the input windows. The built-ins operate on `V = f32`:
//!
//! - [`Ema`] is an exponential moving average (temporal low-pass).
//! - [`Hysteresis`] is the latching two-threshold gate used for binary VAD
//!   smoothing, generalized to any f32 score sequence.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::windowed::Windowed;

#[cfg(all(test, feature = "alloc"))]
mod tests;

/// A policy that smooths a windowed value sequence, preserving each span.
///
/// Generic over the value type `V`; the shipped built-ins implement it for
/// `V = f32`. `V: Clone` is part of the contract so implementors that carry
/// values through unchanged can do so.
#[cfg(feature = "alloc")]
pub trait SmoothPolicy<V: Clone> {
  /// Return a smoothed sequence the same length as `seq`, each element keeping
  /// its input [`Span`](crate::plan::Span).
  fn smooth(&self, seq: &[Windowed<V>]) -> Vec<Windowed<V>>;
}

/// Exponential moving average: `s_t = alpha * x_t + (1 - alpha) * s_{t-1}`.
///
/// Seeded with `s_0 = x_0`. A larger `alpha` tracks the input more closely; a
/// smaller one smooths harder.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ema {
  /// The smoothing factor in `[0, 1]`.
  pub alpha: f32,
}

/// Latching two-threshold gate producing a binary `0.0` / `1.0` sequence.
///
/// The gate turns on when a value rises to `on` or above, turns off when a value
/// falls to `off` or below, and otherwise holds its previous state. It starts
/// off. Configure `on >= off`; the band between them is the hold region that
/// suppresses chatter. This is the binary VAD smoothing generalized to any f32
/// score.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hysteresis {
  /// The turn-on threshold: a value `>= on` latches the gate on.
  pub on: f32,
  /// The turn-off threshold: a value `<= off` latches the gate off.
  pub off: f32,
}

#[cfg(feature = "alloc")]
impl SmoothPolicy<f32> for Ema {
  fn smooth(&self, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> {
    let mut out = Vec::with_capacity(seq.len());
    let mut state = 0.0f32;
    for (i, w) in seq.iter().enumerate() {
      state = if i == 0 {
        w.value
      } else {
        self.alpha * w.value + (1.0 - self.alpha) * state
      };
      out.push(Windowed::new(state, w.span));
    }
    out
  }
}

#[cfg(feature = "alloc")]
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
