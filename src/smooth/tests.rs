use std::{vec, vec::Vec};

use super::{Ema, Hysteresis, SmoothPolicy};
use crate::{plan::Span, windowed::Windowed};

/// One `Windowed<f32>` per value, each covering a single element (window 1).
fn seq(values: &[f32]) -> Vec<Windowed<f32>> {
  values
    .iter()
    .enumerate()
    .map(|(i, &v)| Windowed::new(v, Span::new(i, 1, 1)))
    .collect()
}

fn values(seq: &[Windowed<f32>]) -> Vec<f32> {
  seq.iter().map(|w| w.value).collect()
}

fn spans(seq: &[Windowed<f32>]) -> Vec<Span> {
  seq.iter().map(|w| w.span).collect()
}

/// Elementwise `f32` equality where `NaN == NaN`, so the non-finite goldens
/// below can be pinned exactly. The crate's `assert_close` compares within a
/// tolerance and treats every `NaN` as unequal, so it cannot express these.
fn assert_f32_seq(got: &[f32], want: &[f32]) {
  assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
  for (g, w) in got.iter().zip(want) {
    assert!(
      g == w || (g.is_nan() && w.is_nan()),
      "elementwise mismatch: {got:?} vs {want:?}"
    );
  }
}

#[test]
fn ema_pinned_and_preserves_spans() {
  // alpha 0.5, s_0 = x_0: 0, 0.5*1, 0.5*1 + 0.5*0.5, 0.5*0.75 -> exact dyadics.
  let input = seq(&[0.0, 1.0, 1.0, 0.0]);
  let out = Ema::new(0.5).smooth(&input);
  assert_eq!(values(&out), vec![0.0, 0.5, 0.75, 0.375]);
  assert_eq!(spans(&out), spans(&input));
}

#[test]
fn hysteresis_latches_and_holds() {
  // on=0.6, off=0.3: 0.1 off, 0.7 on, 0.5 hold(on), 0.2 off, 0.6 on.
  let input = seq(&[0.1, 0.7, 0.5, 0.2, 0.6]);
  let out = Hysteresis::new(0.6, 0.3).smooth(&input);
  assert_eq!(values(&out), vec![0.0, 1.0, 1.0, 0.0, 1.0]);
  assert_eq!(spans(&out), spans(&input));
}

#[test]
fn hysteresis_holds_at_off_boundary_instead_of_turning_off() {
  // on=0.6, off=0.3: 0.7 latches on; a value exactly at `off` (0.3) must hold
  // that on state rather than turn off, twice in a row to show it is stable
  // and not a one-step fluke. Only the strictly-below value 0.2 turns it off.
  // This is the strict-below boundary this type's real VAD consumers rely on.
  let input = seq(&[0.7, 0.3, 0.3, 0.2]);
  let out = Hysteresis::new(0.6, 0.3).smooth(&input);
  assert_eq!(values(&out), vec![1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn ema_nan_alpha_clamps_to_hold_seed() {
  // alpha 2.0 clamps to 1.0 (track the input exactly): s_t = x_t.
  let input = seq(&[0.5, 0.8]);
  let out = Ema::new(2.0).smooth(&input);
  assert_eq!(values(&out), vec![0.5, 0.8]);

  // A NaN alpha clamps to 0.0 (hold the seed). With finite inputs the recurrence
  // then holds the seed and every output is finite. This is a statement about the
  // clamped coefficient, not the inputs: a non-finite input still poisons the
  // state (see `ema_nan_input_poisons_the_rest_of_the_call`).
  let out = Ema::new(f32::NAN).smooth(&input);
  assert!(
    out.iter().all(|w| w.value.is_finite()),
    "outputs must be finite"
  );
  assert_eq!(values(&out), vec![0.5, 0.5]);
}

#[test]
fn ema_nan_input_poisons_the_rest_of_the_call() {
  // A NaN score enters the recurrence and, because non-finite absorbs under
  // `+`/`*`, every later output is NaN — for every alpha. Here alpha 0.5 mixes
  // NaN into the state at index 1 and it never washes out.
  let out = Ema::new(0.5).smooth(&seq(&[0.2, f32::NAN, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);

  // A NaN first value seeds the state directly, poisoning from index 0.
  let out = Ema::new(0.5).smooth(&seq(&[f32::NAN, 0.5]));
  assert_f32_seq(&values(&out), &[f32::NAN, f32::NAN]);
}

#[test]
fn ema_infinite_input_saturates_and_zero_coefficients_degrade_to_nan() {
  // With both coefficients nonzero (alpha 0.5), an infinity propagates as that
  // infinity...
  let out = Ema::new(0.5).smooth(&seq(&[0.2, f32::INFINITY, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::INFINITY]);
  let out = Ema::new(0.5).smooth(&seq(&[0.2, f32::NEG_INFINITY, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::NEG_INFINITY, f32::NEG_INFINITY]);

  // ...until opposite infinities meet: `+inf` state, then `0.5*(-inf)` gives
  // `-inf`, and `-inf + (+inf) = NaN`.
  let out = Ema::new(0.5).smooth(&seq(&[0.2, f32::INFINITY, f32::NEG_INFINITY]));
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::NAN]);

  // alpha 1.0 zeroes `1 - alpha`, so the next step computes `1.0*0.8 + 0.0*inf`
  // and `0.0 * inf = NaN` degrades the infinity one step later.
  let out = Ema::new(1.0).smooth(&seq(&[0.2, f32::INFINITY, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::NAN]);

  // alpha 0.0 zeroes `alpha`, so `0.0 * x` degrades a non-finite input to NaN
  // immediately (both an infinity and a NaN), and it then absorbs the rest.
  let out = Ema::new(0.0).smooth(&seq(&[0.2, f32::INFINITY, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);
  let out = Ema::new(0.0).smooth(&seq(&[0.2, f32::NAN, 0.8]));
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);
}

#[test]
fn hysteresis_nan_score_holds_state() {
  // Both gate comparisons are false for NaN, so a NaN score holds the current
  // state — including the initial off state at index 0.
  let out = Hysteresis::new(0.6, 0.3).smooth(&seq(&[f32::NAN, 0.7, f32::NAN, 0.2, f32::NAN]));
  assert_eq!(values(&out), vec![0.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn hysteresis_infinite_scores_latch_and_release() {
  // `+inf >= on` activates; `-inf < off` releases; the finite holds between.
  let out = Hysteresis::new(0.6, 0.3).smooth(&seq(&[f32::INFINITY, 0.5, f32::NEG_INFINITY, 0.5]));
  assert_eq!(values(&out), vec![1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn hysteresis_nan_thresholds_fail_closed_or_never_release() {
  // `on = NaN`: `value >= on` is never true, so the gate can never activate.
  let out = Hysteresis::new(f32::NAN, 0.3).smooth(&seq(&[0.7, f32::INFINITY, 0.1]));
  assert_eq!(values(&out), vec![0.0, 0.0, 0.0]);

  // `off = NaN`: `value < off` is never true, so once on the gate never releases.
  let out = Hysteresis::new(0.6, f32::NAN).smooth(&seq(&[0.7, 0.1, f32::NEG_INFINITY]));
  assert_eq!(values(&out), vec![1.0, 1.0, 1.0]);
}

#[test]
fn hysteresis_infinite_thresholds() {
  // `on = -inf`: every non-NaN score activates (`-inf` itself via `-inf >= -inf`,
  // and a NaN in between holds the on state).
  let out =
    Hysteresis::new(f32::NEG_INFINITY, 0.3).smooth(&seq(&[f32::NEG_INFINITY, f32::NAN, 0.1]));
  assert_eq!(values(&out), vec![1.0, 1.0, 1.0]);

  // `on = +inf`: only a `+inf` score activates; the finite below `off` releases.
  let out = Hysteresis::new(f32::INFINITY, 0.3).smooth(&seq(&[1.0, f32::INFINITY, 0.5, 0.2]));
  assert_eq!(values(&out), vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn hysteresis_exact_on_boundary_activates() {
  // The turn-on test is `value >= on`, so a value exactly at `on` activates.
  // Complements `hysteresis_holds_at_off_boundary_instead_of_turning_off`.
  let out = Hysteresis::new(0.6, 0.3).smooth(&seq(&[0.6]));
  assert_eq!(values(&out), vec![1.0]);
}

#[test]
fn hysteresis_on_below_off_degrades_to_single_threshold() {
  // With `on < off` the turn-on test is evaluated first and wins, so the gate
  // degrades to the pointwise single threshold `value >= on` (here 0.3): 0.4 on,
  // 0.5 on, 0.2 off, 0.35 on — no memory of the previous state survives.
  let out = Hysteresis::new(0.3, 0.6).smooth(&seq(&[0.4, 0.5, 0.2, 0.35]));
  assert_eq!(values(&out), vec![1.0, 1.0, 0.0, 1.0]);
}

#[test]
fn ema_new_clamps_alpha_into_range() {
  // The clamp is applied at construction, so the accessor reports the factor the
  // recurrence actually uses rather than the one that was asked for.
  assert_eq!(Ema::new(0.5).alpha(), 0.5);
  assert_eq!(Ema::new(2.0).alpha(), 1.0);
  assert_eq!(Ema::new(-0.5).alpha(), 0.0);
  assert_eq!(Ema::new(f32::NAN).alpha(), 0.0);
  assert_eq!(Ema::new(f32::INFINITY).alpha(), 1.0);
  assert_eq!(Ema::new(f32::NEG_INFINITY).alpha(), 0.0);
}

#[test]
fn hysteresis_new_exposes_thresholds() {
  let gate = Hysteresis::new(0.6, 0.3);
  assert_eq!((gate.on(), gate.off()), (0.6, 0.3));
}

#[test]
fn empty_input_yields_empty() {
  let input: Vec<Windowed<f32>> = Vec::new();
  assert!(Ema::new(0.5).smooth(&input).is_empty());
  assert!(Hysteresis::new(0.6, 0.3).smooth(&input).is_empty());
}
