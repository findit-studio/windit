use std::{boxed::Box, vec, vec::Vec};

use super::{CadenceEma, Ema, Identity, SmoothPolicy, Smoother};
use crate::{error::WinditError, plan::Span, windowed::Windowed};

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

/// xorshift64 — deterministic and dependency-free; the seed must be nonzero.
fn xorshift(state: &mut u64) -> u64 {
  let mut x = *state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *state = x;
  x
}

/// A pseudo-random `f32` in `[0, 1)` from the generator's next 24 bits.
fn next_unit(state: &mut u64) -> f32 {
  (xorshift(state) >> 40) as f32 / (1u32 << 24) as f32
}

/// The retained 0.1.2 `Ema::smooth` recurrence, verbatim, as the independent
/// differential oracle for the reshaped [`Smoother`]-driven batch method. Takes
/// the already-clamped `alpha` (as [`Ema::alpha`] reports it) and re-clamps it
/// exactly as the shipped batch loop did.
fn ema_oracle(alpha: f32, seq: &[Windowed<f32>]) -> Vec<f32> {
  let alpha = if alpha.is_nan() {
    0.0
  } else {
    alpha.clamp(0.0, 1.0)
  };
  let mut out = Vec::with_capacity(seq.len());
  let mut state = 0.0f32;
  for (i, w) in seq.iter().enumerate() {
    state = if i == 0 {
      w.value
    } else {
      alpha * w.value + (1.0 - alpha) * state
    };
    out.push(state);
  }
  out
}

/// Drive a fresh smoother push-by-push over `seq`, the streaming counterpart of
/// the batch `smooth` method.
fn drive<P: SmoothPolicy<f32>>(policy: &P, seq: &[Windowed<f32>]) -> Vec<Windowed<f32>> {
  let mut s = policy.smoother();
  seq.iter().map(|w| s.push(*w).unwrap()).collect()
}

#[test]
fn ema_pinned_and_preserves_spans() {
  // alpha 0.5, s_0 = x_0: 0, 0.5*1, 0.5*1 + 0.5*0.5, 0.5*0.75 -> exact dyadics.
  let input = seq(&[0.0, 1.0, 1.0, 0.0]);
  let out = Ema::new(0.5).smooth(&input).unwrap();
  assert_eq!(values(&out), vec![0.0, 0.5, 0.75, 0.375]);
  assert_eq!(spans(&out), spans(&input));
}

#[test]
fn ema_batch_equals_streaming_drive() {
  // The batch `smooth` IS a fresh smoother driven over the slice, so the two must
  // agree value for value and span for span.
  let input = seq(&[0.2, 0.9, 0.4, 0.4, 0.1, 0.7]);
  let batch = Ema::new(0.3).smooth(&input).unwrap();
  let streamed = drive(&Ema::new(0.3), &input);
  assert_eq!(values(&batch), values(&streamed));
  assert_eq!(spans(&batch), spans(&streamed));
}

#[test]
fn ema_reset_and_discontinuity_reseed_the_state() {
  // Both return the smoother to `s_0 = x_0`: the value after re-seeding equals the
  // first pushed value, not the recurrence against the pre-break state.
  for reseed in [Smoother::reset, Smoother::discontinuity] {
    let mut s = Ema::new(0.5).smoother();
    let _ = s.push(Windowed::new(1.0, Span::new(0, 1, 1))).unwrap();
    let _ = s.push(Windowed::new(1.0, Span::new(1, 1, 1))).unwrap();
    reseed(&mut s);
    let after = s.push(Windowed::new(0.25, Span::new(2, 1, 1))).unwrap();
    assert_eq!(after.value, 0.25, "re-seed must restore s_0 = x_0");
  }
}

#[test]
fn identity_passes_values_and_spans_through() {
  let input = seq(&[0.1, 0.9, 0.4]);
  let out = Identity::new().smooth(&input).unwrap();
  assert_eq!(values(&out), values(&input));
  assert_eq!(spans(&out), spans(&input));

  // The streaming drive agrees with the batch method.
  assert_eq!(values(&drive(&Identity, &input)), values(&input));
}

#[test]
fn identity_is_generic_over_the_value_type() {
  // `Identity` smooths any `V`, not just `f32` — a genericity contract the
  // score-only smoothers do not carry.
  #[derive(Clone, Debug, PartialEq)]
  struct Payload(u32);
  let input: Vec<Windowed<Payload>> = (0..3)
    .map(|i| Windowed::new(Payload(i * 10), Span::new(i as usize, 1, 1)))
    .collect();
  let out = Identity.smooth(&input).unwrap();
  let got: Vec<Payload> = out.into_iter().map(Windowed::into_value).collect();
  assert_eq!(got, vec![Payload(0), Payload(10), Payload(20)]);
}

#[test]
fn identity_streaming_path_admits_a_non_clone_payload() {
  // Pins the loosened bound: `IdentityState::push` needs no `V: Clone`, unlike
  // the batch `smooth` convenience, which clones at its own method bound.
  #[derive(Debug, PartialEq)]
  struct NotClone(u32);
  let mut s = SmoothPolicy::<NotClone>::smoother(&Identity);
  let out = s
    .push(Windowed::new(NotClone(7), Span::new(0, 1, 1)))
    .unwrap();
  assert_eq!(out.into_value(), NotClone(7));
}

#[test]
fn ema_nan_alpha_clamps_to_hold_seed() {
  // alpha 2.0 clamps to 1.0 (track the input exactly): s_t = x_t.
  let input = seq(&[0.5, 0.8]);
  let out = Ema::new(2.0).smooth(&input).unwrap();
  assert_eq!(values(&out), vec![0.5, 0.8]);

  // A NaN alpha clamps to 0.0 (hold the seed). With finite inputs the recurrence
  // then holds the seed and every output is finite. This is a statement about the
  // clamped coefficient, not the inputs: a non-finite input still poisons the
  // state (see `ema_nan_input_poisons_the_rest_of_the_call`).
  let out = Ema::new(f32::NAN).smooth(&input).unwrap();
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
  let out = Ema::new(0.5).smooth(&seq(&[0.2, f32::NAN, 0.8])).unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);

  // A NaN first value seeds the state directly, poisoning from index 0.
  let out = Ema::new(0.5).smooth(&seq(&[f32::NAN, 0.5])).unwrap();
  assert_f32_seq(&values(&out), &[f32::NAN, f32::NAN]);
}

#[test]
fn ema_infinite_input_saturates_and_zero_coefficients_degrade_to_nan() {
  // With both coefficients nonzero (alpha 0.5), an infinity propagates as that
  // infinity...
  let out = Ema::new(0.5)
    .smooth(&seq(&[0.2, f32::INFINITY, 0.8]))
    .unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::INFINITY]);
  let out = Ema::new(0.5)
    .smooth(&seq(&[0.2, f32::NEG_INFINITY, 0.8]))
    .unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::NEG_INFINITY, f32::NEG_INFINITY]);

  // ...until opposite infinities meet: `+inf` state, then `0.5*(-inf)` gives
  // `-inf`, and `-inf + (+inf) = NaN`.
  let out = Ema::new(0.5)
    .smooth(&seq(&[0.2, f32::INFINITY, f32::NEG_INFINITY]))
    .unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::NAN]);

  // alpha 1.0 zeroes `1 - alpha`, so the next step computes `1.0*0.8 + 0.0*inf`
  // and `0.0 * inf = NaN` degrades the infinity one step later.
  let out = Ema::new(1.0)
    .smooth(&seq(&[0.2, f32::INFINITY, 0.8]))
    .unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::NAN]);

  // alpha 0.0 zeroes `alpha`, so `0.0 * x` degrades a non-finite input to NaN
  // immediately (both an infinity and a NaN), and it then absorbs the rest.
  let out = Ema::new(0.0)
    .smooth(&seq(&[0.2, f32::INFINITY, 0.8]))
    .unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);
  let out = Ema::new(0.0).smooth(&seq(&[0.2, f32::NAN, 0.8])).unwrap();
  assert_f32_seq(&values(&out), &[0.2, f32::NAN, f32::NAN]);
}

#[test]
fn ema_streaming_poisoning_persists_until_reset() {
  // In a stream a NaN poisons every later push until the state is re-seeded.
  let mut s = Ema::new(0.5).smoother();
  assert_eq!(
    s.push(Windowed::new(0.2, Span::new(0, 1, 1)))
      .unwrap()
      .value,
    0.2
  );
  assert!(s
    .push(Windowed::new(f32::NAN, Span::new(1, 1, 1)))
    .unwrap()
    .value
    .is_nan());
  assert!(s
    .push(Windowed::new(0.8, Span::new(2, 1, 1)))
    .unwrap()
    .value
    .is_nan());
  // A discontinuity re-seeds, so the next push starts a clean epoch.
  s.discontinuity();
  assert_eq!(
    s.push(Windowed::new(0.4, Span::new(0, 1, 1)))
      .unwrap()
      .value,
    0.4
  );
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
fn ema_matches_oracle_on_randomized_finite_and_non_finite_inputs() {
  // The reshaped batch `smooth` and a fresh streaming drive must both equal the
  // retained 0.1.2 recurrence, over randomized alphas and sequences with the IEEE
  // tables occasionally exercised.
  let mut state: u64 = 0xD1B5_4A32_D192_ED03;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 65) as usize;
    let alpha = match xorshift(&mut state) % 8 {
      0 => f32::NAN,
      1 => 2.0,
      2 => -1.0,
      _ => next_unit(&mut state),
    };
    let s: Vec<Windowed<f32>> = (0..n)
      .map(|i| {
        let v = match xorshift(&mut state) % 16 {
          0 => f32::NAN,
          1 => f32::INFINITY,
          2 => f32::NEG_INFINITY,
          _ => next_unit(&mut state),
        };
        Windowed::new(v, Span::new(i, 1, 1))
      })
      .collect();

    let ema = Ema::new(alpha);
    let batch = ema.smooth(&s).unwrap();
    let reference = ema_oracle(ema.alpha(), &s);
    assert_f32_seq(&values(&batch), &reference);
    // Batch equals the streaming drive by construction; pin it too.
    assert_f32_seq(&values(&drive(&ema, &s)), &reference);
    // Spans are preserved throughout.
    assert_eq!(spans(&batch), spans(&s));
  }
}

#[test]
fn empty_input_yields_empty() {
  let input: Vec<Windowed<f32>> = Vec::new();
  assert!(Ema::new(0.5).smooth(&input).unwrap().is_empty());
  assert!(Identity::new().smooth(&input).unwrap().is_empty());
}

// --- CadenceEma: element-time-constant smoother ---

/// Build a `Windowed<f32>` sequence from `(span start, value)` pairs, each a
/// unit span (window 1) — the cadence carrier for the element-time-constant
/// tests, where the span start is the timeline the coefficient reads.
fn cadence_seq(samples: &[(usize, f32)]) -> Vec<Windowed<f32>> {
  samples
    .iter()
    .map(|&(start, v)| Windowed::new(v, Span::new(start, 1, 1)))
    .collect()
}

/// The zero-order-hold recurrence in f64 — an independent-precision oracle for
/// `CadenceEma`'s f32 streaming path. Uses `libm::exp` (the f64 exponential)
/// rather than `f64::exp`, which is std-only, so it compiles on the bare-metal
/// `alloc`-only test build.
fn cadence_zoh_reference(tau: f64, samples: &[(usize, f32)]) -> Vec<f32> {
  let mut out = Vec::with_capacity(samples.len());
  let mut prev: Option<(usize, f64)> = None;
  for &(start, x) in samples {
    let s = match prev {
      None => f64::from(x),
      Some((prev_start, prev_val)) => {
        let delta = (start - prev_start) as f64;
        let alpha = 1.0 - libm::exp(-delta / tau);
        alpha * f64::from(x) + (1.0 - alpha) * prev_val
      }
    };
    prev = Some((start, s));
    out.push(s as f32);
  }
  out
}

/// Compare a smoothed sequence against a reference: within `tol` where the
/// reference is finite, exact NaN-aware equality where it is non-finite. The
/// poisoned tails must match a poisoned reference exactly, not merely be close.
fn assert_f32_seq_close(got: &[f32], want: &[f32], tol: f32) {
  assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
  for (i, (g, w)) in got.iter().zip(want).enumerate() {
    let ok = if w.is_finite() {
      (g - w).abs() <= tol
    } else {
      g == w || (g.is_nan() && w.is_nan())
    };
    assert!(
      ok,
      "index {i}: {g} vs {w} (tol {tol})\n{got:?}\nvs\n{want:?}"
    );
  }
}

#[test]
fn cadence_ema_batch_equals_streaming_drive() {
  // The batch `smooth` IS a fresh smoother driven over the slice — parity by
  // construction — even with irregular cadence and a duplicate start.
  let input = cadence_seq(&[
    (0, 0.2),
    (3, 0.9),
    (3, 0.4),
    (12, 0.4),
    (13, 0.1),
    (40, 0.7),
  ]);
  let cfg = CadenceEma::new(9.5);
  let batch = cfg.smooth(&input).unwrap();
  let streamed = drive(&cfg, &input);
  assert_eq!(values(&batch), values(&streamed));
  assert_eq!(spans(&batch), spans(&streamed));
}

#[test]
fn cadence_ema_probe_reversal_agrees_across_hops() {
  // The exact configuration that diverged for `Ema` (parent §4): a per-step
  // alpha reaches 0.9375 at position 40 under hop 10 but only 0.7500 under hop
  // 20. `CadenceEma`'s element time constant makes both hops agree at 0.9375.
  let tau = 10.0 / core::f32::consts::LN_2; // alpha = 0.5 per 10 elements
  let cfg = CadenceEma::new(tau);

  // Hop 10: seed 0.0 at 0, then 1.0 at 10, 20, 30, 40. alpha = 0.5, value at 40
  // = 1 - 0.5^4 = 0.9375.
  let hop10 = drive(
    &cfg,
    &cadence_seq(&[(0, 0.0), (10, 1.0), (20, 1.0), (30, 1.0), (40, 1.0)]),
  );
  let v10 = hop10.last().unwrap().value;

  // Hop 20: seed 0.0 at 0, then 1.0 at 20, 40. alpha = 0.75, value at 40
  // = 1 - 0.25^2 = 0.9375.
  let hop20 = drive(&cfg, &cadence_seq(&[(0, 0.0), (20, 1.0), (40, 1.0)]));
  let v20 = hop20.last().unwrap().value;

  assert!((v10 - 0.9375).abs() < 1e-6, "hop 10 at position 40: {v10}");
  assert!((v20 - 0.9375).abs() < 1e-6, "hop 20 at position 40: {v20}");
  assert!((v10 - v20).abs() < 1e-6, "hops must agree: {v10} vs {v20}");
}

#[test]
fn cadence_ema_half_life_and_step_response() {
  let tau = 10.0 / core::f32::consts::LN_2;
  let cfg = CadenceEma::new(tau);

  // Half-life decay: seed 1.0 at 0, input 0.0 at distance 10 -> (1 - alpha) * 1
  // = 0.5.
  let decay = drive(&cfg, &cadence_seq(&[(0, 1.0), (10, 0.0)]));
  assert!(
    (decay[1].value - 0.5).abs() < 1e-6,
    "half-life decay: {}",
    decay[1].value
  );

  // Rise: seed 0.0, input 1.0 at distance d -> alpha = 1 - exp(-d / tau). The
  // streaming push equals the closed form (checked against `libm`), and
  // `alpha_for` is exactly that shared coefficient.
  for d in [5usize, 10, 20, 37] {
    let rise = drive(&cfg, &cadence_seq(&[(0, 0.0), (d, 1.0)]));
    let want = 1.0 - libm::expf(-(d as f32) / tau);
    assert!(
      (rise[1].value - want).abs() < 1e-6,
      "rise at {d}: {} vs {want}",
      rise[1].value
    );
    assert_eq!(
      cfg.alpha_for(d),
      want,
      "alpha_for must be the applied coefficient"
    );
  }
}

#[test]
fn cadence_ema_partition_invariance() {
  let tau = 17.5f32;
  let cfg = CadenceEma::new(tau);

  // Left-continuous piecewise constant: seed 0.0 at position 0, then value 1.0
  // on (0, 30] and value 0.2 on (30, 60], with the breakpoint at 30.
  fn pw(p: usize) -> f32 {
    if p == 0 {
      0.0
    } else if p <= 30 {
      1.0
    } else {
      0.2
    }
  }

  // Fine sampling: every integer position 0..=60 (so `fine[p]` is position `p`).
  let fine_samples: Vec<(usize, f32)> = (0..=60).map(|p| (p, pw(p))).collect();
  let fine = drive(&cfg, &cadence_seq(&fine_samples));

  // Coarse, irregular sampling — a subset of the fine positions that keeps the
  // breakpoint (30) and the endpoints.
  let coarse_positions = [0usize, 6, 12, 18, 24, 30, 41, 52, 60];
  let coarse_samples: Vec<(usize, f32)> = coarse_positions.iter().map(|&p| (p, pw(p))).collect();
  let coarse = drive(&cfg, &cadence_seq(&coarse_samples));

  // The ZOH recurrence telescopes exactly within each constant segment, so the
  // smoothed value at any shared position is sampling-independent.
  for (ci, &p) in coarse_positions.iter().enumerate() {
    let want = fine[p].value;
    let got = coarse[ci].value;
    assert!(
      (got - want).abs() <= 1e-5,
      "position {p}: coarse {got} vs fine {want}"
    );
  }
}

#[test]
fn cadence_ema_delta_zero_ignores_duplicate_and_nan_still_poisons() {
  let cfg = CadenceEma::new(14.0);

  // A second sample at the same start (delta = 0 => alpha = 0) leaves the state
  // untouched: the smoothed value is the pre-duplicate value, computed via the
  // single recurrence, not a branch.
  let out = drive(&cfg, &cadence_seq(&[(0, 0.5), (5, 0.8), (5, 0.1)]));
  assert_eq!(
    out[2].value, out[1].value,
    "delta-0 duplicate must be ignored"
  );

  // But a NaN pushed at delta = 0 still poisons: `0.0 * NaN = NaN`, so the
  // duplicate is *not* ignored when it is non-finite (mirrors `Ema` at alpha 0).
  let out = drive(&cfg, &cadence_seq(&[(0, 0.5), (5, 0.7), (5, f32::NAN)]));
  assert!(
    out[2].value.is_nan(),
    "NaN at delta 0 must poison: {}",
    out[2].value
  );
}

#[test]
fn cadence_ema_large_gap_forgets_and_washes_infinity_to_nan() {
  let cfg = CadenceEma::new(14.0); // delta / tau >> 88 for the gaps below

  // A gap far past the `expf` underflow (delta / tau ~ 88) makes alpha exactly
  // 1.0, so the state tracks the input exactly — full forget.
  let out = drive(&cfg, &cadence_seq(&[(0, 0.5), (10_000_000, 0.9)]));
  assert_eq!(out[1].value, 0.9, "large gap must track the input exactly");

  // With alpha exactly 1.0, `1 - alpha` is exactly 0.0, and `0.0 * inf = NaN`
  // washes a finite-tracking step over an infinite prior state to NaN.
  let out = drive(&cfg, &cadence_seq(&[(0, f32::INFINITY), (10_000_000, 0.9)]));
  assert!(
    out[1].value.is_nan(),
    "long gap over an infinite state washes to NaN: {}",
    out[1].value
  );
}

#[test]
fn cadence_ema_nan_poisons_until_reset_or_discontinuity() {
  for reseed in [Smoother::reset, Smoother::discontinuity] {
    let mut s = CadenceEma::new(14.0).smoother();
    assert_eq!(
      s.push(Windowed::new(0.5, Span::new(0, 1, 1)))
        .unwrap()
        .value,
      0.5
    );
    assert!(s
      .push(Windowed::new(f32::NAN, Span::new(5, 1, 1)))
      .unwrap()
      .value
      .is_nan());
    // Poisoning persists across a later finite push.
    assert!(s
      .push(Windowed::new(0.8, Span::new(10, 1, 1)))
      .unwrap()
      .value
      .is_nan());
    // Re-seed clears it and re-arms the timeline: a backward start after the
    // break is admitted, since the next push seeds without a monotonicity check.
    reseed(&mut s);
    let after = s.push(Windowed::new(0.4, Span::new(2, 1, 1))).unwrap();
    assert_eq!(
      after.value, 0.4,
      "re-seed must restore s_0 = x_0 on a fresh timeline"
    );
  }
}

#[test]
fn cadence_ema_infinity_propagates_while_coefficients_are_nonzero() {
  // Moderate deltas keep both coefficients strictly in (0, 1), so an infinity
  // propagates as that infinity rather than washing to NaN.
  let cfg = CadenceEma::new(20.0);
  let out = drive(
    &cfg,
    &cadence_seq(&[(0, 0.2), (5, f32::INFINITY), (10, 0.8)]),
  );
  assert_f32_seq(&values(&out), &[0.2, f32::INFINITY, f32::INFINITY]);
  let out = drive(
    &cfg,
    &cadence_seq(&[(0, 0.2), (5, f32::NEG_INFINITY), (10, 0.8)]),
  );
  assert_f32_seq(&values(&out), &[0.2, f32::NEG_INFINITY, f32::NEG_INFINITY]);
}

#[test]
fn cadence_ema_backward_start_errs_and_leaves_state_unchanged() {
  let mut s = CadenceEma::new(14.0).smoother();
  let _ = s.push(Windowed::new(0.5, Span::new(10, 1, 1))).unwrap();
  let _ = s.push(Windowed::new(0.8, Span::new(20, 1, 1))).unwrap();

  // A strictly backward start is rejected with the offending fields, state
  // unchanged.
  let err = s.push(Windowed::new(0.3, Span::new(15, 1, 1))).unwrap_err();
  assert_eq!(
    err,
    WinditError::NonMonotonicSpan {
      prev_start: 20,
      start: 15
    }
  );

  // The offending push was a no-op: a following in-order push produces exactly
  // the value a clean drive that never saw the bad push would.
  let v_after = s
    .push(Windowed::new(0.3, Span::new(25, 1, 1)))
    .unwrap()
    .value;
  let clean = drive(
    &CadenceEma::new(14.0),
    &cadence_seq(&[(10, 0.5), (20, 0.8), (25, 0.3)]),
  );
  assert_eq!(v_after, clean.last().unwrap().value);
}

#[test]
fn cadence_ema_construction_validates_tau() {
  // `try_new` rejects non-finite and non-positive tau.
  for bad in [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    -1.0,
    -f32::MIN_POSITIVE,
  ] {
    assert_eq!(
      CadenceEma::try_new(bad),
      Err(WinditError::TimeConstantOutOfRange),
      "tau {bad} must be rejected"
    );
  }
  // Accepts a positive subnormal and a fractional tau, reported verbatim.
  assert_eq!(
    CadenceEma::try_new(f32::MIN_POSITIVE).unwrap().tau(),
    f32::MIN_POSITIVE
  );
  let subnormal = f32::from_bits(1); // smallest positive subnormal
  assert_eq!(CadenceEma::try_new(subnormal).unwrap().tau(), subnormal);
  assert_eq!(CadenceEma::try_new(0.25).unwrap().tau(), 0.25);
  // `new` agrees with `try_new` on a valid tau.
  assert_eq!(CadenceEma::new(14.0).tau(), 14.0);
}

#[test]
#[should_panic = "cadence time constant"]
fn cadence_ema_new_panics_on_zero_tau() {
  let _ = CadenceEma::new(0.0);
}

#[test]
#[should_panic = "cadence time constant"]
fn cadence_ema_new_panics_on_nan_tau() {
  let _ = CadenceEma::new(f32::NAN);
}

#[test]
fn cadence_ema_alpha_for_is_monotone_and_saturates() {
  let cfg = CadenceEma::new(14.0);
  // delta 0 => alpha exactly 0 (the duplicate-ignoring coefficient).
  assert_eq!(cfg.alpha_for(0), 0.0);
  // Monotonically non-decreasing in delta, and always in [0, 1].
  let mut prev = cfg.alpha_for(0);
  for delta in 1..=200 {
    let a = cfg.alpha_for(delta);
    assert!(
      a >= prev,
      "alpha must be monotone in delta: {a} < {prev} at {delta}"
    );
    assert!((0.0..=1.0).contains(&a), "alpha out of [0, 1]: {a}");
    prev = a;
  }
  // A gap far past the `expf` underflow saturates to exactly 1.0.
  assert_eq!(cfg.alpha_for(100_000), 1.0);
}

#[test]
fn cadence_ema_matches_f64_zoh_reference() {
  // Differential over randomized monotone sequences: the batch `smooth`, a fresh
  // streaming drive, and an independent f64 ZOH reference must agree — within
  // 1e-5 where finite, and exactly on the non-finite tails. Cadence is bounded
  // (delta / tau < 88) so `expf` never underflows here, keeping the f32 and f64
  // paths on the same non-finite trajectory; the f32-specific washout at a huge
  // gap is pinned separately (see the large-gap edge test).
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 40) as usize;
    let tau = 2.0 + next_unit(&mut state) * 48.0; // [2, 50)
    let mut start = 0usize;
    let mut samples: Vec<(usize, f32)> = Vec::with_capacity(n);
    for k in 0..n {
      if k > 0 {
        start += (xorshift(&mut state) % 31) as usize; // gap in [0, 30]
      }
      let v = match xorshift(&mut state) % 16 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        _ => next_unit(&mut state),
      };
      samples.push((start, v));
    }

    let cfg = CadenceEma::new(tau);
    let input = cadence_seq(&samples);
    let reference = cadence_zoh_reference(f64::from(tau), &samples);

    let batch = cfg.smooth(&input).unwrap();
    assert_f32_seq_close(&values(&batch), &reference, 1e-5);
    // Batch equals the streaming drive by construction; pin it too.
    assert_f32_seq_close(&values(&drive(&cfg, &input)), &reference, 1e-5);
    // Spans are preserved throughout.
    assert_eq!(spans(&batch), spans(&input));
  }
}

#[test]
fn cadence_ema_boxed_smoother_drives_through_forwarding_impl() {
  // A `Box<dyn Smoother<f32>>` satisfies `S: Smoother<f32>` only through the
  // forwarding impl (auto-deref would call `push` on the box but would not make
  // the box *itself* a `Smoother`, which is what a generic bound — the `Decoder`
  // manifest path — demands). Passing the box to a generically-bounded helper
  // exercises that impl.
  fn push_through<S: Smoother<f32>>(s: &mut S, v: f32, start: usize) -> f32 {
    s.push(Windowed::new(v, Span::new(start, 1, 1)))
      .unwrap()
      .value
  }
  fn reset_through<S: Smoother<f32>>(s: &mut S) {
    s.reset();
  }

  let mut boxed: Box<dyn Smoother<f32>> = Box::new(CadenceEma::new(14.0).smoother());
  assert_eq!(push_through(&mut boxed, 0.5, 0), 0.5); // seed s_0 = x_0
  assert_eq!(push_through(&mut boxed, 0.9, 10_000_000), 0.9); // huge gap tracks input
  reset_through(&mut boxed);
  assert_eq!(push_through(&mut boxed, 0.4, 0), 0.4); // re-seeded
}

#[test]
fn boxed_smoother_forwards_discontinuity_not_reset() {
  // A smoother that distinguishes `reset` from `discontinuity`, to pin that the
  // `Box` forwarding impl forwards `discontinuity` explicitly (F3) rather than
  // letting the trait default route it to the box's own `reset`.
  #[derive(Default)]
  struct ProbeSmoother {
    resets: u32,
    discontinuities: u32,
  }
  impl Smoother<f32> for ProbeSmoother {
    fn push(&mut self, w: Windowed<f32>) -> Result<Windowed<f32>, WinditError> {
      Ok(w)
    }
    fn reset(&mut self) {
      self.resets += 1;
    }
    fn discontinuity(&mut self) {
      self.discontinuities += 1;
    }
  }

  // `Box<ProbeSmoother>: Smoother<f32>` via the `?Sized` forwarding impl; the
  // fully-qualified calls route through it, and the concrete box lets us read
  // the counters back.
  let mut boxed: Box<ProbeSmoother> = Box::default();
  <Box<ProbeSmoother> as Smoother<f32>>::discontinuity(&mut boxed);
  <Box<ProbeSmoother> as Smoother<f32>>::reset(&mut boxed);
  <Box<ProbeSmoother> as Smoother<f32>>::discontinuity(&mut boxed);
  assert_eq!(
    boxed.discontinuities, 2,
    "Box must forward discontinuity explicitly, not via its own reset"
  );
  assert_eq!(boxed.resets, 1, "Box must forward reset");
}
