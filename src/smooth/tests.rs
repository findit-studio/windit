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
fn ema_sub_epsilon_alpha_accumulates_rather_than_holding() {
  // At `alpha = 2^-25` the f32 `1 - alpha` rounds to exactly 1.0, which deletes
  // the decay term — but not the `alpha * x` injection. The recurrence therefore
  // degenerates into the biased accumulator `s <- s + alpha * x`: the state
  // ramps in the direction of the input rather than holding. These are the
  // values the type docs describe; pinning them keeps the docs from drifting
  // back to the (false) claim that the state simply stops responding.
  const ALPHA: f32 = 1.0 / 33_554_432.0; // 2^-25
  assert_eq!(1.0f32 - ALPHA, 1.0);

  let ema = Ema::new(ALPHA);

  // Seeded at zero, a constant 1.0 input ramps in exact steps of `alpha`. One
  // push lands on exactly 2^-25 — not on 0.0, which a true hold would give.
  let ramp = values(&drive(&ema, &seq(&[0.0, 1.0, 1.0, 1.0, 1.0])));
  assert_eq!(
    ramp,
    vec![0.0, ALPHA, 2.0 * ALPHA, 3.0 * ALPHA, 4.0 * ALPHA]
  );

  // The ramp stalls at `alpha * x * 2^24` — 0.5 for a unit input — where a step
  // of `alpha` is no longer more than half an ulp of the state. Seeding one step
  // below reaches that in a single push, pinning the plateau without the 2^24
  // pushes the climb from zero actually takes. The state never arrives at 1.0.
  assert_eq!(0.5f32, ALPHA * 1.0 * 16_777_216.0);
  let plateau = values(&drive(&ema, &seq(&[0.5 - ALPHA, 1.0, 1.0, 1.0])));
  assert_eq!(plateau, vec![0.5 - ALPHA, 0.5, 0.5, 0.5]);

  // From the stalling magnitude upward it does hold — in every direction, since
  // what was lost is the decay and not the injection. A state of 1.0 is a fixed
  // point against a smaller input, an equal one, and an opposing one alike.
  for x in [0.0f32, 0.5, 1.0, -1.0] {
    assert_eq!(
      values(&drive(&ema, &seq(&[1.0, x, x, x]))),
      vec![1.0, 1.0, 1.0, 1.0],
      "a state of 1.0 must be a fixed point against {x}"
    );
  }

  // The step scales with the input, not with the distance to it: the injection
  // is `alpha * x`, so a large input ramps proportionally faster.
  let scaled = values(&drive(&ema, &seq(&[0.0, 8.0, 8.0])));
  assert_eq!(scaled, vec![0.0, ALPHA * 8.0, 2.0 * ALPHA * 8.0]);
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

  // Exactness is a property of the recurrence, not of a conveniently small
  // prior state: with `alpha` exactly 1.0, `1 * x + 0 * prev` is `x` however
  // far away `prev` is. This pins the product form specifically — the
  // increment-shaped rewrite `prev + alpha * (x - prev)`, which absorbs the
  // sub-ulp decay just as an f32 accumulator does, returns `0.0` here.
  let out = drive(&cfg, &cadence_seq(&[(0, 1e30), (10_000_000, 0.9)]));
  assert_eq!(
    out[1].value, 0.9,
    "exact tracking must not depend on the prior state's magnitude"
  );

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
fn cadence_ema_tiny_cadence_ratio_yields_a_nonzero_coefficient() {
  // The small end of the coefficient. `expf(-x)` rounds to exactly `1.0` once
  // `x` drops below `2^-25` (~3e-8), so deriving alpha as `1 - expf(-x)` there
  // returns exactly `0.0` and the filter stops responding to the data
  // altogether — on finite, valid input. The coefficient is therefore derived
  // with `expm1f`, which is exact to full precision in that regime.
  let tau = 1e8;
  let cfg = CadenceEma::new(tau);
  assert!(
    cfg.alpha_for(1) > 0.0,
    "a unit cadence under tau {tau} must still move the filter: {}",
    cfg.alpha_for(1)
  );
  // Below the epsilon the coefficient is `delta / tau` to within rounding.
  for delta in [1usize, 2, 5, 17, 100, 1000] {
    let want = (delta as f32) / tau;
    let got = cfg.alpha_for(delta);
    assert!(
      (got - want).abs() <= want * 1e-3,
      "alpha_for({delta}) = {got}, want ~{want}"
    );
  }

  // The large end is unchanged: both formulations saturate to exactly `1.0` at
  // the same ratio, where `exp(-x)` falls below half an ulp of 1 (`x` past
  // `ln(2^25)` ~ 17.33), and `1 - alpha` is exactly `0.0` from there on.
  let unit = CadenceEma::new(1.0);
  assert!(unit.alpha_for(17) < 1.0);
  assert_eq!(unit.alpha_for(18), 1.0);
  assert_eq!(1.0 - unit.alpha_for(18), 0.0);
  assert_eq!(cfg.alpha_for(1_800_000_000), 1.0);
}

#[test]
fn cadence_ema_is_cadence_invariant_below_f32_epsilon() {
  // The property the type exists for, asserted directly: one signal sampled at
  // three cadences must smooth to the same value at a shared position. Here
  // `delta / tau` is 2.5e-8 at the finest cadence — under f32's epsilon, the
  // regime where a `1 - expf` coefficient collapses to zero and pins the fine
  // sampling to its seed for the whole horizon while the coarse samplings of
  // the same signal move normally.
  //
  // This geometry rises from a seed of `0.0`, so it constrains the
  // *coefficient* and only weakly the accumulator: the running state stays
  // near zero, where an ulp is minute and every increment lands. The falling
  // counterpart — `cadence_ema_is_cadence_invariant_on_a_falling_step` — is
  // the one that constrains the accumulator, and the two must be kept as a
  // pair; a rising-only invariance test cannot fail on state precision.
  const N: usize = 40_000;
  let tau = 4e7f32;
  let cfg = CadenceEma::new(tau);

  // Seed 0.0 at position 0, then a constant 1.0 over the next `N` elements.
  let sampled = |hop: usize| -> f32 {
    let mut samples: Vec<(usize, f32)> = vec![(0, 0.0)];
    let mut p = 0usize;
    while p + hop <= N {
      p += hop;
      samples.push((p, 1.0));
    }
    drive(&cfg, &cadence_seq(&samples)).last().unwrap().value
  };
  let fine = sampled(1);
  let mid = sampled(100);
  let coarse = sampled(N);

  // The step response at the shared endpoint, in f64 — an independent
  // precision path, not the recurrence under test.
  let want = 1.0 - libm::exp(-(N as f64) / f64::from(tau));
  for (name, got) in [("fine", fine), ("mid", mid), ("coarse", coarse)] {
    assert!(
      (f64::from(got) - want).abs() <= 1e-6,
      "{name} cadence: {got} vs closed form {want}"
    );
  }
  assert!(
    (fine - coarse).abs() <= 1e-6 && (fine - mid).abs() <= 1e-6,
    "cadences must agree: fine {fine} mid {mid} coarse {coarse}"
  );
}

#[test]
fn cadence_ema_is_cadence_invariant_on_a_falling_step() {
  // The mirror of the rising invariance test above, and the geometry that
  // constrains the *accumulator* rather than the coefficient: the same signal
  // sampled at three cadences, but falling from a seed of `1.0` toward `0.0`.
  //
  // At `tau = 4e7` a unit cadence has `alpha = 2.5e-8`, below half an ulp of a
  // state near 1.0 (`2^-25` ~ 2.98e-8). A recurrence that carries its state in
  // f32 therefore rounds `1 - alpha` to exactly `1.0`, subtracts nothing, and
  // leaves the fine cadence pinned at its seed forever — 1.0 against the
  // ~0.9990005 the coarse cadence reaches over the same elapsed distance —
  // while the rising geometry above passes, because a state near zero has ulps
  // far smaller than its increments. Neither the `expm1f` coefficient nor any
  // other algebraic form of the recurrence lifts this; only accumulator
  // precision does.
  const N: usize = 40_000;
  let tau = 4e7f32;
  let cfg = CadenceEma::new(tau);

  // Seed 1.0 at position 0, then a constant 0.0 over the next `N` elements.
  let sampled = |hop: usize| -> f32 {
    let mut samples: Vec<(usize, f32)> = vec![(0, 1.0)];
    let mut p = 0usize;
    while p + hop <= N {
      p += hop;
      samples.push((p, 0.0));
    }
    drive(&cfg, &cadence_seq(&samples)).last().unwrap().value
  };
  let fine = sampled(1);
  let mid = sampled(100);
  let coarse = sampled(N);

  // The decay at the shared endpoint, in f64 — an independent precision path,
  // not the recurrence under test.
  let want = libm::exp(-(N as f64) / f64::from(tau));
  for (name, got) in [("fine", fine), ("mid", mid), ("coarse", coarse)] {
    assert!(
      (f64::from(got) - want).abs() <= 1e-6,
      "{name} cadence: {got} vs closed form {want}"
    );
  }
  assert!(
    (fine - coarse).abs() <= 1e-6 && (fine - mid).abs() <= 1e-6,
    "cadences must agree: fine {fine} mid {mid} coarse {coarse}"
  );
}

#[test]
fn cadence_invariance_is_bounded_by_contrast_not_by_the_coefficient_alone() {
  // The counterexample to any flat `delta / tau` invariance bound, in both
  // directions. A push contributes `alpha * (x - s)` into a state of magnitude
  // `|s|`, so it survives only while `alpha * |x - s|` exceeds half an ulp of
  // `|s|` — a condition on the *product* of coefficient and contrast. At the
  // finest contrast an `f32` caller can express, one ulp, the boundary therefore
  // sits at `alpha = 2^-30` and not at the `2^-53`-ish figure unit contrast
  // suggests: 2^24 times sooner than the `delta / tau > 2^-54` the docs used to
  // claim unconditionally.
  //
  // The observation is `CadenceEmaState`'s `PartialEq`, which compares the
  // retained `f64`. At this contrast every emitted `f32` is the seed on both
  // cadences, so only the retained state separates "absorbed" from "recorded,
  // below `f32` resolution" — and a state advanced solely by absorbed pushes is
  // bit-identical to one merely seeded at the same position.
  const SEED: f32 = 1.0;
  const N: usize = 16;
  const TAU_MOVES: f32 = 536_870_912.0; // 2^29
  const TAU_FREEZES: f32 = 1_073_741_824.0; // 2^30
  const RETIRED_BOUND: f64 = 1.0 / 18_014_398_509_481_984.0; // 2^-54

  // `expm1f` is exact this far below its argument's square, so the coefficient
  // at `delta = 1` is exactly `1 / tau` and the boundary reads off the exponent.
  assert_eq!(CadenceEma::new(TAU_MOVES).alpha_for(1), 1.0 / TAU_MOVES);
  assert_eq!(CadenceEma::new(TAU_FREEZES).alpha_for(1), 1.0 / TAU_FREEZES);
  // Both sit far above the retired bound, so what follows is not its deep tail.
  assert!(f64::from(CadenceEma::new(TAU_FREEZES).alpha_for(1)) > RETIRED_BOUND);

  // A smoother seeded at `at` and never advanced — the "nothing happened" state.
  let bare = |tau: f32, at: usize| {
    let mut s = CadenceEma::new(tau).smoother();
    let _ = s.push(Windowed::new(SEED, Span::new(at, 1, 1))).unwrap();
    s
  };
  // Seed at 0, then the `N` elements at a unit cadence...
  let fine = |tau: f32, x: f32| {
    let mut s = CadenceEma::new(tau).smoother();
    let _ = s.push(Windowed::new(SEED, Span::new(0, 1, 1))).unwrap();
    let outs: Vec<f32> = (1..=N)
      .map(|k| s.push(Windowed::new(x, Span::new(k, 1, 1))).unwrap().value)
      .collect();
    (s, outs)
  };
  // ...and the same `N` elements as one hop.
  let coarse = |tau: f32, x: f32| {
    let mut s = CadenceEma::new(tau).smoother();
    let _ = s.push(Windowed::new(SEED, Span::new(0, 1, 1))).unwrap();
    let out = s.push(Windowed::new(x, Span::new(N, 1, 1))).unwrap().value;
    (s, out)
  };

  // The seed's two `f32` neighbours: the finest falling and rising steps any
  // caller can express — `1 - 2^-24` and `1 + 2^-23`.
  for (dir, x) in [
    ("falling", f32::from_bits(SEED.to_bits() - 1)),
    ("rising", f32::from_bits(SEED.to_bits() + 1)),
  ] {
    // Above the boundary the property holds, and holds exactly: the unit cadence
    // and the single hop reach the same retained `f64`, bit for bit.
    let (fine_state, fine_outs) = fine(TAU_MOVES, x);
    let (coarse_state, _) = coarse(TAU_MOVES, x);
    assert_ne!(
      fine_state,
      bare(TAU_MOVES, N),
      "{dir}: alpha = 2^-29 must still record a one-ulp step"
    );
    assert_eq!(
      fine_state, coarse_state,
      "{dir}: cadences must agree above the boundary"
    );
    assert!(
      fine_outs.iter().all(|&v| v == SEED),
      "{dir}: that movement is below f32 resolution"
    );

    // One binary order further out the step is exactly half an ulp of the state
    // and ties to even, so the fine cadence absorbs every push and is left
    // bit-identical to a bare seed — frozen, and frozen for any number of
    // further pushes, the state being a fixed point of the map. The same
    // elements taken in one hop carry 16x the coefficient and are not absorbed.
    // Same signal, same tau, same elapsed distance, two cadences, two answers.
    let (fine_state, fine_outs) = fine(TAU_FREEZES, x);
    let (coarse_state, coarse_out) = coarse(TAU_FREEZES, x);
    assert_eq!(
      fine_state,
      bare(TAU_FREEZES, N),
      "{dir}: alpha = 2^-30 must absorb a one-ulp step entirely"
    );
    assert_ne!(
      coarse_state,
      bare(TAU_FREEZES, N),
      "{dir}: the single hop over the same elements must still move"
    );
    assert_ne!(
      fine_state, coarse_state,
      "{dir}: cadence invariance must be observably broken here"
    );
    // Both cadences still emit the seed: the divergence is in the retained
    // state, and only a run long enough to accumulate half an `f32` ulp would
    // surface it in the output. That is why the assertions above are on states.
    assert!(fine_outs.iter().all(|&v| v == SEED));
    assert_eq!(coarse_out, SEED);
  }
}

#[test]
fn cadence_agreement_is_absolute_in_the_swing_not_relative_to_the_result() {
  // The narrowed agreement claim. `alpha` is an `f32`, so the retained fraction
  // `1 - alpha` carries an absolute error of about `2^-25`, and that error
  // multiplies the distance between the seed and the input: two cadences over
  // the same elapsed distance agree to a small multiple of `2^-25 * |x - s_0|`,
  // which is an ABSOLUTE bound. Read as ulps of the result it is about one while
  // the result is a healthy fraction of the swing, and thousands once the state
  // has decayed by many `tau` — the residual shrinks exponentially while the
  // error does not. Both ends are pinned so the docs cannot drift back to a flat
  // "cadences agree to about an ulp".
  const TAU: f32 = 1024.0;
  // 8 * 2^-25. The worst ratio measured over `tau` 3..10007 and distances
  // `tau/4`..`12 tau` was 4x; this leaves one binary order of headroom.
  const SWING_ERR: f64 = 8.0 / 33_554_432.0;
  let cfg = CadenceEma::new(TAU);
  let n = TAU as usize;

  // `x0` at position 0, then a constant `x1` over `d` elements at `hop` cadence.
  let sampled = |hop: usize, d: usize, x0: f32, x1: f32| -> f32 {
    let mut samples: Vec<(usize, f32)> = vec![(0, x0)];
    let mut p = 0usize;
    while p + hop <= d {
      p += hop;
      samples.push((p, x1));
    }
    drive(&cfg, &cadence_seq(&samples)).last().unwrap().value
  };

  for mult in [1usize, 2, 4, 8, 12] {
    let d = n * mult;
    for (x0, x1) in [(0.0f32, 1.0f32), (1.0, 0.0)] {
      let fine = sampled(1, d, x0, x1);
      let coarse = sampled(d, d, x0, x1);
      let bound = SWING_ERR * f64::from(x1 - x0).abs();
      assert!(
        (f64::from(fine) - f64::from(coarse)).abs() <= bound,
        "d/tau {mult}, {x0} -> {x1}: |{fine} - {coarse}| exceeds {bound}"
      );
    }
  }

  // The same absolute agreement in ulps of the result, at both ends.
  let ulps = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
  let (fine, coarse) = (sampled(1, n, 0.0, 1.0), sampled(n, n, 0.0, 1.0));
  assert!(
    ulps(fine, coarse) <= 1,
    "a result of order the swing must agree to an ulp: {fine} vs {coarse}"
  );
  let d = n * 12;
  let (fine, coarse) = (sampled(1, d, 1.0, 0.0), sampled(d, d, 1.0, 0.0));
  assert!(
    ulps(fine, coarse) > 1_000,
    "a residual decayed by 12 tau must NOT agree to an ulp: {fine} vs {coarse}"
  );
}

#[test]
fn cadence_ema_falling_step_decays_by_one_over_e_over_one_tau() {
  // The defining decay, over the full time constant, at the cadence where the
  // per-step coefficient is smaller than the state's own resolution: a unit
  // cadence must reach `exp(-1)` after `tau` elements exactly as a single
  // `delta = tau` step does.
  //
  // The horizon is forced, not chosen: absorption at `delta = 1` needs an
  // `alpha` of `2^-25` or below, so covering one `tau` at that cadence always
  // costs at least `2^25` pushes. The sequence is therefore driven push by push
  // rather than materialized — forty million `Windowed`s would be gigabytes.
  let tau = 4e7f32;
  let n = tau as usize;
  let cfg = CadenceEma::new(tau);

  let mut s = cfg.smoother();
  let mut fine = s
    .push(Windowed::new(1.0, Span::new(0, 1, 1)))
    .unwrap()
    .value;
  assert_eq!(fine, 1.0, "seed must be s_0 = x_0");
  for p in 1..=n {
    fine = s
      .push(Windowed::new(0.0, Span::new(p, 1, 1)))
      .unwrap()
      .value;
  }

  // One coarse step over the same elapsed distance: `alpha = 1 - exp(-1)`, so
  // `(1 - alpha) * 1.0 = exp(-1)` directly.
  let coarse = drive(&cfg, &cadence_seq(&[(0, 1.0), (n, 0.0)]))
    .last()
    .unwrap()
    .value;

  let want = 1.0 / core::f64::consts::E; // 0.36787944...
  assert!(
    (f64::from(fine) - want).abs() <= 1e-6,
    "unit cadence over one tau: {fine} vs 1/e {want}"
  );
  assert!(
    (f64::from(coarse) - want).abs() <= 1e-6,
    "single tau-sized step: {coarse} vs 1/e {want}"
  );
  assert!(
    (fine - coarse).abs() <= 1e-6,
    "cadences must agree over one tau: fine {fine} vs coarse {coarse}"
  );
}

#[test]
fn cadence_ema_matches_f64_zoh_reference_at_sub_epsilon_cadence() {
  // The differential below bounds `delta / tau` under 88 to keep `expf` off its
  // large-end underflow; that bound also keeps it clear of the *small* end,
  // where the coefficient rounds away. This is that end: a time constant far
  // above the cadence, over a horizon long enough that the per-step coefficient
  // accumulates into a visible response.
  //
  // The seed is `1.0`, deliberately: a state of order 1 is where an f32
  // accumulator of magnitude `m` fails to record a step smaller than half an
  // ulp of `m`, so seeding there makes this differential a statement about the
  // *accumulator* as well as the coefficient. A seed of `0.0` would keep the
  // running state near zero, where ulps are minute, and could not distinguish
  // the two.
  let mut state: u64 = 0x5DEE_CE66_D1B5_4A32;
  for _ in 0..4 {
    // tau in [3.5e7, 4e7) at a unit cadence: `delta / tau` is about 2.6e-8,
    // just under the threshold below which `expf(-x)` rounds to exactly 1.0.
    let tau = 3.5e7 + next_unit(&mut state) * 5e6;
    let n = 50_000 + (xorshift(&mut state) % 20_000) as usize;
    let mut samples: Vec<(usize, f32)> = Vec::with_capacity(n);
    let mut level = 1.0f32;
    for p in 0..n {
      if p != 0 && p.is_multiple_of(10_000) {
        level = next_unit(&mut state);
      }
      samples.push((p, level));
    }

    let cfg = CadenceEma::new(tau);
    let input = cadence_seq(&samples);
    let reference = cadence_zoh_reference(f64::from(tau), &samples);
    assert_f32_seq_close(&values(&cfg.smooth(&input).unwrap()), &reference, 1e-5);
    assert_f32_seq_close(&values(&drive(&cfg, &input)), &reference, 1e-5);
  }
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
