use std::{boxed::Box, vec, vec::Vec};

use super::{
  cadence_alpha, CadenceEma, CadenceEmaState, Ema, Identity, SmoothPolicy, Smoother, VectorEma,
  VectorEmaState,
};
use crate::{
  error::WinditError,
  plan::Span,
  test_support::{BareI8Emb, RawF64Emb, TestVec},
  windowed::{Vector, Windowed},
};

/// The coefficient floor the accepted `tau` domain is built around: every
/// accepted `tau` derives an `alpha` strictly above this at every `delta >= 1`.
const ALPHA_FLOOR: f32 = 1.0 / 67_108_864.0; // 2^-26

/// The published absorption bound: a step above this many `ulp(s)` must be
/// recorded. Shared by the randomized sweep and the exact witness below, so the
/// two cannot enforce different figures — and so one edit falsifies both.
const PUBLISHED_ABSORPTION_ULPS: f64 = 4.0;

/// The next representable `f32` above `v`, for probing the domain edge from its
/// rejected side.
fn next_up32(v: f32) -> f32 {
  f32::from_bits(v.to_bits() + 1)
}

/// Signed distance between two `f32`s in representable steps. Quantitative
/// accuracy claims are stated in ulps, so they are checked in ulps: an absolute
/// tolerance cannot express "within N representable values" at all — the `1e-6`
/// this replaced was over 33 ulps at `exp(-1)`, far too slack to enforce the
/// claim it accompanied.
fn ulps32(a: f32, b: f32) -> i64 {
  a.to_bits() as i64 - b.to_bits() as i64
}

/// One representable step at `|v|`, as an `f64` so it can be compared against
/// the retained state directly.
fn ulp64(v: f64) -> f64 {
  let m = v.abs();
  f64::from_bits(m.to_bits() + 1) - m
}

/// One representable `f32` step at `|v|` — the resolution of the *emitted*
/// value, `2^29` times coarser than the accumulator's.
fn ulp32(v: f32) -> f32 {
  let m = v.abs();
  f32::from_bits(m.to_bits() + 1) - m
}

/// The retained `f64` inside a seeded [`CadenceEmaState`].
///
/// Absorption is only observable here: the emitted `f32` is `2^29` times
/// coarser than the accumulator, so a step can be recorded, or lost, with the
/// output identical either way.
fn retained(s: &CadenceEmaState) -> f64 {
  s.prev.expect("smoother must be seeded before it is read").1
}

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

/// The retired coefficient spelling, which formed the ratio in `f32` and so
/// rounded `delta` itself above 2^24.
///
/// Kept as the differential the boundary sweep measures the shipped derivation
/// against: the sweep asserts that this one breaches the published two-ulp
/// figure on the very same enumerated points, so "the cast is harmless" cannot
/// be reasserted without a failing test.
fn cadence_alpha_via_f32_cast(tau: f32, delta: usize) -> f32 {
  -libm::expm1f(-(delta as f32) / tau)
}

/// Distance from an `f32` coefficient to its exact value, in `f32` ulps read at
/// the *exact* value's binade.
///
/// Reading the ulp off the returned `f32` instead would halve the measured
/// distance wherever the two straddle a binade edge — precisely the points this
/// sweep exists to enumerate.
fn coefficient_ulps(got: f32, exact: f64) -> f64 {
  (f64::from(got) - exact).abs() / libm::ldexp(1.0, libm::frexp(exact).1 - 24)
}

/// The `tau` edges the coefficient claim is quantified over: the ceiling and the
/// values just under it, every binade edge of the accepted domain with its
/// neighbours, small integers, and non-dyadic values.
///
/// Enumerated, not sampled. The 2.25-ulp breach lived at the ceiling and at
/// `delta`s a random draw over a 16-`tau` span reaches with vanishing
/// probability, which is why the randomized predecessor of this sweep ran
/// 20_000 probes without meeting one.
fn boundary_taus() -> Vec<f32> {
  let mut taus: Vec<f32> = Vec::new();
  // The ceiling and its immediate neighbourhood, where `alpha` is smallest and
  // every figure on the type is tightest.
  for k in 0..32u32 {
    taus.push(f32::from_bits(CadenceEma::MAX_TAU.to_bits() - k));
  }
  // Every binade edge of the accepted domain, each with its two neighbours.
  for e in -30i32..=26 {
    let p = libm::ldexpf(1.0, e);
    for d in [-1i64, 0, 1] {
      let bits = p.to_bits() as i64 + d;
      if bits > 0 {
        let tau = f32::from_bits(bits as u32);
        if tau > 0.0 && tau <= CadenceEma::MAX_TAU {
          taus.push(tau);
        }
      }
    }
  }
  for i in 1..=32u32 {
    taus.push(i as f32);
  }
  for tau in [1.1f32, 3.7, 9.99, 17.5, 14.427, 1e3, 1e5, 1e7, 6.5e7] {
    taus.push(tau);
  }
  taus.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in the ladder"));
  taus.dedup();
  taus
}

/// The `delta` edges for one `tau`: every `f32` cast boundary from 2^24 up, the
/// ratio's own binade edges, a small exhaustive head, and the exact witness.
///
/// Above `delta / tau` of about 17.33 the coefficient is exactly `1.0` on every
/// path, so the accuracy claim has content only below it; the whole enumerated
/// set is finite for that reason.
fn boundary_deltas(tau: f32) -> Vec<usize> {
  let mut deltas: Vec<usize> = (1..=256).collect();
  // The cast boundaries: 2^24 is where an `f32` stops holding every integer,
  // and each binade above it doubles the step it skips. Both the neighbourhood
  // of each boundary and, inside each binade, the counts an `f32` cast rounds
  // WORST — the midpoints of its grid, at odd multiples of half its step, which
  // is where the retired spelling's error peaks.
  for e in 24..=30u32 {
    let p = 1usize << e;
    for j in 0..=64usize {
      deltas.push(p + j);
      deltas.push(p - j);
    }
    let half_step = 1usize << (e - 24);
    for j in 0..=256usize {
      deltas.push(p + (2 * j + 1) * half_step);
    }
  }
  // The ratio's binade edges, where the relative rounding of the narrowing is
  // largest and the reported ulp changes size.
  for k in -20i32..=4 {
    let centre = (f64::from(tau) * libm::ldexp(1.0, k)) as i64;
    for j in -4i64..=4 {
      if centre + j > 0 {
        deltas.push((centre + j) as usize);
      }
    }
  }
  // The witness the published figure was falsified at, and its neighbours.
  for j in -2i64..=2 {
    deltas.push((16_812_203i64 + j) as usize);
  }
  deltas.sort_unstable();
  deltas.dedup();
  deltas
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

  // And the plateau is `alpha * x * 2^24 = x / 2` for any `x`, not only for the
  // unit input the assertions above use — the doc states it in `x`.
  for x in [0.5f32, 1.0, 8.0, -8.0, 1024.0] {
    let plateau = ALPHA * x * 16_777_216.0;
    assert_eq!(plateau, x / 2.0, "the plateau must be x / 2 for x = {x}");
    assert_eq!(
      values(&drive(&ema, &seq(&[plateau - ALPHA * x, x, x, x]))),
      vec![plateau - ALPHA * x, plateau, plateau, plateau],
      "the ramp must stall at x / 2 for x = {x}"
    );
  }
}

#[test]
fn ema_sub_epsilon_ramp_reaches_its_plateau_after_exactly_two_pow_24_pushes() {
  // The push count the type doc states — the ramp "stalls at `alpha * x * 2^24`
  // ... reached after exactly `2^24` pushes". The test above pins the plateau
  // *value* by seeding one step below it, deliberately skipping the climb, so
  // the count itself went unenforced. Driving the climb costs 2^24 `f32`
  // operations and settles it.
  const ALPHA: f32 = 1.0 / 33_554_432.0; // 2^-25
  const PUSHES: u32 = 1 << 24;
  let ramp = |n: u32| {
    let mut s = 0.0f32;
    for _ in 0..n {
      s = ALPHA * 1.0 + (1.0 - ALPHA) * s;
    }
    s
  };

  let plateau = ramp(PUSHES);
  assert_eq!(plateau, 0.5, "the ramp must land exactly on x / 2");
  assert_eq!(
    ALPHA * 1.0 + (1.0 - ALPHA) * plateau,
    plateau,
    "and the very next push must not move it"
  );
  // One push earlier the state is still climbing, so `2^24` is the exact count
  // and not merely an upper bound.
  assert_eq!(ramp(PUSHES - 1), 0.5 - ALPHA);
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
  // `try_new` rejects non-finite and non-positive tau...
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
  // ...and every `tau` past the ceiling, which is the domain restriction this
  // type's accuracy figures are quantified over. `2^55` is the witness that
  // falsified the one-`tau` claim while it still constructed; `2^54` is where
  // the f64 `1 - alpha` first collapses to exactly `1.0`; the two neighbours of
  // `MAX_TAU` pin the boundary itself.
  for bad in [
    next_up32(CadenceEma::MAX_TAU),
    67_108_864.0, // 2^26, the first power of two above the ceiling
    libm::ldexpf(1.0, 54),
    libm::ldexpf(1.0, 55),
    1e8,
    f32::MAX,
  ] {
    assert_eq!(
      CadenceEma::try_new(bad),
      Err(WinditError::TimeConstantOutOfRange),
      "tau {bad} is above MAX_TAU and must be rejected"
    );
  }

  // Both ends of the accepted domain construct, reported verbatim: the smallest
  // positive subnormal and the ceiling itself.
  let subnormal = f32::from_bits(1); // smallest positive subnormal, 2^-149
  assert_eq!(CadenceEma::try_new(subnormal).unwrap().tau(), subnormal);
  assert_eq!(
    CadenceEma::try_new(f32::MIN_POSITIVE).unwrap().tau(),
    f32::MIN_POSITIVE
  );
  assert_eq!(CadenceEma::try_new(0.25).unwrap().tau(), 0.25);
  assert_eq!(
    CadenceEma::try_new(CadenceEma::MAX_TAU).unwrap().tau(),
    CadenceEma::MAX_TAU
  );
  // `new` agrees with `try_new` on a valid tau, at an ordinary value and at the
  // ceiling.
  assert_eq!(CadenceEma::new(14.0).tau(), 14.0);
  assert_eq!(
    CadenceEma::new(CadenceEma::MAX_TAU).tau(),
    CadenceEma::MAX_TAU
  );
}

#[test]
fn cadence_ema_max_tau_is_the_exact_coefficient_boundary() {
  // `MAX_TAU` is not a round number picked for looks: it is the largest `f32`
  // whose `delta = 1` coefficient still clears `2^-26`, the floor every
  // unconditional figure on this type rests on. Both halves are asserted, so a
  // ceiling moved in EITHER direction fails: one f32 step further out the
  // coefficient is exactly `2^-26` — on the `4 * ulp(s)` bar rather than above
  // it — and one step back in, the value below would leave an accepted `tau`
  // wrongly rejected.
  assert_eq!(CadenceEma::MAX_TAU, 67_108_860.0);
  assert_eq!(CadenceEma::MAX_TAU.to_bits(), 0x4C7F_FFFF);
  assert_eq!(next_up32(CadenceEma::MAX_TAU), 67_108_864.0, "2^26");

  let at = cadence_alpha(CadenceEma::MAX_TAU, 1);
  let past = cadence_alpha(next_up32(CadenceEma::MAX_TAU), 1);
  assert!(at > ALPHA_FLOOR, "the ceiling must clear 2^-26: {at:e}");
  assert_eq!(at.to_bits(), 0x3280_0001, "2^-26 + 2^-49, one ulp above");
  assert!(
    past <= ALPHA_FLOOR,
    "one f32 past the ceiling must NOT clear 2^-26: {past:e}"
  );
  assert_eq!(past.to_bits(), ALPHA_FLOOR.to_bits(), "exactly 2^-26");

  // The floor is a property of the whole accepted domain, not just its top: the
  // coefficient falls monotonically with `tau`, so checking it exhaustively over
  // the last 2^20 representable `tau` values below the ceiling — where it is
  // tightest — plus a ladder down to the smallest accepted `tau` covers it.
  let mut prev = 0.0f32;
  for k in 0..(1u32 << 20) {
    let tau = f32::from_bits(CadenceEma::MAX_TAU.to_bits() - k);
    let a = cadence_alpha(tau, 1);
    assert!(a > ALPHA_FLOOR, "tau {tau} yields alpha {a:e} at the floor");
    assert!(a >= prev, "alpha must not rise with tau: {a:e} < {prev:e}");
    prev = a;
  }
  for k in -149i32..=26 {
    let tau = libm::ldexpf(1.0, k);
    if tau > CadenceEma::MAX_TAU {
      continue;
    }
    let cfg = CadenceEma::new(tau);
    assert!(
      cfg.alpha_for(1) > ALPHA_FLOOR,
      "tau 2^{k} yields alpha {:e} at the floor",
      cfg.alpha_for(1)
    );
    // And `alpha` only grows with `delta`, so `delta = 1` is the floor for the
    // whole configuration.
    assert!(cfg.alpha_for(2) >= cfg.alpha_for(1));
    assert!(cfg.alpha_for(1_000_000) >= cfg.alpha_for(1));
  }
}

#[test]
fn cadence_ema_rejected_tau_still_filters_and_the_freeze_is_28_orders_further_out() {
  // The ceiling's rationale, made falsifiable rather than merely worded. It was
  // published for a while as "a `tau` past the ceiling names a filter that
  // cannot move at a unit cadence, which is a silent no-op" — and the very first
  // rejected value refutes that: `tau = 2^26` applies exactly `2^-26` per unit
  // step and the state moves by it. The ceiling is an ACCURACY boundary (the
  // coefficient lands *on* the `4 * ulp(s)` bar instead of above it); the regime
  // where a filter really stops moving is 28 binary orders further out and
  // depends on the state as much as on `tau`. Both halves are pinned here so
  // the two can never be conflated again.
  let first_rejected = next_up32(CadenceEma::MAX_TAU);
  assert_eq!(first_rejected, 67_108_864.0, "2^26");
  assert_eq!(
    CadenceEma::try_new(first_rejected),
    Err(WinditError::TimeConstantOutOfRange),
    "and it is rejected"
  );

  // Its unit coefficient is not zero and not below the floor: it is exactly on
  // it, which is the whole of what the rejection is about. At a state of `1.0`
  // the finest contrast the emitted `f32` can express is half an `f32` ulp, and
  // the step that contrast produces is exactly the published `4 * ulp(s)` —
  // level with the bar, where `MAX_TAU` clears it by the one `f32` ulp that
  // separates the two configurations.
  let alpha = cadence_alpha(first_rejected, 1);
  assert_eq!(alpha.to_bits(), ALPHA_FLOOR.to_bits(), "exactly 2^-26");
  let half_ulp32 = f64::from(ulp32(1.0)) / 2.0;
  assert_eq!(
    f64::from(alpha) * half_ulp32,
    PUBLISHED_ABSORPTION_ULPS * ulp64(1.0),
    "the first rejected tau lands exactly ON the absorption bar"
  );
  assert!(
    f64::from(cadence_alpha(CadenceEma::MAX_TAU, 1)) * half_ulp32
      > PUBLISHED_ABSORPTION_ULPS * ulp64(1.0),
    "while the ceiling itself stays above it"
  );

  // Not a no-op, and not frozen. Driven through the state directly, since the
  // configuration is — correctly — unconstructible.
  let mut from_zero = CadenceEmaState {
    tau: first_rejected,
    prev: Some((0, 0.0)),
  };
  let out = from_zero
    .push(Windowed::new(1.0, Span::new(1, 1, 1)))
    .unwrap();
  assert_eq!(
    retained(&from_zero),
    libm::ldexp(1.0, -26),
    "one unit push must move a zero state to exactly 2^-26"
  );
  assert_eq!(out.value, libm::ldexpf(1.0, -26), "and the output shows it");
  for k in 2..=100usize {
    let _ = from_zero
      .push(Windowed::new(1.0, Span::new(k, 1, 1)))
      .unwrap();
  }
  assert!(
    retained(&from_zero) > 99.0 * libm::ldexp(1.0, -26),
    "and it keeps climbing: {:e}",
    retained(&from_zero)
  );

  // It decays as well as rises: a state of `1.0` loses exactly `2^-26`.
  let mut from_one = CadenceEmaState {
    tau: first_rejected,
    prev: Some((0, 1.0)),
  };
  let _ = from_one
    .push(Windowed::new(0.0, Span::new(1, 1, 1)))
    .unwrap();
  assert_eq!(
    retained(&from_one),
    1.0 - libm::ldexp(1.0, -26),
    "and a unit push must decay a state of 1.0 by exactly 2^-26"
  );

  // The real freeze, 28 binary orders out: at `tau = 2^54` the `f64` `1 - alpha`
  // is exactly `1.0`, so the recurrence keeps no decay term at all.
  let frozen_tau = libm::ldexpf(1.0, 54);
  assert_eq!(
    1.0 - f64::from(cadence_alpha(frozen_tau, 1)),
    1.0,
    "no decay term survives at 2^54"
  );
  assert_ne!(
    1.0 - f64::from(cadence_alpha(libm::ldexpf(1.0, 53), 1)),
    1.0,
    "and 2^53 still decays, so 2^54 is the threshold"
  );
  let mut held = CadenceEmaState {
    tau: frozen_tau,
    prev: Some((0, 1.0)),
  };
  for k in 1..=64usize {
    let _ = held.push(Windowed::new(0.0, Span::new(k, 1, 1))).unwrap();
  }
  assert_eq!(
    retained(&held),
    1.0,
    "a state of order 1 is bit-identical forever at 2^54"
  );
  // But even there the freeze is a statement about STATES, not about `tau`: a
  // state of `0.0` has no ulps to absorb the increment and still moves.
  let mut still_moves = CadenceEmaState {
    tau: frozen_tau,
    prev: Some((0, 0.0)),
  };
  let _ = still_moves
    .push(Windowed::new(1.0, Span::new(1, 1, 1)))
    .unwrap();
  assert_eq!(
    retained(&still_moves),
    libm::ldexp(1.0, -54),
    "a zero state moves by exactly alpha * x even in the frozen regime"
  );
}

#[test]
fn cadence_ema_smallest_accepted_tau_saturates_to_a_pass_through() {
  // The low edge of the accepted domain. A subnormal `tau` drives `delta / tau`
  // past `expf`'s underflow at every `delta >= 1`, so `alpha` saturates to
  // exactly `1.0` and the filter tracks its input exactly — degenerate, but a
  // total and meaningful configuration (no smoothing at all), which is why the
  // domain is bounded above and not below.
  let subnormal = f32::from_bits(1);
  for tau in [subnormal, f32::MIN_POSITIVE, 1e-30] {
    let cfg = CadenceEma::new(tau);
    assert_eq!(cfg.alpha_for(1), 1.0, "tau {tau:e} must saturate alpha");
    assert!(cfg.alpha_for(1) > ALPHA_FLOOR);
    let out = drive(&cfg, &cadence_seq(&[(0, 0.5), (1, 0.25), (2, -3.0)]));
    assert_eq!(
      values(&out),
      vec![0.5, 0.25, -3.0],
      "tau {tau:e} must track the input exactly"
    );
  }
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
#[should_panic = "cadence time constant"]
fn cadence_ema_new_panics_above_max_tau() {
  let _ = CadenceEma::new(next_up32(CadenceEma::MAX_TAU));
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
  //
  // Read at `MAX_TAU`, the extreme of the accepted domain and so the smallest
  // coefficient any configuration can produce; the `1e8` this used to use is
  // above the ceiling and no longer constructs.
  let tau = CadenceEma::MAX_TAU;
  let cfg = CadenceEma::new(tau);
  assert!(
    cfg.alpha_for(1) > ALPHA_FLOOR,
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
fn cadence_ema_coefficient_and_step_floors_are_the_documented_powers_of_two() {
  // The three edge figures the *Fine cadences* bullet quotes but nothing
  // enforced: the smallest coefficient an accepted `tau` can produce, the floor
  // under `|alpha * x|`, and the zero-state escape. All three moved when the
  // domain was bounded — they are properties OF the domain, so they must be read
  // at its edge, `MAX_TAU`, and not at `f32::MAX`, which no longer constructs.

  // "the accepted domain floors it at `2^-26` for `delta = 1`": the largest
  // accepted `tau` is `MAX_TAU`, and the coefficient there is one f32 ulp above
  // `2^-26` — strictly above the bar, which is what the ceiling is chosen for.
  let widest = CadenceEma::new(CadenceEma::MAX_TAU);
  let floor = widest.alpha_for(1);
  assert!(
    floor > ALPHA_FLOOR,
    "the coefficient must clear 2^-26: {floor}"
  );
  assert_eq!(floor, ALPHA_FLOOR + libm::ldexpf(1.0, -49), "2^-26 + 2^-49");

  // "`|alpha * x|` cannot fall below about `2^-175` for a nonzero `f32` input":
  // that floor keeps a single step clear of `f64`'s subnormals (2^-1074), so
  // the relative reading of `ulp(s)` stays the operative one.
  let tiniest = f32::from_bits(1); // f32's smallest positive subnormal, 2^-149
  let product = f64::from(floor) * f64::from(tiniest);
  assert!(
    product > libm::ldexp(1.0, -175) && product < libm::ldexp(1.0, -174),
    "the documented 2^-175: {product:e}"
  );
  assert!(product > f64::MIN_POSITIVE * f64::EPSILON);

  // "a state of exactly `0.0` has no relative resolution to lose, so
  // `s = alpha * x` survives whatever `alpha` is": at the floor coefficient a
  // zero state still moves, where any nonzero state of ordinary magnitude
  // would absorb the same step outright.
  let mut s = widest.smoother();
  assert_eq!(
    s.push(Windowed::new(0.0, Span::new(0, 1, 1)))
      .unwrap()
      .value,
    0.0
  );
  let _ = s.push(Windowed::new(1.0, Span::new(1, 1, 1))).unwrap();
  assert_eq!(
    retained(&s),
    f64::from(floor),
    "zero state records alpha * x"
  );
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
  // The counterexample to any flat `delta / tau` invariance bound, rebuilt
  // INSIDE the accepted domain. A push contributes `alpha * (x - s)` into a
  // state of magnitude `|s|`, so it survives only while `alpha * |x - s|` clears
  // a small multiple of `ulp(s)` — a condition on the *product* of coefficient
  // and contrast, not on `alpha` alone, which is what this test exists to show.
  //
  // Both witnesses run at `MAX_TAU`, the smallest coefficient the domain admits
  // (`alpha_for(1)` one ulp above `2^-26`), so the same `alpha` is held fixed
  // and only the contrast varies. That is the sharpest form of the claim: same
  // tau, same alpha, two contrasts, two outcomes. The pair this replaces used
  // `tau = 2^29` and `2^30` — both now rejected at construction, and that is the
  // point of the ceiling: it removes the configurations at which a contrast the
  // emitted `f32` CAN express was absorbed, and leaves only the sub-resolution
  // ones below.
  const N: usize = 64;
  let cfg = CadenceEma::new(CadenceEma::MAX_TAU);
  assert_eq!(cfg.alpha_for(1).to_bits(), 0x3280_0001, "2^-26 + 2^-49");

  // A reachable non-`f32` state: seed 1.0, then one push of 0.0 across 1836
  // elements leaves the retained `f64` between two `f32` values.
  let seeded = || {
    let mut s = cfg.smoother();
    for (start, x) in [(0usize, 1.0f32), (1836, 0.0)] {
      let _ = s.push(Windowed::new(x, Span::new(start, 1, 1))).unwrap();
    }
    s
  };
  let p = retained(&seeded());
  assert_eq!(p.to_bits(), 0x3FEF_FFC6_A033_4000);

  // The finest contrast that exists at all: the gap between the retained `f64`
  // and its own `f32` rounding, which here is two orders below what the emitted
  // value could express. The corollary says nothing about it — and it is
  // absorbed.
  let sub = p as f32;
  let contrast = (f64::from(sub) - p).abs();
  let half_ulp32 = f64::from(ulp32(sub)) / 2.0;
  assert!(
    contrast < half_ulp32,
    "the absorbed contrast must be below f32 resolution: {contrast:e} vs {half_ulp32:e}"
  );
  assert!(
    f64::from(cfg.alpha_for(1)) * contrast < 0.5 * ulp64(p),
    "and its step must be under half an ulp of the state"
  );

  // A unit cadence absorbs every one of the `N` pushes: the retained state is
  // bit-identical to one that never saw them, and stays so — it is a fixed point
  // of its own map.
  let mut fine = seeded();
  let outs: Vec<f32> = (1..=N)
    .map(|k| {
      fine
        .push(Windowed::new(sub, Span::new(1836 + k, 1, 1)))
        .unwrap()
        .value
    })
    .collect();
  assert_eq!(
    retained(&fine).to_bits(),
    p.to_bits(),
    "a unit cadence must absorb a sub-resolution contrast entirely"
  );

  // The same `N` elements taken as one hop carry `N` times the coefficient and
  // are not absorbed. Same signal, same tau, same elapsed distance, two
  // cadences, two retained states — invariance observably broken, at a `tau` the
  // constructor accepts.
  let mut coarse = seeded();
  let coarse_out = coarse
    .push(Windowed::new(sub, Span::new(1836 + N, 1, 1)))
    .unwrap()
    .value;
  assert_ne!(
    retained(&coarse).to_bits(),
    p.to_bits(),
    "the single hop over the same elements must still move"
  );
  assert_ne!(retained(&fine).to_bits(), retained(&coarse).to_bits());
  // Both cadences still emit the same `f32`: the divergence is in the retained
  // state, which is why the assertions above are on states.
  assert!(outs.iter().all(|&v| v == sub));
  assert_eq!(coarse_out, sub);

  // And the other half of the product: at the SAME tau and the same state, a
  // contrast the emitted `f32` can express is not absorbed even at a unit
  // cadence. That is the guarantee the ceiling makes unconditional.
  for step in [1i64, -1] {
    let x = f32::from_bits((sub.to_bits() as i64 + step) as u32);
    let expressible = (f64::from(x) - p).abs();
    assert!(
      expressible >= half_ulp32,
      "{x} must be an expressible contrast"
    );
    let mut s = seeded();
    let _ = s.push(Windowed::new(x, Span::new(1837, 1, 1))).unwrap();
    assert_ne!(
      retained(&s).to_bits(),
      p.to_bits(),
      "an expressible contrast must move the state at every accepted tau"
    );
  }
}

#[test]
fn cadence_ema_two_rounded_products_absorb_a_step_the_retired_bound_admitted() {
  // Why the absorption bound is `4 * ulp(s)` and not the `ulp(s) / 2` a single
  // correctly-rounded step would give: the recurrence rounds two products and
  // their sum, so a step of nearly a whole ulp can still vanish. Pinned on the
  // exact IEEE bits, at `MAX_TAU` — inside the accepted domain, so this is not a
  // statement about configurations the constructor refuses.
  //
  // The retained state is deliberately non-dyadic and reachable from two
  // ordinary pushes; a dyadic one makes all three roundings exact and cannot
  // exhibit the absorption at all.
  const RETIRED_BOUND: f64 = 0.5;
  let cfg = CadenceEma::new(CadenceEma::MAX_TAU);
  let seed = || {
    let mut s = cfg.smoother();
    for (start, x) in [(0usize, 1000.0f32), (162_488, -1000.0)] {
      let _ = s.push(Windowed::new(x, Span::new(start, 1, 1))).unwrap();
    }
    s
  };
  let mut s = seed();
  let before = retained(&s);
  assert_eq!(before.to_bits(), 0x408F_194E_83CD_0000);

  let x = before as f32;
  assert_eq!(x.to_bits(), 0x4478_CA74);
  let alpha = f64::from(cfg.alpha_for(1));
  let contrast = (f64::from(x) - before).abs();
  let step = alpha * contrast;
  let ulp = ulp64(before);

  // The step clears the retired half-ulp bound with room, and sits inside the
  // published `4 * ulp(s)`.
  assert!(
    step > RETIRED_BOUND * ulp,
    "step {step:e} must exceed the retired half-ulp bound"
  );
  assert!(step > 0.9 * ulp, "step {} ulp", step / ulp);
  assert!(
    step < PUBLISHED_ABSORPTION_ULPS * ulp,
    "step {} ulp is inside the published bound",
    step / ulp
  );

  // And the state does not move — bit for bit, and permanently.
  let out = s.push(Windowed::new(x, Span::new(162_489, 1, 1))).unwrap();
  assert_eq!(
    retained(&s).to_bits(),
    before.to_bits(),
    "the unit step must be absorbed exactly"
  );
  assert_eq!(out.value, x);
  for k in 0..64 {
    let _ = s
      .push(Windowed::new(x, Span::new(162_490 + k, 1, 1)))
      .unwrap();
  }
  assert_eq!(
    retained(&s).to_bits(),
    before.to_bits(),
    "and stay absorbed for every further unit push"
  );

  // The witness that retired the `alpha > 2^-29` corollary is now out of domain
  // rather than merely documented: its `tau` does not construct, and at the
  // largest `tau` that does, its own contrast — one the emitted `f32` can
  // express — moves the state instead of vanishing.
  assert_eq!(
    CadenceEma::try_new(f32::from_bits(0x4DFF_FFFF)),
    Err(WinditError::TimeConstantOutOfRange),
    "the retired witness tau (just under 2^29) is above MAX_TAU"
  );
  let retired_state = f64::from_bits(0x40E7_F76B_0747_0000);
  let retired_x = f32::from_bits(0x473F_BB59);
  let retired_contrast = (f64::from(retired_x) - retired_state).abs();
  assert!(
    retired_contrast >= f64::from(ulp32(retired_state as f32)) / 2.0,
    "that contrast is expressible in the emitted f32"
  );
  let mut at_ceiling = CadenceEmaState {
    tau: CadenceEma::MAX_TAU,
    prev: Some((0, retired_state)),
  };
  let _ = at_ceiling
    .push(Windowed::new(retired_x, Span::new(1, 1, 1)))
    .unwrap();
  assert_ne!(
    retained(&at_ceiling).to_bits(),
    retired_state.to_bits(),
    "at MAX_TAU the same expressible contrast moves the state"
  );
}

#[test]
fn cadence_ema_published_absorption_bound_survives_a_randomized_sweep() {
  // The published bound, enforced rather than described: a step above
  // `4 * ulp(s)` must leave the retained `f64` different, and — now
  // unconditionally, at every accepted configuration — a contrast the emitted
  // `f32` can express must do so too.
  //
  // The sweep runs over the ACCEPTED DOMAIN and nothing else: `tau` in every
  // binade from the smallest subnormal to `MAX_TAU`, with the ceiling itself
  // forced in, rather than the `2^0..2^41` convenience range this used to walk
  // — a range that was partly outside what the constructor now admits and left
  // the domain's own edge unprobed. States are built with full 52-bit mantissas
  // at exponents spanning the useful range, and `x` is engineered to straddle
  // the bar.
  let mut rng: u64 = 0x243F_6A88_85A3_08D3;
  let mut enforced = 0u32;
  let mut corollary = 0u32;
  let mut at_ceiling = 0u32;

  for probe in 0..3_000u32 {
    // A `tau` spanning the accepted domain, and a cadence to match. Every
    // eighth probe sits exactly at the ceiling, where `alpha` is smallest and
    // the corollary is tightest.
    let tau = if probe % 8 == 0 {
      at_ceiling += 1;
      CadenceEma::MAX_TAU
    } else {
      let exp = (xorshift(&mut rng) % 176) as i32 - 149; // 2^-149 ..= 2^26
      libm::ldexpf(1.0 + next_unit(&mut rng), exp).min(CadenceEma::MAX_TAU)
    };
    let delta = 1 + (xorshift(&mut rng) % 64) as usize;
    let cfg = CadenceEma::new(tau);
    let alpha = f64::from(cfg.alpha_for(delta));
    // The domain invariant every unconditional figure rests on, asserted for
    // every probe rather than used as a filter: no accepted `tau` may derive a
    // coefficient at or below `2^-26` for `delta >= 1`.
    assert!(
      cfg.alpha_for(delta) > ALPHA_FLOOR,
      "tau {tau:e} delta {delta} derived alpha {alpha:e}, at or below the 2^-26 floor"
    );
    // A saturated coefficient has no absorption boundary to probe.
    if alpha >= 1.0 {
      continue;
    }

    // A non-dyadic retained state: full mantissa, random sign and exponent.
    let bits = xorshift(&mut rng);
    let pexp = ((xorshift(&mut rng) % 60) as i64 - 30 + 1023) as u64;
    let p = f64::from_bits(((bits >> 63) << 63) | (pexp << 52) | (bits & ((1u64 << 52) - 1)));
    if !p.is_finite() || p == 0.0 {
      continue;
    }
    let ulp = ulp64(p);

    // Offsets that put the exact step right across the published bar.
    for i in 0..24u32 {
      let target = 0.5 + f64::from(i) * 0.5; // 0.5 .. 12 ulp
      for sign in [1.0f64, -1.0] {
        let x = (p + sign * target * ulp / alpha) as f32;
        if !x.is_finite() {
          continue;
        }
        let contrast = (f64::from(x) - p).abs();
        let step = alpha * contrast;
        if !step.is_finite() || step == 0.0 {
          continue;
        }

        let mut st = CadenceEmaState {
          tau,
          prev: Some((0, p)),
        };
        let _ = st.push(Windowed::new(x, Span::new(delta, 1, 1))).unwrap();
        let moved = retained(&st).to_bits() != p.to_bits();

        if step > PUBLISHED_ABSORPTION_ULPS * ulp {
          enforced += 1;
          assert!(
            moved,
            "published bound violated: tau {tau} delta {delta} p {p:?} x {x:?} \
             step {step:?} = {:.4} ulp",
            step / ulp
          );
        }
        // Half an `f32` ulp at this magnitude is the finest contrast the
        // emitted value can express. No `alpha` guard: the accepted domain
        // supplies it, which is the whole point of bounding `tau`.
        let half_ulp32 = f64::from(ulp32(p as f32)) / 2.0;
        if contrast >= half_ulp32 {
          corollary += 1;
          assert!(
            moved,
            "corollary violated: tau {tau} delta {delta} alpha {alpha:?} with an \
             expressible contrast {contrast:?} on p {p:?} did not move the state"
          );
        }
      }
    }
  }

  // No arm may pass vacuously, and the ceiling must actually have been visited.
  assert!(
    enforced > 10_000,
    "too few probes above the bar: {enforced}"
  );
  assert!(corollary > 1_000, "too few corollary probes: {corollary}");
  assert_eq!(at_ceiling, 375, "the domain edge must be swept");
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
  // 8 * 2^-25. The worst ratio measured over the whole swept grid — `tau` 3 to
  // 10007 at `tau/4`..`12 tau`, and the power-of-two ladder to 2^20 at
  // `tau/4`..`4 tau` — was 2x; this leaves two binary orders of headroom.
  const SWING_ERR: f64 = 8.0 / 33_554_432.0;

  // `x0` at position 0, then a constant `x1` over `d` elements at `hop` cadence.
  let sampled = |tau: f32, hop: usize, d: usize, x0: f32, x1: f32| -> f32 {
    let cfg = CadenceEma::new(tau);
    let mut samples: Vec<(usize, f32)> = vec![(0, x0)];
    let mut p = 0usize;
    while p + hop <= d {
      p += hop;
      samples.push((p, x1));
    }
    drive(&cfg, &cadence_seq(&samples)).last().unwrap().value
  };

  // The absolute bound, over the `tau` range and distances the claim names —
  // not at one `tau`, which is all this used to check and which cannot enforce
  // a statement quantified over a range.
  for tau in [3.0f32, 7.0, 17.5, 64.0, 251.0, 1024.0, 4093.0, 10007.0] {
    let n = tau as usize;
    for mult in [1usize, 2, 4, 8, 12] {
      let d = n * mult;
      for (x0, x1) in [(0.0f32, 1.0f32), (1.0, 0.0), (-1.0, 1.0), (2.0, -2.0)] {
        let fine = sampled(tau, 1, d, x0, x1);
        let coarse = sampled(tau, d, d, x0, x1);
        let bound = SWING_ERR * f64::from(x1 - x0).abs();
        assert!(
          (f64::from(fine) - f64::from(coarse)).abs() <= bound,
          "tau {tau}, d/tau {mult}, {x0} -> {x1}: |{fine} - {coarse}| exceeds {bound}"
        );
      }
    }
  }
  // The quarter-tau distance the claim also names, kept separate because
  // `n * mult` cannot express it.
  for tau in [64.0f32, 1024.0, 10007.0] {
    let d = (tau as usize) / 4;
    for (x0, x1) in [(0.0f32, 1.0f32), (1.0, 0.0)] {
      let fine = sampled(tau, 1, d, x0, x1);
      let coarse = sampled(tau, d, d, x0, x1);
      let bound = SWING_ERR * f64::from(x1 - x0).abs();
      assert!(
        (f64::from(fine) - f64::from(coarse)).abs() <= bound,
        "tau {tau}, d = tau/4, {x0} -> {x1}: |{fine} - {coarse}| exceeds {bound}"
      );
    }
  }
  // Up the `tau` ladder, driven push by push rather than materialized: a fine
  // cadence over `4 tau` at `tau = 2^20` is four million windows. The claim is
  // about the whole accepted domain, and the top of it — one time constant at
  // `MAX_TAU`, where the error peaks — is enforced by
  // `cadence_ema_one_tau_decay_lands_within_four_ulps_of_exp_minus_one`, whose
  // `4`-ulp bound at `exp(-1)` IS this bound at `d = tau` for a unit swing.
  for k in 10..=20u32 {
    let tau = libm::ldexpf(1.0, k as i32);
    let n = tau as usize;
    for (num, den) in [(1usize, 4usize), (1, 1), (2, 1), (4, 1)] {
      let d = n * num / den;
      for (x0, x1) in [(0.0f32, 1.0f32), (1.0, 0.0)] {
        let fine = streamed(tau, 1, d, x0, x1);
        let coarse = streamed(tau, d, d, x0, x1);
        let bound = SWING_ERR * f64::from(x1 - x0).abs();
        assert!(
          (f64::from(fine) - f64::from(coarse)).abs() <= bound,
          "tau 2^{k}, d = {num}/{den} tau, {x0} -> {x1}: |{fine} - {coarse}| exceeds {bound}"
        );
      }
    }
  }

  let n = TAU as usize;
  let sampled = |hop: usize, d: usize, x0: f32, x1: f32| sampled(TAU, hop, d, x0, x1);

  // The same absolute agreement in ulps of the result, at both ends. The
  // rising direction happens to agree to an ulp at this `tau`; that is a
  // property of this geometry, NOT the general one-tau bound — the falling
  // direction reaches 2 ulps at `tau = 238`, which
  // `cadence_ema_one_tau_decay_lands_within_four_ulps_of_exp_minus_one`
  // measures across the range.
  let (fine, coarse) = (sampled(1, n, 0.0, 1.0), sampled(n, n, 0.0, 1.0));
  assert!(
    ulps32(fine, coarse).abs() <= 1,
    "a result of order the swing must agree to an ulp here: {fine} vs {coarse}"
  );
  // At `d / tau = 12` the residual is exponentially smaller than the error, so
  // the same absolute agreement reads as ~10^4 ulps. Bounded on BOTH sides: a
  // bare `> 1_000` would also admit 10^6 and so could not enforce the figure
  // the docs quote. Swept over the `tau` range rather than read at one `tau`,
  // since the docs quote it for the range (measured 10832..15746, the top of
  // that from a fractional `tau` whose `d / tau` lands at 11.66).
  for tau in [3.0f32, 17.5, 251.0, 1024.0, 10007.0, 65_536.0] {
    let d = (tau as usize) * 12;
    let (fine, coarse) = (streamed(tau, 1, d, 1.0, 0.0), streamed(tau, d, d, 1.0, 0.0));
    let apart = ulps32(fine, coarse).abs();
    assert!(
      (5_000..=20_000).contains(&apart),
      "tau {tau}: a residual decayed by 12 tau must be ~10^4 ulps apart, \
       got {apart}: {fine} vs {coarse}"
    );
  }
}

/// Seed `x0` at position 0, then a constant `x1` every `hop` elements out to
/// `d`, driven push by push rather than materialized: the long cadences below
/// run to millions of windows, which as a `Vec` would be gigabytes.
fn streamed(tau: f32, hop: usize, d: usize, x0: f32, x1: f32) -> f32 {
  let mut s = CadenceEma::new(tau).smoother();
  let mut out = s.push(Windowed::new(x0, Span::new(0, 1, 1))).unwrap().value;
  let mut p = 0usize;
  while p + hop <= d {
    p += hop;
    out = s.push(Windowed::new(x1, Span::new(p, 1, 1))).unwrap().value;
  }
  out
}

/// Drive a falling unit cadence from a seed of `1.0` over `n` elements, push by
/// push rather than materialized: one `tau` at a unit cadence can run to tens of
/// millions of pushes, and that many `Windowed`s would be gigabytes.
fn one_tau_fine(cfg: &CadenceEma, n: usize) -> f32 {
  let mut s = cfg.smoother();
  let mut out = s
    .push(Windowed::new(1.0, Span::new(0, 1, 1)))
    .unwrap()
    .value;
  assert_eq!(out, 1.0, "seed must be s_0 = x_0");
  for p in 1..=n {
    out = s
      .push(Windowed::new(0.0, Span::new(p, 1, 1)))
      .unwrap()
      .value;
  }
  out
}

/// The same elapsed distance taken as a single `delta = n` step.
fn one_tau_coarse(cfg: &CadenceEma, n: usize) -> f32 {
  drive(cfg, &cadence_seq(&[(0, 1.0), (n, 0.0)]))
    .last()
    .unwrap()
    .value
}

#[test]
fn cadence_ema_one_tau_decay_lands_within_four_ulps_of_exp_minus_one() {
  // The defining decay, over the full time constant: a unit cadence must reach
  // `exp(-1)` after `tau` elements much as a single `delta = tau` step does.
  //
  // "Much as", not "to an ulp". The claim this replaces said the two land on
  // `exp(-1)` *within one* ulp, and was enforced by a `1e-6` tolerance — over
  // 33 ulps at this magnitude, so slack enough to admit any of the answers
  // below.
  //
  // The sweep is quantified over the ACCEPTED DOMAIN, not over a convenience
  // range: every integer `tau` from 1 to 1024, sampled fractional `tau`, a
  // power-of-two ladder, and `MAX_TAU` itself — the largest `tau` this type
  // admits. That is the fix for how this claim was falsified: it was measured
  // over `tau` to 4e7 and stated without a limit, while the constructor took
  // every positive finite `f32`, so `tau = 2^55` — where a unit cadence cannot
  // move the state at all and the two cadences end millions of ulps apart —
  // satisfied the type and contradicted the figure. The domain now stops below
  // that regime and this sweep runs to the new edge. Measured worst case over
  // it: 2 ulps from the closed form and 2 between the cadences; `4` is the
  // conservative figure published, enforced here in exact representable steps
  // rather than through a tolerance.
  const BOUND: i64 = 4;
  let inv_e = (1.0f64 / core::f64::consts::E) as f32;
  assert_eq!(inv_e.to_bits(), 0x3EBC_5AB2, "nearest f32 to exp(-1)");

  // The two exact witnesses that falsify "within one": both land two
  // representable values below `exp(-1)`, and two below the coarse step.
  for tau in [14usize, 238] {
    let cfg = CadenceEma::new(tau as f32);
    let fine = one_tau_fine(&cfg, tau);
    let coarse = one_tau_coarse(&cfg, tau);
    assert_eq!(fine.to_bits(), 0x3EBC_5AB0, "tau {tau} fine");
    assert_eq!(coarse.to_bits(), 0x3EBC_5AB2, "tau {tau} coarse");
    assert_eq!(ulps32(fine, inv_e), -2, "tau {tau} vs exp(-1)");
    assert_eq!(ulps32(fine, coarse), -2, "tau {tau} fine vs coarse");
  }

  // Every integer tau across the swept range, plus fractional ones, in exact
  // ulps. `n = tau` keeps the elapsed distance at exactly one time constant.
  for tau in 1..=1_024usize {
    let cfg = CadenceEma::new(tau as f32);
    let fine = one_tau_fine(&cfg, tau);
    let coarse = one_tau_coarse(&cfg, tau);
    assert!(
      ulps32(fine, inv_e).abs() <= BOUND,
      "tau {tau}: fine {fine} is {} ulp from exp(-1)",
      ulps32(fine, inv_e)
    );
    assert!(
      ulps32(fine, coarse).abs() <= BOUND,
      "tau {tau}: cadences {} ulp apart ({fine} vs {coarse})",
      ulps32(fine, coarse)
    );
  }

  // Fractional taus, where one time constant is not a whole number of elements
  // and the target is `exp(-n / tau)` rather than `exp(-1)`.
  let mut rng: u64 = 0xB5AD_4ECE_DA10_1010;
  for _ in 0..256 {
    let tau = 1.0 + next_unit(&mut rng) * 1_023.0;
    let n = tau as usize;
    let cfg = CadenceEma::new(tau);
    let fine = one_tau_fine(&cfg, n);
    let coarse = one_tau_coarse(&cfg, n);
    let want = libm::exp(-(n as f64) / f64::from(tau)) as f32;
    assert!(
      ulps32(fine, want).abs() <= BOUND,
      "tau {tau}: fine {fine} is {} ulp from exp(-{n}/tau) {want}",
      ulps32(fine, want)
    );
    assert!(
      ulps32(fine, coarse).abs() <= BOUND,
      "tau {tau}: cadences {} ulp apart ({fine} vs {coarse})",
      ulps32(fine, coarse)
    );
  }

  // The deep regime the type exists for, walked up to the edge of the domain.
  // Past `tau = 2^25` the per-step coefficient is smaller than the state's own
  // `f32` resolution, so an `f32` accumulator would not move at all and one
  // `tau` at a unit cadence costs tens of millions of pushes — which is exactly
  // why this end went unswept before, and exactly where the claim died.
  for k in 11..=25u32 {
    let tau = libm::ldexpf(1.0, k as i32);
    let cfg = CadenceEma::new(tau);
    let n = tau as usize;
    let fine = one_tau_fine(&cfg, n);
    let coarse = one_tau_coarse(&cfg, n);
    assert!(
      ulps32(fine, inv_e).abs() <= BOUND,
      "tau 2^{k}: fine {fine} is {} ulp from exp(-1)",
      ulps32(fine, inv_e)
    );
    assert!(
      ulps32(fine, coarse).abs() <= BOUND,
      "tau 2^{k}: cadences {} ulp apart ({fine} vs {coarse})",
      ulps32(fine, coarse)
    );
  }

  // The edge itself: the largest `tau` the constructor accepts, where the
  // coefficient is one ulp above the `2^-26` floor the whole domain is built
  // around. Pinned on exact bits — one representable value below `exp(-1)`, and
  // the two cadences one apart — so this cannot regress silently.
  let cfg = CadenceEma::new(CadenceEma::MAX_TAU);
  let n = CadenceEma::MAX_TAU as usize;
  assert_eq!(n, 67_108_860);
  let fine = one_tau_fine(&cfg, n);
  let coarse = one_tau_coarse(&cfg, n);
  assert_eq!(ulps32(fine, inv_e), -1, "MAX_TAU vs exp(-1)");
  assert_eq!(ulps32(fine, coarse), -1, "MAX_TAU fine vs coarse");
}

#[test]
fn cadence_ema_coefficient_stays_within_two_ulps_of_the_exact_one() {
  // The `cadence_alpha` comment's accuracy figure, enforced over an ENUMERATED
  // boundary set rather than a random draw: the `f32` cast boundaries from 2^24
  // up, the ratio's binade edges, the accepted domain's binade edges, and the
  // ceiling itself. The randomized predecessor of this sweep drew 20_000 probes
  // and reported the figure held; it never landed on a cast boundary, where it
  // does not. Sampling the interior of a domain cannot enforce a claim about
  // its edges, so this one walks the edges and nothing else.
  //
  // The reference is the same function evaluated in `f64`, an independent
  // precision path: `delta` is exact there for every count below 2^53, so its
  // own error is under 2^-28 of an `f32` ulp and cannot colour the figure.
  const BOUND: f64 = 2.0;
  // Below this the sweep would not be reaching the hard region at all — the
  // narrowing alone costs up to a full ulp and `expm1f` adds its own.
  const REACHED: f64 = 1.4;
  let mut worst = 0.0f64;
  let mut worst_at = (0.0f32, 0usize);
  let mut retired_worst = 0.0f64;
  let mut retired_breaches = 0u32;
  let mut probed = 0u32;
  let mut past_cast = 0u32;

  for tau in boundary_taus() {
    let cfg = CadenceEma::new(tau);
    for delta in boundary_deltas(tau) {
      let want = -libm::expm1(-(delta as f64) / f64::from(tau));
      // A saturated coefficient is exactly 1.0 on every path, with no ulp to
      // measure against; a zero one cannot arise, since `expm1f` is exact in
      // that regime.
      if !(0.0..1.0).contains(&want) || want == 0.0 {
        continue;
      }
      probed += 1;
      if delta > (1 << 24) {
        past_cast += 1;
      }

      let apart = coefficient_ulps(cfg.alpha_for(delta), want);
      assert!(
        apart <= BOUND,
        "tau {tau:e} delta {delta}: alpha is {apart:.4} f32 ulps from {want:e}"
      );
      if apart > worst {
        worst = apart;
        worst_at = (tau, delta);
      }

      // The same point measured against the spelling this derivation replaced.
      // Its breaches are counted, not asserted away: they are what makes this
      // enumeration a regression rather than a description.
      let retired = coefficient_ulps(cadence_alpha_via_f32_cast(tau, delta), want);
      retired_worst = retired_worst.max(retired);
      if retired > BOUND {
        retired_breaches += 1;
      }
    }
  }

  // No arm may pass vacuously: the enumeration must be large, must reach past
  // the cast boundary, and must reach the hard region rather than skirt it.
  assert!(probed > 100_000, "too few unsaturated probes: {probed}");
  assert!(
    past_cast > 50_000,
    "too few probes past the cast: {past_cast}"
  );
  assert!(
    worst > REACHED,
    "the sweep never reached the hard region: worst {worst:.4} ulps at tau \
     {:e} delta {}",
    worst_at.0,
    worst_at.1
  );
  // And the retired spelling must fail on this very set — the sweep is only
  // evidence for the shipped derivation if it can tell the two apart.
  assert!(
    retired_breaches > 50 && retired_worst > 2.2,
    "the f32-cast spelling must breach the bound on the enumerated boundaries: \
     {retired_breaches} breaches, worst {retired_worst:.4} ulps"
  );
}

#[test]
fn cadence_ema_coefficient_never_rounds_delta_through_f32() {
  // The cast-boundary regression, on the exact witness that falsified the
  // published two-ulp figure. `delta` is a usize count of elements; putting it
  // through an `f32` rounded the caller's *data* before the configuration was
  // applied to it, and no `f32` holds an odd integer above 2^24.
  const DELTA: usize = 16_812_203;
  let cfg = CadenceEma::new(CadenceEma::MAX_TAU);

  assert_eq!(DELTA as f32, 16_812_204.0, "the cast lands an element away");
  const { assert!(DELTA > (1 << 24) && DELTA % 2 == 1) };
  assert_eq!(DELTA as f64, 16_812_203.0, "f64 holds it exactly");

  let want = -libm::expm1(-(DELTA as f64) / f64::from(CadenceEma::MAX_TAU));
  let retired = cadence_alpha_via_f32_cast(CadenceEma::MAX_TAU, DELTA);
  let shipped = cfg.alpha_for(DELTA);
  assert_eq!(retired.to_bits(), 0x3E62_EC78, "the retired cast spelling");
  assert_eq!(shipped.to_bits(), 0x3E62_EC76, "the shipped f64 ratio");
  assert!(
    coefficient_ulps(retired, want) > 2.2,
    "the retired spelling must be the failing one: {:.4} ulps",
    coefficient_ulps(retired, want)
  );
  assert!(
    coefficient_ulps(shipped, want) < 0.5,
    "the shipped one must be within half an ulp here: {:.4} ulps",
    coefficient_ulps(shipped, want)
  );

  // `alpha_for` still reports exactly the coefficient the state applies, read
  // in the region the derivation changed: the emitted value and the retained
  // `f64` are both reproduced from the reported coefficient alone.
  let mut s = cfg.smoother();
  assert_eq!(
    s.push(Windowed::new(0.25, Span::new(0, 1, 1)))
      .unwrap()
      .value,
    0.25,
    "seed s_0 = x_0"
  );
  let out = s
    .push(Windowed::new(1.0, Span::new(DELTA, 1, 1)))
    .unwrap()
    .value;
  let alpha = f64::from(shipped);
  let expected = alpha * 1.0 + (1.0 - alpha) * 0.25;
  assert_eq!(
    retained(&s),
    expected,
    "the applied coefficient is alpha_for"
  );
  assert_eq!(out, expected as f32);

  // Below the cast boundary the two spellings agree bit for bit at every
  // enumerated `tau`, which is why nothing else on the type moved: the change
  // is confined to the counts an `f32` cannot hold. The agreement is a theorem
  // rather than a lucky sample — `delta` is exactly representable there, so both
  // round the same quotient of two `f32`s, and `f64` carries more than twice
  // `f32`'s significand — and this checks the corner of it the rest of the suite
  // depends on.
  let below: Vec<usize> = (1..=512)
    .chain((1 << 24) - 512..=(1 << 24))
    .chain([1 << 20, 1 << 23, 12_345_678])
    .collect();
  for tau in boundary_taus() {
    for &delta in &below {
      assert_eq!(
        cadence_alpha(tau, delta).to_bits(),
        cadence_alpha_via_f32_cast(tau, delta).to_bits(),
        "tau {tau:e} delta {delta} must be unchanged below the cast boundary"
      );
    }
  }

  // Above it they diverge on enumerated boundaries, so the agreement below is a
  // property of the domain and not of the two spellings being the same code.
  let mut diverged = 0u32;
  for tau in boundary_taus() {
    for delta in boundary_deltas(tau) {
      if delta > (1 << 24)
        && cadence_alpha(tau, delta).to_bits() != cadence_alpha_via_f32_cast(tau, delta).to_bits()
      {
        diverged += 1;
      }
    }
  }
  assert!(
    diverged > 1_000,
    "the two spellings must differ above 2^24: {diverged}"
  );
}

#[test]
fn one_minus_alpha_collapses_at_two_pow_minus_25_in_f32_and_two_pow_minus_54_in_f64() {
  // The degeneracy thresholds both smoothers' docs quote. `Ema` says an `alpha`
  // of `2^-25` "or below" makes the `f32` `1 - alpha` round to exactly 1.0, and
  // that `CadenceEma`'s `f64` state pushes the same degeneracy 29 binary orders
  // further out. Only the single `f32` value at `2^-25` was pinned, and the
  // `f64` threshold was quoted as `2^-53`, which is wrong: `1 - 2^-53` is
  // exactly representable, so it does not collapse. `2^-54` is the tie that
  // rounds to even, and 25 to 54 is 29 orders, not 28.
  assert_ne!(1.0f32 - libm::ldexpf(1.0, -24), 1.0, "2^-24 must survive");
  assert_ne!(1.0f64 - libm::ldexp(1.0, -53), 1.0, "2^-53 must survive");
  assert_eq!(
    1.0f64 - libm::ldexp(1.0, -53),
    f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)
  );

  // "or below", swept rather than sampled at one exponent: from each threshold
  // down to the smallest subnormal of its type.
  for k in 25..=149i32 {
    assert_eq!(
      1.0f32 - libm::ldexpf(1.0, -k),
      1.0,
      "f32: 1 - 2^-{k} must collapse"
    );
  }
  for k in 54..=1074i32 {
    assert_eq!(
      1.0f64 - libm::ldexp(1.0, -k),
      1.0,
      "f64: 1 - 2^-{k} must collapse"
    );
  }
  assert_eq!(54 - 25, 29, "29 binary orders, as the Ema doc states");

  // And the collapse is unreachable through `CadenceEma`: the `tau` that first
  // produces it does not construct, and the coefficient at the ceiling clears
  // the `f64` threshold by the 2^28 the domain is built to keep.
  assert_eq!(
    CadenceEma::try_new(libm::ldexpf(1.0, 54)),
    Err(WinditError::TimeConstantOutOfRange)
  );
  let at_ceiling = f64::from(CadenceEma::new(CadenceEma::MAX_TAU).alpha_for(1));
  assert_ne!(1.0 - at_ceiling, 1.0, "the ceiling must still decay");
  assert!(at_ceiling > libm::ldexp(1.0, -54) * f64::from(1u32 << 28));
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

// ---------------------------------------------------------------------------
// VectorEma: the renormalizing, span-preserving sibling of the aggregation
// half's `EmaRenormalized`.
// ---------------------------------------------------------------------------

/// One `Windowed<RawF64Emb>` covering a single element at `start`.
///
/// [`RawF64Emb`] rather than [`TestVec`] on purpose. Its `from_unnormalized`
/// captures the compute-domain slice **verbatim** instead of normalizing it, so
/// these tests read exactly what the smoother emitted. An embedding double that
/// normalized in its own `from_unnormalized` would return a unit vector whether
/// or not the smoother renormalized, and could not falsify the renormalization
/// at all — the property that separates this type from a plain component-wise
/// vector EMA would be untestable through it.
fn vec_window(values: &[f64], start: usize) -> Windowed<RawF64Emb> {
  Windowed::new(
    RawF64Emb {
      data: values.to_vec(),
      captured: Vec::new(),
    },
    Span::new(start, 1, 1),
  )
}

/// What a pushed window emitted, at full `f64` precision.
fn emitted(w: &Windowed<RawF64Emb>) -> &[f64] {
  &w.value.captured
}

/// The L2 norm, spelled independently of the crate's own scale-aware routine so
/// a unit-norm assertion is not checked by the code it is checking.
fn l2(v: &[f64]) -> f64 {
  libm::sqrt(v.iter().map(|x| x * x).sum::<f64>())
}

#[test]
fn vector_ema_renormalizes_every_window() {
  // THE falsifier for "renormalized", the one property that separates this from
  // a plain component-wise vector EMA. `[1, 0]` then `[0, 1]` at alpha 0.5
  // leaves the accumulator at exactly `[0.5, 0.5]` — norm 2^-0.5, emphatically
  // not 1 — so an emit that skipped the renormalization reads back as
  // `[0.5, 0.5]` here and both assertions below fail. Non-vacuous by
  // construction: the seed at index 0 is already unit-norm, so only the second
  // window can distinguish the two implementations, and it is the one asserted.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let first = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  let second = s.push(vec_window(&[0.0, 1.0], 1)).unwrap();

  // One ulp below `core::f64::consts::FRAC_1_SQRT_2`, and pinned as the
  // division rather than as the constant because that is what the arithmetic
  // is: renormalization divides by the norm, and `1.0 / sqrt(2)` is not the
  // correctly rounded `1/sqrt(2)`.
  let want = 1.0 / libm::sqrt(2.0);
  assert_eq!(
    emitted(&first),
    &[1.0, 0.0],
    "the seed emits its own unit direction"
  );
  assert_eq!(
    emitted(&second),
    &[want, want],
    "the accumulator is [0.5, 0.5]; the emitted window must be its direction"
  );
  // Restated as the norm, so a failure names the property rather than only the
  // golden: an unrenormalized emit has norm 2^-0.5.
  let norm = l2(emitted(&second));
  assert!(
    (norm - 1.0).abs() <= f64::EPSILON,
    "emitted window must be unit-norm, got {norm}"
  );
}

#[test]
fn vector_ema_seeds_the_accumulator_with_the_first_window() {
  // `s_0 = x_0`, not `alpha * x_0` and not a zero start. At `alpha = 0` the
  // accumulator can never move, so every later window must still emit the FIRST
  // window's direction however far the input has travelled — and an
  // accumulator that started at zero would stay at zero and be gated as
  // indeterminate instead of emitting anything at all.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.0));
  for (i, x) in [[3.0, 4.0], [0.0, -1.0], [-5.0, 12.0]].iter().enumerate() {
    let out = s.push(vec_window(x, i)).unwrap();
    assert_eq!(
      emitted(&out),
      &[0.6, 0.8],
      "alpha 0 holds the seed direction at window {i}"
    );
  }

  // The other endpoint: at `alpha = 1` the accumulator IS the input, so every
  // window emits its own direction. Together the two pin the clamp's ends
  // behaviourally rather than only through the `alpha()` accessor.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(1.0));
  assert_eq!(
    emitted(&s.push(vec_window(&[3.0, 4.0], 0)).unwrap()),
    &[0.6, 0.8]
  );
  assert_eq!(
    emitted(&s.push(vec_window(&[0.0, -2.0], 1)).unwrap()),
    &[0.0, -1.0]
  );
}

#[test]
fn vector_ema_reset_and_discontinuity_reseed_the_accumulator() {
  // Both return the smoother to `s_0 = x_0`: the window after the break emits
  // its own direction, not the recurrence against the pre-break accumulator.
  // Without the re-seed the third push would mix `[0, 1]` into the retained
  // `[0.5, 0.5]`, giving `[0.25, 0.75]` — direction `[0.316…, 0.948…]`, which
  // is neither `[0, 1]` nor within any tolerance of it.
  for reseed in [Smoother::reset, Smoother::discontinuity] {
    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
    let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
    let _ = s.push(vec_window(&[0.0, 1.0], 1)).unwrap();
    reseed(&mut s);
    let after = s.push(vec_window(&[0.0, 4.0], 2)).unwrap();
    assert_eq!(
      emitted(&after),
      &[0.0, 1.0],
      "a re-seed must restore s_0 = x_0"
    );
  }
}

#[test]
fn vector_ema_preserves_spans_and_batch_equals_streaming() {
  // Deliberately irregular spans (0, 5, 11) with a window wider than one
  // element: a stage that regenerated spans instead of carrying the input's
  // would have to reproduce this exact sequence by accident.
  let input = vec![
    Windowed::new(
      RawF64Emb {
        data: vec![1.0, 0.0, 0.0],
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      RawF64Emb {
        data: vec![0.0, 1.0, 0.0],
        captured: Vec::new(),
      },
      Span::new(5, 4, 4),
    ),
    Windowed::new(
      RawF64Emb {
        data: vec![0.0, 0.0, 1.0],
        captured: Vec::new(),
      },
      Span::new(11, 2, 4),
    ),
  ];
  let batch = VectorEma::new(0.4).smooth(&input).unwrap();
  assert_eq!(
    batch.iter().map(|w| w.span).collect::<Vec<_>>(),
    input.iter().map(|w| w.span).collect::<Vec<_>>(),
    "spans must survive the rewrite untouched"
  );

  // The batch method IS a fresh smoother driven over the slice, so the two must
  // agree component for component.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.4));
  let streamed: Vec<Windowed<RawF64Emb>> =
    input.iter().map(|w| s.push(w.clone()).unwrap()).collect();
  for (b, t) in batch.iter().zip(&streamed) {
    assert_eq!(emitted(b), emitted(t));
    assert_eq!(b.span, t.span);
  }
  // Non-vacuity: the sequence genuinely mixes, so the three outputs differ from
  // each other and from the inputs.
  assert_ne!(emitted(&batch[1]), &[0.0, 1.0, 0.0]);
  assert_ne!(emitted(&batch[1]), emitted(&batch[2]));
}

#[test]
fn vector_ema_rejects_a_width_change_and_leaves_the_accumulator_unchanged() {
  // The first push fixes the epoch's dimension; a later window of another width
  // has no component to mix into and is refused.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  assert_eq!(
    s.push(vec_window(&[1.0, 0.0, 0.0], 1)).unwrap_err(),
    WinditError::DimMismatch {
      got: 3,
      expected: 2
    }
  );
  assert_eq!(
    s.push(vec_window(&[1.0], 2)).unwrap_err(),
    WinditError::DimMismatch {
      got: 1,
      expected: 2
    }
  );

  // Checked BEFORE any component is written, so the two refused pushes are
  // no-ops: the next in-width window emits exactly what it would have emitted
  // had they never happened. Without this assertion a guard that rejected only
  // after mutating the accumulator would still pass.
  let want = 1.0 / libm::sqrt(2.0);
  let after = s.push(vec_window(&[0.0, 1.0], 3)).unwrap();
  assert_eq!(emitted(&after), &[want, want]);
}

#[test]
fn vector_ema_rejects_a_zero_width_window() {
  // A zero-dimension embedding is structurally empty rather than
  // direction-less, so it reports `Empty` — and refusing it is what keeps an
  // epoch from "seeding" with an accumulator that is still empty, which is
  // exactly the unseeded state.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  assert_eq!(s.push(vec_window(&[], 0)).unwrap_err(), WinditError::Empty);
  // Still unseeded: the next window seeds the epoch itself.
  assert_eq!(
    emitted(&s.push(vec_window(&[0.0, 3.0], 1)).unwrap()),
    &[0.0, 1.0]
  );
}

#[test]
fn vector_ema_new_clamps_alpha_into_range() {
  assert_eq!(VectorEma::new(2.0).alpha(), 1.0);
  assert_eq!(VectorEma::new(f32::INFINITY).alpha(), 1.0);
  assert_eq!(VectorEma::new(-1.0).alpha(), 0.0);
  assert_eq!(VectorEma::new(f32::NEG_INFINITY).alpha(), 0.0);
  assert_eq!(VectorEma::new(f32::NAN).alpha(), 0.0);
  assert_eq!(VectorEma::new(0.25).alpha(), 0.25);

  // The re-clamp inside `smoother()` is unreachable through `new`, exactly as
  // `Ema`'s is, so it is exercised the only way it can be: by building the
  // config through its private field, which this module can do and downstream
  // cannot. Without it a future construction path that bypassed `new` (a serde
  // derive, say) would hand the recurrence a NaN coefficient.
  let bypassed: VectorEmaState<RawF64Emb> = VectorEma { alpha: f32::NAN }.smoother();
  assert_eq!(bypassed.alpha, 0.0);
  let bypassed: VectorEmaState<RawF64Emb> = VectorEma { alpha: 4.0 }.smoother();
  assert_eq!(bypassed.alpha, 1.0);
  let bypassed: VectorEmaState<RawF64Emb> = VectorEma { alpha: -4.0 }.smoother();
  assert_eq!(bypassed.alpha, 0.0);
}

#[test]
fn vector_ema_nan_alpha_holds_the_seed_rather_than_poisoning_the_stream() {
  // A NaN `alpha` clamps to 0.0 — hold the seed — rather than propagating. It
  // matters that this is asserted through a *second* push: a NaN coefficient
  // would make the accumulator all-NaN, whose largest magnitude is zero, so the
  // determinacy gate would reject it and this would be `Err(NonFinite)` rather
  // than a wrong value.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(f32::NAN));
  assert_eq!(
    emitted(&s.push(vec_window(&[0.0, 2.0], 0)).unwrap()),
    &[0.0, 1.0]
  );
  assert_eq!(
    emitted(&s.push(vec_window(&[7.0, 0.0], 1)).unwrap()),
    &[0.0, 1.0]
  );
}

#[test]
fn vector_ema_rejects_a_non_finite_window_and_leaves_the_accumulator_unchanged() {
  // Unlike the scalar `Ema`, which documents a non-finite input poisoning its
  // state until a reset, a non-finite component is refused here before the
  // accumulator is touched.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  for bad in [
    [f64::NAN, 0.0],
    [0.0, f64::INFINITY],
    [f64::NEG_INFINITY, 1.0],
  ] {
    assert_eq!(
      s.push(vec_window(&bad, 1)).unwrap_err(),
      WinditError::NonFinite
    );
  }
  // The epoch is intact: the next finite window sees the seed, not a poisoned
  // accumulator. Drop the finiteness guard and the accumulator is NaN here, the
  // gate rejects it, and this `unwrap` panics.
  let want = 1.0 / libm::sqrt(2.0);
  let after = s.push(vec_window(&[0.0, 1.0], 2)).unwrap();
  assert_eq!(emitted(&after), &[want, want]);

  // A non-finite *seed* is refused the same way, leaving the smoother unseeded
  // so the next window seeds the epoch itself.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  assert_eq!(
    s.push(vec_window(&[f64::NAN, 1.0], 0)).unwrap_err(),
    WinditError::NonFinite
  );
  assert_eq!(
    emitted(&s.push(vec_window(&[0.0, 2.0], 1)).unwrap()),
    &[0.0, 1.0]
  );
}

#[test]
fn vector_ema_gate_rejects_an_indeterminate_direction() {
  // Exact cancellation: `[1, 0]` then `[-1, 0]` at alpha 0.5 leaves the
  // accumulator at exactly zero, which has no direction to emit.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  assert_eq!(
    s.push(vec_window(&[-1.0, 0.0], 1)).unwrap_err(),
    WinditError::NonFinite
  );

  // And the case an exact-zero test could never catch: a residue that is
  // NONZERO but below the fold's own rounding floor. The accumulator here is
  // ~5.0e-16 while the gate stands at 16 * EPSILON * ||M|| ~= 3.55e-15, so
  // without the gate `l2_renorm` would amplify it into the fabricated unit
  // direction `[1, 0]` and the push would succeed.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  assert_eq!(
    s.push(vec_window(&[-1.0 + 1e-15, 0.0], 1)).unwrap_err(),
    WinditError::NonFinite
  );

  // Non-vacuity: the gate has a real boundary rather than rejecting everything
  // small. One decimal order out — an accumulator of ~5.0e-15 against the same
  // ~3.55e-15 threshold — the residue is determinate and the direction is
  // emitted.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  let out = s.push(vec_window(&[-1.0 + 1e-14, 0.0], 1)).unwrap();
  assert_eq!(emitted(&out), &[1.0, 0.0]);
}

#[test]
fn vector_ema_gate_failure_still_advances_the_accumulator() {
  // The one push that is NOT a no-op on failure, and deliberately so: a window
  // whose output was gated was still a real observation, and the epoch the
  // aggregate would fold over includes it. After the exact cancellation above,
  // the accumulator is zero, so the next window's direction is its own —
  // whereas a rolled-back accumulator would still hold `[1, 0]` and mix into
  // `[0.5, 0.5]`.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let _ = s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  let _ = s.push(vec_window(&[-1.0, 0.0], 1)).unwrap_err();
  let out = s.push(vec_window(&[0.0, 1.0], 2)).unwrap();
  assert_eq!(emitted(&out), &[0.0, 1.0]);
}

#[test]
fn vector_ema_refuses_raw_quantization_codes() {
  // The projection is `compute_components`, the same value surface `aggregate`
  // reads, so quantized storage with no dequantization override fails closed
  // here too. Reading `as_slice` and widening it directly would be one line
  // shorter and would silently smooth raw codes as if they were values.
  let mut s = SmoothPolicy::<BareI8Emb>::smoother(&VectorEma::new(0.5));
  assert_eq!(
    s.push(Windowed::new(BareI8Emb(vec![1, 2, 3]), Span::new(0, 1, 1)))
      .unwrap_err(),
    WinditError::MissingDequantization
  );
}

#[test]
fn vector_ema_drives_an_f32_storage_embedding_through_the_widening_projection() {
  // `TestVec` stores `f32` and normalizes inside its own `from_unnormalized`,
  // which is the shape of every real embedding (a 512-wide CLAP vector, say):
  // it drives the `Cow::Owned` elementwise-widening branch of
  // `compute_components` rather than the `f64` zero-copy borrow the other
  // vector tests take, and it proves the emitted value survives a narrowing
  // reconstruction.
  let mut s = SmoothPolicy::<TestVec>::smoother(&VectorEma::new(0.5));
  let first = s
    .push(Windowed::new(
      TestVec::from_unnormalized(&[1.0, 0.0]).unwrap(),
      Span::new(0, 1, 1),
    ))
    .unwrap();
  let second = s
    .push(Windowed::new(
      TestVec::from_unnormalized(&[0.0, 1.0]).unwrap(),
      Span::new(1, 1, 1),
    ))
    .unwrap();
  assert_eq!(first.value.as_slice(), &[1.0f32, 0.0]);
  let want = (1.0 / libm::sqrt(2.0)) as f32;
  assert_eq!(second.value.as_slice(), &[want, want]);
}

#[test]
fn vector_ema_emits_the_aggregate_ema_of_every_prefix() {
  // The defining equivalence, and the reason the renormalization is applied to
  // an emitted COPY rather than to the accumulator: the recurrence this
  // smoother carries is exactly the recency weighting
  // `w_0 = (1 - alpha)^t`, `w_i = alpha * (1 - alpha)^(t - i)` that
  // `EmaRenormalized` builds explicitly, so window `i` must emit the direction
  // the aggregate folds over the prefix `[0..=i]`.
  //
  // A genuine differential: the two sides share no code below `l2_renorm` —
  // the aggregate materializes the weights and runs a Neumaier-compensated
  // sum over the whole prefix, this one runs a two-term recurrence — so
  // agreement is evidence rather than a tautology. Renormalizing the
  // accumulator in place instead (the "spherical EMA" this type deliberately
  // is not) breaks it in the second decimal place, thousands of times the
  // tolerance below.
  let fixture: [[f64; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [-0.25, 0.75, 0.5],
    [0.125, -0.625, 0.25],
  ];

  for alpha in [0.5f32, 0.3, 0.75, 1.0] {
    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(alpha));
    for i in 0..fixture.len() {
      let got = s.push(vec_window(&fixture[i], i)).unwrap();
      let prefix: Vec<Windowed<RawF64Emb>> = fixture[..=i]
        .iter()
        .enumerate()
        .map(|(j, v)| vec_window(v, j))
        .collect();
      let want =
        crate::aggregate::aggregate(&crate::aggregate::EmaRenormalized::new(alpha), &prefix)
          .unwrap();
      let got = emitted(&got);
      assert_eq!(got.len(), want.captured.len());
      for (g, w) in got.iter().zip(&want.captured) {
        let diff = if g > w { g - w } else { w - g };
        assert!(
          diff < 1e-15,
          "alpha {alpha}, prefix {i}: {got:?} vs {:?}",
          want.captured
        );
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The streaming gate against its aggregate sibling.
//
// Every gate test above is two windows long, and at length two the two
// magnitude definitions coincide — which is why none of them can see a gate
// that carries only the current step's mass. These measure the two sides
// against each other at every prefix length instead.
// ---------------------------------------------------------------------------

/// The direction [`EmaRenormalized`](crate::aggregate::EmaRenormalized) folds
/// over `fixture[..=upto]`, or the error its own determinacy gate raised.
///
/// The oracle the streaming sibling is measured against: same windows, same
/// coefficient, the aggregation shape.
fn aggregate_prefix(
  alpha: f32,
  fixture: &[Vec<f64>],
  upto: usize,
) -> Result<Vec<f64>, WinditError> {
  let prefix: Vec<Windowed<RawF64Emb>> = fixture[..=upto]
    .iter()
    .enumerate()
    .map(|(j, v)| vec_window(v, j))
    .collect();
  crate::aggregate::aggregate(&crate::aggregate::EmaRenormalized::new(alpha), &prefix)
    .map(|e| e.captured)
}

/// The three candidate mass definitions after folding `xs`, by a local
/// recurrence written independently of the smoother's own.
///
/// - `two_term` is `|alpha * x_t| + |(1 - alpha) * s_{t-1}|`: this step's two
///   products and nothing older.
/// - `fold` is `alpha * |x_t| + (1 - alpha) * fold_{t-1}`, the recency-weighted
///   sum of term magnitudes an IDEAL fold would accumulate over the whole
///   prefix. It is *not* what `EmaRenormalized` computes — that rematerializes
///   its weights and sums `|w_i * x_i|` in window order, which is
///   [`aggregate_mass`]. Mistaking the two is what produced a threshold
///   ordering that the shipped fold's roundings do not obey; this field is kept
///   only to aim a residue at a band, never to assert one.
/// - `propagated` is `|alpha * x_t| + |(1 - alpha) * s_{t-1}| + (1 - alpha) *
///   propagated_{t-1}` from `propagated_0 = 0`, which additionally carries the
///   recurrence's own propagated rounding error — and starts at zero because
///   the seed is a copy and rounds by nothing.
///
/// The shipped `ema_step` exempts a step that rounds nothing (`alpha` of `0` or
/// `1`) from charging any mass; this replica does not model that, and asserts
/// it is never asked to.
struct Masses {
  state: Vec<f64>,
  two_term: Vec<f64>,
  fold: Vec<f64>,
  propagated: Vec<f64>,
}

fn masses(alpha: f32, xs: &[Vec<f64>]) -> Masses {
  let a = f64::from(alpha);
  let c = 1.0 - a;
  assert!(
    a != 0.0 && a != 1.0 && c != 1.0,
    "this replica does not model the exact-step exemption; alpha {alpha} needs it"
  );
  let mut state = xs[0].clone();
  let mut two_term: Vec<f64> = xs[0].iter().map(|v| v.abs()).collect();
  let mut fold = two_term.clone();
  let mut propagated = vec![0.0; xs[0].len()];
  for x in &xs[1..] {
    for j in 0..state.len() {
      let recent = a * x[j];
      let carried = c * state[j];
      state[j] = recent + carried;
      two_term[j] = recent.abs() + carried.abs();
      fold[j] = recent.abs() + (c * fold[j]).abs();
      propagated[j] = recent.abs() + carried.abs() + c * propagated[j];
    }
  }
  Masses {
    state,
    two_term,
    fold,
    propagated,
  }
}

/// The gate threshold for a mass vector: the crate's own
/// `16 * EPSILON * ||M|| + MIN_GATE_THRESHOLD`, spelled here so a test can say
/// where a residue must land.
fn tau(mass: &[f64]) -> f64 {
  16.0 * f64::EPSILON * l2(mass) + f64::from_bits(0x0170_0000_0000_0000)
}

/// The mass [`EmaRenormalized`](crate::aggregate::EmaRenormalized) **actually
/// computes** over `xs`: its own rematerialized weights, its own index order,
/// its own additions.
///
/// Not the recurrence `alpha * |x_t| + (1 - alpha) * M_{t-1}` that
/// [`masses`]'s `fold` carries. The two agree in exact arithmetic and are not
/// the same computation, and it is precisely that difference — the aggregate
/// builds `weights[i]` by iterated multiplication and sums `|w_i * x_i|` in
/// window order — that a differential against the aggregate's *output* has to
/// read. Reading the recurrence instead is how an induction over the ideal fold
/// came to be mistaken for a statement about the shipped one.
fn aggregate_mass(alpha: f32, xs: &[Vec<f64>]) -> Vec<f64> {
  let a = f64::from(alpha);
  let c = 1.0 - a;
  let n = xs.len();
  let mut weights = vec![0.0f64; n];
  let mut power = 1.0f64;
  for i in (1..n).rev() {
    weights[i] = a * power;
    power *= c;
  }
  weights[0] = power;
  let mut mag = vec![0.0f64; xs[0].len()];
  for (i, x) in xs.iter().enumerate() {
    for (m, &e) in mag.iter_mut().zip(x) {
      *m += (weights[i] * e).abs();
    }
  }
  mag
}

#[test]
fn vector_ema_and_the_aggregate_compute_the_same_mass_at_length_two() {
  // The rebuilt length-two regression. What stood here before was a tautology:
  // its fixture pushed ONE window, and the local `masses` helper then
  // initialised `two_term` and `fold` identically from that seed, so the
  // assertion compared a value with itself — invoking neither `M_1` nor either
  // production implementation. The blind spot the test was written to pin was
  // not pinned.
  //
  // This drives TWO real windows through the shipped smoother, reads the
  // production `VectorEmaState.mag` back, and compares it against the mass
  // `EmaRenormalized` actually computes over the same pair. At length two the
  // two are bit-identical and provably so: `M_0` is zero because the seed is a
  // copy, so `M_1` is `|alpha * x_1| + |(1 - alpha) * x_0| + (1 - alpha) * 0`,
  // and the aggregate's `weights[1] = alpha`, `weights[0] = 1 - alpha` make its
  // mass the same two products, summed in the other order. Past length two the
  // agreement stops — see the two `can_*` witnesses below — which is exactly
  // why this one is asserted here and nowhere else.
  const DIM: usize = 3;
  let mut rng = 0x0f1e_2d3c_4b5a_6978u64;
  for alpha in [1.0f32, 0.75, 0.5, 0.3, 0.1, 0.0] {
    let xs: Vec<Vec<f64>> = (0..2)
      .map(|_| {
        (0..DIM)
          .map(|_| f64::from(next_unit(&mut rng)) - 0.5)
          .collect()
      })
      .collect();
    assert_ne!(xs[0], xs[1], "the two windows must differ");

    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(alpha));
    s.push(vec_window(&xs[0], 0)).expect("seed");
    s.push(vec_window(&xs[1], 1)).expect("second window");

    let want = aggregate_mass(alpha, &xs);
    // The two coefficient ends are the deliberate exception, pinned rather than
    // skipped: at `alpha = 1` the step is an exact pass-through and at
    // `alpha = 0` an exact hold, so neither rounds and neither charges. The
    // aggregate's mass is `|x_1|` and `|x_0|` there, so this is a real
    // divergence — in the emitting direction, and sound, because the step it
    // declines to charge for committed nothing to charge.
    if alpha == 1.0 || alpha == 0.0 {
      assert!(
        l2(&want) > 0.1,
        "alpha {alpha}: the aggregate does charge here"
      );
      assert_eq!(
        s.mag,
        vec![0.0; DIM],
        "alpha {alpha}: an exact step charges nothing"
      );
      continue;
    }
    // Non-vacuity: a mass of zero would make every comparison below hold for
    // the wrong reason.
    assert!(l2(&want) > 0.1, "alpha {alpha}: fixture mass must be real");
    assert_eq!(
      s.mag, want,
      "alpha {alpha}: the production mass after two windows must be the mass the \
       aggregate computes over the same pair, bit for bit"
    );
    assert_eq!(
      tau(&s.mag),
      tau(&want),
      "alpha {alpha}: and therefore the same threshold"
    );
  }
}

#[test]
fn vector_ema_gate_agrees_with_the_aggregate_over_a_cancelling_three_window_prefix() {
  // THE falsifier for "streaming sibling of `EmaRenormalized`". Three windows,
  // not two: at length two the streaming gate's magnitude and the aggregate's
  // coincide, so every existing gate test is blind to a gate that carries only
  // the current step's two terms. At length three they part, and this prefix is
  // built to land in the gap — both sides fold to a norm of ~3.43e-15, which is
  // ABOVE a two-term threshold and BELOW the aggregate's.
  //
  // `x1` is chosen so that `||s_1||` is `alpha / (1 - alpha)`, which makes the
  // unit vector `x2 = -s_1 / ||s_1||` cancel the accumulator exactly: the
  // prefix has no direction, and the aggregate says so.
  const ALPHA: f32 = 0.3;
  let fixture: Vec<Vec<f64>> = vec![
    vec![1.0, 0.0],
    vec![-0.9436345029127625, 0.33098931238422735],
    vec![-0.9727890720095563, -0.23169251472325642],
  ];

  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(ALPHA));
  for i in 0..fixture.len() {
    let got = s.push(vec_window(&fixture[i], i));
    let want = aggregate_prefix(ALPHA, &fixture, i);
    match (got, want) {
      (Ok(g), Ok(w)) => {
        let g = emitted(&g);
        assert_eq!(g.len(), w.len(), "prefix {i}");
        for (a, b) in g.iter().zip(&w) {
          assert!(
            (a - b).abs() < 1e-12,
            "prefix {i}: streaming {g:?} vs aggregate {w:?}"
          );
        }
      }
      (Err(g), Err(w)) => assert_eq!(g, w, "prefix {i}"),
      (Ok(g), Err(w)) => panic!(
        "prefix {i}: the streaming gate EMITTED {:?} from a prefix its aggregate sibling \
         calls indeterminate ({w})",
        emitted(&g)
      ),
      (Err(g), Ok(w)) => panic!("prefix {i}: streaming rejected ({g}) but aggregate emitted {w:?}"),
    }
  }
}

#[test]
fn vector_ema_gate_and_the_aggregate_agree_at_every_prefix_length() {
  // The single counterexample above, generalized: at every prefix length from
  // three to ten, and at five smoothing factors, a final window is tuned so the
  // accumulator's residue lands strictly between the two-term threshold and the
  // aggregate's own. Every one of those is a prefix the aggregate calls
  // indeterminate, and the streaming sibling calls it indeterminate too.
  //
  // CHARACTERIZATION of an observed agreement, not a guarantee. The residue is
  // aimed at the GEOMETRIC MEAN of the two thresholds and the band is required
  // to be at least 1.2 wide before the case counts, so every case here clears
  // both edges by the same factor — deliberately away from the last-bit region
  // where the two verdicts are known to split in either direction (the two
  // `can_*` witnesses below).
  //
  // Length two is not in the sweep: it has its own test, which drives two real
  // windows through the shipped smoother instead of computing both sides of an
  // equality in this file. The version of this loop that carried a length-two
  // arm asserted a tautology — its fixture pushed one window and compared two
  // helper fields initialised identically from that seed.
  const DIM: usize = 4;
  let mut rng = 0x5eed_1234_9abc_def0u64;
  for alpha in [0.75f32, 0.5, 0.3, 0.2, 0.1] {
    let a = f64::from(alpha);
    let c = 1.0 - a;
    for len in 3..=10usize {
      // A prefix that cancels most of the way, so its collapsed accumulator
      // carries far less magnitude than the mass that produced it — the regime
      // in which the two definitions differ at all. Random windows, then one
      // that cancels nine tenths of what they built.
      let mut xs: Vec<Vec<f64>> = (0..len - 2)
        .map(|_| {
          (0..DIM)
            .map(|_| 0.5 + f64::from(next_unit(&mut rng)))
            .collect()
        })
        .collect();
      let built = masses(alpha, &xs);
      xs.push(
        built
          .state
          .iter()
          .map(|v| -(c / a) * v * 0.9)
          .collect::<Vec<f64>>(),
      );
      let m = masses(alpha, &xs);
      // The final window is `-((1 - alpha) / alpha) * s`, scaled just off exact
      // cancellation, so the residue it leaves is `(1 - alpha) * scale * ||s||`
      // — a dial. Its own term magnitude is `|(1 - alpha) * s|` per component,
      // which makes the closing step's two-term mass twice that and the fold's
      // that plus the damped history. The residue is aimed at the geometric
      // mean of the two thresholds, so it clears both edges by the same factor
      // however wide the band is at this alpha.
      let closing_two: Vec<f64> = m.state.iter().map(|v| 2.0 * (c * v).abs()).collect();
      let closing_fold: Vec<f64> = m
        .state
        .iter()
        .zip(&m.fold)
        .map(|(v, f)| (c * v).abs() + c * f)
        .collect();
      let (lo, hi) = (tau(&closing_two), tau(&closing_fold));
      assert!(
        hi > 1.2 * lo,
        "alpha {alpha}, len {len}: the two thresholds must differ for this to test anything \
         ({lo:e} vs {hi:e})"
      );
      let target = libm::sqrt(lo * hi);
      let scale = target / (c * l2(&m.state));
      let last: Vec<f64> = m
        .state
        .iter()
        .map(|v| -(c / a) * v * (1.0 + scale))
        .collect();
      xs.push(last);

      // Where the residue actually landed: above a two-term threshold, at or
      // below the aggregate's. Asserted rather than assumed, so the fixture
      // cannot quietly stop testing anything.
      let fin = masses(alpha, &xs);
      let residue = l2(&fin.state);
      assert!(
        tau(&fin.two_term) < residue && residue <= tau(&fin.fold),
        "alpha {alpha}, len {len}: residue {residue:e} must sit between {:e} and {:e}",
        tau(&fin.two_term),
        tau(&fin.fold)
      );

      let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(alpha));
      let mut got = None;
      for (i, x) in xs.iter().enumerate() {
        got = Some(s.push(vec_window(x, i)));
      }
      let got = got.unwrap();
      let want = aggregate_prefix(alpha, &xs, len - 1);
      assert!(
        want.is_err(),
        "alpha {alpha}, len {len}: fixture must be indeterminate for the aggregate too, got \
         {want:?}"
      );
      assert_eq!(
        got.map(|w| emitted(&w).to_vec()).err(),
        Some(WinditError::NonFinite),
        "alpha {alpha}, len {len}: the streaming gate emitted a direction from a prefix its \
         aggregate sibling calls indeterminate"
      );
    }
  }
}

#[test]
fn vector_ema_and_the_aggregate_refuse_the_same_out_of_domain_component() {
  // `[f64::MAX, f64::MAX]` is an ordinary diagonal — every component finite,
  // direction plainly `[2^-0.5, 2^-0.5]` — but its NORM is not representable.
  // The aggregate refuses it up front as outside the aggregation magnitude
  // domain; the streaming gate must give the same answer rather than let its
  // own `l2_norm` overflow to infinity and reject through `inf <= inf`.
  let big = vec![f64::MAX, f64::MAX];
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  assert_eq!(
    s.push(vec_window(&big, 0)).unwrap_err(),
    WinditError::MagnitudeOutOfRange {
      window: 0,
      component: 0
    }
  );
  assert_eq!(
    aggregate_prefix(0.5, &[big], 0).unwrap_err(),
    WinditError::MagnitudeOutOfRange {
      window: 0,
      component: 0
    }
  );

  // Non-vacuity: the boundary itself is inside the domain, so the check rejects
  // a domain violation rather than everything large. `2^400` is
  // `Real::MAX_AGG_MAGNITUDE`, and a diagonal of it still normalizes.
  let edge = libm::ldexp(1.0, 400);
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  let out = s.push(vec_window(&[edge, edge], 0)).unwrap();
  let want = 1.0 / libm::sqrt(2.0);
  assert_eq!(emitted(&out), &[want, want]);

  // And the lower edge, the other half of the same domain.
  let tiny = libm::ldexp(1.0, -400);
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  assert!(s.push(vec_window(&[tiny, tiny], 0)).is_ok());
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  assert_eq!(
    s.push(vec_window(&[libm::ldexp(1.0, -401), 0.0], 0))
      .unwrap_err(),
    WinditError::MagnitudeOutOfRange {
      window: 0,
      component: 0
    }
  );
}

#[test]
fn vector_ema_gate_mass_carries_the_whole_prefix_not_one_step() {
  // The three candidate definitions, separated by hand on one four-window
  // fixture at alpha 0.5 over `[1, 0]`, `[-1, 0]`, `[1, 0]`, `[-1, 0]`, whose
  // accumulator runs `1`, `0`, `0.5`, `-0.25`:
  //
  //   this step's two terms  ->  [0.75, 0]
  //   the aggregate's fold   ->  [1.0,  0]
  //   propagated error mass  ->  [1.25, 0]
  //
  // The gate must measure against the last: the accumulator is a recurrence, so
  // the rounding it carries is the whole damped history of its own steps, not
  // the two products of the current one.
  let fixture: Vec<Vec<f64>> = vec![
    vec![1.0, 0.0],
    vec![-1.0, 0.0],
    vec![1.0, 0.0],
    vec![-1.0, 0.0],
  ];
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  for (i, x) in fixture.iter().enumerate() {
    let _ = s.push(vec_window(x, i));
  }
  assert_eq!(s.mag, vec![1.25, 0.0]);

  // The local recurrence agrees, so the fixture reads the same from both sides.
  let m = masses(0.5, &fixture);
  assert_eq!(m.two_term, vec![0.75, 0.0]);
  assert_eq!(m.fold, vec![1.0, 0.0]);
  assert_eq!(m.propagated, vec![1.25, 0.0]);
}

#[test]
fn vector_ema_can_refuse_a_prefix_the_aggregate_emits() {
  // One of the two directions a near-threshold verdict can run. CHARACTERIZATION
  // of what is observed, not a promise: the crate no longer claims an ordering
  // between the two thresholds, and the companion test below exhibits the other
  // direction at exact bits.
  //
  // This prefix is the cancelling one above with its third window rotated by
  // 4e-15, which lifts the residue to ~3.63e-15 — above an ideal fold's
  // threshold (~3.52e-15) and below the streaming sibling's (~6.30e-15). Those
  // two figures only POSITION the fixture; both verdicts below are read off the
  // shipped implementations.
  //
  // The aggregate emits a direction; the recurrence refuses, because at this
  // residue its own propagated rounding is not provably smaller than the
  // result. A gate carrying only the fold's mass would emit here too, and would
  // be claiming an error bound the recurrence does not have — which is the
  // property this case still pins.
  const ALPHA: f32 = 0.3;
  let fixture: Vec<Vec<f64>> = vec![
    vec![1.0, 0.0],
    vec![-0.9436345029127625, 0.33098931238422735],
    vec![-0.9727890720095554, -0.2316925147232603],
  ];
  let m = masses(ALPHA, &fixture);
  let residue = l2(&m.state);
  assert!(
    tau(&m.fold) < residue && residue <= tau(&m.propagated),
    "fixture must sit in the band: {residue:e} against {:e} and {:e}",
    tau(&m.fold),
    tau(&m.propagated)
  );

  assert!(aggregate_prefix(ALPHA, &fixture, 2).is_ok());
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(ALPHA));
  for (i, x) in fixture.iter().enumerate().take(2) {
    s.push(vec_window(x, i)).unwrap();
  }
  assert_eq!(
    s.push(vec_window(&fixture[2], 2)).unwrap_err(),
    WinditError::NonFinite
  );
}

#[test]
fn vector_ema_can_emit_a_prefix_the_aggregate_refuses() {
  // The OTHER direction, at exact bits, and the case that killed the claim
  // "this side's threshold is never below the aggregate's". The induction
  // behind that claim compared the streaming mass against the recurrence
  // `alpha * |x_t| + (1 - alpha) * M_{t-1}` — the mass an IDEAL fold would
  // accumulate. `EmaRenormalized` does not evaluate that recurrence. It
  // rematerializes `weights[i]` by iterated multiplication and sums
  // `|w_i * x_i|` in window order, and those roundings do not have to land
  // where the recurrence's do.
  //
  // Here they land one ulp apart in the wrong direction. Three one-dimensional
  // windows at `alpha = 0.3f32`: both sides reach the same accumulator, bit for
  // bit, and the streaming mass comes out one ulp BELOW the aggregate's — so
  // the streaming threshold is one ulp below the aggregate's, the accumulator
  // falls exactly in that one-ulp gap, and the streaming gate emits a direction
  // its aggregate sibling calls indeterminate.
  //
  // Recorded as the witness that near-threshold verdicts differ in BOTH
  // directions. There is nothing to fix here: each gate is sound against its
  // own error bound, and neither bound was ever a statement about the other
  // side's rounding.
  const ALPHA: f32 = f32::from_bits(0x3e99_999a);
  let fixture: Vec<Vec<f64>> = vec![
    vec![f64::from_bits(0x3f0c_a8ca_2820_0000)],
    vec![f64::from_bits(0xbf20_b7cb_3226_ac2d)],
    vec![f64::from_bits(0xbc27_67b6_0c53_0643)],
  ];

  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(ALPHA));
  s.push(vec_window(&fixture[0], 0)).expect("the seed");
  // The second window cancels the seed to EXACTLY zero, so both sides refuse it
  // — and the streaming accumulator still advances through the refusal, which is
  // what leaves the third window's residue in the gap.
  assert_eq!(
    s.push(vec_window(&fixture[1], 1)).unwrap_err(),
    WinditError::NonFinite
  );
  assert_eq!(s.state, vec![0.0], "the two-window prefix cancels exactly");
  assert_eq!(
    aggregate_prefix(ALPHA, &fixture, 1).unwrap_err(),
    WinditError::NonFinite,
    "and the aggregate agrees about that one"
  );
  let got = s.push(vec_window(&fixture[2], 2));

  // The accumulators meet, bit for bit — so the disagreement is entirely in the
  // two masses and not in the value being gated.
  let ours = s.state[0];
  assert_eq!(
    ours.abs().to_bits(),
    0x3c0c_160d_bb1c_ff8d,
    "the streaming accumulator must land on the witness bits"
  );

  // The masses, and therefore the thresholds, differ by one ulp — the streaming
  // one BELOW the aggregate's, which is the ordering the retired claim forbade.
  let theirs = aggregate_mass(ALPHA, &fixture);
  assert!(
    s.mag[0] < theirs[0],
    "the streaming mass must be the smaller here: {:e} against {:e}",
    s.mag[0],
    theirs[0]
  );
  assert_eq!(tau(&s.mag).to_bits(), 0x3c0c_160d_bb1c_ff8c);
  assert_eq!(tau(&theirs).to_bits(), 0x3c0c_160d_bb1c_ff8d);

  // And the verdicts split, in the direction the retired claim ruled out.
  assert_eq!(
    emitted(&got.expect("the streaming gate emits at this residue")),
    &[-1.0],
    "the accumulator is negative, so its unit direction is -1"
  );
  assert_eq!(
    aggregate_prefix(ALPHA, &fixture, 2).unwrap_err(),
    WinditError::NonFinite,
    "the aggregate calls the same prefix indeterminate"
  );
}

#[test]
fn vector_ema_matches_the_aggregate_on_determinate_prefixes_at_both_storage_widths() {
  // The other half of the relationship, and the one that has to hold
  // everywhere: away from the gate's band the two siblings agree. Twelve
  // windows, five smoothing factors, both storage widths — the `f64` verbatim
  // double that reads the emitted value exactly, and the `f32` double that
  // drives the widening projection and a narrowing reconstruction.
  const DIM: usize = 5;
  let mut rng = 0x1234_5678_9abc_def0u64;
  let fixture: Vec<Vec<f64>> = (0..12)
    .map(|_| {
      (0..DIM)
        .map(|_| f64::from(next_unit(&mut rng)) - 0.5)
        .collect()
    })
    .collect();

  for alpha in [1.0f32, 0.75, 0.5, 0.3, 0.1] {
    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(alpha));
    for i in 0..fixture.len() {
      let got = s.push(vec_window(&fixture[i], i)).expect("determinate");
      let want = aggregate_prefix(alpha, &fixture, i).expect("determinate");
      for (g, w) in emitted(&got).iter().zip(&want) {
        assert!(
          (g - w).abs() < 1e-12,
          "alpha {alpha}, prefix {i}: {:?} vs {want:?}",
          emitted(&got)
        );
      }
    }

    let mut s = SmoothPolicy::<TestVec>::smoother(&VectorEma::new(alpha));
    for i in 0..fixture.len() {
      let got = s
        .push(Windowed::new(
          TestVec::from_unnormalized(&fixture[i]).unwrap(),
          Span::new(i, 1, 1),
        ))
        .expect("determinate");
      let prefix: Vec<Windowed<TestVec>> = fixture[..=i]
        .iter()
        .enumerate()
        .map(|(j, v)| Windowed::new(TestVec::from_unnormalized(v).unwrap(), Span::new(j, 1, 1)))
        .collect();
      let want =
        crate::aggregate::aggregate(&crate::aggregate::EmaRenormalized::new(alpha), &prefix)
          .expect("determinate");
      for (g, w) in got.value.as_slice().iter().zip(want.as_slice()) {
        assert!(
          (g - w).abs() < 1e-6,
          "f32 storage, alpha {alpha}, prefix {i}: {:?} vs {:?}",
          got.value.as_slice(),
          want.as_slice()
        );
      }
    }
  }
}

#[test]
fn vector_ema_alpha_zero_holds_the_seed_and_accumulates_no_mass() {
  // FALSIFIER for the branch's horizon argument. At `alpha = 0` the recurrence
  // is exact — `1 - alpha` is `1`, so the carry is a rounding-free copy and the
  // injection is exactly zero — yet the gate's mass still charged
  // `|(1 - alpha) * s|` a push, so `M` grew LINEARLY by `|s_0|` per window.
  // That is the growth the horizon argument assumed away: at `2^48` post-seed
  // pushes the threshold reaches the held seed's own magnitude and the
  // inclusive comparison refuses it from then on, forever.
  //
  // Measured through the internal mass rather than by running `2^48` pushes: a
  // step that rounds nothing must charge nothing, so after any number of pushes
  // the mass of an exact hold is exactly zero.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.0));
  s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  for i in 1..64usize {
    let out = s.push(vec_window(&[0.25, -0.5], i)).unwrap();
    assert_eq!(
      emitted(&out),
      &[1.0, 0.0],
      "push {i}: alpha 0 holds the seed direction"
    );
  }
  assert_eq!(
    s.state,
    vec![1.0, 0.0],
    "the accumulator is the seed, bit for bit"
  );
  assert_eq!(
    s.mag,
    vec![0.0, 0.0],
    "an exact hold commits no rounding, so it must charge no mass; linear growth \
     here is what makes the 2^48 horizon reachable"
  );
}

#[test]
fn vector_ema_a_mass_at_the_horizon_refuses_a_held_seed() {
  // Why a zero mass is the cure and not a detail. The consequence is reached by
  // writing the mass the old linear growth left after `2^48` post-seed pushes
  // rather than by running them: `16 * EPSILON * 2^48` is exactly `1`, the
  // seed's own magnitude, and the gate's comparison is inclusive.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.0));
  s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  s.push(vec_window(&[0.0, 0.0], 1)).unwrap();
  s.mag = vec![libm::ldexp(1.0, 48), 0.0];
  assert_eq!(
    s.push(vec_window(&[0.0, 0.0], 2)).unwrap_err(),
    WinditError::NonFinite,
    "a mass of 2^48 seeds puts the threshold at the seed's own magnitude"
  );
}

#[test]
fn vector_ema_a_nonzero_alpha_below_the_complement_collapse_still_grows_its_mass() {
  // The other half of the alpha-zero analysis, pinned so the cure cannot be
  // widened into a lie. At `alpha = 2^-60` the complement rounds to exactly
  // `1.0` just as it does at zero, so the carry is again a rounding-free copy —
  // but the injection is NOT zero, and `recent + carried` genuinely rounds
  // every push. That error really does accumulate undamped, so the mass really
  // does grow linearly here, and the growth is a true bound rather than an
  // artifact. Only the exact hold may charge nothing.
  // `2^-54` is the FIRST alpha whose complement collapses — the tie that rounds
  // to even — so this sits exactly on the documented edge rather than safely
  // inside it, and the neighbour that does not collapse is pinned beside it.
  const ALPHA: f32 = 1.0 / 18_014_398_509_481_984.0; // 2^-54
  assert_eq!(
    1.0f64 - f64::from(ALPHA),
    1.0,
    "the complement must collapse"
  );
  assert_ne!(
    1.0f64 - libm::ldexp(1.0, -53),
    1.0,
    "and 2^-53 must NOT: the doc's constant is 2^-54"
  );
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(ALPHA));
  s.push(vec_window(&[1.0, 0.0], 0)).unwrap();
  for i in 1..64usize {
    s.push(vec_window(&[0.25, 0.0], i)).unwrap();
  }
  assert!(
    s.mag[0] > 62.0 && s.mag[0] < 64.0,
    "a rounding step per push must still be charged: {:?}",
    s.mag
  );
}

#[test]
fn vector_ema_error_bound_needs_an_absolute_term_in_the_subnormal_regime() {
  // FALSIFIER for the published bound `|e_t| <= 2u * M_t`. At alpha 0.5 a seed
  // of `2^-400` and a run of all-zero windows halves the accumulator every
  // push, walking it through the normal range and down into the subnormals. At
  // the 675th push the exact recurrence is `2^-1075` and the computed
  // accumulator rounds it to zero — an error of `2^-1075`, half the smallest
  // subnormal `eta = 2^-1074`.
  //
  // The mass by then is `337 * eta`, so the RELATIVE allowance `2u * M` is
  // `337 * 2^-52` eta — about 42 binary orders too small to cover it. Subnormal
  // rounding is ABSOLUTE, and a bound with only a relative term cannot see it.
  //
  // Everything below is stated in units of `eta`, because `2u * M` itself is
  // `~2^-1126` and has no f64 to be stated in.
  const SEED_EXP: i32 = -400; // exactly `Real::MIN_AGG_MAGNITUDE`
  const STEPS: usize = 675;
  let eta = f64::from_bits(1);

  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  s.push(vec_window(&[libm::ldexp(1.0, SEED_EXP)], 0))
    .expect("the seed is in domain and determinate");
  for i in 1..=STEPS {
    // Refused from the moment the accumulator falls under the gate's absolute
    // floor, and advancing the accumulator through every refusal — which is the
    // documented behaviour and what carries the state into the subnormals.
    let _ = s.push(vec_window(&[0.0], i));
  }

  assert_eq!(
    s.state,
    vec![0.0],
    "the computed accumulator must have rounded to zero"
  );
  let error_in_eta = 0.5; // the exact recurrence is `2^-1075`, half of eta
  let mass_in_eta = s.mag[0] / eta;
  assert_eq!(mass_in_eta, 337.0, "the mass at the underflow step");

  // The relative term alone does NOT cover it — that is the whole finding, and
  // it is asserted rather than described so the published bound can never be
  // narrowed back.
  let relative_in_eta = mass_in_eta * f64::EPSILON;
  assert!(
    relative_in_eta < error_in_eta,
    "`2u * M` must be the term that fails here: {relative_in_eta:e} eta"
  );

  // The absolute term covers it, with room to spare: each step's two products
  // can each round by at most `eta / 2` when their result is subnormal (a
  // floating-point ADDITION never underflows), and those contributions
  // accumulate undamped in the worst case, so `E_t <= t * eta`.
  let absolute_in_eta = STEPS as f64;
  assert!(
    relative_in_eta + absolute_in_eta >= error_in_eta,
    "the mixed bound must cover the error"
  );

  // And the absolute term never reaches a verdict, because the gate's own
  // absolute floor is `2^-1000` — 74 binary orders above `t * eta` at this
  // epoch, and above it for every epoch shorter than `2^74` windows.
  let floor = f64::from_bits(0x0170_0000_0000_0000);
  assert!(
    absolute_in_eta * eta < floor,
    "MIN_GATE_THRESHOLD must dominate the accumulated absolute term"
  );
  assert_eq!(
    s.push(vec_window(&[0.0], STEPS + 1)).unwrap_err(),
    WinditError::NonFinite,
    "the verdict is refusal, and correctly so: the exact value is under the floor too"
  );
}

#[test]
fn vector_ema_gate_floor_decides_below_two_pow_minus_1000() {
  // The absolute floor's own boundary, and the one regime where it decides a
  // verdict rather than sitting far under the relative term. Same walk as the
  // subnormal case above: alpha 0.5, a seed of `2^-400`, all-zero windows, so
  // the accumulator is `2^-(400 + t)` exactly and the mass is `t * 2^-(400 + t)`
  // — which puts `16 * EPSILON * ||M||` around `2^-1039` while the accumulator
  // is still around `2^-1000`. The relative term is 39 binary orders too small
  // to reach anything here; `MIN_GATE_THRESHOLD` is the whole threshold.
  //
  // Non-vacuous from both sides: one push earlier the same gate emits.
  const SEED_EXP: i32 = -400;
  let floor_exp = -1000;

  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  s.push(vec_window(&[libm::ldexp(1.0, SEED_EXP)], 0))
    .expect("the seed");
  // 599 halvings leave the accumulator at `2^-999`, one binade ABOVE the floor.
  for i in 1..=599usize {
    s.push(vec_window(&[0.0], i))
      .expect("still above the floor");
  }
  assert_eq!(s.state, vec![libm::ldexp(1.0, floor_exp + 1)]);
  assert!(
    16.0 * f64::EPSILON * s.mag[0] < libm::ldexp(1.0, floor_exp) / 1e9,
    "the relative term must be irrelevant here: {:e}",
    16.0 * f64::EPSILON * s.mag[0]
  );

  // One more halving puts it AT the floor, and the comparison is inclusive.
  assert_eq!(
    s.push(vec_window(&[0.0], 600)).unwrap_err(),
    WinditError::NonFinite,
    "at 2^-1000 the absolute floor refuses"
  );
  assert_eq!(s.state, vec![libm::ldexp(1.0, floor_exp)]);
}

// ---------------------------------------------------------------------------
// The epoch horizon: `M` is itself a floating-point accumulation, and past a
// proven number of charging steps it stops dominating the mass an exact
// recurrence carries.
// ---------------------------------------------------------------------------

/// The `alpha` whose complement is the first to round to exactly `1.0`, and the
/// coefficient the stagnation counterexample runs at.
const COLLAPSING_ALPHA: f32 = 1.0 / 18_014_398_509_481_984.0; // 2^-54

#[test]
fn vector_ema_mass_stagnation_is_reachable_by_pushing() {
  // The reachability argument for the fixture the two tests below start from,
  // asserted rather than described: every claim here is an actual `f64`
  // operation or an actual production push.
  //
  // At `alpha = 2^-54` the complement is exactly `1.0`, so with `x = 2^-24`:
  //
  //   recent    = alpha * x            = 2^-78
  //   carried   = 1.0 * s              = s
  //   s         = 2^-78 + 2^-24        = 2^-24        (absorbed, a quarter ulp)
  //   committed = |2^-78| + |2^-24|    = 2^-24        (the sum rounds back too)
  //   m         = 2^-24 + m
  //
  // so the accumulator is stationary and the mass walks the `2^-24` grid, which
  // is exact for every partial sum of at most 53 significant bits — that is
  // `k * 2^-24` for `k <= 2^53`, landing on exactly `2^29` at `k = 2^53`. From
  // there each further charge is exactly half an ulp of `2^29` and ties back to
  // even, so `(s, m) = (2^-24, 2^29)` is not a passing value but the epoch's
  // fixed point, held for every step after the `2^53`rd.
  let a = f64::from(COLLAPSING_ALPHA);
  assert_eq!(
    1.0f64 - a,
    1.0,
    "the complement must collapse to exactly one"
  );
  let x = libm::ldexp(1.0, -24);
  let recent = a * x;
  assert_eq!(recent, libm::ldexp(1.0, -78), "the injection is 2^-78");
  assert_eq!(recent + x, x, "and is absorbed whole by a state of 2^-24");
  let committed = recent.abs() + x.abs();
  assert_eq!(committed, x, "the charge rounds back to exactly 2^-24");

  // The grid is exact right up to the stagnation point, and stagnant after it.
  let m_before = (libm::ldexp(1.0, 53) - 1.0) * x; // k = 2^53 - 1
  assert_eq!(
    m_before / x,
    libm::ldexp(1.0, 53) - 1.0,
    "still on the grid"
  );
  let m_at = m_before + committed;
  assert_eq!(m_at, libm::ldexp(1.0, 29), "k = 2^53 lands on exactly 2^29");
  assert_eq!(
    m_at + committed,
    m_at,
    "and every charge after it ties back: the mass STAGNATES while the exact \
     mass keeps climbing"
  );

  // And the same walk on the production path, so the closed form above is not a
  // second implementation but a description of what `push` does.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(COLLAPSING_ALPHA));
  s.push(vec_window(&[x], 0)).expect("the seed");
  for i in 1..=16usize {
    s.push(vec_window(&[x], i)).expect("determinate throughout");
    assert_eq!(s.state, vec![x], "push {i}: the accumulator is stationary");
    assert_eq!(s.mag, vec![i as f64 * x], "push {i}: the mass is k * 2^-24");
    assert_eq!(s.steps, i as u64, "push {i}: every one of these charges");
  }
}

#[test]
fn vector_ema_past_the_horizon_the_gate_emits_a_direction_from_an_exact_zero() {
  // THE finding, on the production path. The fixture is the stagnated pair the
  // test above proves reachable — `s = 2^-24`, `M = 2^29` — reached after `2^53`
  // charging steps and held for every step after, so the exact recurrence has
  // been climbing by `2^-78` a step while neither the accumulator nor the mass
  // moved. After `2^60` steps the exact recurrence stands at `65 * 2^-24`.
  //
  // `steps` is set BELOW the shipped limit here on purpose: this test is the
  // characterization of what the limit exists to prevent, so it must run the
  // arithmetic the limit refuses. The companion test below starts from the same
  // fixture with the step count the fixture actually implies, and gets a
  // refusal.
  let x = libm::ldexp(1.0, -24);
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(COLLAPSING_ALPHA));
  s.push(vec_window(&[x], 0)).expect("the seed");
  s.state = vec![x];
  s.mag = vec![libm::ldexp(1.0, 29)];
  s.steps = 0;

  // `2^60 * 2^-78` is `64 * 2^-24`, so the exact recurrence is at `65 * 2^-24`
  // and `2_129_920` injections of `-2^-39` take it to exactly zero.
  const PUSHES: usize = 2_129_920;
  let injection = libm::ldexp(1.0, -39);
  assert_eq!(
    65.0 * x - PUSHES as f64 * injection,
    0.0,
    "the exact recurrence must land on zero"
  );

  let window = -libm::ldexp(1.0, 15); // alpha * -2^15 = -2^-39
  let mut last = Err(WinditError::Empty);
  for i in 0..PUSHES {
    last = s.push(vec_window(&[window], i));
  }

  assert_eq!(
    s.state,
    vec![-libm::ldexp(1.0, -18)],
    "the computed accumulator ends at -2^-18 where the exact one is zero"
  );
  assert_eq!(
    s.mag[0].to_bits(),
    0x41c0_0000_0200_0002,
    "the mass barely moved off its stagnation point: 0x1.0000002000002p29"
  );
  let threshold = tau(&s.mag);
  assert_eq!(
    threshold.to_bits(),
    0x3ec0_0000_0200_0002,
    "which puts the gate's threshold at 0x1.0000002000002p-19"
  );
  assert!(
    s.state[0].abs() > threshold,
    "and the accumulator is ABOVE it: {:e} > {threshold:e}",
    s.state[0].abs()
  );
  assert_eq!(
    last.map(|w| emitted(&w).to_vec()),
    Ok(vec![-1.0]),
    "so the gate emits a direction for a prefix whose exact value is zero — the \
     published `|e| <= 2u * M` bound broken by a factor of 32"
  );
  // `2u * M` is the bound this type publishes on its own error, and the gate
  // reads it with a factor of sixteen of headroom. The error here is `2^-18`
  // against a bound of `2^-23`: past both, which is what makes the emission a
  // soundness failure rather than a tight call.
  let bound = f64::EPSILON * s.mag[0];
  assert!(
    s.state[0].abs() > 16.0 * bound,
    "the error {:e} must break the published `2u * M` bound {bound:e} by more \
     than the gate's headroom",
    s.state[0].abs()
  );
}

#[test]
fn vector_ema_refuses_the_step_that_would_leave_the_proven_epoch() {
  // The cure, against the very same fixture. Reaching `(2^-24, 2^29)` takes
  // `2^53` charging steps — every push of the walk charges — and `2^53` is past
  // `MAX_EPOCH_STEPS`, so the epoch that produced this state is already over.
  // Setting the count to the limit is therefore the honest completion of the
  // fixture, not a second hypothesis: no smaller count can produce this pair.
  let x = libm::ldexp(1.0, -24);
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(COLLAPSING_ALPHA));
  s.push(vec_window(&[x], 0)).expect("the seed");
  s.state = vec![x];
  s.mag = vec![libm::ldexp(1.0, 29)];
  s.steps = VectorEma::MAX_EPOCH_STEPS;

  let before = (s.state.clone(), s.mag.clone(), s.steps);
  assert_eq!(
    s.push(vec_window(&[-libm::ldexp(1.0, 15)], 1)).unwrap_err(),
    WinditError::EpochTooLong,
    "the step that would leave the proven range is refused"
  );
  assert_eq!(
    (s.state.clone(), s.mag.clone(), s.steps),
    before,
    "and refused before any write: the accumulator, the mass and the count are \
     exactly as they were"
  );
  // Permanent, not per-window: nothing a caller pushes re-enters the range.
  for i in 2..6usize {
    assert_eq!(
      s.push(vec_window(&[0.0], i)).unwrap_err(),
      WinditError::EpochTooLong,
      "push {i} is refused too"
    );
  }
  // A clone carries the exhausted epoch with it — a `Clone` that dropped the
  // count would hand out a smoother that resumes fabricating.
  let mut copy = s.clone();
  assert_eq!(
    copy.push(vec_window(&[0.0], 6)).unwrap_err(),
    WinditError::EpochTooLong,
    "the clone is exhausted too"
  );
}

#[test]
fn vector_ema_horizon_is_exactly_max_epoch_steps() {
  // Non-vacuous from both sides, and pinned to the number rather than to the
  // constant so that moving the constant is a test failure rather than a silent
  // widening of the accepted regime.
  assert_eq!(
    VectorEma::MAX_EPOCH_STEPS,
    1u64 << 50,
    "the enforced limit is 2^50 charging steps — inside the ~2^53.4 the mass's \
     own rounding is proven over, and inside the subnormal term's 2^74"
  );

  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
  assert_eq!(s.steps, 0, "the seed rounds nothing, so it charges no step");
  s.steps = VectorEma::MAX_EPOCH_STEPS - 1;
  s.push(vec_window(&[0.0, 1.0], 1))
    .expect("the last step inside the proven range still runs");
  assert_eq!(s.steps, VectorEma::MAX_EPOCH_STEPS, "and it counted");
  assert_eq!(
    s.push(vec_window(&[0.0, 1.0], 2)).unwrap_err(),
    WinditError::EpochTooLong,
    "the next one is refused"
  );
}

#[test]
fn vector_ema_reset_rearms_an_exhausted_epoch() {
  // What the caller is expected to do about `EpochTooLong`, and the proof that
  // the path works: a new epoch starts from a zero mass and a zero count.
  for rearm in [0u8, 1] {
    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
    s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
    s.steps = VectorEma::MAX_EPOCH_STEPS;
    assert_eq!(
      s.push(vec_window(&[0.0, 1.0], 1)).unwrap_err(),
      WinditError::EpochTooLong
    );

    // `discontinuity` is the trait default and routes to `reset`; both are
    // exercised so neither can regress alone.
    if rearm == 0 {
      s.reset();
    } else {
      s.discontinuity();
    }
    assert_eq!(s.steps, 0, "rearm {rearm}: the count is cleared");

    let out = s
      .push(vec_window(&[0.0, 1.0], 2))
      .expect("the next window seeds a fresh epoch");
    assert_eq!(
      emitted(&out),
      &[0.0, 1.0],
      "rearm {rearm}: and it is a seed, not a continuation of the old prefix"
    );
    assert_eq!(s.mag, vec![0.0, 0.0], "rearm {rearm}: with a zero mass");
    assert_eq!(s.steps, 0, "rearm {rearm}: and a zero count");
  }
}

#[test]
fn vector_ema_only_charging_steps_count_against_the_horizon() {
  // The horizon counts the quantity the induction is stated over — steps that
  // round `M` — and nothing else. That is what keeps the exact-step exemption's
  // liveness: an `alpha` of `0` or `1` never spends any of the mass's relative
  // precision, so its epoch stays unbounded. A count of pushes rather than of
  // charges would make an exact hold terminal at `2^50` windows, reintroducing
  // at a further horizon exactly the defect the exemption removed.
  for alpha in [0.0f32, 1.0] {
    let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(alpha));
    s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
    for i in 1..32usize {
      s.push(vec_window(&[0.25, -0.5], i)).expect("determinate");
    }
    assert_eq!(
      s.mag,
      vec![0.0, 0.0],
      "alpha {alpha}: an exact step charges nothing"
    );
    assert_eq!(s.steps, 0, "alpha {alpha}: so it advances no count either");
  }

  // The third exact case: a collapsed complement against an all-zero window.
  // `recent` is exactly zero and the carry is a copy, so the step is exact even
  // though `alpha` is neither end of the range.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(COLLAPSING_ALPHA));
  s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
  for i in 1..8usize {
    s.push(vec_window(&[0.0, 0.0], i)).expect("determinate");
  }
  assert_eq!(
    s.steps, 0,
    "an all-zero window under a collapsed complement is exact"
  );
  // And the neighbouring inexact case does count, so the exemption is narrow.
  s.push(vec_window(&[0.25, 0.0], 8)).expect("determinate");
  assert_eq!(s.steps, 1, "a nonzero injection rounds, and is counted");
}

#[test]
fn vector_ema_a_gate_refused_push_still_counts_against_the_horizon() {
  // The accumulator advances through a determinacy refusal — that is documented
  // behaviour — so the mass grew and the step must be charged to the epoch.
  // Counting only emitted windows would let an epoch of refusals run past the
  // range the bound is proven over and then start emitting again.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
  assert_eq!(
    s.push(vec_window(&[-1.0, 0.0], 1)).unwrap_err(),
    WinditError::NonFinite,
    "exact cancellation is refused"
  );
  assert_eq!(s.steps, 1, "and still counts");
}

#[test]
fn vector_ema_horizon_does_not_mask_a_malformed_window() {
  // Ordering, pinned: the epoch check runs after the window's own validation, so
  // a caller past the horizon is still told what is wrong with the window it
  // pushed rather than being handed an epoch-level error for a dimension bug.
  let mut s = SmoothPolicy::<RawF64Emb>::smoother(&VectorEma::new(0.5));
  s.push(vec_window(&[1.0, 0.0], 0)).expect("the seed");
  s.steps = VectorEma::MAX_EPOCH_STEPS;
  assert_eq!(
    s.push(vec_window(&[1.0, 0.0, 0.0], 1)).unwrap_err(),
    WinditError::DimMismatch {
      got: 3,
      expected: 2
    },
    "the window's own defect is reported first"
  );
  assert_eq!(
    s.push(vec_window(&[f64::NAN, 0.0], 2)).unwrap_err(),
    WinditError::NonFinite,
    "and so is a non-finite component"
  );
  assert_eq!(
    s.push(vec_window(&[1.0, 0.0], 3)).unwrap_err(),
    WinditError::EpochTooLong,
    "a well-formed window is what reaches the epoch check"
  );
}

#[test]
fn epoch_too_long_names_the_limit_it_enforces() {
  // The message reads the constant, so it cannot drift from the bound it
  // reports. Spelled here as the formatted number rather than as the constant so
  // that a message rewritten to a stale literal fails.
  let rendered = std::format!("{}", WinditError::EpochTooLong);
  assert!(
    rendered.contains("1125899906842624"),
    "the limit must appear in the message: {rendered}"
  );
}
