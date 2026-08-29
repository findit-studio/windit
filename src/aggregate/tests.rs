use std::{vec, vec::Vec};

#[cfg(feature = "serde")]
use super::AggregatePolicyKind;
use super::{
  aggregate, keep_separate, l2_renorm, max_magnitude, normalizing_shift, weighted_sum_renorm,
  AggregatePolicy, CoverageWeightedMean, EmaRenormalized, MeanRenormalized, SaliencyWeighted,
  MIN_NORMAL_EXPONENT,
};
use crate::{
  plan::Span,
  scalar::{Real, TestQuant},
  test_support::{
    assert_close, assert_close_f64, BareI8Emb, QuantEmb, RawF64Emb, TestQuantVec, TestVec,
  },
  windowed::{Vector, WindowEmbedding, Windowed},
  WinditError,
};

/// Build a windowed embedding from a raw vector and a span with the given real
/// length and window size (so `coverage() == len / window`).
///
/// `raw` is `&[f64]`: `TestVec` stores `f32` but computes in `f64`, so its
/// `from_unnormalized` takes the compute scalar and narrows on the way in.
fn win(raw: &[f64], len: usize, window: usize) -> WindowEmbedding<TestVec> {
  Windowed::new(
    TestVec::from_unnormalized(raw).unwrap(),
    Span::new(0, len, window),
  )
}

#[test]
fn coverage_weighted_mean_pinned() {
  // cov 1.0 * [1,0] + cov 0.5 * [0,1] = [1, 0.5]; renorm by sqrt(1.25). The
  // aggregation runs in f64 and the result is narrowed into f32 storage, so the
  // pinned values are the correctly-rounded 2/sqrt(5) and 1/sqrt(5) either way.
  let windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 1.0], 2, 4)];
  let out = aggregate(&CoverageWeightedMean, &windows).unwrap();
  assert_close(out.as_slice(), &[0.8944272, 0.4472136]);
}

#[test]
fn coverage_weighted_mean_f64_pinned() {
  // The same fixture as `coverage_weighted_mean_pinned`, pinned to f64
  // precision. The tolerance is the point: the f32 answer (0.894_427_2) sits
  // about 1e-8 away from this value, so an implementation that computed in f32
  // and widened afterwards fails here by four orders of magnitude.
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0, 0.5];
  let out = CoverageWeightedMean
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();
  assert_close_f64(&out, &[0.894_427_190_999_915_9, 0.447_213_595_499_957_9]);

  // Stated as a comparison rather than left implicit: the f32 result is not
  // within the f64 tolerance, which is what makes the assertion above evidence.
  let widened_f32 = f64::from(0.894_427_2_f32);
  let diff = (widened_f32 - 0.894_427_190_999_915_9_f64).abs();
  assert!(diff > 1e-12, "f32 precision must not satisfy the f64 pin");
}

/// Two window geometries whose true coverages differ by far less than an `f32`
/// ulp must reach the fold as *different* weights.
///
/// Every operand is at most `2^24` and so exactly representable in `f32`, which
/// is what makes this a statement about the quotient alone — nothing rounds on
/// the way into the division, unlike the geometry
/// `coverage_past_f32_integer_range_is_the_exact_ratio` pins. The two ratios
/// `8388607/16777213` and `8388608/16777215` differ by exactly
/// `1/(16777213 * 16777215)`, about `3.6e-15`: 32 `f64` ulps, and `6e-8` of the
/// `f32` ulp at `0.5`. A coverage channel narrower than the fold rounds both to
/// `0.50000006` and weighs two different windows identically.
#[test]
fn sub_ulp_coverages_reach_the_fold_as_distinct_weights() {
  let cov_a = Span::new(0, 8_388_607, 16_777_213).coverage();
  let cov_b = Span::new(0, 8_388_608, 16_777_215).coverage();

  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let out_a = CoverageWeightedMean
    .aggregate_values(&embeddings, &[1.0, cov_a], 2)
    .unwrap();
  let out_b = CoverageWeightedMean
    .aggregate_values(&embeddings, &[1.0, cov_b], 2)
    .unwrap();

  assert_ne!(
    out_a, out_b,
    "spans (len 8388607, window 16777213) and (len 8388608, window 16777215) \
     have true coverages 3.6e-15 apart, yet both folded at {cov_a:?} / {cov_b:?} \
     and produced one vector"
  );
  assert_ne!(
    cov_a, cov_b,
    "two window geometries 3.6e-15 apart in true coverage must not share a coverage"
  );

  // And through `aggregate`, which collects the coverages itself: the seam above
  // is the one a caller reaches directly, this is the one the crate walks, and a
  // narrowing anywhere along it collapses the two geometries again.
  let mk = |len, window| {
    [
      Windowed::new(
        RawF64Emb {
          data: vec![1.0, 0.0],
          captured: Vec::new(),
        },
        Span::new(0, 4, 4),
      ),
      Windowed::new(
        RawF64Emb {
          data: vec![0.0, 1.0],
          captured: Vec::new(),
        },
        Span::new(0, len, window),
      ),
    ]
  };
  let folded_a = aggregate(&CoverageWeightedMean, &mk(8_388_607, 16_777_213)).unwrap();
  let folded_b = aggregate(&CoverageWeightedMean, &mk(8_388_608, 16_777_215)).unwrap();
  assert_ne!(
    folded_a.captured, folded_b.captured,
    "`aggregate` must carry the distinction its spans hold"
  );
}

/// Multiplying every coverage by a common positive factor must not change a
/// normalized weighted mean.
///
/// The weights of a *normalized* weighted mean are defined only up to a common
/// positive factor: `sum_i (s * c_i) * e_i` is `s * sum_i c_i * e_i`, and the
/// renormalization that ends the fold divides `s` back out. So the scale of the
/// coverage slice carries no information about the answer, and any part of the
/// policy that reads it is reading noise.
#[test]
fn coverage_weights_are_scale_invariant() {
  // One window, one component, and the fold's exact result is the normal value
  // `[2^-1001]`, whose direction is `[1.0]` — the same direction the same fold
  // has at coverage `1.0`. Nothing here is ill-conditioned: there is one term,
  // no cancellation, and the result is finite, nonzero and safely normalizable.
  let one: [&[f64]; 1] = [&[1.0]];
  let at_one = CoverageWeightedMean.aggregate_values(&one, &[1.0], 1);
  let at_tiny = CoverageWeightedMean.aggregate_values(&one, &[libm::ldexp(1.0, -1001)], 1);
  assert_eq!(
    at_tiny.as_ref().ok(),
    at_one.as_ref().ok(),
    "one window's direction cannot depend on the scale of the single weight it \
     carries: at 1.0 {at_one:?}, at 2^-1001 {at_tiny:?}"
  );

  // And across a family of factors on a two-window fold whose weights differ:
  // `1.0`, `2^-1001`, and a factor that carries the smaller coverage all the way
  // to the minimum `f64` subnormal. Each factor is a power of two and no product
  // underflows, so each scaled slice is *exactly* the base one times the factor —
  // which is what makes a difference in the output a statement about the policy.
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let base = [1.0_f64, 0.5];
  let reference = CoverageWeightedMean
    .aggregate_values(&embeddings, &base, 2)
    .expect("the unscaled fold resolves");
  for exp in [0, -1001, -1073] {
    let factor = libm::ldexp(1.0, exp);
    let scaled = [base[0] * factor, base[1] * factor];
    assert_eq!(
      (scaled[0], scaled[1]),
      (factor, libm::ldexp(1.0, exp - 1)),
      "the scaling must be exact for this to be a statement about the fold"
    );
    let got = CoverageWeightedMean.aggregate_values(&embeddings, &scaled, 2);
    assert_eq!(
      got.as_ref().ok(),
      Some(&reference),
      "scaling every coverage by 2^{exp} changed the fold, got {got:?}"
    );
  }
  assert_eq!(
    libm::ldexp(1.0, -1074),
    f64::from_bits(1),
    "the last factor must reach the minimum f64 subnormal"
  );

  // Beyond the powers of two: a factor whose products are all exactly
  // representable keeps the two slices exactly proportional, so the contract
  // still binds. `0.75` is not a power of two, and `[1.0, 0.5] * 0.75` is
  // `[0.75, 0.375]` with no rounding anywhere.
  let exact_factor = 0.75_f64;
  let exactly_proportional = [base[0] * exact_factor, base[1] * exact_factor];
  assert_eq!(
    exactly_proportional,
    [0.75, 0.375],
    "both products must be exact for this to be a statement about the fold"
  );
  let got = CoverageWeightedMean.aggregate_values(&embeddings, &exactly_proportional, 2);
  assert_eq!(
    got.as_ref().ok(),
    Some(&reference),
    "scaling every coverage by an exactly representable {exact_factor} changed the fold, got {got:?}"
  );

  // And an ordinary floating factor, which is where the *bit-identical* contract
  // stops — the boundary being a property of the products, not of the factor.
  // `0.1` is representable and in range, and neither product leaves the domain or
  // rounds to zero, so a contract keyed on the factor would have to cover this
  // one. It cannot: `0.1 * 0.1` is not exactly representable, so the scaled slice
  // is not the base slice times a constant and its second weight is a different
  // number. The invariance that survives is approximate, and the assertion below
  // says which is which rather than leaving the stronger claim to be assumed.
  let ordinary = [1.0_f64, 0.1];
  let ordinary_reference = CoverageWeightedMean
    .aggregate_values(&embeddings, &ordinary, 2)
    .expect("the unscaled fold resolves");
  let ordinary_scaled = [ordinary[0] * 0.1, ordinary[1] * 0.1];
  assert_ne!(
    ordinary_scaled[1] / ordinary_scaled[0],
    ordinary[1] / ordinary[0],
    "this row is only evidence while the scaled slice is *not* proportional"
  );
  let got = CoverageWeightedMean
    .aggregate_values(&embeddings, &ordinary_scaled, 2)
    .expect("the scaled fold resolves too");
  assert_ne!(
    got, ordinary_reference,
    "an inexactly scaled slice is a different slice, and the bit-identical \
     contract must not be claimed for it"
  );
  assert_close_f64(&got, &ordinary_reference);

  // All-zero coverage is not a scale: no positive factor produces it, and the
  // zero vector it folds to has no direction. It stays `NonFinite`.
  let all_zero = CoverageWeightedMean.aggregate_values(&embeddings, &[0.0, 0.0], 2);
  assert!(
    matches!(all_zero, Err(WinditError::NonFinite)),
    "an all-zero coverage slice has no direction to report, got {all_zero:?}"
  );
}

/// A weight must not be *rounded into existence* before it multiplies its
/// component.
///
/// FALSIFIER. The weights are ratios, and a ratio of two in-domain coverages can
/// land anywhere in `f64` — including the subnormal range, where rounding stops
/// being relative and becomes absolute. Materializing such a ratio as a value
/// replaces it with one up to a factor of two away, and the fold then answers a
/// question nobody asked. Neither the compensated sum nor the determinacy gate
/// can recover information destroyed before the multiply.
#[test]
fn subnormal_coverage_ratios_do_not_fabricate_a_direction() {
  // `eta` is the minimum `f64` subnormal, and all three coverages are ordinary
  // in-domain fractions. Against a largest coverage of `0.75` the intended
  // weights are `1`, `(4/3)eta` and `(8/3)eta`, and against these components the
  // exact weighted sum is `(4/3)eta * -2^400 + (8/3)eta * 2^399`, which is
  // exactly zero. There is no direction here to report.
  let eta = f64::from_bits(1);
  let coverages = [0.75, eta, 2.0 * eta];
  let embeddings: [&[f64]; 3] = [&[0.0], &[-libm::ldexp(1.0, 400)], &[libm::ldexp(1.0, 399)]];
  let got = CoverageWeightedMean.aggregate_values(&embeddings, &coverages, 1);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "an exactly cancelling in-domain fold has no direction, got {got:?}"
  );

  // The blunter witness, where the fabrication is a wrong answer rather than a
  // wrong verdict: the same coverages against two orthogonal components. The
  // exact sum is `(4/3)eta * [2^100, 0] + (8/3)eta * [0, 2^100]`, whose
  // direction is that of `[1, 2]`. Rounding the two ratios independently to
  // `eta` and `3 * eta` turns it into the direction of `[1, 3]`.
  let two = libm::ldexp(1.0, 100);
  let orthogonal: [&[f64]; 3] = [&[0.0, 0.0], &[two, 0.0], &[0.0, two]];
  let want = MeanRenormalized
    .aggregate_values(&[&[1.0, 2.0]], &[1.0], 2)
    .expect("[1, 2] has a direction");
  let got = CoverageWeightedMean
    .aggregate_values(&orthogonal, &coverages, 2)
    .expect("two nonzero terms in one quadrant have a direction");
  assert_eq!(
    got, want,
    "the fold must point where the exact weighted sum does, not where the \
     rounded ratios do"
  );

  // The same shape at eight ratios, so the fix is a property rather than one
  // arithmetic coincidence. The ideal weights are `(4/3)eta` and `(4k/3)eta`, so
  // the answer is the direction of `[1, k]` whatever `k` is; independently
  // rounded ratios give `[1, round(4k/3)]`, which agrees only at `k = 1`.
  for k in 1..=8_u32 {
    let coverages = [0.75, eta, f64::from(k) * eta];
    let want = MeanRenormalized
      .aggregate_values(&[&[1.0, f64::from(k)]], &[1.0], 2)
      .expect("[1, k] has a direction");
    let got = CoverageWeightedMean
      .aggregate_values(&orthogonal, &coverages, 2)
      .expect("two nonzero terms in one quadrant have a direction");
    for (g, w) in got.iter().zip(&want) {
      let gap = if g > w { g - w } else { w - g };
      assert!(
        gap <= 4.0 * f64::EPSILON * w.abs(),
        "at k = {k} the fold must reach the direction of [1, {k}] to within the \
         rounding its weights carry: got {got:?}, want {want:?}"
      );
    }
  }

  // A window that carries no coverage at all must not disable the lift for the
  // others. It is the *smallest nonzero* weight that decides whether a ratio can
  // be represented, and a zero weight is not a ratio that has to be — it is a
  // window that drops out. A reduction that let the zero through comes back with
  // the un-lifted answer for everything else.
  let with_a_gap = [0.75, eta, 2.0 * eta, 0.0];
  let four: [&[f64]; 4] = [&[0.0, 0.0], &[two, 0.0], &[0.0, two], &[0.0, 0.0]];
  let got = CoverageWeightedMean
    .aggregate_values(&four, &with_a_gap, 2)
    .expect("the zero-coverage window drops out; the rest still has a direction");
  assert_eq!(
    got, want,
    "a zero coverage must not decide the lift for the weights that are ratios"
  );
}

/// The lift that keeps a weight out of the subnormal range is the identity
/// everywhere the weights were already sound.
///
/// Which is the whole of its design: a correction that moved the common case
/// would be a fourth re-measure of a fold this release has already re-measured
/// three times, bought to fix a regime no plan can reach. So the engagement
/// boundary is pinned from both sides, and so is the bound on how far the lift
/// can ever go.
#[test]
fn the_weight_lift_engages_only_below_the_normal_boundary() {
  assert_eq!(
    f64::MIN_POSITIVE.exponent(),
    MIN_NORMAL_EXPONENT,
    "the boundary constant is the smallest normal f64's exponent, not a literal \
     that happens to look like one"
  );

  // The ratio decides, and nothing else. Sweeping the smaller coverage across
  // every exponent an `f64` has puts the boundary at exactly the place a weight
  // stops being normal, and bounds the lift by 53 there rather than by argument.
  for exp in -1074..=0 {
    let coverages = [1.0, libm::ldexp(1.0, exp)];
    let shift = normalizing_shift(&coverages, 1.0);
    assert_eq!(
      shift == 0,
      exp >= MIN_NORMAL_EXPONENT,
      "at a weight of 2^{exp} the lift must engage if and only if that weight is \
       subnormal, got {shift}"
    );
    assert!(
      (0..=53).contains(&shift),
      "no in-domain slice can ask for a lift outside [0, 53], got {shift} at \
       2^{exp}"
    );
  }

  // The lift is computed before `check_inputs` runs, so an out-of-domain largest
  // can drive the quotient to zero — and `Real::exponent` is documented for a
  // finite *nonzero* value. What it returns for zero is therefore pinned here
  // rather than relied on: whatever it is, it must not read as a subnormal
  // weight, so no lift is attempted on a slice that is about to be rejected.
  assert!(
    0.0_f64.exponent() >= MIN_NORMAL_EXPONENT,
    "a quotient that underflowed to zero must not be read as a subnormal weight"
  );
  let out_of_domain = [1e300, f64::from_bits(1)];
  assert_eq!(
    f64::from_bits(1) / 1e300,
    0.0,
    "this row is only evidence while the quotient really does underflow"
  );
  assert_eq!(
    normalizing_shift(&out_of_domain, max_magnitude(&out_of_domain)),
    0,
    "no lift before the rejection"
  );
  assert!(matches!(
    CoverageWeightedMean.aggregate_values(&[&[1.0], &[1.0]], &out_of_domain, 1),
    Err(WinditError::CoverageOutOfRange { window: 0 })
  ));

  // And the lift it does ask for keeps every product inside the domain's own
  // ceiling, which is what makes 53 a bound rather than a hope.
  assert!(
    (libm::ldexp(1.0, 53) * <f64 as Real>::MAX_AGG_MAGNITUDE).is_finite(),
    "the largest lift must leave the largest in-domain product representable"
  );

  // The largest weight stays exactly a power of two through the lift: `m` scaled
  // by `2^s` and divided by `m` is `2^s` to the bit, at every `m` and every `s`
  // the policy can reach. That is what keeps the fold reading ratios only.
  for exp in [0, -1, -7, -52, -53, -400, -1000, -1021, -1022, -1074] {
    let largest = libm::ldexp(0.75, exp);
    for shift in [0, 1, 2, 52, 53] {
      assert_eq!(
        libm::ldexp(largest, shift) / largest,
        libm::ldexp(1.0, shift),
        "the largest weight must be exactly 2^{shift} at a largest coverage of \
         {largest:?}"
      );
    }
  }

  // A few whole slices, including the two boundary neighbours and the witness
  // the falsifier above is built from.
  let eta = f64::from_bits(1);
  let rows: [(&[f64], i32); 7] = [
    (&[1.0, 0.5], 0),
    (&[1.0 / 3.0, 1.0], 0),
    (&[eta, 4.0 * eta], 0),
    (&[1.0, libm::ldexp(1.0, MIN_NORMAL_EXPONENT)], 0),
    (&[1.0, libm::ldexp(1.0, MIN_NORMAL_EXPONENT - 1)], 2),
    (&[0.75, eta, 2.0 * eta], 53),
    (&[0.0, 0.0], 0),
  ];
  for (coverages, want) in rows {
    let shift = normalizing_shift(coverages, max_magnitude(coverages));
    assert_eq!(shift, want, "wrong lift for {coverages:?}");
  }
}

/// The lift changes nothing it does not have to change.
///
/// **Real plan output never engages the lift, and that is structural:** a
/// plan's non-final windows all carry coverage exactly `1.0`, so a real slice's
/// largest coverage is always `1.0`, and its smallest is bounded below by
/// `1 / usize::MAX` — a plan's coverages are at worst that far apart — which
/// keeps `shift` at `0` (the lift engages only past a ratio of `2^1022`) on
/// every slice a plan can produce, independent of this or any other sample.
///
/// This is a broader characterization check on top of that proof, not the
/// source of it: a sweep over the same 20736 *synthetic* four-window coverage
/// slices this release's previous `CoverageWeightedMean` change was measured
/// over — arbitrary four-tuples of the twelve `len / 12` ratios
/// `Span::coverage` can produce, pushed through the policy directly rather than
/// through a `WindowPlan`, so most of them are not slices any plan would
/// actually emit. Every one of them still folds bit-identically against the
/// weighting as it stood before the lift. A fourth re-measure of this fold
/// would need its own entry in the changelog; this test is what says there is
/// not one.
#[test]
fn the_weight_lift_is_the_identity_on_every_synthetic_direct_api_slice() {
  // The weighting verbatim as it was, folded through the very same routine, so
  // the weight is the only thing that differs.
  fn unlifted(
    embeddings: &[&[f64]],
    coverages: &[f64],
    dim: usize,
  ) -> Result<Vec<f64>, WinditError> {
    let largest = max_magnitude(coverages);
    weighted_sum_renorm(embeddings, coverages, dim, move |i, _| {
      if largest > 0.0 {
        coverages[i] / largest
      } else {
        0.0
      }
    })
  }

  let ratios: Vec<f64> = (1..=12)
    .map(|len| Span::new(0, len, 12).coverage())
    .collect();
  let embeddings: [&[f64]; 4] = [&[1.0, 0.0], &[0.0, 1.0], &[0.6, 0.8], &[0.8, -0.6]];
  let mut folded = 0_u32;
  for &a in &ratios {
    for &b in &ratios {
      for &c in &ratios {
        for &d in &ratios {
          let coverages = [a, b, c, d];
          assert_eq!(
            normalizing_shift(&coverages, max_magnitude(&coverages)),
            0,
            "no plan-reachable slice engages the lift, {coverages:?} did"
          );
          let want = unlifted(&embeddings, &coverages, 2);
          let got = CoverageWeightedMean.aggregate_values(&embeddings, &coverages, 2);
          assert_eq!(
            got.as_ref().ok(),
            want.as_ref().ok(),
            "the lift moved a fold it does not engage on, at {coverages:?}"
          );
          folded += 1;
        }
      }
    }
  }
  assert_eq!(
    folded, 20736,
    "the sweep must cover every four-window combination"
  );

  // Non-vacuity: where the lift *does* engage the two disagree, and the lifted
  // answer is the right one. Without this the assertion above would be satisfied
  // by a lift that never engaged at all.
  let eta = f64::from_bits(1);
  let coverages = [0.75, eta, 2.0 * eta];
  let two = libm::ldexp(1.0, 100);
  let orthogonal: [&[f64]; 3] = [&[0.0, 0.0], &[two, 0.0], &[0.0, two]];
  let before = unlifted(&orthogonal, &coverages, 2).expect("the un-lifted fold answers");
  let after = CoverageWeightedMean
    .aggregate_values(&orthogonal, &coverages, 2)
    .expect("the lifted fold answers");
  assert_ne!(before, after, "the lift must engage somewhere");
  let want = MeanRenormalized
    .aggregate_values(&[&[1.0, 2.0]], &[1.0], 2)
    .expect("[1, 2] has a direction");
  assert_eq!(after, want, "and where it engages it must be right");
}

/// Two consequences of the weights being normalized, each the kind of thing a
/// caller can rely on.
///
/// A single window has nothing to weigh against, so its coverage cannot matter
/// at all: whatever it is, the answer is that window's own direction, which is
/// exactly what [`MeanRenormalized`] returns. Before normalization the fold
/// multiplied by the coverage first and renormalized after, so a coverage that
/// is not a power of two cost an ulp for nothing — `2/3` on `[3, 4]` returned
/// `[0.6, 0.7999999999999999]`.
///
/// And a slice that contains a full window — every plan with one does — divides
/// by exactly `1.0`, so normalization is the identity and those folds are
/// bit-identical to the un-normalized ones.
#[test]
fn normalized_weights_leave_the_common_geometries_where_they_were() {
  let raw: [&[f64]; 1] = [&[3.0, 4.0]];
  let reference = MeanRenormalized
    .aggregate_values(&raw, &[1.0], 2)
    .expect("one window has a direction");
  assert_eq!(reference, vec![0.6, 0.8], "the exact direction of [3, 4]");
  for len in 1..=3_usize {
    let coverage = Span::new(0, len, 3).coverage();
    let got = CoverageWeightedMean
      .aggregate_values(&raw, &[coverage], 2)
      .expect("one window has a direction at every coverage");
    assert_eq!(
      got, reference,
      "one window at coverage {coverage:?} must be its own direction"
    );
  }

  // A full window present: the divisor is exactly 1.0 and every weight is its own
  // coverage, unrounded.
  let four: [&[f64]; 4] = [&[1.0, 0.0], &[0.0, 1.0], &[0.6, 0.8], &[0.8, -0.6]];
  let coverages = [1.0, 1.0, 1.0, 1.0 / 3.0];
  let got = CoverageWeightedMean
    .aggregate_values(&four, &coverages, 2)
    .unwrap();
  let mut acc = [0.0_f64; 2];
  for (e, c) in four.iter().zip(coverages) {
    for (a, x) in acc.iter_mut().zip(e.iter()) {
      *a += c * x;
    }
  }
  let norm = libm::sqrt(acc[0] * acc[0] + acc[1] * acc[1]);
  assert_close_f64(&got, &[acc[0] / norm, acc[1] / norm]);
}

/// The divisor is the *largest* coverage, and no other entry will do.
///
/// Any fixed positive divisor preserves the ratios a weighted mean reads, so the
/// choice looks free. It is not: dividing by the largest is what puts every
/// weight in `(0, 2^shift]` with the largest exactly `2^shift`, and a largest
/// weight the caller's scale cannot move is the property the determinacy gate's
/// absolute floor is read against. Dividing by, say, the first coverage instead
/// fails in both directions — a leading zero annihilates a fold that has a
/// direction, and a leading *smallest* sends the other weights past `f64`'s
/// range.
#[test]
fn the_divisor_is_the_largest_coverage() {
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];

  // A window with no coverage contributes nothing, and the rest still folds.
  let got = CoverageWeightedMean
    .aggregate_values(&embeddings, &[0.0, 1.0], 2)
    .expect("a zero-coverage window drops out; the other still has a direction");
  assert_close_f64(&got, &[0.0, 1.0]);

  // The smallest coverage first: dividing by it would make the other weight
  // `2^1074`, which is not an `f64` at all.
  let got = CoverageWeightedMean
    .aggregate_values(&embeddings, &[f64::from_bits(1), 1.0], 2)
    .expect("no weight exceeds 1, so nothing here can leave the range");
  assert_close_f64(&got, &[0.0, 1.0]);
}

/// The recurrence's oldest window carries no `alpha` factor, and the weights
/// that fact produces do not generally sum to one in `f64`.
///
/// Both halves are corrections to prose, and both are pinned here because prose
/// is where they went wrong. The module's Input domain note displayed
/// `w_i = alpha * (1 - alpha)^(n - 1 - i)` for every `i`, which is not the split
/// `s_i = alpha * e_i + (1 - alpha) * s_{i-1}` from `s_0 = e_0` produces — nothing
/// preceded the first window for it to blend with — and it claimed the
/// materialized weights sum to exactly `1`. The *ideal* weights do, in exact
/// arithmetic. The `f64` ones do not, and a spot check at a dyadic `alpha` cannot
/// see the difference.
#[test]
fn ema_weights_are_the_split_the_recurrence_produces() {
  // The implementation's own backward pass, replicated so the two claims below
  // are about the numbers the fold actually carries. The basis fold that follows
  // is what ties this replica to the policy.
  fn materialized(alpha: f64, n: usize) -> Vec<f64> {
    let complement = 1.0 - alpha;
    let mut w = vec![0.0; n];
    let mut power = 1.0;
    for i in (1..n).rev() {
      w[i] = alpha * power;
      power *= complement;
    }
    if n > 0 {
      w[0] = power;
    }
    w
  }

  // Folding the standard basis makes the weight vector itself observable, up to
  // the L2 normalization every policy ends with. The two candidate formulas
  // differ only in the oldest window and there by exactly a factor of `alpha`,
  // so the fold can tell them apart.
  let alpha = 0.3_f64;
  let n = 4_usize;
  let basis: Vec<Vec<f64>> = (0..n)
    .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  let refs: Vec<&[f64]> = basis.iter().map(Vec::as_slice).collect();
  let coverages = vec![1.0; n];
  let got = EmaRenormalized::new(alpha)
    .aggregate_values(&refs, &coverages, n)
    .expect("a convex EMA over the basis has a direction");

  let split = materialized(alpha, n);
  let mut uniform_alpha = split.clone();
  uniform_alpha[0] *= alpha;
  let mut want_split = split.clone();
  let mut want_uniform = uniform_alpha.clone();
  l2_renorm(&mut want_split).expect("the split weights have a direction");
  l2_renorm(&mut want_uniform).expect("the uniform-alpha weights have a direction");
  assert_close_f64(&got, &want_split);
  let gap = got[0] - want_uniform[0];
  assert!(
    gap > 0.1,
    "the displayed formula must be the one the fold uses: {got:?} against the \
     uniform-alpha {want_uniform:?}"
  );

  // And the sum. At a dyadic `alpha` every weight is exact and the sum is exactly
  // one, which is why checking `0.5` proves nothing about `0.3`.
  let sum = |w: &[f64]| w.iter().fold(0.0_f64, |a, x| a + x);
  for n in 2..=8 {
    assert_eq!(
      sum(&materialized(0.5, n)),
      1.0,
      "a dyadic alpha keeps every weight exact"
    );
  }
  assert_eq!(
    sum(&materialized(0.3, 3)),
    1.0,
    "and 0.3 survives to n = 3, which is how a partial check passes"
  );
  assert_ne!(
    sum(&materialized(0.3, 4)),
    1.0,
    "but the materialized weights do not generally sum to one, and no part of \
     the policy needs them to"
  );
  assert_eq!(sum(&materialized(0.3, 4)), 0.999_999_999_999_999_8);
}

/// `a * b` as an exact unevaluated `product + error` pair.
///
/// Dekker's exact product: splitting each operand at 27 bits makes the four
/// partial products exact, so `e` is the part of `a * b` that the rounded
/// product dropped. Carrying that alongside the product gives about 106 bits,
/// which is what it takes to see an error of tens of `u` for what it is. Only
/// ever called on `|b| < 1` and its own shrinking powers, so no split overflows.
fn two_product(a: f64, b: f64) -> (f64, f64) {
  let split = |x: f64| {
    let c = 134_217_729.0 * x;
    let hi = c - (c - x);
    (hi, x - hi)
  };
  let p = a * b;
  let (ah, al) = split(a);
  let (bh, bl) = split(b);
  (p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl)
}

/// `(1 - alpha)^k` as a `hi + lo` pair, from the complement carried the same
/// way.
///
/// The complement must be a pair and not an `f64`: `1 - alpha` is *not*
/// generally representable (`1 - 0.46` is not), and the documented ideal weight
/// is `alpha * (1 - alpha)^k` at the exact difference. Rounding the complement
/// once and then raising it multiplies that single rounding by `k`, which is the
/// larger half of the error this whole test is about. `Fast2Sum` splits it
/// exactly: `1` dominates `alpha`, so `(1 - hi) - alpha` is the part `hi` could
/// not hold.
fn dd_complement(alpha: f64) -> (f64, f64) {
  let hi = 1.0 - alpha;
  (hi, (1.0 - hi) - alpha)
}

fn dd_power(bhi: f64, blo: f64, k: usize) -> (f64, f64) {
  let (mut hi, mut lo) = (1.0_f64, 0.0_f64);
  for _ in 0..k {
    let (p, e) = two_product(hi, bhi);
    let t = ((e + hi * blo) + lo * bhi) + lo * blo;
    // Renormalize so `hi` stays the leading term.
    let s = p + t;
    lo = (p - s) + t;
    hi = s;
  }
  (hi, lo)
}

/// The EMA weight ladder, and the fold it drives, replicated so the two tests
/// below measure the numbers the policy actually carries rather than a model of
/// them.
///
/// `ema_ladder` is `EmaRenormalized::aggregate_values`' backward pass;
/// `ema_gate_ratio` is [`weighted_sum_renorm`]'s Neumaier fold and determinacy
/// threshold at `dim == 1`, returning the residue over the threshold that judges
/// it. A ratio at or under `1` is the verdict [`WinditError::NonFinite`].
fn ema_ladder(alpha: f64, n: usize) -> Vec<f64> {
  let complement = 1.0 - alpha;
  let mut w = vec![0.0; n];
  let mut power = 1.0;
  for i in (1..n).rev() {
    w[i] = alpha * power;
    power *= complement;
  }
  if n > 0 {
    w[0] = power;
  }
  w
}

fn ema_gate_ratio(weights: &[f64], components: &[f64]) -> f64 {
  let (mut acc, mut comp, mut mag) = (0.0_f64, 0.0_f64, 0.0_f64);
  for (w, e) in weights.iter().zip(components) {
    let term = w * e;
    let sum = acc + term;
    comp += if acc.abs() >= term.abs() {
      (acc - sum) + term
    } else {
      (term - sum) + acc
    };
    acc = sum;
    mag += term.abs();
  }
  acc += comp;
  (acc + 0.0).abs() / (16.0 * f64::EPSILON * mag + f64::MIN_GATE_THRESHOLD)
}

/// The accumulated weight error is real and **no input can deliver it to the
/// determinacy gate**. It is a bound, not a defect.
///
/// FALSIFIER for the bound's *reach*. The weights are built by repeated
/// multiplication, so weight `k` carries the error of every multiply before it —
/// about `0.7 * n * u`, past the gate's own `16 * EPSILON = 32u` somewhere near
/// `n = 32`. That crossing is what a fold would have to exploit, and the
/// exploitation is what does not exist.
///
/// For any input whose *ideal* weighted sum is exactly zero, write `t_i` for the
/// ideal terms (`sum_i t_i = 0`) and `d_i` for each weight's relative error. The
/// residue is `sum_i t_i * d_i`, and the gate measures it against
/// `32u * sum_i |t_i|`; because the `t_i` sum to zero, any constant may be
/// subtracted from `d`, so
///
/// ```text
/// residue / mass <= (max_i d_i - min_i d_i) / 2
/// ```
///
/// — the *spread* of the weight error over the input's support, never its size.
/// So a witness needs a support carrying a spread above `64u`, and an exact
/// cancellation across it.
///
/// A two-window pair at chain distance `d` cancels exactly only when its ideal
/// weight ratio `(1 - alpha)^d` is an exact ratio of two `f64` significands.
/// Writing the complement as `B * 2^-q` with `B` odd, that ratio's odd part is
/// `B^d`, so it needs `B^d < 2^53` — a lever capped at `d <= 53 / log2(B)`. The
/// same `B` that buys a long lever makes `(1 - alpha)^k` exactly representable
/// for `k` up to that same cap, which is precisely where the chain has no error
/// yet. That tension is the whole result, and it was measured rather than
/// argued: over every complement `B * 2^-q` with `B` odd up to `2^40 - 1` and
/// every `q` in `1..=53`, at every chain index whose materialized weight is a
/// normal `f64`, the largest reachable `|d_k - d_{k+d}|` with `d` inside the
/// lever cap is **10.0u** — against the `64u` a witness needs. The complement
/// whose *whole* error spread does clear `64u` needs a support spanning 131 to
/// 7609 chain steps to realize it, 65 to 1300 times its own lever cap; and a
/// three-window chain, which is how a support reaches past its own lever,
/// extends it by a measured **2.2x** and no further, so closing that gap takes
/// on the order of 30 to 600 support windows whose interior terms all vanish
/// against the two carrying the mass.
///
/// Driving the fold rather than the arithmetic says the same thing: **5 072 311**
/// exactly cancelling pairs, every short-mantissa complement at every chain
/// index whose weights are both normal, and the largest residue any of them
/// leaves is **0.162** of the threshold. This pins that maximum.
/// `alpha = 0.625` puts the complement at `3/8`, the odd part that buys the
/// longest lever (`3^33 < 2^53`), and chain indices 651 and 684 are where the
/// accumulated error between two such indices is widest.
#[test]
fn ema_weight_error_accumulates_but_no_input_can_reach_the_gate() {
  // The measured growth first, so the bound is on record as a number rather
  // than a claim. It is against the *ideal* `(1 - alpha)^k`, so it needs a
  // reference wider than the `f64` under test: `dd_power` carries the power as
  // an unevaluated `hi + lo` pair (about 106 bits), which the self-check below
  // pins against the one case where the plain chain is exact.
  assert_eq!(
    dd_complement(0.5),
    (0.5, 0.0),
    "a dyadic complement is exact"
  );
  let (dhi, dlo) = dd_power(0.5, 0.0, 40);
  assert_eq!(
    (dhi, dlo),
    (libm::ldexp(1.0, -40), 0.0),
    "and a dyadic power is exact, which is what pins the reference"
  );

  let n = 64_usize;
  let alpha = 0.46_f64;

  // The replica is only evidence while it is the policy's own ladder. Folding
  // the standard basis makes the weight vector observable up to the L2
  // normalization every policy ends with, and the comparison is bit-for-bit:
  // each product is `w_i * 1` or `w_i * 0`, so the fold reproduces the ladder
  // exactly and `l2_renorm` is then the same call on the same values. Any change
  // to how the policy forms its weights parts the two.
  let basis: Vec<Vec<f64>> = (0..n)
    .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  let refs: Vec<&[f64]> = basis.iter().map(Vec::as_slice).collect();
  let folded = EmaRenormalized::new(alpha)
    .aggregate_values(&refs, &vec![1.0; n], n)
    .expect("a convex EMA over the basis has a direction");
  let mut replica = ema_ladder(alpha, n);
  l2_renorm(&mut replica).expect("the ladder has a direction");
  assert_eq!(
    folded, replica,
    "the replica must be the ladder the policy actually builds"
  );

  let (bhi, blo) = dd_complement(alpha);
  assert!(
    blo != 0.0,
    "1 - 0.46 is not an f64, and that rounding is the larger half of the error"
  );
  let w = ema_ladder(alpha, n);
  let mut worst = 0.0_f64;
  let mut worst_powi = 0.0_f64;
  for k in 0..n - 1 {
    let (hi, lo) = dd_power(bhi, blo, k);
    // `alpha * hi` and `alpha * lo` scale the pair; the relative error of the
    // materialized weight against it is `((w - a*hi) - a*lo) / (a*hi)`.
    let (rhi, rlo) = (alpha * hi, alpha * lo);
    worst = worst.max((((w[n - 1 - k] - rhi) - rlo) / rhi).abs());
    let by_powi = alpha * (1.0 - alpha).powi(k as i32);
    worst_powi = worst_powi.max((((by_powi - rhi) - rlo) / rhi).abs());
  }
  let u = f64::EPSILON / 2.0;
  assert!(
    (58.0..60.0).contains(&(worst / u)),
    "repeated multiplication reaches tens of u by n = 64, got {} u",
    worst / u
  );

  // And the cure the issue named, measured rather than repeated.
  // `alpha * (1 - alpha).powi(k)` is *not* "one rounding instead of k". `powi`
  // is exponentiation by squaring, so it is `O(log k)` roundings and not
  // correctly rounded; and, decisively, it raises the same `fl(1 - alpha)` the
  // chain does. That single complement rounding, multiplied by `k`, is the
  // larger part of the error, and `powi` does not touch it. It buys 18% here,
  // not `k -> 1`. At a dyadic alpha, where the complement is exact, both are
  // exact and there is nothing to buy.
  assert!(
    (47.0..50.0).contains(&(worst_powi / u)),
    "powi is O(log k) roundings on top of the complement's, not one: {} u",
    worst_powi / u
  );
  assert!(
    worst / worst_powi > 1.20 && worst / worst_powi < 1.25,
    "so it improves the weights by about a fifth, not by a factor of k: {}x",
    worst / worst_powi
  );
  for k in [4_usize, 64, 199] {
    let (hi, lo) = dd_power(0.5, 0.0, k);
    assert_eq!(
      (ema_ladder(0.5, k + 2)[1], 0.0),
      (0.5 * hi, 0.5 * lo),
      "and at the dyadic alpha the chain is already exact, at k = {k}"
    );
  }

  // And the reach. `3^33 < 2^53`, so `(3/8)^33 = 3^33 / 2^99` is an exact `f64`
  // and `c_hi = -c_lo / (3/8)^33` is exactly `-2^399`; the pair then cancels to
  // zero in exact arithmetic, the terms being
  // `alpha * b^651 * (c_lo + b^33 * c_hi)` with the bracket exactly zero. Both
  // components sit inside the input domain's `2^400`.
  let alpha = 0.625_f64;
  let b = 1.0 - alpha;
  assert_eq!(
    b, 0.375,
    "the complement must be exact, or the ideal is not b"
  );
  let (k1, d) = (651_usize, 33_usize);
  let n = k1 + d + 2;
  let c_lo = libm::ldexp(3.0_f64.powi(33), 300);
  assert_eq!(
    3.0_f64.powi(33),
    5_559_060_566_555_523.0,
    "3^33 must be an exact f64, or the lever is not exact"
  );
  let ratio = b.powi(d as i32);
  let c_hi = -c_lo / ratio;
  assert_eq!(
    c_lo + c_hi * ratio,
    0.0,
    "the ideal weighted sum must be exactly zero"
  );

  let w = ema_ladder(alpha, n);
  let (i1, i2) = (n - 1 - k1, n - 1 - (k1 + d));
  assert!(
    w[i1] >= f64::MIN_POSITIVE && w[i2] >= f64::MIN_POSITIVE,
    "both weights must be normal, or this is the underflow regime and not this \
     one: {:e}, {:e}",
    w[i1],
    w[i2]
  );
  let mut components = vec![0.0_f64; n];
  components[i1] = c_lo;
  components[i2] = c_hi;

  let ratio = ema_gate_ratio(&w, &components);
  assert!(
    (0.16..0.17).contains(&ratio),
    "the accumulated error's strongest reachable residue is a sixth of the \
     gate, got {ratio}"
  );
  let lever: Vec<&[f64]> = components.iter().map(core::slice::from_ref).collect();
  let got = EmaRenormalized::new(alpha).aggregate_values(&lever, &vec![1.0; n], 1);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "an exactly cancelling in-domain fold has no direction, got {got:?}"
  );
}

/// A weight whose *ideal* value has left `f64`'s exponent range is rounded
/// **absolutely**, and the determinacy gate's absolute floor does not cover it.
///
/// CHARACTERIZATION of a live gap, not a guarantee — see
/// <https://github.com/findit-studio/windit/issues/17>. The module note used to
/// claim this regime was sound because a subnormal weight drives the fold's own
/// products subnormal, leaving `MIN_GATE_THRESHOLD` to gate alone. That holds
/// only while the components are `O(1)`. The [input domain](super#input-domain)
/// admits components up to `2^400`, and one of those lifts the product of a
/// subnormal weight back into the ordinary range, where the floor is nowhere
/// near it.
///
/// Nothing here is the repeated multiplication: at `alpha = 0.9` the chain's
/// accumulated error is a few `u`, and the same input works at `alpha = 0.5`
/// where every representable weight is exact to the bit. What breaks is that
/// the ideal ratio between two adjacent weights — `1 / (1 - alpha)`, a factor
/// of ten here — is not representable at the bottom of the subnormal grid, so
/// the older weight rounds to zero while the newer one survives. Evaluating
/// each weight once (`powi`) reaches the same zero: `alpha * 0.1^324` is not an
/// `f64`, however it is computed.
#[test]
fn ema_weights_below_the_exponent_range_fabricate_a_direction() {
  let alpha = 0.9_f64;
  let b = 1.0 - alpha;
  assert_eq!(
    b, 0.099_999_999_999_999_98,
    "1 - 0.9 is exact (Sterbenz), so the ideal complement is the one the fold \
     uses and every weight error below is the exponent range and nothing else"
  );

  // `b` is `900_719_925_474_099 * 2^-53`, so building the newer component out of
  // that significand makes `c_hi = -c_lo / b` the exact power of two `-2^83`,
  // and the ideal pair cancels to zero by construction rather than by luck.
  assert_eq!(
    libm::ldexp(900_719_925_474_099.0, -53),
    b,
    "b's significand"
  );
  let c_lo = libm::ldexp(900_719_925_474_099.0, 30);
  let c_hi = -c_lo / b;
  assert_eq!(c_hi, -libm::ldexp(1.0, 83), "so the quotient is exact");
  assert_eq!(
    c_lo + c_hi * b,
    0.0,
    "the ideal weighted sum must be exactly zero"
  );
  assert!(
    c_lo.abs() < f64::MAX_AGG_MAGNITUDE && c_hi.abs() < f64::MAX_AGG_MAGNITUDE,
    "and both components must be ordinary in-domain values: {c_lo:e}, {c_hi:e}"
  );

  let k1 = 323_usize;
  let n = k1 + 3;
  let w = ema_ladder(alpha, n);
  let (i1, i2) = (n - 1 - k1, n - 1 - (k1 + 1));
  assert!(
    w[i1] > 0.0 && w[i1] < f64::MIN_POSITIVE,
    "the surviving weight is a subnormal, one step above the flush: {:e}",
    w[i1]
  );
  assert_eq!(
    w[i2], 0.0,
    "and its ideal partner, a tenth of it, has no f64 at all"
  );

  let mut components = vec![0.0_f64; n];
  components[i1] = c_lo;
  components[i2] = c_hi;
  let ratio = ema_gate_ratio(&w, &components);
  assert!(
    ratio > 100.0,
    "the residue clears the gate by two orders of magnitude even with the \
     absolute floor fully engaged, got {ratio:e}"
  );

  // The verdict this pins is the defect. When #17 is fixed this assertion is
  // what has to change, and the numbers above are what it has to change to.
  let refs: Vec<&[f64]> = components.iter().map(core::slice::from_ref).collect();
  let got = EmaRenormalized::new(alpha).aggregate_values(&refs, &vec![1.0; n], 1);
  assert!(
    matches!(got.as_deref(), Ok([1.0])),
    "TODAY a direction is fabricated from exact cancellation, at n = {n} and \
     ordinary components; got {got:?}"
  );
}

#[test]
fn ema_alpha_range_is_rejected_at_f64() {
  // The alpha range check runs on the configuration field, which is now the
  // compute scalar itself, so the bounds are `C::ZERO` and `C::ONE` and the
  // rejection is the same at every `Real`.
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0, 1.0];
  assert!(matches!(
    EmaRenormalized::new(2.0).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  assert!(matches!(
    EmaRenormalized::new(-0.5).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  assert!(matches!(
    EmaRenormalized::new(f64::NAN).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  assert!(EmaRenormalized::new(0.5)
    .aggregate_values(&embeddings, &coverages, 2)
    .is_ok());
}

#[test]
fn ema_renormalized_carries_a_coefficient_no_f32_can_hold() {
  // FALSIFIER for the coefficient's precision, the aggregation twin of the
  // vector smoother's. `aggregate_values` is `impl<C: Real>` and `Real` has one
  // implementor, `f64`, so the weights, their products and the compensated sum
  // are all `f64`; a coefficient on the `f32` grid was the one narrow value in
  // the whole fold.
  //
  // `1 - 2^-30` is the witness. The `f32` grid immediately below `1` is spaced
  // `2^-24`, so the nearest `f32` to it is exactly `1.0`, at which the recency
  // weights collapse to `[0, 0, 1]` and the fold returns the last window
  // verbatim — the entire prefix deleted rather than merely down-weighted.
  let two_m30 = libm::ldexp(1.0, -30);
  const ALPHA: f64 = 0.999_999_999_068_677_425_384_521_484_375; // 1 - 2^-30
  assert_eq!(ALPHA, 1.0 - two_m30, "the literal is exactly 1 - 2^-30");
  assert_eq!(
    ALPHA as f32, 1.0,
    "and no f32 is nearer to it than 1.0 itself"
  );
  assert_eq!(EmaRenormalized::new(ALPHA).alpha(), ALPHA, "and it is kept");

  let embeddings: [&[f64]; 3] = [&[0.0, 1.0], &[0.0, 1.0], &[1.0, 0.0]];
  let coverages = [1.0, 1.0, 1.0];
  let got = EmaRenormalized::new(ALPHA)
    .aggregate_values(&embeddings, &coverages, 2)
    .expect("in-domain fold");

  // With `c = 1 - alpha = 2^-30` the weights are `[c^2, alpha * c, alpha]`, so
  // the unnormalized fold is `[alpha, c^2 + alpha * c]` and its second
  // component is the whole of what an `f32` coefficient destroys.
  assert!(got[1] > 0.0, "the recency tail must survive: {got:?}");
  assert!(
    (got[1] / two_m30 - 1.0).abs() < 1e-8,
    "and it must be the coefficient's own 2^-30: {got:?}"
  );

  // Measured against the alpha an `f32` field would have rounded this to,
  // rather than merely asserted.
  let collapsed = EmaRenormalized::new(1.0)
    .aggregate_values(&embeddings, &coverages, 2)
    .expect("in-domain fold");
  assert_eq!(
    collapsed,
    vec![1.0, 0.0],
    "at alpha = 1 the whole prefix is gone"
  );
}

#[test]
fn ema_renormalized_resolves_two_coefficients_the_f32_grid_merges() {
  // The re-measure hazard this release carries, pinned as behaviour rather than
  // left to the changelog. `0.3` and the nearest `f32` to it are two distinct
  // `f64` coefficients that an `f32` field stores identically. A `0.2.x` caller
  // who wrote `EmaRenormalized::new(0.3)` folded with the second; the same
  // source now folds with the first, and the two do not agree.
  let exact = 0.3_f64;
  let via_f32 = f64::from(0.3_f32);
  assert_ne!(exact, via_f32, "the two coefficients differ in f64");
  assert_eq!(exact as f32, via_f32 as f32, "and coincide in f32");

  let embeddings: [&[f64]; 4] = [&[1.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0; 4];
  let fold = |alpha: f64| {
    EmaRenormalized::new(alpha)
      .aggregate_values(&embeddings, &coverages, 2)
      .expect("in-domain fold")
  };
  let a = fold(exact);
  let b = fold(via_f32);
  assert_ne!(a, b, "two coefficients, two folds");
  // Non-vacuity: the separation is the coefficients' own relative gap, about
  // `1.2e-8` — far above `f64` noise and far below anything an eyeball tolerance
  // would absorb.
  let gap = (a[0] - b[0]).abs() / a[0];
  assert!(
    (1e-10..1e-7).contains(&gap),
    "the separation must be the coefficient's own, got {gap:e}"
  );
}

#[test]
fn quantized_storage_takes_widening_path() {
  // `TestQuantVec` stores i8 and computes in f64, so `as_compute_slice` returns
  // `None` and `aggregate` widens elementwise instead of borrowing. The pinned
  // output also documents the requantization contract: the aggregate unit vector
  // [0.894_427…, 0.447_213…] scales to [113.59, 56.80], which rounds half away
  // from zero to [114, 57].
  let windows = [
    Windowed::new(
      TestQuantVec::from_unnormalized(&[1.0, 0.0]).unwrap(),
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      TestQuantVec::from_unnormalized(&[0.0, 1.0]).unwrap(),
      Span::new(0, 2, 4),
    ),
  ];
  let out = aggregate(&CoverageWeightedMean, &windows).unwrap();
  assert_eq!(out.as_slice(), &[TestQuant(114), TestQuant(57)]);

  // The same geometry through the f32 storage type agrees before quantization.
  // Both storage types now widen to f64, so this pins that the requantization is
  // the only thing separating them.
  let f32_windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 1.0], 2, 4)];
  let f32_out = aggregate(&CoverageWeightedMean, &f32_windows).unwrap();
  let requantized: Vec<TestQuant> = f32_out
    .as_slice()
    .iter()
    .map(|x| TestQuant(libm::roundf(x * 127.0) as i8))
    .collect();
  assert_eq!(out.as_slice(), requantized.as_slice());

  // The determinacy gate holds at the widening path too: an exactly-cancelling
  // quantized pair [1, 0] and [-1, 0] quantizes to [127, 0] and [-127, 0], widens
  // to f64 [1, 0] and [-1, 0], and folds to the zero vector -> NonFinite. This is
  // the F1 class reached through i8 storage rather than the f64 borrow path.
  // (TestQuant's image {0} U +-[1/127, 1] cannot express an out-of-domain
  // magnitude, so MagnitudeOutOfRange genericity stays with the direct-f64 corner
  // tests.)
  let cancelling = [
    Windowed::new(
      TestQuantVec::from_unnormalized(&[1.0, 0.0]).unwrap(),
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      TestQuantVec::from_unnormalized(&[-1.0, 0.0]).unwrap(),
      Span::new(0, 4, 4),
    ),
  ];
  assert!(matches!(
    aggregate(&CoverageWeightedMean, &cancelling),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn mean_renormalized_equals_coverage_when_all_full() {
  // Every coverage is 1.0, so the coverage weights are uniform: both policies
  // reduce to the renormalized arithmetic mean.
  let windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 1.0], 4, 4)];
  let cov = aggregate(&CoverageWeightedMean, &windows).unwrap();
  let mean = aggregate(&MeanRenormalized, &windows).unwrap();
  assert_close(cov.as_slice(), mean.as_slice());
  assert_close(cov.as_slice(), &[0.70710677, 0.70710677]);
}

#[test]
fn saliency_weights_higher_norm_more() {
  // Raw slices with unequal norms (3 vs 1): saliency weights by ||emb||, so the
  // high-norm vector dominates the direction more than a plain mean would.
  let embeddings: [&[f64]; 2] = [&[3.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0, 1.0];
  let sal = SaliencyWeighted
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();
  assert_close_f64(&sal, &[0.993_883_734_673_618_8, 0.110_431_526_074_846_53]);

  let mean = MeanRenormalized
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();
  assert!(
    sal[0] > mean[0],
    "saliency should pull harder toward the high-norm vector"
  );
}

#[test]
fn ema_renormalized_recency() {
  // alpha 0.5 over three basis vectors: the most recent dominates. Reversing the
  // order reverses the emphasis, proving ordering matters.
  let coverages = [1.0, 1.0, 1.0];
  let fwd: [&[f64]; 3] = [&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
  let out = EmaRenormalized::new(0.5)
    .aggregate_values(&fwd, &coverages, 3)
    .unwrap();
  assert_close_f64(
    &out,
    &[
      0.408_248_290_463_863_1,
      0.408_248_290_463_863_1,
      0.816_496_580_927_726_1,
    ],
  );

  let rev: [&[f64]; 3] = [&[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]];
  let out_rev = EmaRenormalized::new(0.5)
    .aggregate_values(&rev, &coverages, 3)
    .unwrap();
  assert_close_f64(
    &out_rev,
    &[
      0.816_496_580_927_726_1,
      0.408_248_290_463_863_1,
      0.408_248_290_463_863_1,
    ],
  );
}

#[test]
fn ema_renormalized_rejects_out_of_range_alpha() {
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0, 1.0];
  // alpha 2.0 previously produced a sign-flipping "average" silently; now typed.
  assert!(matches!(
    EmaRenormalized::new(2.0).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  // A negative alpha is likewise rejected.
  assert!(matches!(
    EmaRenormalized::new(-0.5).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  // NaN alpha is out of range and is caught before it can yield a NaN vector.
  assert!(matches!(
    EmaRenormalized::new(f64::NAN).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  // The closed-interval endpoints stay valid.
  assert!(EmaRenormalized::new(0.0)
    .aggregate_values(&embeddings, &coverages, 2)
    .is_ok());
  assert!(EmaRenormalized::new(1.0)
    .aggregate_values(&embeddings, &coverages, 2)
    .is_ok());
}

/// A built-in policy's `aggregate_values`, as a plain function pointer so a test
/// can iterate over the four of them.
type PolicyRun = fn(&[&[f64]], &[f64], usize) -> Result<Vec<f64>, WinditError>;

/// The four built-in policies, paired with a name for failure messages, over one
/// compute scalar. Saves each magnitude test from spelling the list four times.
fn builtin_policies() -> [(&'static str, PolicyRun); 4] {
  [
    ("MeanRenormalized", |e, c, d| {
      MeanRenormalized.aggregate_values(e, c, d)
    }),
    ("CoverageWeightedMean", |e, c, d| {
      CoverageWeightedMean.aggregate_values(e, c, d)
    }),
    ("EmaRenormalized", |e, c, d| {
      EmaRenormalized::new(0.5).aggregate_values(e, c, d)
    }),
    ("SaliencyWeighted", |e, c, d| {
      SaliencyWeighted.aggregate_values(e, c, d)
    }),
  ]
}

#[test]
fn every_builtin_policy_normalizes_f32_range_vectors() {
  // These are the magnitudes that broke the f32 fold: `1e20` and `f32::MAX`
  // square past the f32 ceiling, `f32::from_bits(1)` is a subnormal, and
  // `[3e38, 3e38]` has a norm (`sqrt(2) * 3e38`) that overflows f32. Widened to
  // f64 they are all ordinary, which is the whole point of computing there — the
  // caller gets the unit vector every one of these has.
  let coverages = [1.0];
  for (raw, want) in [
    ([3e38_f64, 3e38], [core::f64::consts::FRAC_1_SQRT_2; 2]),
    ([1e20, 0.0], [1.0, 0.0]),
    ([f64::from(f32::MAX), 0.0], [1.0, 0.0]),
    ([1e-30, 0.0], [1.0, 0.0]),
    ([f64::from(f32::from_bits(1)), 0.0], [1.0, 0.0]),
  ] {
    let embeddings: [&[f64]; 1] = [&raw];
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, &coverages, 2)
        .unwrap_or_else(|e| panic!("{name} rejected the finite vector {raw:?}: {e:?}"));
      assert_close_f64(&got, &want);
    }
  }
}

#[test]
fn linear_policies_normalize_f64_range_vectors() {
  // REVOKED (settlement §4.5.6, deliberate): the former promise that any finite
  // f64 normalizes is retracted. These synthetic f64-range extremes all sit
  // outside the enforced input domain [2^-400, 2^400], so every row is now
  // rejected before arithmetic with MagnitudeOutOfRange rather than normalized.
  // No consumer produces them: an f32-storage embedding stays 250+ binary orders
  // inside the domain on both sides.
  let coverages = [1.0];
  let linear: [(&str, PolicyRun); 3] = [
    ("MeanRenormalized", |e, c, d| {
      MeanRenormalized.aggregate_values(e, c, d)
    }),
    ("CoverageWeightedMean", |e, c, d| {
      CoverageWeightedMean.aggregate_values(e, c, d)
    }),
    ("EmaRenormalized", |e, c, d| {
      EmaRenormalized::new(0.5).aggregate_values(e, c, d)
    }),
  ];
  for raw in [
    [1.5e308_f64, 1.5e308],
    [1e200, 0.0],
    [f64::MAX, 0.0],
    [1e-300, 0.0],
    [f64::from_bits(1), 0.0],
  ] {
    let embeddings: [&[f64]; 1] = [&raw];
    for (name, run) in linear {
      let got = run(&embeddings, &coverages, 2);
      assert!(
        matches!(got, Err(WinditError::MagnitudeOutOfRange { .. })),
        "{name} must reject out-of-domain {raw:?}, got {got:?}"
      );
    }
  }
}

#[test]
fn saliency_normalizes_within_its_f64_magnitude_window() {
  // REVOKED (settlement §4.5.6, deliberate): SaliencyWeighted loses its special
  // per-policy window. The crate-level input domain [2^-400, 2^400] replaces it,
  // so the former "inside the window" extremes (1e150, 1e-150) are now out of
  // domain and rejected before the square is formed, alongside the former
  // out-of-window ones (1e200, 1e-200).
  for raw in [[1e150_f64, 0.0], [1e-150, 0.0], [1e200, 0.0], [1e-200, 0.0]] {
    let embeddings: [&[f64]; 1] = [&raw];
    assert!(
      matches!(
        SaliencyWeighted.aggregate_values(&embeddings, &[1.0], 2),
        Err(WinditError::MagnitudeOutOfRange { .. })
      ),
      "SaliencyWeighted must reject out-of-domain {raw:?}"
    );
  }

  // The in-domain row stays: 1e100 is well within [2^-400, 2^400], so its square
  // (1e200) is an ordinary normal f64 and the window still normalizes.
  let in_domain: [&[f64]; 1] = [&[1e100, 1e100]];
  let got = SaliencyWeighted
    .aggregate_values(&in_domain, &[1.0], 2)
    .unwrap();
  assert_close_f64(&got, &[core::f64::consts::FRAC_1_SQRT_2; 2]);
}

#[test]
fn every_builtin_policy_normalizes_f32_range_sequences() {
  // A single window never exercises a policy's fold. Two do, and the fold is
  // where the remaining magnitude hazards lived: an accumulation that overflowed
  // at the top, and — the EMA halving a subnormal toward zero — one that was
  // flushed away at the bottom. Widened to f64, two windows pointing the same way
  // combine to that direction whatever their f32-range magnitude.
  let coverages = [1.0, 1.0];
  for raw in [
    [[3e38_f64, 0.0], [3e38, 0.0]],
    [[f64::from(f32::MAX), 0.0], [f64::from(f32::MAX), 0.0]],
    [[1e-30, 0.0], [1e-30, 0.0]],
    [
      [f64::from(f32::from_bits(1)), 0.0],
      [f64::from(f32::from_bits(1)), 0.0],
    ],
  ] {
    let embeddings: [&[f64]; 2] = [&raw[0], &raw[1]];
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, &coverages, 2)
        .unwrap_or_else(|e| panic!("{name} rejected the finite sequence {raw:?}: {e:?}"));
      assert_close_f64(&got, &[1.0, 0.0]);
    }
  }
}

#[test]
fn non_normalizable_vectors_are_rejected() {
  // The rejections that must survive the redesign: a zero vector has no
  // direction, and a non-finite component cannot be normalized. Both come back
  // as `NonFinite`, on the one path there is.
  let coverages = [1.0];
  let zero: [&[f64]; 1] = [&[0.0, 0.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&zero, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
  let nan: [&[f64]; 1] = [&[f64::NAN, 1.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&nan, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
  let inf: [&[f64]; 1] = [&[f64::INFINITY, 1.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&inf, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn saliency_magnitude_weights_survive_their_own_scale() {
  // SaliencyWeighted weights each input by its own L2 norm, so a 1e20 input
  // contributes 1e20 * 1e20 = 1e40 to the accumulator. That is finite in f64
  // (it was infinite in f32), so the accumulator's direction survives the
  // renormalization that ends the policy.
  let embeddings: [&[f64]; 2] = [&[1e20, 0.0], &[0.0, 1e20]];
  let out = SaliencyWeighted
    .aggregate_values(&embeddings, &[1.0, 1.0], 2)
    .unwrap();
  assert_close_f64(&out, &[core::f64::consts::FRAC_1_SQRT_2; 2]);

  // Equal magnitudes weight equally, so the result matches the plain mean over
  // the same directions.
  let unit: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let mean = MeanRenormalized
    .aggregate_values(&unit, &[1.0, 1.0], 2)
    .unwrap();
  assert_close_f64(&out, &mean);

  // And the weighting still ranks by magnitude: a 1e20 input against a 1e10 one
  // leans almost entirely toward the larger.
  let uneven: [&[f64]; 2] = [&[1e20, 0.0], &[0.0, 1e10]];
  let skewed = SaliencyWeighted
    .aggregate_values(&uneven, &[1.0, 1.0], 2)
    .unwrap();
  assert!(
    skewed[0] > 0.999_999 && skewed[1] > 0.0,
    "the 1e20 input must dominate without erasing the 1e10 one, got {skewed:?}"
  );
}

#[test]
fn saliency_ranks_by_magnitude_across_a_wide_range() {
  // The weights only ever matter by ratio, so a window whose norm reaches for the
  // range must not reorder them. A 3e38 window against a 1e-38 one spans 76
  // orders of magnitude — well within f64 and its saliency ceiling: the larger
  // has to dominate, and the smaller still has to survive as a non-zero
  // contribution.
  let embeddings: [&[f64]; 2] = [&[3e38, 0.0], &[0.0, 1e-38]];
  let out = SaliencyWeighted
    .aggregate_values(&embeddings, &[1.0, 1.0], 2)
    .unwrap();
  assert!(
    out[0] > 0.999_999 && out[1] >= 0.0,
    "the 3e38 window must dominate, got {out:?}"
  );

  // Reversing the magnitudes reverses which axis dominates, which is what makes
  // the assertion above about the weighting rather than about the order.
  let flipped: [&[f64]; 2] = [&[1e-38, 0.0], &[0.0, 3e38]];
  let out_flipped = SaliencyWeighted
    .aggregate_values(&flipped, &[1.0, 1.0], 2)
    .unwrap();
  assert!(
    out_flipped[1] > 0.999_999 && out_flipped[0] >= 0.0,
    "the 3e38 window must dominate, got {out_flipped:?}"
  );
}

#[test]
fn normal_magnitude_normalization_is_within_one_ulp_and_no_worse_than_naive() {
  // The scale-aware renorm replaced the bit-for-bit-with-naive-f32 guard, which
  // could not survive computing in f64: rounding once at f64 and narrowing is
  // deliberately *more* accurate than a naive f32 fold. What it still owes is a
  // bounded account against the naive f64 fold — the correctly-rounded direct
  // computation for an in-range vector. Dividing by a power of two is exact, so
  // the scale-aware result is within one ulp of that fold (in fact bit-identical
  // whenever the vector's own scale is an exact power of two, as here), and the
  // result is L2-unit to f64 precision.
  let raw = [0.6_f64, 0.8, 0.1, 3.0, -2.5];
  let mut naive_ss = 0.0_f64;
  for x in raw {
    naive_ss += x * x;
  }
  let naive_norm = libm::sqrt(naive_ss);
  let naive: Vec<f64> = raw.iter().map(|x| x / naive_norm).collect();

  let embeddings: [&[f64]; 1] = [&raw];
  let got = MeanRenormalized
    .aggregate_values(&embeddings, &[1.0], 5)
    .unwrap();
  // Never worse than naive: within one ulp of it, componentwise.
  for (g, n) in got.iter().zip(&naive) {
    assert!(
      (g - n).abs() <= f64::EPSILON * n.abs(),
      "scale-aware {g} is more than one ulp from the naive fold {n}"
    );
  }
  // Correctly rounded: the result is a unit vector to f64 precision.
  let out_norm = libm::sqrt(got.iter().map(|x| x * x).sum::<f64>());
  assert!(
    (out_norm - 1.0).abs() < 1e-15,
    "expected a unit vector, got norm {out_norm}"
  );

  // Coverage 1.0 widens to an exact 1.0 weight, so this must agree with the mean.
  let cov = CoverageWeightedMean
    .aggregate_values(&embeddings, &[1.0], 5)
    .unwrap();
  assert_eq!(cov, got);
}

/// Every permutation of `0..n`, so a cancellation fixture can be asserted
/// independently of the association order the fold happens to use (Heap's
/// algorithm).
fn permutations(n: usize) -> Vec<Vec<usize>> {
  fn recur(a: &mut [usize], k: usize, out: &mut Vec<Vec<usize>>) {
    if k == 1 {
      out.push(a.to_vec());
      return;
    }
    for i in 0..k {
      recur(a, k - 1, out);
      if k.is_multiple_of(2) {
        a.swap(i, k - 1);
      } else {
        a.swap(0, k - 1);
      }
    }
  }
  let mut a: Vec<usize> = (0..n).collect();
  let mut out = Vec::new();
  recur(&mut a, n, &mut out);
  out
}

#[test]
fn exact_cancellation_is_rejected() {
  // 94/512 + 2592/512 + 113/512 - 2799/512: 94 + 2592 + 113 - 2799 is zero, and
  // every value is a multiple of 2^-9 below 8, so the fold is exact in f64 and
  // lands on zero at all 24 association orders. A vector that cancels has no
  // direction, and the only honest answer is a rejection — not the small residue
  // an inexact rescale used to round it to.
  let raw = [
    94.0_f64 / 512.0,
    2592.0 / 512.0,
    113.0 / 512.0,
    -2799.0 / 512.0,
  ];
  let coverages = [1.0_f64; 4];
  for order in permutations(4) {
    let cols = [
      [raw[order[0]]],
      [raw[order[1]]],
      [raw[order[2]]],
      [raw[order[3]]],
    ];
    let embeddings: [&[f64]; 4] = [&cols[0], &cols[1], &cols[2], &cols[3]];
    let mean = MeanRenormalized.aggregate_values(&embeddings, &coverages, 1);
    assert!(
      matches!(mean, Err(WinditError::NonFinite)),
      "MeanRenormalized at order {order:?} must reject exact cancellation, got {mean:?}"
    );
    // Uniform coverage widens to an exactly-1.0 weight, so the coverage policy
    // folds the same values and must reach the same verdict.
    let cov = CoverageWeightedMean.aggregate_values(&embeddings, &coverages, 1);
    assert!(
      matches!(cov, Err(WinditError::NonFinite)),
      "CoverageWeightedMean at order {order:?} must reject exact cancellation, got {cov:?}"
    );
  }
}

/// The R4 counterexample: a sum that cancels only because tiny terms ride on a
/// huge one. `e` is the huge component and `d*e` the tiny term; the five window
/// values `e, -e, 48d*e, -32d*e, -16d*e` have an exact sum of zero (`e - e = 0`
/// and `48 - 32 - 16 = 0`). A naive fold destroys the second half by absorbing
/// `48d*e` into `e` and then subtracting `e` away, fabricating an
/// order-dependent residue that normalizes to a fictitious `[1]`.
///
/// The uniform-weight policies must reject the directionless vector under every
/// permutation. When the values are in-domain (`domain_rejected == false`) the
/// fold cannot fabricate a direction from the exact cancellation and returns
/// [`WinditError::NonFinite`]; when the huge component is out of the input
/// domain (`domain_rejected == true`) the input check rejects it first as
/// [`WinditError::MagnitudeOutOfRange`].
///
/// The cancellation is carried in the (f64) embedding values, not the (f32)
/// coverages: at the f64 scale the coverage weights `48d` would be far below
/// f32's subnormal floor, so only the compute-domain channel can express it.
fn assert_wide_spread_cancellation_rejected(e: f64, d: f64, domain_rejected: bool) {
  let de = d * e; // exact: the tiny term that rides on `e`
  let values = [e, -e, 48.0 * de, -32.0 * de, -16.0 * de];
  let coverages = [1.0_f64; 5];
  for order in permutations(5) {
    let cols: Vec<[f64; 1]> = order.iter().map(|&i| [values[i]]).collect();
    let embeddings: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    // Uniform weight at both policies (every coverage is 1.0), so each folds the
    // same five values and must reach the same verdict.
    for (name, got) in [
      (
        "MeanRenormalized",
        MeanRenormalized.aggregate_values(&embeddings, &coverages, 1),
      ),
      (
        "CoverageWeightedMean",
        CoverageWeightedMean.aggregate_values(&embeddings, &coverages, 1),
      ),
    ] {
      let rejected = if domain_rejected {
        matches!(got, Err(WinditError::MagnitudeOutOfRange { .. }))
      } else {
        matches!(got, Err(WinditError::NonFinite))
      };
      assert!(
        rejected,
        "{name} order {order:?} at e={e:e} d={d:e} must reject cancellation, got {got:?}"
      );
    }
  }
}

#[test]
fn wide_spread_cancellation_rejected_at_f32_scale() {
  // The R4 f32 counterexample, computed where f32 embeddings now compute: f64.
  // e = 2^127 and d = the smallest f32 subnormal (2^-149), so d*e = 2^-22. Even
  // here — a spread the old f32 fold could never survive — the tiny term is
  // absorbed into 2^127 under a naive f64 fold; only the compensated sum keeps
  // the cancellation exact.
  assert_wide_spread_cancellation_rejected(
    f64::from(f32::from_bits(0x7f00_0000)), // 2^127
    f64::from(f32::from_bits(1)),           // 2^-149
    false,                                  // in-domain: rejected as directionless
  );
}

#[test]
fn wide_spread_cancellation_rejected_at_f64_scale() {
  // REVOKED (settlement §4.5.6, deliberate): the huge component e = 2^1023 is now
  // outside the input domain [2^-400, 2^400], so this fixture is rejected before
  // any arithmetic with MagnitudeOutOfRange rather than by compensated summation.
  // The in-domain cancellation class stays covered by the f32-scale sibling.
  assert_wide_spread_cancellation_rejected(
    f64::from_bits(0x7fe0_0000_0000_0000), // 2^1023
    f64::from_bits(1),                     // 2^-1074
    true,                                  // out-of-domain: rejected by the input check
  );
}

#[test]
fn ema_subnormal_survives_at_f32_scale() {
  // REVOKED (settlement §4.5.6(b), R-A — expressly signed off by the Fable 5
  // adjudication): re-pinned to Err(NonFinite) for both window orders. Here
  // d = 2^-149 is in-domain, so the domain check passes it and the determinacy
  // gate does the rejecting.
  //
  // Justification: `alpha 0.5` over `[1, d]` then `[-1, d]` has the exact convex
  // sum `[0, d]`, but that `d` sits 2^-97 below the fold's provable error floor
  // (||R|| / (eps*||M||) = 2^-97). A one-ulp change to either unit component
  // moves the exact direction anywhere in the rejection band — [0,1], [+1,0],
  // and [-1,0] are all members (2^96 conditioning) — so no direction is
  // determined at working precision; the old [0,1] was a per-dimension-recurrence
  // artifact. The fixture's gate signature is moreover indistinguishable from a
  // real in-domain three-tier EMA fabrication whose exact direction is orthogonal
  // to [0,1], so keeping it green would require special-casing the fixture (the
  // actual gaming). This is the f32-scale face of the promise its own f64-scale
  // sibling already revokes. Genuine in-domain subnormal survival is pinned by
  // the cancellation-free companion below.
  let d = f64::from(f32::from_bits(1)); // 2^-149
  for windows in [[[1.0, d], [-1.0, d]], [[-1.0, d], [1.0, d]]] {
    let embeddings: [&[f64]; 2] = [&windows[0], &windows[1]];
    let got = EmaRenormalized::new(0.5).aggregate_values(&embeddings, &[1.0, 1.0], 2);
    assert!(
      matches!(got, Err(WinditError::NonFinite)),
      "an in-domain subnormal below the gate floor must be NonFinite, got {got:?}"
    );
  }
}

#[test]
fn ema_subnormal_survives_without_cancellation_at_f32_scale() {
  // The companion the R-A revocation adds, pinning the property the revoked
  // fixture actually owned: f64 keeps a halved f32-subnormal representable
  // (`0.5 * 2^-149 = 2^-150`, a normal f64, no flush), decoupled from the
  // cancellation rider. Windows `[d, 2d]` twice (d = 2^-149) fold without
  // cancellation to `[d, 2d]`, so `||R|| = ||M|| = sqrt(5)*d` clears the gate by
  // ~2^48 and the direction is `[1/sqrt(5), 2/sqrt(5)]`. A flushing fold would
  // return `[0, 1]` and fail this assertion; note `[3d, 4d]` would not exercise
  // the floor, since `0.5 * 3d = 1.5 * 2^-149` never dips below it.
  let d = f64::from(f32::from_bits(1)); // 2^-149
  let windows: [&[f64]; 2] = [&[d, 2.0 * d], &[d, 2.0 * d]];
  let out = EmaRenormalized::new(0.5)
    .aggregate_values(&windows, &[1.0, 1.0], 2)
    .unwrap();
  assert_close_f64(&out, &[0.447_213_595_499_957_9, 0.894_427_190_999_915_9]);
}

#[test]
fn exact_cancellation_across_three_tiers_is_gated() {
  // R5 F1: the Neumaier fold is exact per step, but its final fold-back leaves an
  // order-dependent residue when tiny terms ride on a magnitude-1 cancellation
  // across three exponent tiers. A = 2^-60 and d = 3*2^-120 are both in-domain,
  // so the domain check passes them; the six windows have an exact componentwise
  // sum of zero, yet the fold leaves ~3*2^-120 in dimension 0 that a naive renorm
  // would amplify to a fabricated [1, 0]. The determinacy gate rejects it
  // (||R|| ~ 2.3e-36 vs tau ~ 1.6e-14) under every one of the 720 window orders,
  // at both uniform-weight policies.
  let a = f64::from_bits(0x3c30_0000_0000_0000); // 2^-60
  let d = 3.0 * f64::from_bits(0x3870_0000_0000_0000); // 3 * 2^-120
  let windows = [
    [a, 1.0],
    [-d, 1.0],
    [-1.0, 0.0],
    [-a, -1.0],
    [d, -1.0],
    [1.0, 0.0],
  ];
  let coverages = [1.0_f64; 6];
  let mut orders = 0;
  for order in permutations(6) {
    orders += 1;
    let cols: Vec<[f64; 2]> = order.iter().map(|&i| windows[i]).collect();
    let embeddings: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    for (name, got) in [
      (
        "MeanRenormalized",
        MeanRenormalized.aggregate_values(&embeddings, &coverages, 2),
      ),
      (
        "CoverageWeightedMean",
        CoverageWeightedMean.aggregate_values(&embeddings, &coverages, 2),
      ),
    ] {
      assert!(
        matches!(got, Err(WinditError::NonFinite)),
        "{name} order {order:?} must gate the F1 fabrication, got {got:?}"
      );
    }
  }
  assert_eq!(orders, 720, "F1 must be checked at all 720 permutations");
}

#[test]
fn determinacy_gate_rejects_opposite_units_and_admits_a_real_residual() {
  // Two exactly-opposite unit windows sum to the zero vector; the gate reports
  // NonFinite (no direction) at both uniform-weight policies.
  let opposite: [&[f64]; 2] = [&[1.0, 0.0], &[-1.0, 0.0]];
  for (name, got) in [
    (
      "MeanRenormalized",
      MeanRenormalized.aggregate_values(&opposite, &[1.0, 1.0], 2),
    ),
    (
      "CoverageWeightedMean",
      CoverageWeightedMean.aggregate_values(&opposite, &[1.0, 1.0], 2),
    ),
  ] {
    assert!(
      matches!(got, Err(WinditError::NonFinite)),
      "{name} must gate exactly-opposite units, got {got:?}"
    );
  }

  // A genuine residual well above the gate floor is admitted: [1, 0] and
  // [-1, 1e-8] cancel in dimension 0 but leave 1e-8 in dimension 1 (~13 orders
  // above tau ~ 7e-15), so the honest aggregate direction [0, 1] is returned.
  let near: [&[f64]; 2] = [&[1.0, 0.0], &[-1.0, 1e-8]];
  let out = MeanRenormalized
    .aggregate_values(&near, &[1.0, 1.0], 2)
    .unwrap();
  assert_close_f64(&out, &[0.0, 1.0]);
}

#[test]
fn ema_subnormal_product_cancellation_family_a_is_gated() {
  // B1 family A (independent review counterexample, confirmed on the real crate):
  // EmaRenormalized(0.5) over n = 700 fabricated Ok([1, 0]) from an exactly
  // cancelling in-domain input before the determinacy threshold carried an
  // absolute floor. Windows 0..3 carry values ~2^-349 (all in [2^-400, 2^400]);
  // the recency weights w0 = w1 = 2^-699 and w2 = 2^-698 drive the products to
  // ~2^-1048 (subnormal), where the relative gate 16*eps*||M|| underflowed to 0.0
  // and the nonzero subnormal rounding residue slipped past. e0 = -(e1 + 2*e2) is
  // exact, so the weighted sum is exactly zero; MIN_GATE_THRESHOLD now gates it.
  let two_m40 = libm::ldexp(1.0, -40);
  let e1 = libm::ldexp(1.0 + 19600.0 * two_m40, -350);
  let e2 = libm::ldexp(1.0 + 9800.0 * two_m40, -350);
  let e0 = -(e1 + 2.0 * e2);
  let mut cols: Vec<[f64; 2]> = vec![[e0, 0.0], [e1, 0.0], [e2, 0.0]];
  cols.resize(700, [0.0, 0.0]);
  let embeddings: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
  let coverages = vec![1.0_f64; 700];
  let got = EmaRenormalized::new(0.5).aggregate_values(&embeddings, &coverages, 2);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "family A exact cancellation must be gated, got {got:?}"
  );
}

#[test]
fn ema_subnormal_product_cancellation_family_b_is_gated() {
  // B1 family B: alpha = 255/256 (a dyadic, exact in every binary float), n = 84
  // — an ordinary window count,
  // confirmed on the real crate to fabricate Ok([-1, 0]) before the floor. The
  // weights (w0 = 2^-664, w1 = 255*2^-664, w2 = 255*2^-656) push the products
  // subnormal; e0 = -255*(e1 + 256*e2) is exact, so the weighted sum is exactly
  // zero. Confirms the fix depends on neither a dyadic alpha nor a large n.
  let alpha = 255.0 / 256.0; // 0.996_093_75
  let two_m30 = libm::ldexp(1.0, -30);
  let e1 = libm::ldexp(1.0 + 7.0 * two_m30, -392);
  let e2 = libm::ldexp(1.0 + 9.0 * two_m30, -392);
  let e0 = -255.0 * (e1 + 256.0 * e2);
  let mut cols: Vec<[f64; 2]> = vec![[e0, 0.0], [e1, 0.0], [e2, 0.0]];
  cols.resize(84, [0.0, 0.0]);
  let embeddings: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
  let coverages = vec![1.0_f64; 84];
  let got = EmaRenormalized::new(alpha).aggregate_values(&embeddings, &coverages, 2);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "family B exact cancellation must be gated, got {got:?}"
  );
}

#[test]
fn ema_single_subnormal_product_term_is_gated() {
  // B1 family C (no cancellation): a single in-domain window whose tiny EMA weight
  // drives its product to a nonzero *subnormal* — falsifying the old claim that
  // "every nonzero intermediate is a normal f64" in domain. alpha = 0.5, n = 701,
  // so window 0's weight is w0 = 2^-700; its value e ~ 2^-370 is in domain, but
  // w0*e ~ 2^-1070 is subnormal, rounded absolutely rather than by 4*eps*||M||.
  // The old gate's 16*eps*||M|| underflowed to 0.0 and admitted the sub-precision
  // result; the floor now gates it (||acc|| ~ 2^-1070 <= MIN_GATE_THRESHOLD =
  // 2^-1000). The restated error bound 4*eps*||M|| + K_abs holds — |R - exact| <=
  // 2^-1075 <= K_abs — which the gate threshold subsumes.
  let e = (libm::ldexp(1.0, 53) - 1.0) * libm::ldexp(1.0, -423); // (2^53 - 1) * 2^-423 ~ 2^-370
  let mut cols: Vec<[f64; 2]> = vec![[e, 0.0]];
  cols.resize(701, [0.0, 0.0]);
  let embeddings: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
  let coverages = vec![1.0_f64; 701];
  let got = EmaRenormalized::new(0.5).aggregate_values(&embeddings, &coverages, 2);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "a single in-domain subnormal-product term must be gated, got {got:?}"
  );
}

#[test]
fn determinacy_gate_floor_boundary_is_monotone() {
  // The MIN_GATE_THRESHOLD floor's engagement boundary, in the normal-product
  // regime so it isolates the floor itself. alpha = 0.5, n = 606 puts window 0's
  // weight at w0 = 2^-605; window 0 = [v, 2v] (a genuine direction [1, 2]/sqrt5),
  // the other 605 windows zero, so the aggregated mass is ~ 2^-605 * ||[v, 2v]||:
  //  - v = 2^-390 (in domain): mass ~ 2^-994 > floor, and the real direction
  //    [1/sqrt5, 2/sqrt5] is returned (products 2^-995 / 2^-994 are normal);
  //  - v = 2^-400 (in domain, the input-domain floor): mass ~ 2^-1004 < floor, so
  //    NonFinite.
  // Both sides carry the same geometry, so crossing the floor is direction ->
  // NonFinite (monotone): never a flipped or fabricated direction, and the
  // sub-floor side is honestly "no direction at working precision". No f32 input
  // reaches this regime (605 zero recent windows beside one ancient window).
  let above = libm::ldexp(1.0, -390);
  let mut cols_above: Vec<[f64; 2]> = vec![[above, 2.0 * above]];
  cols_above.resize(606, [0.0, 0.0]);
  let e_above: Vec<&[f64]> = cols_above.iter().map(|c| c.as_slice()).collect();
  let out = EmaRenormalized::new(0.5)
    .aggregate_values(&e_above, &vec![1.0_f64; 606], 2)
    .unwrap();
  assert_close_f64(&out, &[0.447_213_595_499_957_9, 0.894_427_190_999_915_9]);

  let below = libm::ldexp(1.0, -400);
  let mut cols_below: Vec<[f64; 2]> = vec![[below, 2.0 * below]];
  cols_below.resize(606, [0.0, 0.0]);
  let e_below: Vec<&[f64]> = cols_below.iter().map(|c| c.as_slice()).collect();
  let got = EmaRenormalized::new(0.5).aggregate_values(&e_below, &vec![1.0_f64; 606], 2);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "sub-floor mass must be NonFinite, got {got:?}"
  );
}

#[test]
fn ema_subnormal_survives_at_f64_scale() {
  // REVOKED (settlement §4.5.6, deliberate): d = 2^-1073 is below the input
  // domain floor 2^-400, so this fixture is now rejected before any arithmetic
  // with MagnitudeOutOfRange rather than normalized to [0, 1]. The genuine
  // in-domain subnormal-survival property is pinned by a separate,
  // cancellation-free fixture.
  let d = f64::from_bits(2); // 2^-1073
  for windows in [[[1.0, d], [-1.0, d]], [[-1.0, d], [1.0, d]]] {
    let embeddings: [&[f64]; 2] = [&windows[0], &windows[1]];
    let got = EmaRenormalized::new(0.5).aggregate_values(&embeddings, &[1.0, 1.0], 2);
    assert!(
      matches!(got, Err(WinditError::MagnitudeOutOfRange { .. })),
      "out-of-domain EMA subnormal must be rejected, got {got:?}"
    );
  }
}

#[test]
fn domain_corners_are_accepted_at_the_boundary_and_rejected_beyond() {
  // The inclusive domain boundary: ±2^400 and ±2^-400 (Real::MAX_AGG_MAGNITUDE /
  // MIN_AGG_MAGNITUDE for f64) are accepted by all four policies and normalize to
  // the axis they point along; one binary order past either edge (2^401, 2^-401)
  // is rejected with MagnitudeOutOfRange carrying the offending window and
  // component.
  let max = f64::from_bits(0x58F0_0000_0000_0000); // 2^400
  let min = f64::from_bits(0x26F0_0000_0000_0000); // 2^-400
  for (value, want) in [
    (max, [1.0, 0.0]),
    (-max, [-1.0, 0.0]),
    (min, [1.0, 0.0]),
    (-min, [-1.0, 0.0]),
  ] {
    let embeddings: [&[f64]; 1] = [&[value, 0.0]];
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, &[1.0], 2)
        .unwrap_or_else(|e| panic!("{name} rejected boundary {value:e}: {e:?}"));
      assert_close_f64(&got, &want);
    }
  }

  // One binary order past the boundary, with the offending value at a known index
  // so the reported window/component can be pinned exactly. `2 * 2^400 = 2^401`
  // and `2^-400 / 2 = 2^-401` are both exact powers of two.
  for beyond in [2.0 * max, min / 2.0] {
    let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, beyond]];
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, &[1.0, 1.0], 2);
      assert!(
        matches!(
          got,
          Err(WinditError::MagnitudeOutOfRange {
            window: 1,
            component: 1
          })
        ),
        "{name} must reject {beyond:e} at window 1 component 1, got {got:?}"
      );
    }
  }
}

/// Where the determinacy gate's absolute floor still decides a
/// `CoverageWeightedMean` verdict — and where it was made to decide one it had
/// no business deciding.
///
/// REVOKED, and this note is the record. `0.3.0` widened the coverage channel to
/// `f64`, which widened the smallest positive coverage the input domain admits
/// from `2^-149` to `2^-1074`; the first round answered that by declaring the
/// policy a member of `EmaRenormalized`'s regime, so a fold whose whole mass sat
/// under [`MIN_GATE_THRESHOLD`](crate::scalar::Real::MIN_GATE_THRESHOLD) came
/// back `NonFinite`. That test asserted the wrong thing. `2^-1074` is not a
/// degraded `1.0`; it is `1.0` times a positive factor, and a normalized weighted
/// mean does not depend on that factor. The floor is an *absolute* bound
/// borrowed from a gate whose quantity is a norm in the embedding's own units,
/// and a weight has no units to measure it in.
///
/// So the first half below is the old fixture with its verdict corrected: the
/// smallest coverage `f64` can hold, on both windows, is a scale and resolves to
/// the same direction `1.0` does.
///
/// The floor is not thereby unreachable here — it is reached by an unbounded
/// *ratio* rather than by a scale, exactly as it is for EMA. After
/// normalization the largest weight is `1.0`, so a fold's mass falls under the
/// floor only when the windows carrying the largest coverages contribute no mass
/// of their own. The second half builds that: a zero-valued window at coverage
/// `1.0` beside a real one at `2^-1000`, whose entire accumulated mass is
/// `2^-1000` and so at or below the floor. That is a statement about the
/// embeddings, and scaling both coverages does not change it.
#[test]
fn a_coverage_scale_is_not_a_loss_of_precision_but_a_ratio_can_be() {
  let subnormal = f64::from_bits(1); // 2^-1074, unrepresentable as a nonzero f32
  assert!(subnormal.is_finite() && subnormal > 0.0 && subnormal < 1.0);
  assert_eq!(subnormal as f32, 0.0, "no f32 carries this coverage");

  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let at_subnormal = CoverageWeightedMean
    .aggregate_values(&embeddings, &[subnormal; 2], 2)
    .expect("a uniform coverage resolves whatever its scale");
  assert_close_f64(&at_subnormal, &[core::f64::consts::FRAC_1_SQRT_2; 2]);
  let at_one = CoverageWeightedMean
    .aggregate_values(&embeddings, &[1.0; 2], 2)
    .expect("the same fold at coverage 1.0");
  assert_eq!(
    at_subnormal, at_one,
    "the smallest positive coverage is a scale of the largest, not a degradation of it"
  );

  // The ratio regime. Window 0 carries the largest coverage and no mass; window 1
  // carries all the mass at a coverage `2^-1000` of it, so the fold accumulates
  // `2^-1000` against a floor of `2^-1000` and there is no direction at working
  // precision. Both windows are in domain (zero is, and `1.0` is).
  let ratio: [&[f64]; 2] = [&[0.0, 0.0], &[1.0, 0.0]];
  let tiny = libm::ldexp(1.0, -1000);
  let got = CoverageWeightedMean.aggregate_values(&ratio, &[1.0, tiny], 2);
  assert!(
    matches!(got, Err(WinditError::NonFinite)),
    "a fold whose whole mass sits at the floor has no direction, got {got:?}"
  );
  // And that verdict is itself scale-invariant: halving both coverages keeps the
  // ratio, and so keeps the verdict.
  let scaled = CoverageWeightedMean.aggregate_values(&ratio, &[0.5, tiny / 2.0], 2);
  assert!(
    matches!(scaled, Err(WinditError::NonFinite)),
    "the ratio regime must not depend on the scale either, got {scaled:?}"
  );

  // One window above it, to show the floor is a boundary and not a blanket: the
  // same geometry at a ratio of `2^-900` resolves to window 1's direction.
  let above = libm::ldexp(1.0, -900);
  let out = CoverageWeightedMean
    .aggregate_values(&ratio, &[1.0, above], 2)
    .expect("mass well above the floor must resolve");
  assert_close_f64(&out, &[1.0, 0.0]);
}

/// The coverage slice must be as long as the window sequence, at every policy.
///
/// The shared input check enforces it, so even a policy that never reads a
/// coverage rejects a mismatched one rather than folding a sequence it was
/// handed no geometry for. Both directions are checked: a short slice is the one
/// that would index out of bounds inside `CoverageWeightedMean`, and a long one
/// is the one the per-window zip would silently truncate.
#[test]
fn a_coverage_slice_that_does_not_match_the_windows_is_rejected() {
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  for (label, coverages) in [("short", &[1.0][..]), ("long", &[1.0, 1.0, 1.0][..])] {
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, coverages, 2);
      assert!(
        matches!(
          got,
          Err(WinditError::DimMismatch {
            got: g,
            expected: 2
          }) if g == coverages.len()
        ),
        "{name} must reject a {label} coverage slice, got {got:?}"
      );
    }
  }
}

#[test]
fn out_of_range_coverage_is_rejected() {
  // A coverage is a geometric fraction in [0, 1]; NaN, above 1, or below 0 is
  // rejected with CoverageOutOfRange before any fold, at every policy — the check
  // lives in the shared input path, so even the policies that ignore coverage
  // enforce its range.
  let embeddings: [&[f64]; 1] = [&[1.0, 0.0]];
  for bad in [f64::NAN, 1.5, -0.1] {
    for (name, run) in builtin_policies() {
      let got = run(&embeddings, &[bad], 2);
      assert!(
        matches!(got, Err(WinditError::CoverageOutOfRange { window: 0 })),
        "{name} must reject coverage {bad} at window 0, got {got:?}"
      );
    }
  }
}

#[test]
fn out_of_domain_saliency_coverage_and_ema_inputs_are_rejected() {
  // The R5 F2/F3 findings, now designed rejections: each fixture carries a
  // component outside the input domain, so it is rejected before the arithmetic
  // that used to overflow (F2a) or flush a subnormal (F2b, F3) could run.
  let m = f64::from_bits(1); // 2^-1074, below the domain floor

  // F2a: SaliencyWeighted squared 1e154 past f64's range; 1e154 is out of domain.
  let f2a: [&[f64]; 2] = [&[1e154, 0.0], &[1e154, 0.0]];
  assert!(matches!(
    SaliencyWeighted.aggregate_values(&f2a, &[1.0, 1.0], 2),
    Err(WinditError::MagnitudeOutOfRange { .. })
  ));

  // F2b: CoverageWeightedMean flushed 0.5 * m to zero; m is out of domain.
  let f2b: [&[f64]; 2] = [&[1.0, m], &[-1.0, m]];
  assert!(matches!(
    CoverageWeightedMean.aggregate_values(&f2b, &[0.5, 0.5], 2),
    Err(WinditError::MagnitudeOutOfRange { .. })
  ));

  // F3: EmaRenormalized halved m toward zero; m is out of domain.
  let f3: [&[f64]; 2] = [&[1.0, m], &[-1.0, m]];
  assert!(matches!(
    EmaRenormalized::new(0.5).aggregate_values(&f3, &[1.0, 1.0], 2),
    Err(WinditError::MagnitudeOutOfRange { .. })
  ));
}

#[test]
fn keep_separate_returns_all_windows() {
  let windows = vec![
    win(&[1.0, 0.0], 4, 4),
    win(&[0.0, 1.0], 4, 4),
    win(&[1.0, 1.0], 2, 4),
  ];
  let spans: Vec<Span> = windows.iter().map(|w| w.span).collect();
  let kept = keep_separate(windows);
  assert_eq!(kept.len(), 3);
  assert_eq!(kept.iter().map(|w| w.span).collect::<Vec<_>>(), spans);
}

#[test]
fn empty_windows_errors() {
  let windows: [WindowEmbedding<TestVec>; 0] = [];
  assert!(matches!(
    aggregate(&CoverageWeightedMean, &windows),
    Err(WinditError::Empty)
  ));
  // Two empty slices pin no compute scalar on their own, so the element type is
  // named here. This is the one place the generalization adds inference
  // friction, and it is confined to a fully-empty call.
  let empty: [&[f64]; 0] = [];
  assert!(matches!(
    CoverageWeightedMean.aggregate_values(&empty, &[], 2),
    Err(WinditError::Empty)
  ));
}

#[test]
fn dim_mismatch_errors() {
  let windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 0.0, 1.0], 4, 4)];
  assert!(matches!(
    aggregate(&CoverageWeightedMean, &windows),
    Err(WinditError::DimMismatch { .. })
  ));
}

#[cfg(feature = "serde")]
#[test]
fn kind_into_policy_matches_builtin() {
  let windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 1.0], 2, 4)];
  let via_kind = aggregate(
    AggregatePolicyKind::CoverageWeightedMean
      .into_policy()
      .as_ref(),
    &windows,
  )
  .unwrap();
  let via_builtin = aggregate(&CoverageWeightedMean, &windows).unwrap();
  assert_close(via_kind.as_slice(), via_builtin.as_slice());

  let windows3 = [
    win(&[1.0, 0.0, 0.0], 4, 4),
    win(&[0.0, 1.0, 0.0], 4, 4),
    win(&[0.0, 0.0, 1.0], 4, 4),
  ];
  let via_kind = aggregate(
    AggregatePolicyKind::Ema { alpha: 0.5 }
      .into_policy()
      .as_ref(),
    &windows3,
  )
  .unwrap();
  let via_builtin = aggregate(&EmaRenormalized::new(0.5), &windows3).unwrap();
  assert_close(via_kind.as_slice(), via_builtin.as_slice());

  // The selector is generic over the compute scalar. Both calls above infer `C`
  // from the embeddings (f64) with no turbofish; here nothing else pins it,
  // which is the one case that needs the annotation.
  // A coefficient no `f32` can hold, carried from the wire value all the way to
  // the fold. `into_policy` is the one place a configured `f64` still crosses
  // into the compute scalar (through `Real::from_f64`), so it is the one place
  // that crossing can silently narrow: at `1 - 2^-30` a narrowing widener
  // delivers exactly `1.0`, whose weights are `[0, 0, 1]`.
  //
  // Read through `aggregate_values` on raw `f64` slices rather than through
  // `aggregate`: `win` builds a `TestVec`, which stores `f32` and would narrow
  // the very difference this is measuring.
  let fine = 1.0 - libm::ldexp(1.0, -30);
  assert_eq!(fine as f32, 1.0, "no f32 is nearer to this alpha than 1.0");
  let raw: [&[f64]; 3] = [&[0.0, 1.0], &[0.0, 1.0], &[1.0, 0.0]];
  let cov = [1.0f64; 3];
  let via_kind = AggregatePolicyKind::Ema { alpha: fine }
    .into_policy()
    .aggregate_values(&raw, &cov, 2)
    .unwrap();
  let via_builtin = EmaRenormalized::new(fine)
    .aggregate_values(&raw, &cov, 2)
    .unwrap();
  assert_eq!(
    via_kind, via_builtin,
    "the wire value must reach the fold unrounded"
  );
  let collapsed = EmaRenormalized::new(1.0)
    .aggregate_values(&raw, &cov, 2)
    .unwrap();
  assert_ne!(
    via_kind, collapsed,
    "and it must not be the alpha an f32 field would have stored"
  );

  let p64 = AggregatePolicyKind::CoverageWeightedMean.into_policy::<f64>();
  let embeddings: [&[f64]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let out = p64.aggregate_values(&embeddings, &[1.0, 0.5], 2).unwrap();
  assert_close_f64(&out, &[0.894_427_190_999_915_9, 0.447_213_595_499_957_9]);
}

#[cfg(feature = "serde")]
#[test]
fn kind_serde_wire_format_is_pinned() {
  // The compute scalar became generic, but `AggregatePolicyKind` did not:
  // policy configuration stays f32 and is widened at use. These exact strings
  // are the guarantee that persisted configurations keep deserializing — a
  // renamed variant, a restructured payload, or an `alpha` widened to f64 would
  // all fail here.
  assert_eq!(
    serde_json::to_string(&AggregatePolicyKind::CoverageWeightedMean).unwrap(),
    r#""CoverageWeightedMean""#
  );
  assert_eq!(
    serde_json::to_string(&AggregatePolicyKind::MeanRenormalized).unwrap(),
    r#""MeanRenormalized""#
  );
  assert_eq!(
    serde_json::to_string(&AggregatePolicyKind::Ema { alpha: 0.25 }).unwrap(),
    r#"{"Ema":{"alpha":0.25}}"#
  );
  assert_eq!(
    serde_json::to_string(&AggregatePolicyKind::SaliencyWeighted).unwrap(),
    r#""SaliencyWeighted""#
  );

  // And the same strings decode back, so a config written by an older build is
  // still readable by this one.
  let decoded: Vec<AggregatePolicyKind> = [
    r#""CoverageWeightedMean""#,
    r#""MeanRenormalized""#,
    r#"{"Ema":{"alpha":0.25}}"#,
    r#""SaliencyWeighted""#,
  ]
  .iter()
  .map(|s| serde_json::from_str(s).unwrap())
  .collect();
  assert_eq!(
    decoded,
    vec![
      AggregatePolicyKind::CoverageWeightedMean,
      AggregatePolicyKind::MeanRenormalized,
      AggregatePolicyKind::Ema { alpha: 0.25 },
      AggregatePolicyKind::SaliencyWeighted,
    ]
  );
}

#[cfg(feature = "serde")]
#[test]
fn kind_serde_round_trip() {
  for kind in [
    AggregatePolicyKind::CoverageWeightedMean,
    AggregatePolicyKind::MeanRenormalized,
    AggregatePolicyKind::Ema { alpha: 0.25 },
    AggregatePolicyKind::SaliencyWeighted,
  ] {
    let json = serde_json::to_string(&kind).unwrap();
    let back: AggregatePolicyKind = serde_json::from_str(&json).unwrap();
    assert_eq!(kind, back);
  }

  // The Ema variant carries its `alpha` field through serialization intact.
  let json = serde_json::to_string(&AggregatePolicyKind::Ema { alpha: 0.75 }).unwrap();
  let back: AggregatePolicyKind = serde_json::from_str(&json).unwrap();
  assert!(matches!(back, AggregatePolicyKind::Ema { alpha } if alpha == 0.75));
}

// Shared quantized fixture: W = 3, D = 8, mixed-sign non-cancelling codes with
// distinct per-window (scale, zero_point) and coverages (1.0, 0.75, 0.5 at
// window 4), exercising asymmetric and per-window-scale dequantization together.
const CODES: [[i8; 8]; 3] = [
  [40, -12, 5, 60, -30, 18, -7, 25],
  [15, 50, -20, -8, 33, -45, 10, 22],
  [-25, 30, 12, -40, 8, 19, -33, 5],
];
const SCALES: [f64; 3] = [0.011, 0.0125, 0.02];
const ZPS: [i8; 3] = [0, -3, 5];
const LENS: [usize; 3] = [4, 3, 2];
const WINDOW: usize = 4;

/// Dequantize one window with the affine formula the design pins:
/// `scale * (q - zero_point)`, the exact `i16` subtraction widened to `f64`.
fn dequant(codes: &[i8], scale: f64, zp: i8) -> Vec<f64> {
  codes
    .iter()
    .map(|&q| scale * f64::from(i16::from(q) - i16::from(zp)))
    .collect()
}

/// The fixture as quantized `i8` embeddings (path Q): each carries its own scale
/// and zero point and dequantizes through the `compute_components` override.
fn quant_windows() -> Vec<WindowEmbedding<QuantEmb>> {
  (0..3)
    .map(|i| {
      Windowed::new(
        QuantEmb {
          codes: CODES[i].to_vec(),
          scale: SCALES[i],
          zero_point: ZPS[i],
          captured: Vec::new(),
        },
        Span::new(0, LENS[i], WINDOW),
      )
    })
    .collect()
}

/// The same fixture hand-dequantized into `f64` storage (path R): the reference
/// side, aggregated through the default zero-copy `f64` projection.
fn raw_windows() -> Vec<WindowEmbedding<RawF64Emb>> {
  (0..3)
    .map(|i| {
      Windowed::new(
        RawF64Emb {
          data: dequant(&CODES[i], SCALES[i], ZPS[i]),
          captured: Vec::new(),
        },
        Span::new(0, LENS[i], WINDOW),
      )
    })
    .collect()
}

/// The fixture dequantized through `f32` into `TestVec` storage (path R2): the
/// f32-precision reference, which agrees with the full-precision `i8` path only
/// to about f32 epsilon.
fn r2_windows() -> Vec<WindowEmbedding<TestVec>> {
  (0..3)
    .map(|i| {
      let data: Vec<f32> = CODES[i]
        .iter()
        .map(|&q| (SCALES[i] * f64::from(i16::from(q) - i16::from(ZPS[i]))) as f32)
        .collect();
      Windowed::new(TestVec(data), Span::new(0, LENS[i], WINDOW))
    })
    .collect()
}

/// Aggregate the quantized and hand-dequantized fixtures with `policy` and assert
/// the captured results are bitwise identical: the same f64 inputs traverse the
/// same deterministic pipeline to the same bits.
fn assert_bitwise_identical<P: AggregatePolicy<f64>>(policy: &P, name: &str) {
  let q = aggregate(policy, &quant_windows()).unwrap();
  let r = aggregate(policy, &raw_windows()).unwrap();
  assert_eq!(q.captured.len(), 8, "{name}: unexpected dim");
  assert_eq!(q.captured.len(), r.captured.len(), "{name}: length");
  for (a, b) in q.captured.iter().zip(&r.captured) {
    assert_eq!(
      a.to_bits(),
      b.to_bits(),
      "{name}: {a} vs {b} not bitwise-equal"
    );
  }
}

#[test]
fn quantized_projection_matches_hand_dequantized_f64_bitwise() {
  // The primary differential: aggregating quantized i8 windows (through their
  // compute_components override) must feed the deterministic pipeline exactly the
  // f64 values a hand-dequantized f64 aggregation feeds it, so the captured unit
  // vector is bitwise identical at every policy — compensated sum, fixed order,
  // determinacy gate, and input-domain check included. Any divergence is a
  // projection-path defect by construction.
  assert_bitwise_identical(&CoverageWeightedMean, "CoverageWeightedMean");
  assert_bitwise_identical(&MeanRenormalized, "MeanRenormalized");
  assert_bitwise_identical(&EmaRenormalized::new(0.3), "EmaRenormalized");
  assert_bitwise_identical(&SaliencyWeighted, "SaliencyWeighted");
}

#[test]
fn quantized_projection_tracks_f32_dequant_reference() {
  // The prompt's stated differential: the i8 projection against an f32-dequant
  // reference. R2 rounds each dequantized component to f32 and narrows the
  // aggregate to f32 storage, so it agrees with the full-precision i8 path only
  // to about f32 epsilon. On this well-conditioned fixture the unit directions
  // still align to within 1e-6.
  let q = aggregate(&CoverageWeightedMean, &quant_windows()).unwrap();
  let r2 = aggregate(&CoverageWeightedMean, &r2_windows()).unwrap();
  // Both are unit vectors, so their dot product is the direction cosine: q is
  // f64-unit, r2 is f32-unit widened back to f64.
  let dot: f64 = q
    .captured
    .iter()
    .zip(r2.as_slice())
    .map(|(a, b)| *a * f64::from(*b))
    .sum();
  assert!(dot >= 1.0 - 1e-6, "direction cosine {dot} below 1 - 1e-6");
}

#[test]
fn per_row_scale_feeds_true_magnitudes_to_saliency() {
  // Per-row scale: window A has a small scale (1e-3, ||A|| ~ 0.1), window B a
  // large one (1e-2, ||B|| ~ 1.0). Dequantization precedes weighting, so
  // SaliencyWeighted weighs the TRUE magnitudes and B dominates. Negative control
  // (stated, not run): the raw-code norms are near-equal (~100.04 vs ~100.02), so
  // a scale-blind fold would land near [0.707, .., 0.707, 0] — the r[0] < 0.2
  // bound kills that.
  let windows = vec![
    Windowed::new(
      QuantEmb {
        codes: vec![100, 3, 0, 0],
        scale: 1e-3,
        zero_point: 0,
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      QuantEmb {
        codes: vec![0, 2, 100, 0],
        scale: 1e-2,
        zero_point: 0,
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
  ];
  let sal = aggregate(&SaliencyWeighted, &windows).unwrap();
  let r = &sal.captured;
  assert!(
    r[2] > 0.99,
    "B's true magnitude must dominate saliency, got {r:?}"
  );
  assert!(
    r[0] < 0.2,
    "a scale-blind fold would put ~0.707 here, got {r:?}"
  );

  // Per-row scale breaks even the plain mean: B's 10x scale restores its ratio,
  // so this proves the point is about dequant-before-weight, not about saliency.
  let mean = aggregate(&MeanRenormalized, &windows).unwrap();
  let m = &mean.captured;
  assert!(
    m[2] > 9.0 * m[0],
    "the 10x scale ratio must survive the mean, got {m:?}"
  );
}

#[test]
fn asymmetric_zero_point_shifts_direction() {
  // A single asymmetric window (zp = -20): dequantization shifts every code by
  // +20 before scaling, so the true direction differs from the normalized raw
  // codes. Guards a future "optimization" that drops the zero point.
  let codes = vec![10i8, -5, 30, 0, -20, 15];
  let scale = 0.01;
  let zp = -20i8;
  let out = aggregate(
    &MeanRenormalized,
    &[Windowed::new(
      QuantEmb {
        codes: codes.clone(),
        scale,
        zero_point: zp,
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    )],
  )
  .unwrap();

  // The bitwise identity from the primary differential, on this asymmetric
  // window: the override must match hand-dequantization to the bit.
  let raw = aggregate(
    &MeanRenormalized,
    &[Windowed::new(
      RawF64Emb {
        data: dequant(&codes, scale, zp),
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    )],
  )
  .unwrap();
  for (a, b) in out.captured.iter().zip(&raw.captured) {
    assert_eq!(a.to_bits(), b.to_bits(), "override must match hand-dequant");
  }

  // The normalized RAW codes (zero point dropped) point a measurably different
  // way — at least 1e-2 apart in some component.
  let raw_norm: Vec<f64> = {
    let ss: f64 = codes.iter().map(|&q| f64::from(q) * f64::from(q)).sum();
    let n = libm::sqrt(ss);
    codes.iter().map(|&q| f64::from(q) / n).collect()
  };
  let mut max_diff = 0.0_f64;
  for (a, b) in out.captured.iter().zip(&raw_norm) {
    let d = libm::fabs(a - b);
    if d > max_diff {
      max_diff = d;
    }
  }
  assert!(
    max_diff > 1e-2,
    "dropping the zero point must shift the direction, max diff {max_diff}"
  );
}

#[test]
fn i8_without_projection_is_refused() {
  // The footgun guard: a bare i8 embedding with no compute_components override.
  // The default projection refuses to fold raw codes and returns
  // MissingDequantization, before any policy math. It is a monomorphization
  // constant, so it fires in every build profile and a plain #[test] pins it.
  let windows = vec![Windowed::new(
    BareI8Emb(vec![1, 2, 3, 4]),
    Span::new(0, 4, 4),
  )];
  assert!(matches!(
    aggregate(&MeanRenormalized, &windows),
    Err(WinditError::MissingDequantization)
  ));
  // Every policy refuses it identically — the guard is in the shared projection,
  // not in one policy.
  assert!(matches!(
    aggregate(&SaliencyWeighted, &windows),
    Err(WinditError::MissingDequantization)
  ));
}

#[test]
fn poisoned_or_out_of_domain_quant_params_fail_closed() {
  // Every hole in the quantization parameters fails closed. NaN/Inf scale poisons
  // components non-finite -> the input-domain check returns NonFinite; a zero
  // scale dequantizes to the all-zero (directionless) vector -> the determinacy
  // gate returns NonFinite; an absurd but finite scale drives a component past
  // 2^400 -> the genuine MagnitudeOutOfRange (settlement §5 domain arithmetic);
  // and a sane per-tensor scale lands ~390 binary orders inside the domain and
  // aggregates cleanly. No new validation code — the settlement catches it all.
  let span = Span::new(0, 4, 4);
  let mk = |codes: Vec<i8>, scale: f64, zp: i8| -> Vec<WindowEmbedding<QuantEmb>> {
    vec![Windowed::new(
      QuantEmb {
        codes,
        scale,
        zero_point: zp,
        captured: Vec::new(),
      },
      span,
    )]
  };
  assert!(matches!(
    aggregate(&MeanRenormalized, &mk(vec![10, -3, 5, 7], f64::NAN, 0)),
    Err(WinditError::NonFinite)
  ));
  assert!(matches!(
    aggregate(&MeanRenormalized, &mk(vec![10, -3, 5, 7], f64::INFINITY, 0)),
    Err(WinditError::NonFinite)
  ));
  assert!(matches!(
    aggregate(&MeanRenormalized, &mk(vec![10, -3, 5, 7], 0.0, 0)),
    Err(WinditError::NonFinite)
  ));
  assert!(matches!(
    aggregate(&MeanRenormalized, &mk(vec![100, 0, 0, 0], 1e200, 0)),
    Err(WinditError::MagnitudeOutOfRange {
      window: 0,
      component: 0
    })
  ));
  assert!(aggregate(&MeanRenormalized, &mk(vec![100, -50, 30, 7], 1.2e-3, 0)).is_ok());
}

#[cfg(feature = "half")]
#[test]
fn half_projection_matches_hand_widened_f64_bitwise() {
  // Part A composed with aggregation: f16/bf16 storage widened by the default
  // projection feeds the pipeline exactly what a hand-widened f64 aggregation
  // feeds it. Every finite f16/bf16 is exact in f64, so the captured result is
  // bitwise identical — the §4.1 differential pattern, in the half registers.
  use crate::{
    scalar::{bf16, f16},
    test_support::{Bf16Emb, HalfEmb},
  };

  // Small dyadic values, exact in both half formats and in f64.
  let w0 = [0.5f32, -0.25, 1.5, 0.75];
  let w1 = [-1.0f32, 0.125, 0.5, -2.0];

  // f16 storage vs its hand-widened f64 reference.
  let f16_windows = [
    Windowed::new(
      HalfEmb {
        data: w0.iter().map(|&x| f16::from_f32(x)).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      HalfEmb {
        data: w1.iter().map(|&x| f16::from_f32(x)).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 3, 4),
    ),
  ];
  let f16_reference = [
    Windowed::new(
      RawF64Emb {
        data: w0.iter().map(|&x| f64::from(f16::from_f32(x))).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      RawF64Emb {
        data: w1.iter().map(|&x| f64::from(f16::from_f32(x))).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 3, 4),
    ),
  ];
  let h = aggregate(&CoverageWeightedMean, &f16_windows).unwrap();
  let r = aggregate(&CoverageWeightedMean, &f16_reference).unwrap();
  assert_eq!(h.captured.len(), r.captured.len());
  for (a, b) in h.captured.iter().zip(&r.captured) {
    assert_eq!(
      a.to_bits(),
      b.to_bits(),
      "f16 projection must match hand-widened f64 bitwise"
    );
  }

  // bf16 storage vs its hand-widened f64 reference.
  let bf16_windows = [
    Windowed::new(
      Bf16Emb {
        data: w0.iter().map(|&x| bf16::from_f32(x)).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      Bf16Emb {
        data: w1.iter().map(|&x| bf16::from_f32(x)).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 3, 4),
    ),
  ];
  let bf16_reference = [
    Windowed::new(
      RawF64Emb {
        data: w0.iter().map(|&x| f64::from(bf16::from_f32(x))).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 4, 4),
    ),
    Windowed::new(
      RawF64Emb {
        data: w1.iter().map(|&x| f64::from(bf16::from_f32(x))).collect(),
        captured: Vec::new(),
      },
      Span::new(0, 3, 4),
    ),
  ];
  let hb = aggregate(&CoverageWeightedMean, &bf16_windows).unwrap();
  let rb = aggregate(&CoverageWeightedMean, &bf16_reference).unwrap();
  assert_eq!(hb.captured.len(), rb.captured.len());
  for (a, b) in hb.captured.iter().zip(&rb.captured) {
    assert_eq!(
      a.to_bits(),
      b.to_bits(),
      "bf16 projection must match hand-widened f64 bitwise"
    );
  }
}

#[cfg(feature = "half")]
#[test]
fn half_stored_non_finite_is_rejected() {
  // f16/bf16 storage composes with the input-domain check: a stored NaN or
  // infinity widens to a non-finite f64 and is rejected with NonFinite, exactly
  // as a poisoned quant scale is. Only stored non-finites can reject a half
  // embedding — its entire finite range sits inside the domain.
  use crate::{
    scalar::{bf16, f16},
    test_support::{Bf16Emb, HalfEmb},
  };

  let span = Span::new(0, 4, 4);
  for bad in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
    let windows = vec![Windowed::new(
      HalfEmb {
        data: vec![
          f16::from_f32(0.5),
          bad,
          f16::from_f32(-0.25),
          f16::from_f32(1.0),
        ],
        captured: Vec::new(),
      },
      span,
    )];
    assert!(matches!(
      aggregate(&MeanRenormalized, &windows),
      Err(WinditError::NonFinite)
    ));
  }
  for bad in [bf16::NAN, bf16::INFINITY, bf16::NEG_INFINITY] {
    let windows = vec![Windowed::new(
      Bf16Emb {
        data: vec![
          bf16::from_f32(0.5),
          bad,
          bf16::from_f32(-0.25),
          bf16::from_f32(1.0),
        ],
        captured: Vec::new(),
      },
      span,
    )];
    assert!(matches!(
      aggregate(&MeanRenormalized, &windows),
      Err(WinditError::NonFinite)
    ));
  }
}
