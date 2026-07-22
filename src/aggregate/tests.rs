use std::{vec, vec::Vec};

#[cfg(feature = "serde")]
use super::AggregatePolicyKind;
use super::{
  aggregate, keep_separate, AggregatePolicy, CoverageWeightedMean, EmaRenormalized,
  MeanRenormalized, SaliencyWeighted,
};
use crate::{
  plan::Span,
  scalar::TestQuant,
  test_support::{assert_close, assert_close_f64, TestQuantVec, TestVec},
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

#[test]
fn ema_alpha_range_is_rejected_at_f64() {
  // The alpha range check runs in f32 on the f32 configuration field, before
  // widening, so it must fire identically at the f64 compute scalar.
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
    EmaRenormalized::new(f32::NAN).aggregate_values(&embeddings, &coverages, 2),
    Err(WinditError::AlphaOutOfRange)
  ));
  assert!(EmaRenormalized::new(0.5)
    .aggregate_values(&embeddings, &coverages, 2)
    .is_ok());
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
    EmaRenormalized::new(f32::NAN).aggregate_values(&embeddings, &coverages, 2),
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
type PolicyRun = fn(&[&[f64]], &[f32], usize) -> Result<Vec<f64>, WinditError>;

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
  let coverages = [1.0_f32; 4];
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
  let coverages = [1.0_f32; 5];
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
  let coverages = [1.0_f32; 6];
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

#[test]
fn out_of_range_coverage_is_rejected() {
  // A coverage is a geometric fraction in [0, 1]; NaN, above 1, or below 0 is
  // rejected with CoverageOutOfRange before any fold, at every policy — the check
  // lives in the shared input path, so even the policies that ignore coverage
  // enforce its range.
  let embeddings: [&[f64]; 1] = [&[1.0, 0.0]];
  for bad in [f32::NAN, 1.5, -0.1] {
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
