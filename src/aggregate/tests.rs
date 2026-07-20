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
fn win(raw: &[f32], len: usize, window: usize) -> WindowEmbedding<TestVec> {
  Windowed::new(
    TestVec::from_unnormalized(raw).unwrap(),
    Span::new(0, len, window),
  )
}

#[test]
fn coverage_weighted_mean_pinned() {
  // cov 1.0 * [1,0] + cov 0.5 * [0,1] = [1, 0.5]; renorm by sqrt(1.25).
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
  // widening, so it must fire identically at a non-f32 compute scalar.
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
  // `TestQuantVec` stores i8 and computes in f32, so `as_compute_slice` returns
  // `None` and `aggregate` must widen elementwise instead of borrowing. The
  // pinned output also documents the requantization contract: the aggregate
  // unit vector [0.894_427_2, 0.447_213_6] scales to [113.59, 56.80], which
  // rounds half away from zero to [114, 57].
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

  // The same geometry through the f32 storage type agrees before quantization,
  // proving the widening branch is not a different computation.
  let f32_windows = [win(&[1.0, 0.0], 4, 4), win(&[0.0, 1.0], 2, 4)];
  let f32_out = aggregate(&CoverageWeightedMean, &f32_windows).unwrap();
  let requantized: Vec<TestQuant> = f32_out
    .as_slice()
    .iter()
    .map(|x| TestQuant(libm::roundf(x * 127.0) as i8))
    .collect();
  assert_eq!(out.as_slice(), requantized.as_slice());
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
  let embeddings: [&[f32]; 2] = [&[3.0, 0.0], &[0.0, 1.0]];
  let coverages = [1.0, 1.0];
  let sal = SaliencyWeighted
    .aggregate_values(&embeddings, &coverages, 2)
    .unwrap();
  assert_close(&sal, &[0.9938837, 0.1104315]);

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
  let fwd: [&[f32]; 3] = [&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
  let out = EmaRenormalized::new(0.5)
    .aggregate_values(&fwd, &coverages, 3)
    .unwrap();
  assert_close(&out, &[0.4082483, 0.4082483, 0.8164966]);

  let rev: [&[f32]; 3] = [&[0.0, 0.0, 1.0], &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]];
  let out_rev = EmaRenormalized::new(0.5)
    .aggregate_values(&rev, &coverages, 3)
    .unwrap();
  assert_close(&out_rev, &[0.8164966, 0.4082483, 0.4082483]);
}

#[test]
fn ema_renormalized_rejects_out_of_range_alpha() {
  let embeddings: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
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

#[test]
fn large_magnitude_and_underflowing_vectors_normalize_at_f32() {
  // Nothing about these vectors is invalid: every component is an ordinary
  // finite f32 and every normalized result is exactly [1, 0]. It was squaring
  // them that left the scalar -- 1e20 squares to infinity, 1e-30 to zero -- so a
  // norm taken as sqrt(sum(x*x)) reported NonFinite (or a zero norm) for input
  // the caller was entitled to have normalized.
  let coverages = [1.0];
  for big in [1e20_f32, 1e30, f32::MAX] {
    let embeddings: [&[f32]; 1] = [&[big, 0.0]];
    assert_eq!(
      MeanRenormalized
        .aggregate_values(&embeddings, &coverages, 2)
        .unwrap(),
      vec![1.0_f32, 0.0],
      "magnitude {big:e} must normalize, not be rejected"
    );
  }
  for tiny in [1e-30_f32, f32::MIN_POSITIVE, f32::from_bits(1)] {
    let embeddings: [&[f32]; 1] = [&[tiny, 0.0]];
    assert_eq!(
      MeanRenormalized
        .aggregate_values(&embeddings, &coverages, 2)
        .unwrap(),
      vec![1.0_f32, 0.0],
      "magnitude {tiny:e} must normalize, not be rejected"
    );
  }

  // A vector whose norm is not representable even though its normalization is:
  // sqrt(2) * 3e38 overflows f32, but the unit vector is the ordinary diagonal.
  let embeddings: [&[f32]; 1] = [&[3e38, 3e38]];
  assert_close(
    &MeanRenormalized
      .aggregate_values(&embeddings, &coverages, 2)
      .unwrap(),
    &[0.70710677, 0.70710677],
  );

  // A genuinely unnormalizable vector is still rejected, by the same path.
  let zero: [&[f32]; 1] = [&[0.0, 0.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&zero, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
  let nan: [&[f32]; 1] = [&[f32::NAN, 1.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&nan, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
  let inf: [&[f32]; 1] = [&[f32::INFINITY, 1.0]];
  assert!(matches!(
    MeanRenormalized.aggregate_values(&inf, &coverages, 2),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn large_magnitude_and_underflowing_vectors_normalize_at_f64() {
  // The same defect at the other shipped scalar. 1e20 and 1e-30 square
  // comfortably inside f64 and were never affected there, which is the point of
  // pinning them alongside the exponents that do reach it (~1e200).
  let coverages = [1.0];
  for big in [1e200_f64, f64::MAX] {
    let embeddings: [&[f64]; 1] = [&[big, 0.0]];
    assert_eq!(
      MeanRenormalized
        .aggregate_values(&embeddings, &coverages, 2)
        .unwrap(),
      vec![1.0_f64, 0.0],
      "magnitude {big:e} must normalize, not be rejected"
    );
  }
  for tiny in [1e-200_f64, f64::MIN_POSITIVE, f64::from_bits(1)] {
    let embeddings: [&[f64]; 1] = [&[tiny, 0.0]];
    assert_eq!(
      MeanRenormalized
        .aggregate_values(&embeddings, &coverages, 2)
        .unwrap(),
      vec![1.0_f64, 0.0],
      "magnitude {tiny:e} must normalize, not be rejected"
    );
  }
  for in_range in [1e20_f64, 1e-30_f64] {
    let embeddings: [&[f64]; 1] = [&[in_range, 0.0]];
    assert_eq!(
      MeanRenormalized
        .aggregate_values(&embeddings, &coverages, 2)
        .unwrap(),
      vec![1.0_f64, 0.0]
    );
  }
}

#[test]
fn saliency_magnitude_weights_survive_their_own_scale() {
  // SaliencyWeighted weights each input by its own L2 norm, so a 1e20 input
  // contributes 1e20 * 1e20 = 1e40 to the accumulator -- infinity in f32 -- even
  // once the norm itself is computed safely. Only the accumulator's direction
  // survives renormalization, so it is rescaled rather than surrendered.
  let embeddings: [&[f32]; 2] = [&[1e20, 0.0], &[0.0, 1e20]];
  let out = SaliencyWeighted
    .aggregate_values(&embeddings, &[1.0, 1.0], 2)
    .unwrap();
  assert_close(&out, &[0.70710677, 0.70710677]);

  // Equal magnitudes weight equally, so the result matches the plain mean over
  // the same directions -- the saliency is doing nothing here except overflow.
  let unit: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 1.0]];
  let mean = MeanRenormalized
    .aggregate_values(&unit, &[1.0, 1.0], 2)
    .unwrap();
  assert_close(&out, &mean);

  // And the weighting still ranks by magnitude once rescaled: a 1e20 input
  // against a 1e10 one leans almost entirely toward the larger.
  let uneven: [&[f32]; 2] = [&[1e20, 0.0], &[0.0, 1e10]];
  let skewed = SaliencyWeighted
    .aggregate_values(&uneven, &[1.0, 1.0], 2)
    .unwrap();
  assert!(
    skewed[0] > 0.999_999 && skewed[1] > 0.0,
    "the 1e20 input must dominate without erasing the 1e10 one, got {skewed:?}"
  );
}

#[test]
fn normal_magnitude_normalization_stays_bit_identical() {
  // The scale-aware path is a fallback, not a replacement: whenever the direct
  // sum of squares lands in range it is still the answer, so every pinned f32
  // value keeps its exact bits. Asserted against the naive computation with
  // `assert_eq!` rather than a tolerance, since a tolerance is precisely what
  // would hide a changed rounding.
  let raw = [0.6_f32, 0.8, 0.1, 3.0, -2.5];
  let mut naive = 0.0_f32;
  for x in raw {
    naive += x * x;
  }
  let norm = libm::sqrtf(naive);
  let want: Vec<f32> = raw.iter().map(|x| x / norm).collect();

  let embeddings: [&[f32]; 1] = [&raw];
  assert_eq!(
    MeanRenormalized
      .aggregate_values(&embeddings, &[1.0], 5)
      .unwrap(),
    want
  );
  // Coverage 1.0 widens to an exact 1.0 weight, so this must agree to the bit.
  assert_eq!(
    CoverageWeightedMean
      .aggregate_values(&embeddings, &[1.0], 5)
      .unwrap(),
    want
  );

  // Saliency scales the accumulator by the vector's own norm before
  // renormalizing, so its direct path is reproduced with that weight in place
  // rather than compared to the unweighted one.
  let weighted: Vec<f32> = raw.iter().map(|x| norm * x).collect();
  let mut naive_weighted = 0.0_f32;
  for x in &weighted {
    naive_weighted += x * x;
  }
  let weighted_norm = libm::sqrtf(naive_weighted);
  let want_saliency: Vec<f32> = weighted.iter().map(|x| x / weighted_norm).collect();
  assert_eq!(
    SaliencyWeighted
      .aggregate_values(&embeddings, &[1.0], 5)
      .unwrap(),
    want_saliency
  );
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
  let empty: [&[f32]; 0] = [];
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

  // The selector is generic over the compute scalar too. Both calls above infer
  // `C` from the embeddings with no turbofish; here nothing else pins it, which
  // is the one case that needs the annotation.
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
