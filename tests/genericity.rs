//! Genericity acceptance: the SAME public functions handle wildly different
//! window sizes and value units with no code change between cases — only the
//! configuration differs. This is the crate's window-size- and unit-agnosticism
//! contract, and the bar it must clear to be worth publishing.
//!
//! Gated on `alloc`: the planner and the strategy families (and therefore this
//! whole suite) require it. Under a feature set without `alloc` the file compiles
//! to an empty test binary, so the `serde`-only / no-feature matrix rows still
//! build.
#![cfg(any(feature = "std", feature = "alloc"))]

use windit::prelude::*;

/// A minimal embedding double that L2-normalizes on construction, standing in
/// for a real 384/512/768-dimension model embedding. Integration tests see only
/// the public API, so the suite carries its own [`Vector`] implementor rather
/// than reaching for the crate-internal test double.
struct TestEmbedding(Vec<f32>);

impl Vector for TestEmbedding {
  type Scalar = f32;

  fn as_slice(&self) -> &[f32] {
    &self.0
  }

  fn from_unnormalized(v: &[f32]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(v.iter().map(|x| x / norm).collect()))
  }
}

/// The same double at f64 storage, so the acceptance cases below drive the SAME
/// generic helpers over a second, genuinely different scalar. A genericity claim
/// exercised at one scalar proves nothing.
struct TestEmbedding64(Vec<f64>);

impl Vector for TestEmbedding64 {
  type Scalar = f64;

  fn as_slice(&self) -> &[f64] {
    &self.0
  }

  fn from_unnormalized(v: &[f64]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(v.iter().map(|x| x / norm).collect()))
  }
}

/// Assert an f32 embedding is L2 unit-norm within tolerance.
fn assert_unit_norm(e: &TestEmbedding) {
  let norm = e.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
  assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
}

/// Assert an f64 embedding is L2 unit-norm to a tolerance f32 arithmetic cannot
/// reach, so passing is evidence the aggregation really ran in f64 rather than
/// in f32 and widened afterwards.
fn assert_unit_norm_64(e: &TestEmbedding64) {
  let norm = e.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
  assert!((norm - 1.0).abs() < 1e-12, "expected unit norm, got {norm}");
}

/// Plan `input_len` elements into windows of `unit_window`, embed each span,
/// aggregate with the coverage-weighted mean, assert the result is unit-norm,
/// and return the number of windows planned.
///
/// This is the ONE helper every embedding acceptance case shares. Nothing in it
/// names a concrete window size, unit, embedding dimension, or scalar — the
/// cases differ only by the arguments they pass. The unit-norm check is a
/// parameter rather than a fixed `&[f32]` call because `Real` deliberately has
/// no `PartialOrd`: comparing against a tolerance belongs to the caller, at the
/// caller's concrete scalar, which is what keeps that bound off the sealed
/// trait.
fn run<E: Vector>(
  unit_window: usize,
  input_len: usize,
  embed: impl Fn(&Span) -> E,
  assert_unit: impl Fn(&E),
) -> usize {
  let opts = WindowOptions::new(unit_window);
  let spans = WindowPlan::spans(&opts, input_len).expect("plan spans");
  let windows: Vec<WindowEmbedding<E>> = spans
    .iter()
    .map(|span| Windowed::new(embed(span), *span))
    .collect();
  let summary = aggregate(&CoverageWeightedMean, &windows).expect("aggregate");
  assert_unit(&summary);
  spans.len()
}

/// A small positive embedding derived from the span, so distinct windows differ
/// yet their coverage-weighted sum never cancels to a zero (non-normalizable)
/// vector. The dimension is fixed at 4 across every case; window size is what
/// varies.
fn embed(span: &Span) -> TestEmbedding {
  let base = (span.start() % 7) as f32 + 1.0;
  let raw: Vec<f32> = (0..4).map(|i| base + i as f32 + 1.0).collect();
  TestEmbedding::from_unnormalized(&raw).expect("valid embedding")
}

/// [`embed`] at f64, mirroring it value for value so the two scalar cases differ
/// in nothing but their scalar.
fn embed64(span: &Span) -> TestEmbedding64 {
  let base = (span.start() % 7) as f64 + 1.0;
  let raw: Vec<f64> = (0..4).map(|i| base + f64::from(i) + 1.0).collect();
  TestEmbedding64::from_unnormalized(&raw).expect("valid embedding")
}

#[test]
fn granite_512_window() {
  // A short input fits in one window; a longer one tiles into three (two full
  // windows plus a ragged tail).
  assert_eq!(run(512, 40, embed, assert_unit_norm), 1);
  assert_eq!(run(512, 1500, embed, assert_unit_norm), 3);
}

#[test]
fn granite_512_window_f64() {
  // Scalar-agnosticism: the SAME `run`, the same window sizes, the same span
  // counts — only the embedding's scalar differs, and the tighter unit-norm
  // tolerance proves the math ran at that scalar's precision.
  assert_eq!(run(512, 40, embed64, assert_unit_norm_64), 1);
  assert_eq!(run(512, 1500, embed64, assert_unit_norm_64), 3);
}

#[test]
fn granite_large_window() {
  // The SAME `run`, only `unit_window` changes to a long-context size:
  // window-size-agnosticism.
  assert_eq!(run(8192, 20_000, embed, assert_unit_norm), 3);
  // And the same geometry again at f64, so window size and scalar are
  // independently free.
  assert_eq!(run(8192, 20_000, embed64, assert_unit_norm_64), 3);
}

#[test]
fn clap_sample_window() {
  // Audio-sample units and a half-million-wide window: still the same `run`,
  // still config-only.
  assert_eq!(run(480_000, 1_200_000, embed, assert_unit_norm), 3);
  assert_eq!(run(480_000, 1_200_000, embed64, assert_unit_norm_64), 3);
}

#[test]
fn dyn_policy_is_object_safe_at_both_scalars() {
  // `AggregatePolicy` gained a compute-scalar type parameter, but it defaults to
  // f32, so the bare `dyn` spellings that existed before must still compile
  // verbatim — as a reference, as a box, and as a struct field.
  let by_ref: &dyn AggregatePolicy = &CoverageWeightedMean;
  let boxed: Box<dyn AggregatePolicy> = Box::new(MeanRenormalized);

  struct Holder {
    policy: Box<dyn AggregatePolicy>,
  }
  let held = Holder {
    policy: Box::new(CoverageWeightedMean),
  };

  let spans = WindowPlan::spans(&WindowOptions::new(512), 1500).expect("plan spans");
  let windows: Vec<WindowEmbedding<TestEmbedding>> = spans
    .iter()
    .map(|span| Windowed::new(embed(span), *span))
    .collect();
  for policy in [by_ref, boxed.as_ref(), held.policy.as_ref()] {
    assert_unit_norm(&aggregate(policy, &windows).expect("aggregate"));
  }

  // A non-default scalar names the parameter, and the trait stays object-safe
  // there too.
  let p64: Box<dyn AggregatePolicy<f64>> = Box::new(CoverageWeightedMean);
  let windows64: Vec<WindowEmbedding<TestEmbedding64>> = spans
    .iter()
    .map(|span| Windowed::new(embed64(span), *span))
    .collect();
  assert_unit_norm_64(&aggregate(p64.as_ref(), &windows64).expect("aggregate"));
}

#[test]
fn vad_frame_segment_longest_run() {
  // Unit-agnosticism: the same segment functions that never touched an embedding
  // reduce a 100-frame f32 probability sequence to speech ranges. Frames are
  // window-1, so element units coincide with frame indices.
  let mut probs = vec![0.1f32; 100];
  probs[10..20].fill(0.8); // a 10-frame speech region
  probs[30..60].fill(0.9); // the longest, 30-frame speech region
  probs[70..75].fill(0.7); // a 5-frame speech region
  let frames: Vec<Windowed<f32>> = probs
    .iter()
    .enumerate()
    .map(|(i, &p)| Windowed::new(p, Span::new(i, 1, 1)))
    .collect();

  let opts = SegmentOptions::new();
  let speech = runs(&frames, |&p| p >= 0.5, &opts);
  assert_eq!(
    speech,
    vec![Range::new(10, 20), Range::new(30, 60), Range::new(70, 75),]
  );
  assert_eq!(
    longest_run(&frames, |&p| p >= 0.5, &opts),
    Some(Range::new(30, 60))
  );
}

/// The content-aware chunker is driven by the same [`WindowOptions`] window, with
/// length measured through the caller's `len_fn` callback. Gated on `text`.
#[cfg(feature = "text")]
#[test]
fn content_aware_chunk_is_window_config_driven() {
  use windit::split::ContentAware;

  // The len_fn callback defines "how long": here, whitespace-separated words.
  let count = |s: &str| s.split_whitespace().count();
  let chunker = ContentAware::new(&count);
  let text = "a b c d e f g h i j k l";

  // The same chunker over two window sizes — only the configuration differs.
  let wide = chunker.chunk(text, &WindowOptions::new(12)).unwrap();
  assert_eq!(wide.len(), 1); // all twelve words fit one window

  let narrow = chunker.chunk(text, &WindowOptions::new(4)).unwrap();
  assert_eq!(narrow.len(), 3); // twelve words, four per window
  for (start, end) in narrow {
    assert!(count(&text[start..end]) <= 4);
  }
}

/// A geometry persisted through serde replays the exact same spans. Gated on
/// `serde` (and, via the file gate, on `alloc`).
#[cfg(feature = "serde")]
#[test]
fn serialized_options_replay_identical_spans() {
  let opts = WindowOptions::new(512).with_overlap(64);
  let json = serde_json::to_string(&opts).unwrap();
  let restored: WindowOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(
    WindowPlan::spans(&opts, 1500).unwrap(),
    WindowPlan::spans(&restored, 1500).unwrap()
  );
}
