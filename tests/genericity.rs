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

use std::{cell::Cell, rc::Rc};

use windit::prelude::*;

/// A minimal embedding double that L2-normalizes on construction, standing in
/// for a real 384/512/768-dimension model embedding. Integration tests see only
/// the public API, so the suite carries its own [`Vector`] implementor rather
/// than reaching for the crate-internal test double.
///
/// It stores `f32` but computes in `f64` (`f32`'s compute type), so
/// `from_unnormalized` takes `&[f64]` and narrows into storage.
///
/// `Clone` because the batch `SmoothPolicy::smooth` convenience clones each
/// window at its method bound.
#[derive(Clone)]
struct TestEmbedding(Vec<f32>);

impl Vector for TestEmbedding {
  type Scalar = f32;

  fn as_slice(&self) -> &[f32] {
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
    Ok(Self(v.iter().map(|x| (x / norm) as f32).collect()))
  }
}

/// The same double at f64 storage, so the acceptance cases below drive the SAME
/// generic helpers over a second, genuinely different scalar. A genericity claim
/// exercised at one scalar proves nothing.
#[derive(Clone)]
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
  let base = (span.start() % 7) as f64 + 1.0;
  let raw: Vec<f64> = (0..4).map(|i| base + f64::from(i) + 1.0).collect();
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
fn f32_widening_path_matches_f64_fast_path_by_value() {
  // `assert_unit_norm`/`assert_unit_norm_64` are shape-only: any finite,
  // non-zero, dimension-4 output passes them, so neither can tell "the
  // widening branch computed the right f64 values" from "the widening branch
  // computed some other finite, dimension-preserving values" (e.g. a
  // wrong-but-uniform transform). `embed`/`embed64` compute the identical
  // logical input from one shared `spans` geometry, so this pins the two
  // `aggregate()` branches — f32 storage (the widening path) and f64 storage
  // (the zero-copy fast path) — to the SAME direction by value, not just by
  // shape.
  let opts = WindowOptions::new(512);
  let spans = WindowPlan::spans(&opts, 1500).expect("plan spans");

  let windows: Vec<WindowEmbedding<TestEmbedding>> = spans
    .iter()
    .map(|span| Windowed::new(embed(span), *span))
    .collect();
  let summary = aggregate(&CoverageWeightedMean, &windows).expect("aggregate");

  let windows64: Vec<WindowEmbedding<TestEmbedding64>> = spans
    .iter()
    .map(|span| Windowed::new(embed64(span), *span))
    .collect();
  let summary64 = aggregate(&CoverageWeightedMean, &windows64).expect("aggregate");

  // The f32 path narrows to f32 at every window (`from_unnormalized`) and
  // again at the aggregate's own reconstruction, so the comparison tolerance
  // is f32 narrowing precision (~1e-6, matching the crate's own
  // `assert_close`), not f64's ~1e-12.
  let widened: Vec<f64> = summary.as_slice().iter().map(|&x| f64::from(x)).collect();
  let want64 = summary64.as_slice();
  assert_eq!(widened.len(), want64.len());
  for (got, want) in widened.iter().zip(want64) {
    assert!(
      (got - want).abs() < 1e-6,
      "f32 widening path diverged from f64 fast path in value: {widened:?} vs {want64:?}"
    );
  }
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
  // `AggregatePolicy` has a compute-scalar type parameter defaulting to `f64` —
  // the domain both shipped scalars compute in — so the bare `dyn` spellings
  // still compile verbatim, and now aggregate the ordinary `f32`-stored
  // embedding (which computes in `f64`) directly, as a reference, a box, and a
  // struct field.
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

  // Naming the scalar explicitly (here `f64`, the default) compiles to the same
  // object and drives an f64-stored embedding, so the trait stays object-safe
  // whether the parameter is spelled or left off.
  let p64: Box<dyn AggregatePolicy<f64>> = Box::new(CoverageWeightedMean);
  let windows64: Vec<WindowEmbedding<TestEmbedding64>> = spans
    .iter()
    .map(|span| Windowed::new(embed64(span), *span))
    .collect();
  assert_unit_norm_64(&aggregate(p64.as_ref(), &windows64).expect("aggregate"));
}

#[test]
fn dyn_smoother_and_gate_are_object_safe() {
  // The streaming state traits are object-safe (no generic methods, no `Self` by
  // value), so a run-time-selected smoother or gate is a boxed trait object — the
  // manifest-driven selection path, mirroring `dyn AggregatePolicy`.
  let mut ema: Box<dyn Smoother<f32>> = Box::new(Ema::new(0.5).smoother());
  // s_0 = x_0 = 1.0.
  assert_eq!(
    ema
      .push(Windowed::new(1.0, Span::new(0, 1, 1)))
      .unwrap()
      .value(),
    &1.0
  );
  ema.reset();
  assert_eq!(
    ema
      .push(Windowed::new(0.4, Span::new(0, 1, 1)))
      .unwrap()
      .value(),
    &0.4
  );

  // `Identity` is generic over `V`, so name the value type when boxing its
  // state — the same disambiguation any `dyn` selection over a generic config uses.
  let mut ident: Box<dyn Smoother<f32>> = Box::new(SmoothPolicy::<f32>::smoother(&Identity::new()));
  let passed = ident.push(Windowed::new(0.25, Span::new(0, 1, 1))).unwrap();
  assert_eq!(passed.value(), &0.25);

  let mut thr: Box<dyn Gate<f32>> = Box::new(Threshold::new(0.5).gate());
  assert!(thr.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(!thr.push(&Windowed::new(0.1, Span::new(1, 1, 1))).unwrap());

  let mut hy: Box<dyn Gate<f32>> = Box::new(Hysteresis::new(0.6, 0.3).gate());
  assert!(!hy.push(&Windowed::new(0.4, Span::new(0, 1, 1))).unwrap()); // starts off
  assert!(hy.push(&Windowed::new(0.7, Span::new(1, 1, 1))).unwrap()); // latches on
  assert!(hy.push(&Windowed::new(0.5, Span::new(2, 1, 1))).unwrap()); // holds
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
  let speech = runs(&frames, |&p| p >= 0.5, &opts).unwrap();
  assert_eq!(
    speech,
    vec![Range::new(10, 20), Range::new(30, 60), Range::new(70, 75),]
  );
  assert_eq!(
    longest_run(&frames, |&p| p >= 0.5, &opts).unwrap(),
    Some(Range::new(30, 60))
  );
}

/// The content-aware chunker is driven by the same [`WindowOptions`] window, with
/// length measured through the caller's `MeasureText`. Gated on `text`.
#[cfg(feature = "text")]
#[test]
fn content_aware_chunk_is_window_config_driven() {
  use windit::split::ContentAware;

  // The MeasureText defines "how long": here, whitespace-separated words.
  let count = |s: &str| s.split_whitespace().count();
  let chunker = ContentAware::new(&count);
  let text = "a b c d e f g h i j k l";

  // The same chunker over two window sizes — only the configuration differs.
  let wide = chunker.chunk(text, &WindowOptions::new(12)).unwrap();
  assert_eq!(wide.len(), 1); // all twelve words fit one window

  let narrow = chunker.chunk(text, &WindowOptions::new(4)).unwrap();
  assert_eq!(narrow.len(), 3); // twelve words, four per window
  for chunk in &narrow {
    assert!(count(chunk.as_str(text).unwrap()) <= 4);
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

// ── P3: dyn manifest path, discontinuity forwarding, and non-f32 genericity ──

/// Drive a decoder to exhaustion, returning its `(causal flags, finalized
/// ranges)`. Takes the decoder by value because [`Decoder::finish`] consumes it.
fn drive_decoder<S: Smoother<f32>, G: Gate<f32>>(
  mut dec: Decoder<S, G, f32>,
  seq: &[Windowed<f32>],
) -> (Vec<bool>, Vec<Range>) {
  let mut flags = Vec::new();
  let mut ranges = Vec::new();
  for w in seq {
    let step = dec.push(*w).expect("push");
    flags.push(step.active());
    ranges.extend(step.finalized());
  }
  ranges.extend(dec.finish());
  (flags, ranges)
}

/// Drive a value-free gate policy through a decoder over ANY value type, using
/// the pass-through `Identity` smoother, and collect the causal flags. The
/// `V`-generic signature is the point: the same body runs for `f32` and for a
/// non-`f32`, non-`Clone` payload.
fn decode_flags<V, CG: GatePolicy<V>>(
  cg: &CG,
  opts: SegmentOptions,
  seq: Vec<Windowed<V>>,
) -> Vec<bool> {
  // `Identity` is a value-free smoother (`Smoother<V>` for every `V`), so the
  // decoder threads `V` through without cloning or formatting it.
  let mut dec = Decoder::new(
    SmoothPolicy::<V>::smoother(&Identity::new()),
    cg.gate(),
    opts,
  );
  seq
    .into_iter()
    .map(|w| dec.push(w).expect("push").active())
    .collect()
}

#[test]
fn dyn_p3_smoothers_and_gates_are_object_safe() {
  // The P3 stages behind boxed trait objects — the run-time-selected manifest
  // path, extending the P2 dyn suite to CadenceEma, Vote, and the combinators.

  // CadenceEma's element-time-constant smoother; the first push seeds `s_0 = x_0`.
  let mut ce: Box<dyn Smoother<f32>> = Box::new(CadenceEma::new(8.0).smoother());
  assert_eq!(
    ce.push(Windowed::new(1.0, Span::new(0, 1, 1)))
      .unwrap()
      .value(),
    &1.0
  );

  // Vote: 2-of-3 activates on the second consecutive true (earliest activation at
  // push index `need - 1`).
  let mut vote: Box<dyn Gate<f32>> = Box::new(Vote::new(2, 3, 0.5).gate());
  assert!(!vote.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(vote.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap());

  // Dwell of a threshold, confirm = 2 elements: with unit spans the second
  // consecutive active window confirms (`end 2 - origin 0 >= 2`).
  let mut dwell: Box<dyn Gate<f32>> = Box::new(Dwell::new(Threshold::new(0.5), 2).gate());
  assert!(!dwell.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(dwell.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap());

  // Hangover of a threshold, hold = 1 element: holds one element past the last
  // active window, then releases (strict `<`).
  let mut hang: Box<dyn Gate<f32>> = Box::new(Hangover::new(Threshold::new(0.5), 1).gate());
  assert!(hang.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap()); // active
  assert!(hang.push(&Windowed::new(0.1, Span::new(1, 1, 1))).unwrap()); // gap 0 < 1: held
  assert!(!hang.push(&Windowed::new(0.1, Span::new(2, 1, 1))).unwrap()); // gap 1 not < 1: released
}

#[test]
fn dyn_decoder_manifest_path_with_p3_policies_matches_concrete_and_batch() {
  // The full manifest path with P3 stages held INSIDE a decoder: a boxed
  // `CadenceEma` smoother and a boxed `Hangover(Dwell(Vote))` gate — the canonical
  // nesting. Instantiable only because `Box<dyn Smoother<f32>>` and `Box<dyn
  // Gate<f32>>` themselves implement the stage traits (the forwarding impls). Its
  // output must match both the concrete decoder over the same policies and the
  // batch composition, on both planes.
  let scores = [
    0.1f32, 0.8, 0.9, 0.9, 0.2, 0.7, 0.8, 0.1, 0.9, 0.9, 0.9, 0.2, 0.85,
  ];
  let seq: Vec<Windowed<f32>> = scores
    .iter()
    .enumerate()
    .map(|(i, &s)| Windowed::new(s, Span::new(i, 1, 1)))
    .collect();
  let opts = SegmentOptions::new().with_min_len(2).with_merge_gap(1);

  // Cheap `Copy` configs, reused across the three drives.
  let smoother_cfg = CadenceEma::new(4.0);
  let gate_cfg = Hangover::new(Dwell::new(Vote::new(2, 3, 0.5), 2), 3);

  // The run-time-selected manifest decoder.
  let sm: Box<dyn Smoother<f32>> = Box::new(smoother_cfg.smoother());
  let ga: Box<dyn Gate<f32>> = Box::new(gate_cfg.gate());
  let dyn_dec: Decoder<Box<dyn Smoother<f32>>, Box<dyn Gate<f32>>, f32> =
    Decoder::new(sm, ga, opts);
  let (dyn_flags, dyn_ranges) = drive_decoder(dyn_dec, &seq);

  // The concrete decoder over the identical policies.
  let concrete = Decoder::new(smoother_cfg.smoother(), gate_cfg.gate(), opts);
  let (cc_flags, cc_ranges) = drive_decoder(concrete, &seq);
  assert_eq!(
    dyn_flags, cc_flags,
    "causal plane: dyn manifest vs concrete decoder"
  );
  assert_eq!(
    dyn_ranges, cc_ranges,
    "finalized plane: dyn manifest vs concrete decoder"
  );

  // The batch composition of the same policies, both planes.
  let smoothed = smoother_cfg.smooth(&seq).expect("batch smooth");
  let batch_ranges = gate_cfg.segment(&opts, &smoothed).expect("batch segment");
  assert_eq!(
    dyn_ranges, batch_ranges,
    "finalized plane: dyn manifest vs batch composition"
  );
  let mut batch_gate = gate_cfg.gate();
  let batch_flags: Vec<bool> = smoothed
    .iter()
    .map(|w| batch_gate.push(w).expect("batch gate"))
    .collect();
  assert_eq!(
    dyn_flags, batch_flags,
    "causal plane: dyn manifest vs batch gate drive"
  );
}

/// A public-API discontinuity probe gate: a [`Gate<f32>`] recording which
/// lifecycle call fired. The crate has an internal twin; this one pins the
/// forwarding through the PUBLIC `Box<dyn Gate<f32>>` impl an external caller
/// hits.
struct DiscProbeGate {
  reset_calls: Rc<Cell<usize>>,
  discontinuity_calls: Rc<Cell<usize>>,
}

impl Gate<f32> for DiscProbeGate {
  fn push(&mut self, _w: &Windowed<f32>) -> Result<bool, WinditError> {
    Ok(true)
  }

  fn reset(&mut self) {
    self.reset_calls.set(self.reset_calls.get() + 1);
  }

  fn discontinuity(&mut self) {
    self
      .discontinuity_calls
      .set(self.discontinuity_calls.get() + 1);
  }
}

#[test]
fn dyn_decoder_forwards_discontinuity_through_box() {
  let reset_calls = Rc::new(Cell::new(0usize));
  let discontinuity_calls = Rc::new(Cell::new(0usize));
  let ga: Box<dyn Gate<f32>> = Box::new(DiscProbeGate {
    reset_calls: reset_calls.clone(),
    discontinuity_calls: discontinuity_calls.clone(),
  });
  let sm: Box<dyn Smoother<f32>> = Box::new(Ema::new(1.0).smoother());
  let mut dec: Decoder<Box<dyn Smoother<f32>>, Box<dyn Gate<f32>>, f32> =
    Decoder::new(sm, ga, SegmentOptions::new());

  let _ = dec.push(Windowed::new(0.5, Span::new(0, 1, 1))).unwrap();
  // `discontinuity` must reach the concrete gate through the box AS
  // `discontinuity`, not be flattened to the box's default `reset`.
  let _ = dec.discontinuity();
  assert_eq!(
    discontinuity_calls.get(),
    1,
    "discontinuity reached the boxed gate"
  );
  assert_eq!(reset_calls.get(), 0, "reset must not fire on discontinuity");

  // `reset`, in contrast, forwards `reset`.
  dec.reset();
  assert_eq!(reset_calls.get(), 1, "reset reached the boxed gate");
  assert_eq!(
    discontinuity_calls.get(),
    1,
    "discontinuity count unchanged by reset"
  );
}

/// A payload deliberately without `Clone`, `Copy`, or `Debug`. The value-free
/// stages — `Identity`, `Dwell`, `Hangover`, and the `Decoder` threading them —
/// must carry it with none of those bounds; any stage that required `V: Clone`
/// (etc.) would fail to compile through this type.
struct Marker(#[allow(dead_code)] u32);

/// A gate policy replaying a fixed decision script, generic over EVERY value type
/// — it reads neither spans nor values. It is the non-`f32` witness's inner gate,
/// since the shipped leaf gates (`Threshold`, `Vote`, ...) are `Gate<f32>` only.
#[derive(Clone)]
struct ScriptPolicy {
  flags: Vec<bool>,
}

/// The streaming state of [`ScriptPolicy`], replaying its script by push index.
struct ScriptGate {
  flags: Vec<bool>,
  next: usize,
}

impl<V> GatePolicy<V> for ScriptPolicy {
  type Gate = ScriptGate;

  fn gate(&self) -> ScriptGate {
    ScriptGate {
      flags: self.flags.clone(),
      next: 0,
    }
  }
}

impl<V> Gate<V> for ScriptGate {
  fn push(&mut self, _w: &Windowed<V>) -> Result<bool, WinditError> {
    // Replay the script; past its end hold the last decision (inactive if empty),
    // so a sequence longer than the script stays defined.
    let flag = self
      .flags
      .get(self.next)
      .copied()
      .unwrap_or_else(|| self.flags.last().copied().unwrap_or(false));
    self.next += 1;
    Ok(flag)
  }

  fn reset(&mut self) {
    self.next = 0;
  }
}

#[test]
fn value_free_stages_carry_a_non_f32_payload() {
  // The value-free stages carry a payload that is deliberately NOT
  // Clone/Copy/Debug. A leading lone `true` the Dwell suppresses guarantees the
  // shaped output differs from the raw script, so the pipeline demonstrably
  // engaged.
  let script = vec![
    true, false, true, true, true, false, false, true, true, false,
  ];
  let gate_cfg = Hangover::new(Dwell::new(ScriptPolicy { flags: script }, 2), 2);
  let opts = SegmentOptions::new();
  let n = 10;

  // Drive over the non-f32, non-Clone Marker payload.
  let marker_seq: Vec<Windowed<Marker>> = (0..n)
    .map(|i| Windowed::new(Marker(i as u32), Span::new(i, 1, 1)))
    .collect();
  let marker_flags = decode_flags(&gate_cfg, opts, marker_seq);

  // Drive the IDENTICAL gate config and spans over f32: only the value type
  // differs, so equal decisions prove the stages never read the value.
  let f32_seq: Vec<Windowed<f32>> = (0..n)
    .map(|i| Windowed::new(0.0f32, Span::new(i, 1, 1)))
    .collect();
  let f32_flags = decode_flags(&gate_cfg, opts, f32_seq);
  assert_eq!(
    marker_flags, f32_flags,
    "value-free stages decided differently on a non-f32 payload"
  );

  // The exact shaped sequence: Dwell(2) suppresses onsets shorter than two
  // elements, Hangover(2) holds two elements past each active run.
  let expected = vec![
    false, false, false, true, true, true, true, false, true, true,
  ];
  assert_eq!(
    marker_flags, expected,
    "the value-free pipeline shaped the script wrong"
  );
}

#[test]
fn clap_per_window_embedding_stream_stays_per_window() {
  // The shape the CLAP audio-embedding consumer needs and that
  // `aggregate` cannot give it: one 512-wide unit-norm embedding per sliding
  // window, denoised against the windows before it, with EVERY window still
  // emitting. `VectorEma` is `Smoother<E>` for any `Vector` `E`, so the
  // consumer's own embedding type flows through `Windowed<E>` with no
  // conversion at any window — the property that makes this smoother's home
  // this crate rather than the consumer's (a downstream
  // `impl Smoother<TheirEmbedding>` would be an orphan impl).
  const DIM: usize = 512;
  let opts = WindowOptions::new(480_000).with_overlap(240_000);
  let spans = WindowPlan::spans(&opts, 1_200_000).expect("plan spans");
  assert!(spans.len() >= 4, "need a real stream, got {}", spans.len());

  // A 512-wide embedding per span whose direction rotates window to window, so
  // the smoothed stream is genuinely different from the raw one.
  let windows: Vec<Windowed<TestEmbedding>> = spans
    .iter()
    .enumerate()
    .map(|(i, span)| {
      let raw: Vec<f64> = (0..DIM)
        .map(|d| ((d + i) % 17) as f64 - 8.0 + (i as f64) * 0.25)
        .collect();
      Windowed::new(
        TestEmbedding::from_unnormalized(&raw).expect("valid embedding"),
        *span,
      )
    })
    .collect();

  let smoothed = VectorEma::new(0.3).smooth(&windows).expect("smooth");

  // Span-preserving: one window in, one window out, geometry untouched. This is
  // the whole difference from `aggregate`, which returns a single embedding.
  assert_eq!(smoothed.len(), windows.len());
  for (out, inp) in smoothed.iter().zip(&windows) {
    assert_eq!(out.span(), inp.span());
    assert_eq!(out.value().dim(), DIM);
    assert_unit_norm(out.value());
  }

  // Denoised, not passed through: after the seed, each window differs from its
  // own input because it carries the ones before it.
  assert_eq!(
    smoothed[0].value().as_slice(),
    windows[0].value().as_slice()
  );
  assert_ne!(
    smoothed[1].value().as_slice(),
    windows[1].value().as_slice()
  );

  // And smoothing genuinely reduces window-to-window movement: the mean squared
  // step across the smoothed stream is smaller than across the raw one.
  fn mean_sq_step(seq: &[Windowed<TestEmbedding>]) -> f64 {
    let mut total = 0.0;
    for pair in seq.windows(2) {
      total += pair[0]
        .value()
        .as_slice()
        .iter()
        .zip(pair[1].value().as_slice())
        .map(|(a, b)| f64::from(a - b) * f64::from(a - b))
        .sum::<f64>();
    }
    total / (seq.len() - 1) as f64
  }
  let raw_step = mean_sq_step(&windows);
  let smooth_step = mean_sq_step(&smoothed);
  assert!(
    smooth_step < raw_step,
    "smoothing must reduce window-to-window movement: {smooth_step} vs {raw_step}"
  );

  // The same public smoother over a genuinely different scalar: `f64` storage,
  // config unchanged. Genericity, not an f32 special case.
  let windows64: Vec<Windowed<TestEmbedding64>> = spans
    .iter()
    .enumerate()
    .map(|(i, span)| {
      let raw: Vec<f64> = (0..DIM)
        .map(|d| ((d + i) % 17) as f64 - 8.0 + (i as f64) * 0.25)
        .collect();
      Windowed::new(
        TestEmbedding64::from_unnormalized(&raw).expect("valid embedding"),
        *span,
      )
    })
    .collect();
  let smoothed64 = VectorEma::new(0.3).smooth(&windows64).expect("smooth");
  assert_eq!(smoothed64.len(), windows64.len());
  for out in &smoothed64 {
    assert_unit_norm_64(out.value());
  }

  // Run-time selection: the vector smoother is a `Box<dyn Smoother<E>>` too, so
  // a manifest can pick it the way one picks a `dyn AggregatePolicy`.
  let mut boxed: Box<dyn Smoother<TestEmbedding64>> = Box::new(VectorEma::new(0.3).smoother());
  let first = boxed
    .push(Windowed::new(
      TestEmbedding64::from_unnormalized(&[3.0, 4.0]).unwrap(),
      Span::new(0, 1, 1),
    ))
    .unwrap();
  assert_eq!(first.value().as_slice(), &[0.6, 0.8]);
  boxed.reset();
}
