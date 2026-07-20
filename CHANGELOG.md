# Changelog

All notable changes to `windit` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

Initial release: the generic windowed-sequence processing core (pre + post) for
embeddings, VAD, and ASR.

### Added

- Window geometry: `WindowOptions`, `WindowPlan`, `Span`, and `TailPolicy` for
  turning an input length into unit-agnostic window spans. `Span` and
  `segment::Range` each pair a panicking `new` with a checked `try_new`, and
  both enforce their invariants — `0 < len <= window` with a representable
  `Span::end`, and `start <= end` — identically in debug and release.
- Pre-processing: `slice_pad_mask` / `try_slice_pad_mask` to slice, right-pad,
  and mask a span into a fixed-width window.
- Scalars: the sealed `Scalar` and `Real` traits and the `ComputeOf` alias,
  implemented for `f32` and `f64` (neither feature-gated). `Vector` carries an
  associated `Scalar`, so embeddings are generic over what they store, while
  `Scalar::Compute` keeps the storage and compute domains separable for a future
  narrow type such as `f16`.
- Aggregation policies: the object-safe `AggregatePolicy<C: Real = f32>` with the
  built-ins `CoverageWeightedMean`, `MeanRenormalized`, `EmaRenormalized`, and
  `SaliencyWeighted`, the multi-vector `keep_separate`, and the serde
  `AggregatePolicyKind` selector. The `f32` default keeps `dyn AggregatePolicy`
  spelling the `f32` policy object; policy configuration stays `f32` at every
  compute scalar, so `AggregatePolicyKind`'s wire format is scalar-independent.
- Smoothing policies: the `SmoothPolicy` trait with `Ema` and `Hysteresis`.
- Segmentation: `runs`, `longest_run`, `runs_sorted`, and the `SegmentPolicy`
  built-ins `Threshold` and `HysteresisSegment`. A run is continuous in the
  input geometry as well as in the sequence, so a plan whose hop exceeds its
  window never fuses two accepted spans across the elements it strided past;
  only `merge_gap` bridges them.
- Split policies: `FixedWindow`, and the tokenizer-free `ContentAware` string
  chunker behind the `text` feature. `ContentAware::chunk` is fallible: it
  reports invalid geometry and honours `WindowOptions::max_windows` exactly as
  `WindowPlan::spans` does, so the one configured bound on how much work a
  chunking may cost reaches the chunker too. Packing calls `len_fn` `O(a log a)`
  times for `a` atoms — it never re-measures a range whose measure it already
  knows, and it locates each overlap boundary by probing rather than by walking
  one atom at a time — which keeps a near-window overlap over untrusted text off
  the cubic path. Boundaries are still decided by measuring the real contiguous
  text, never by summing per-atom measurements, so a non-additive (BPE,
  wordpiece) `len_fn` keeps its exact chunk boundaries.
- `no_std + alloc` support with optional `std`, `text`, and `serde` features;
  minimum supported Rust version 1.95. `libm` is an unconditional dependency:
  `Real::sqrt` lives on the ungated core tier, so even a `--no-default-features`
  build needs it.
