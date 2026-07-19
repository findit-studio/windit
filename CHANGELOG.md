# Changelog

All notable changes to `windit` are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

Initial release: the generic windowed-sequence processing core (pre + post) for
embeddings, VAD, and ASR.

### Added

- Window geometry: `WindowOptions`, `WindowPlan`, `Span`, and `TailPolicy` for
  turning an input length into unit-agnostic window spans.
- Pre-processing: `slice_pad_mask` / `try_slice_pad_mask` to slice, right-pad,
  and mask a span into a fixed-width window.
- Aggregation policies: the object-safe `AggregatePolicy` with the built-ins
  `CoverageWeightedMean`, `MeanRenormalized`, `EmaRenormalized`, and
  `SaliencyWeighted`, the multi-vector `keep_separate`, and the serde
  `AggregatePolicyKind` selector.
- Smoothing policies: the `SmoothPolicy` trait with `Ema` and `Hysteresis`.
- Segmentation: `runs`, `longest_run`, `runs_sorted`, and the `SegmentPolicy`
  built-ins `Threshold` and `HysteresisSegment`.
- Split policies: `FixedWindow`, and the tokenizer-free `ContentAware` string
  chunker behind the `text` feature.
- `no_std + alloc` support with optional `std`, `text`, and `serde` features;
  minimum supported Rust version 1.95.
