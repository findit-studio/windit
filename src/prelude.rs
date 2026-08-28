//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```
//!
//! The value, geometry, and scalar types are always available, as are the
//! featureless smoothing, gating, segmentation, and decoding cores: the
//! `Smoother` / `SmoothPolicy` and `Gate` / `GatePolicy` traits, the `Identity`,
//! `Ema`, and `CadenceEma` smoother configs, the `Threshold`, `Hysteresis`, and
//! `Vote` gate configs with the `Dwell` and `Hangover` gate combinators,
//! `Segmenter`, `SegmentTail`, `Range`, `SegmentOptions`, and the `Decoder`
//! pipeline with its `Step` output. The heap-tier items — the planner, the
//! pre-processing helpers, the aggregation and split families, and the batch
//! segmentation drivers (`runs`, `longest_run`, `runs_sorted`) — are re-exported
//! under the `alloc` feature, matching where they are defined. The content-aware
//! chunker `ContentAware`, its `Chunk` payload, and the `MeasureText` measurer it
//! reads length through join them under the `text` feature, and
//! `AggregatePolicyKind` under `serde`.
//!
//! The vector smoother `smooth::VectorEma` is deliberately **not** here, and is
//! imported by path. A crate that globs both this module and one of its own
//! carrying the same name stops compiling with `E0659`, ambiguity being
//! reported at the use site rather than at the import — and
//! `cargo-semver-checks` does not model that, so its verdict is not evidence
//! either way. Adding the name to `smooth` carries the same hazard for anyone
//! who globs *that* module, so this is a reduction in exposure rather than a
//! guarantee: this prelude is the glob the crate documents and asks every
//! dependent to write, and `smooth::*` is a glob it suggests nowhere. Whether
//! the name joins is a decision of its own, not a consequence of any release
//! that happens to announce a break elsewhere.

pub use crate::{
  // Featureless and always available: the value, geometry, and scalar types, the
  // smoothing, gating, segmentation, and decoding cores, and the value surface.
  // The `alloc`-gated block below adds the `Vec`-returning algorithms.
  decode::{Decoder, Step},
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
  scalar::{Real, Scalar},
  segment::{
    Dwell, Gate, GatePolicy, Hangover, Hysteresis, Range, SegmentOptions, SegmentTail, Segmenter,
    Threshold, Vote,
  },
  smooth::{CadenceEma, Ema, Identity, SmoothPolicy, Smoother},
  windowed::{ComputeOf, Vector, WindowEmbedding, Windowed},
};

#[cfg(any(feature = "std", feature = "alloc"))]
pub use crate::{
  aggregate::{
    aggregate, keep_separate, AggregatePolicy, CoverageWeightedMean, EmaRenormalized,
    MeanRenormalized, SaliencyWeighted,
  },
  plan::WindowPlan,
  pre::{slice_pad_mask, try_slice_pad_mask},
  segment::{longest_run, runs, runs_sorted},
  split::{FixedWindow, SplitPolicy},
};

#[cfg(feature = "text")]
pub use crate::split::{Chunk, ContentAware, MeasureText};

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
pub use crate::aggregate::AggregatePolicyKind;
