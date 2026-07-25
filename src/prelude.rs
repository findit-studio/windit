//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```
//!
//! The value, geometry, and scalar types are always available, as are the
//! featureless smoothing, gating, and segmentation cores: the `Smoother` /
//! `SmoothPolicy` and `Gate` / `GatePolicy` traits, the `Identity`, `Ema`,
//! `Threshold`, and `Hysteresis` configs, and `Segmenter`, `SegmentTail`,
//! `Range`, `SegmentOptions`. The `Vec`-returning algorithms — the planner, the
//! pre-processing helpers, the aggregation and split families, and the batch
//! segmentation drivers (`runs`, `longest_run`, `runs_sorted`) — are re-exported
//! under the `alloc` feature, matching where they are defined. The content-aware
//! chunker `ContentAware`, its `Chunk` payload, and the `MeasureText` measurer it
//! reads length through join them under the `text` feature, and
//! `AggregatePolicyKind` under `serde`.

pub use crate::{
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
  scalar::{Real, Scalar},
  // The smoothing, gating, and segmentation cores are featureless, so they are
  // always available.
  segment::{
    Gate, GatePolicy, Hysteresis, Range, SegmentOptions, SegmentTail, Segmenter, Threshold,
  },
  smooth::{Ema, Identity, SmoothPolicy, Smoother},
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
