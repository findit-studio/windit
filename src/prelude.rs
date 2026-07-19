//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```
//!
//! The value and geometry types are always available. The `Vec`-returning
//! algorithms — the planner, the pre-processing helpers, and the four strategy
//! families — are re-exported under the `alloc` feature, matching where they are
//! defined. The content-aware chunker [`ContentAware`] joins them under the
//! `text` feature, and [`AggregatePolicyKind`] under `serde`.

pub use crate::{
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
  windowed::{Vector, WindowEmbedding, Windowed},
};

#[cfg(any(feature = "std", feature = "alloc"))]
pub use crate::{
  aggregate::{
    aggregate, keep_separate, AggregatePolicy, CoverageWeightedMean, EmaRenormalized,
    MeanRenormalized, SaliencyWeighted,
  },
  plan::WindowPlan,
  pre::{slice_pad_mask, try_slice_pad_mask},
  segment::{
    longest_run, runs, runs_sorted, HysteresisSegment, Range, SegmentOptions, SegmentPolicy,
    Threshold,
  },
  smooth::{Ema, Hysteresis, SmoothPolicy},
  split::{FixedWindow, SplitPolicy},
};

#[cfg(feature = "text")]
pub use crate::split::ContentAware;

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
pub use crate::aggregate::AggregatePolicyKind;
