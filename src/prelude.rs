//! Commonly used `windit` types, re-exported for a single glob import.
//!
//! ```
//! use windit::prelude::*;
//! ```
//!
//! The value and geometry types are always available. The `Vec`-returning
//! algorithms (the planner and the four strategy families) are re-exported only
//! under the `alloc` feature, matching where they are defined.

pub use crate::{
  error::WinditError,
  plan::{Span, TailPolicy, WindowOptions},
  windowed::{Vector, WindowEmbedding, Windowed},
};

#[cfg(feature = "alloc")]
pub use crate::{
  aggregate::{
    aggregate, keep_separate, AggregatePolicy, CoverageWeightedMean, EmaRenormalized,
    MeanRenormalized, SaliencyWeighted,
  },
  plan::WindowPlan,
  segment::{
    longest_run, runs, runs_sorted, HysteresisSegment, Range, SegmentOptions, SegmentPolicy,
    Threshold,
  },
  smooth::{Ema, Hysteresis, SmoothPolicy},
  split::{FixedWindow, SplitPolicy},
};

#[cfg(all(feature = "serde", feature = "alloc"))]
pub use crate::aggregate::AggregatePolicyKind;
