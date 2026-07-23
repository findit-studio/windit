//! Segmentation: reduce a windowed score sequence to continuous element ranges.
//!
//! The core is `runs`: it walks a `&[Windowed<V>]`, groups the windows a
//! caller-supplied predicate accepts into runs that are continuous in both the
//! sequence and the input geometry, maps each run to a half-open `Range` in
//! input-element units through its spans, then applies
//! two `SegmentOptions` passes — merge runs separated by at most `merge_gap`
//! elements, then drop runs shorter than `min_len`. `longest_run` and
//! `runs_sorted` rank those ranges; the find-longest-continuous-range case
//! (the longest speech region, say) is `longest_run`.
//!
//! `SegmentPolicy` packages a predicate with its options. `Threshold` admits
//! values at or above a cutoff; `HysteresisSegment` first latches the sequence
//! through `smooth::Hysteresis` and then segments, which is the binary-VAD
//! path.

use std::vec::Vec;

use crate::{
  error::WinditError,
  smooth::{Hysteresis, SmoothPolicy},
  windowed::Windowed,
};

#[cfg(test)]
mod tests;

/// A half-open range of input elements, `[start, end)`.
///
/// Units are input elements (samples, tokens, patches, frames) — the same units
/// as [`Span`](crate::plan::Span) — so a range is independent of the window
/// geometry that produced it. A well-formed range has `start <= end`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Range {
  start: usize,
  end: usize,
}

impl Range {
  /// The half-open range covering the elements from `start` up to (but not
  /// including) `end`.
  ///
  /// The infallible counterpart to [`try_new`](Range::try_new), for the callers
  /// that know their bounds and would only unwrap.
  ///
  /// # Panics
  ///
  /// Panics, in every build, if `start > end`. Use
  /// [`try_new`](Range::try_new) to handle untrusted bounds instead.
  #[must_use]
  pub const fn new(start: usize, end: usize) -> Self {
    match Self::try_new(start, end) {
      Ok(range) => range,
      Err(_) => panic!("a range must satisfy start <= end"),
    }
  }

  /// The checked counterpart of [`new`](Range::new): validate the bounds rather
  /// than panic on them.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::InvalidRange`] if `start > end`. That is the range
  /// invariant [`len`](Range::len) relies on, so it is enforced identically in
  /// debug and release.
  pub const fn try_new(start: usize, end: usize) -> Result<Self, WinditError> {
    if start > end {
      return Err(WinditError::InvalidRange { start, end });
    }
    Ok(Self { start, end })
  }

  /// The first element in the range.
  #[must_use]
  pub const fn start(&self) -> usize {
    self.start
  }

  /// One past the last element in the range.
  #[must_use]
  pub const fn end(&self) -> usize {
    self.end
  }

  /// The number of elements the range covers (`end - start`).
  ///
  /// Never underflows: both constructors reject `start > end` in every build,
  /// and the only writes that bypass them — [`runs`] and `merge_adjacent`
  /// extending a run — move `end` upward alone.
  #[must_use]
  pub const fn len(&self) -> usize {
    // The saturation is therefore unreachable. It is kept as the last line of
    // defence for those in-crate field writes: a future one that moved `end`
    // downward would report a zero-length range rather than underflow into a
    // near-`usize::MAX` length that `min_len` would then wave through.
    self.end.saturating_sub(self.start)
  }

  /// Whether the range covers no elements (`start >= end`).
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.start >= self.end
  }
}

/// Post-processing applied to raw runs: gap merging and minimum length.
///
/// Both values are in input elements. Construct with [`SegmentOptions::new`]
/// (keep everything, merge only touching runs) and refine with the `with_*`
/// builders.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SegmentOptions {
  min_len: usize,
  merge_gap: usize,
}

impl SegmentOptions {
  /// Options that keep every run and merge only touching or overlapping runs
  /// (`min_len == 0`, `merge_gap == 0`).
  #[must_use]
  pub const fn new() -> Self {
    Self {
      min_len: 0,
      merge_gap: 0,
    }
  }

  /// Set the minimum kept run length, in input elements.
  #[must_use]
  pub const fn with_min_len(mut self, min_len: usize) -> Self {
    self.min_len = min_len;
    self
  }

  /// Set the largest inter-run gap to bridge, in input elements.
  #[must_use]
  pub const fn with_merge_gap(mut self, merge_gap: usize) -> Self {
    self.merge_gap = merge_gap;
    self
  }

  /// The minimum kept run length, in input elements.
  #[must_use]
  pub const fn min_len(&self) -> usize {
    self.min_len
  }

  /// The largest inter-run gap to bridge, in input elements.
  #[must_use]
  pub const fn merge_gap(&self) -> usize {
    self.merge_gap
  }
}

impl Default for SegmentOptions {
  fn default() -> Self {
    Self::new()
  }
}

/// Group the windows `predicate` accepts into merged, length-filtered element
/// ranges, in input order.
///
/// The sequence must be in span order (ascending `span.start`), as planners
/// produce; a run's start is taken from its first window, not the minimum over
/// the run.
///
/// A run is a maximal block of accepted windows that is also *geometrically
/// continuous*: a window is added to the open run only when its `span.start` is
/// at or before the run's current end. A plan whose hop exceeds its window
/// strides over the input, so two accepted windows can be separated by elements
/// no span covers; such a pair starts a new run rather than fusing into one that
/// would claim the uncovered elements. Bridging that separation is
/// [`SegmentOptions::merge_gap`]'s decision alone.
///
/// Each run becomes the [`Range`] from its first window's `span.start` to the
/// largest [`Span::end`](crate::plan::Span::end) among its windows (so
/// overlapping-window runs cover the union of their spans). The runs are then
/// merged when separated by at most [`SegmentOptions::merge_gap`] elements, and
/// any run shorter than [`SegmentOptions::min_len`] is dropped.
pub fn runs<V, F>(seq: &[Windowed<V>], predicate: F, opts: &SegmentOptions) -> Vec<Range>
where
  F: Fn(&V) -> bool,
{
  let mut raw: Vec<Range> = Vec::new();
  let mut current: Option<Range> = None;
  for w in seq {
    if predicate(&w.value) {
      let (start, end) = (w.span.start(), w.span.end());
      match current {
        // A span beginning past the open run's end leaves elements that no span
        // covers. Closing the run here keeps `merge_gap` the only thing that can
        // bridge them; extending instead would select them unconditionally.
        Some(run) if start > run.end => {
          raw.push(run);
          current = Some(Range::new(start, end));
        }
        Some(ref mut run) => run.end = run.end.max(end),
        // `Span::end` is `start + len` with a non-zero `len`, so `start < end`
        // and the range is well formed.
        None => current = Some(Range::new(start, end)),
      }
    } else if let Some(run) = current.take() {
      raw.push(run);
    }
  }
  if let Some(run) = current.take() {
    raw.push(run);
  }

  let mut merged = merge_adjacent(raw, opts.merge_gap());
  merged.retain(|r| r.len() >= opts.min_len());
  merged
}

/// The longest range from [`runs`], breaking ties toward the earliest.
///
/// Returns `None` when [`runs`] is empty.
pub fn longest_run<V, F>(seq: &[Windowed<V>], predicate: F, opts: &SegmentOptions) -> Option<Range>
where
  F: Fn(&V) -> bool,
{
  let mut best: Option<Range> = None;
  for r in runs(seq, predicate, opts) {
    match best {
      // Keep the incumbent on a tie, so the earliest of equal-length runs wins.
      Some(b) if b.len() >= r.len() => {}
      _ => best = Some(r),
    }
  }
  best
}

/// The ranges from [`runs`], sorted by length descending.
///
/// The sort is stable, so equal-length ranges keep their input order.
pub fn runs_sorted<V, F>(seq: &[Windowed<V>], predicate: F, opts: &SegmentOptions) -> Vec<Range>
where
  F: Fn(&V) -> bool,
{
  let mut all = runs(seq, predicate, opts);
  all.sort_by_key(|r| core::cmp::Reverse(r.len()));
  all
}

/// Merge runs whose gap to the previous run is at most `merge_gap`.
///
/// `ranges` must be sorted by `start` (as [`runs`] produces them).
fn merge_adjacent(ranges: Vec<Range>, merge_gap: usize) -> Vec<Range> {
  let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
  for r in ranges {
    match out.last_mut() {
      // `saturating_sub` folds an overlapping range (a start before the previous
      // end, reachable with overlapping windows) into a zero gap, so it merges.
      Some(last) if r.start.saturating_sub(last.end) <= merge_gap => {
        last.end = last.end.max(r.end);
      }
      _ => out.push(r),
    }
  }
  out
}

/// A policy that segments a windowed value sequence into element [`Range`]s.
///
/// Generic over the value type `V`; the shipped built-ins implement it for
/// `V = f32` (speech probabilities, energies, logits).
pub trait SegmentPolicy<V> {
  /// Segment `seq` into the element ranges it selects, in input order.
  fn segment(&self, seq: &[Windowed<V>]) -> Vec<Range>;
}

/// Segment where the score is at or above a fixed threshold.
///
/// A window is in-segment when `value >= thr`; the resulting runs are shaped by
/// the policy's [`SegmentOptions`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Threshold {
  thr: f32,
  opts: SegmentOptions,
}

impl Threshold {
  /// Admit values at or above `thr`, shaping the runs with the default
  /// [`SegmentOptions`].
  #[must_use]
  pub const fn new(thr: f32) -> Self {
    Self {
      thr,
      opts: SegmentOptions::new(),
    }
  }

  /// Shape the runs with `opts` rather than the default.
  #[must_use]
  pub const fn with_opts(mut self, opts: SegmentOptions) -> Self {
    self.opts = opts;
    self
  }

  /// The cutoff at or above which a value is in-segment.
  #[must_use]
  pub const fn thr(&self) -> f32 {
    self.thr
  }

  /// The gap-merging and minimum-length options applied to the runs.
  #[must_use]
  pub const fn opts(&self) -> SegmentOptions {
    self.opts
  }
}

impl SegmentPolicy<f32> for Threshold {
  fn segment(&self, seq: &[Windowed<f32>]) -> Vec<Range> {
    runs(seq, |&v| v >= self.thr, &self.opts)
  }
}

/// Segment through a latching two-threshold gate: the binary-VAD path.
///
/// The sequence is first smoothed by [`Hysteresis`]
/// with these `on` / `off` thresholds (turn on at `>= on`, off strictly below
/// `off`, hold between — a value exactly at `off` holds), then the latched-on
/// windows are grouped by [`runs`] under `opts`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HysteresisSegment {
  on: f32,
  off: f32,
  opts: SegmentOptions,
}

impl HysteresisSegment {
  /// Latch on at `>= on` and off strictly below `off`, shaping the resulting
  /// runs with the default [`SegmentOptions`].
  ///
  /// Configure `on >= off`; [`Hysteresis`] documents the hold region and how
  /// the gate degrades otherwise.
  #[must_use]
  pub const fn new(on: f32, off: f32) -> Self {
    Self {
      on,
      off,
      opts: SegmentOptions::new(),
    }
  }

  /// Shape the runs with `opts` rather than the default.
  #[must_use]
  pub const fn with_opts(mut self, opts: SegmentOptions) -> Self {
    self.opts = opts;
    self
  }

  /// The turn-on threshold, forwarded to [`Hysteresis`].
  #[must_use]
  pub const fn on(&self) -> f32 {
    self.on
  }

  /// The turn-off threshold, forwarded to [`Hysteresis`].
  #[must_use]
  pub const fn off(&self) -> f32 {
    self.off
  }

  /// The gap-merging and minimum-length options applied to the runs.
  #[must_use]
  pub const fn opts(&self) -> SegmentOptions {
    self.opts
  }
}

impl SegmentPolicy<f32> for HysteresisSegment {
  fn segment(&self, seq: &[Windowed<f32>]) -> Vec<Range> {
    let gated = Hysteresis::new(self.on, self.off).smooth(seq);
    // The gate emits exactly 0.0 / 1.0, so `>= 0.5` selects the latched-on runs.
    runs(&gated, |&v| v >= 0.5, &self.opts)
  }
}
