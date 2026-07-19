//! Segmentation: reduce a windowed score sequence to continuous element ranges.
//!
//! The core is `runs`: it walks a `&[Windowed<V>]`, groups the windows a
//! caller-supplied predicate accepts into contiguous runs, maps each run to a
//! half-open `Range` in input-element units through its spans, then applies
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
  /// # Panics
  ///
  /// Panics via a debug assertion (debug builds only) if `start > end`. The
  /// check is compiled out in release, so [`len`](Range::len) still saturates an
  /// inverted range to `0` there.
  #[must_use]
  pub const fn new(start: usize, end: usize) -> Self {
    debug_assert!(start <= end, "a range must satisfy start <= end");
    Self { start, end }
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
  /// Saturates to `0` for an inverted range (`start > end`). [`new`](Range::new)
  /// rejects one only through a debug assertion, so a release build can still
  /// hold such a range; the crate itself never produces one.
  #[must_use]
  pub const fn len(&self) -> usize {
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
/// A run is a maximal block of consecutive accepted windows. Each run becomes
/// the [`Range`] from its first window's `span.start` to the largest
/// `span.start + span.len` among its windows (so overlapping-window runs cover
/// the union of their spans). The runs are then merged when separated by at most
/// [`SegmentOptions::merge_gap`] elements, and any run shorter than
/// [`SegmentOptions::min_len`] is dropped.
pub fn runs<V, F>(seq: &[Windowed<V>], predicate: F, opts: &SegmentOptions) -> Vec<Range>
where
  F: Fn(&V) -> bool,
{
  let mut raw: Vec<Range> = Vec::new();
  let mut current: Option<Range> = None;
  for w in seq {
    if predicate(&w.value) {
      let end = w.span.start() + w.span.len();
      if let Some(run) = current.as_mut() {
        run.end = run.end.max(end);
      } else {
        // `end` is `span.start() + span.len()` and `len` is non-zero, so the
        // range is well formed.
        current = Some(Range::new(w.span.start(), end));
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
/// with these `on` / `off` thresholds (turn on at `>= on`, off at `<= off`, hold
/// between), then the latched-on windows are grouped by [`runs`] under `opts`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HysteresisSegment {
  on: f32,
  off: f32,
  opts: SegmentOptions,
}

impl HysteresisSegment {
  /// Latch on at `>= on` and off at `<= off`, shaping the resulting runs with
  /// the default [`SegmentOptions`].
  ///
  /// Configure `on >= off`; [`Hysteresis`] documents how the gate degrades
  /// otherwise.
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
