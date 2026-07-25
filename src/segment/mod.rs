//! Segmentation: gate a windowed score sequence and reduce it to element ranges.
//!
//! Two stages: a [`Gate`] turns each windowed value into a binary active/inactive
//! decision, and the incremental [`Segmenter`] groups those decisions into
//! finalized element [`Range`]s. The `Segmenter` is a bounded, O(1) state machine
//! that consumes `(active, span)` decisions one at a time and emits ranges no
//! future input can change: it groups the accepted windows into runs that are
//! continuous in both the sequence and the input geometry, maps each run to a
//! half-open `Range` in input-element units through its spans, and applies the two
//! [`SegmentOptions`] morphology passes — merge runs separated by at most
//! `merge_gap` elements, then drop runs shorter than `min_len`.
//!
//! The shipped gates are [`Threshold`] (active at or above a fixed cutoff) and
//! [`Hysteresis`] (a latching two-threshold gate). A [`GatePolicy`] is the
//! configuration that constructs a [`Gate`]; it also
#![cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "drives that gate through the `Segmenter` as the batch"
)]
#![cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`segment`](GatePolicy::segment) convenience. [`runs`], [`longest_run`], and"
)]
#![cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "drives that gate through the `Segmenter` as the batch `segment` convenience."
)]
#![cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`runs_sorted`] are the predicate-driven counterparts. Every batch driver"
)]
#![cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "`runs`, `longest_run`, and `runs_sorted` are the predicate-driven counterparts. Every batch driver"
)]
//! *drives* a fresh `Segmenter` over a slice and collects what it emits, so batch
//! output equals the streaming core plus [`finish`](Segmenter::finish) by
//! construction rather than by two implementations kept in sync. Each restarts
//! from fresh state on every call — a batch convenience, not an incremental
//! decoder. For streaming, drive a [`Gate`] into a `Segmenter` directly.
//!
//! The [`Gate`]/[`GatePolicy`] traits, the gate configs, the `Segmenter`,
//! `SegmentTail`, `Range`, and `SegmentOptions` all live in the featureless core
//! tier (they allocate nothing); only the `Vec`-returning batch drivers gate on
//! the `alloc` feature.

use crate::{error::WinditError, plan::Span, windowed::Windowed};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// A half-open range of input elements, `[start, end)`.
///
/// Units are input elements (samples, tokens, patches, frames) — the same units
/// as [`Span`] — so a range is independent of the window
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
  /// and the only writes that bypass them — the in-crate [`Segmenter`] core
  /// extending a run or folding a merge — move `end` upward alone.
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

/// Incremental run builder and morphology: consume `(active, span)` gate
/// decisions and emit finalized element [`Range`]s with exact batch parity.
///
/// One concrete semantics, parameterized by [`SegmentOptions`] — there is no
#[cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "trait, because there is exactly one geometry. The batch drivers ([`runs`],"
)]
#[cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "trait, because there is exactly one geometry. The batch drivers (`runs`,"
)]
#[cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`longest_run`], [`runs_sorted`]) *are* this state machine driven over a"
)]
#[cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "`longest_run`, `runs_sorted`) *are* this state machine driven over a"
)]
/// slice, so streaming and batch cannot drift apart.
///
/// # State and bound
///
/// The state is four fields (a fixed 80 bytes) — `opts`, the currently-extending
/// run (`open`), the closed-and-merged candidate awaiting its gap verdict
/// (`pending`), and the last start seen (`last_start`) — and is **O(1) for
/// every configuration**, including an unbounded `merge_gap`. A large
/// `merge_gap` never grows the state: `pending` only ever widens, and its
/// emission simply defers to [`finish`](Segmenter::finish). Every
/// [`push`](Segmenter::push) allocates nothing.
///
/// # Emission and commit latency
///
/// A pushed decision emits at most one finalized range; [`finish`](Segmenter::finish)
/// emits at most two. A range `pending` is emitted on the first pushed span
/// whose `start` clears `pending.end + merge_gap` — the earliest witness that
/// no future run can merge into it, since starts only grow — or at `finish`.
///
/// # Contract
///
/// Spans must arrive in ascending `start` order (equal starts admitted); a
/// strictly backward start returns [`WinditError::NonMonotonicSpan`]. A genuine
/// timeline break is declared with [`discontinuity`](Segmenter::discontinuity),
/// which finalizes pending output and never bridges `merge_gap` across the
/// break, then re-arms fresh state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segmenter {
  opts: SegmentOptions,
  /// The run currently being extended by geometrically continuous accepted
  /// spans; `None` between runs.
  open: Option<Range>,
  /// The closed, gap-merged accumulator awaiting the verdict of whether a later
  /// run merges into it. `None` until the first run closes, and again after it
  /// is finalized.
  pending: Option<Range>,
  /// The `start` of the most recent pushed span, for the monotonicity check.
  last_start: Option<usize>,
}

impl Segmenter {
  /// A fresh segmenter that shapes its runs with `opts`.
  #[must_use]
  pub const fn new(opts: SegmentOptions) -> Self {
    Self {
      opts,
      open: None,
      pending: None,
      last_start: None,
    }
  }

  /// The morphology options this segmenter applies.
  #[must_use]
  pub const fn opts(&self) -> SegmentOptions {
    self.opts
  }

  /// Feed one gate decision with the [`Span`] it covers.
  ///
  /// `Ok(Some(range))` is a finalized range that no future input can change;
  /// `Ok(None)` means this decision extended a run, closed one into the merge
  /// fold without finalizing anything, or was rejected with nothing pending.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::NonMonotonicSpan`] if `span.start()` is strictly
  /// less than the previous pushed span's start, with no intervening
  /// [`discontinuity`](Segmenter::discontinuity) or [`reset`](Segmenter::reset).
  /// The offending push leaves the segmenter unchanged.
  pub fn push(&mut self, active: bool, span: Span) -> Result<Option<Range>, WinditError> {
    let start = span.start();
    // Rule 1 — monotonicity. Checked before any state mutation, so an
    // out-of-order push is a no-op that reports the violation.
    if let Some(prev) = self.last_start {
      if start < prev {
        return Err(WinditError::NonMonotonicSpan {
          prev_start: prev,
          start,
        });
      }
    }
    self.last_start = Some(start);

    // Rule 3 — an accepted span geometrically continuous with the open run
    // (its start at or before the run's current end) extends it and emits
    // nothing: the run is still growing, so nothing can be finalized yet.
    if active {
      if let Some(run) = self.open.as_mut() {
        if start <= run.end {
          run.end = run.end.max(span.end());
          return Ok(None);
        }
      }
    }

    // Otherwise the open run (if any) closes into the merge fold (rules 3–5).
    let closed_emit = match self.open.take() {
      Some(closed) => self.feed_raw_run(closed),
      None => None,
    };
    // Rule 2 — early finalization. With nothing now open, a span beyond the gap
    // horizon finalizes `pending`. Only when the fold above emitted nothing, so
    // a push finalizes at most one range; a deferred finalization surfaces on a
    // later push or at `finish`.
    let emitted = match closed_emit {
      Some(_) => closed_emit,
      None => self.early_finalize(start),
    };
    // Rule 3 — a non-continuous accepted span opens a fresh run.
    if active {
      self.open = Some(Range {
        start,
        end: span.end(),
      });
    }
    Ok(emitted)
  }

  /// End of stream: close the open run, resolve the merge fold, and emit
  /// everything pending — at most two ranges — as a fixed-size iterator.
  ///
  /// Consuming `self` makes use-after-finish unrepresentable; the continue-past
  /// case is [`discontinuity`](Segmenter::discontinuity).
  #[must_use]
  pub fn finish(mut self) -> SegmentTail {
    let first = match self.open.take() {
      Some(run) => self.feed_raw_run(run),
      None => None,
    };
    let second = match self.pending.take() {
      Some(p) => self.keep(p),
      None => None,
    };
    SegmentTail::new(first, second)
  }

  /// Declared timeline break: emit as [`finish`](Segmenter::finish) would, never
  /// bridging `merge_gap` across the break, then re-arm fresh state for the next
  /// epoch.
  ///
  /// Span positions may restart after the break; the monotonicity check is
  /// re-armed too. The caller owns the epoch bookkeeping — it is the one
  /// declaring the break — so windit stays unit-agnostic.
  #[must_use]
  pub fn discontinuity(&mut self) -> SegmentTail {
    let first = match self.open.take() {
      Some(run) => self.feed_raw_run(run),
      None => None,
    };
    let second = match self.pending.take() {
      Some(p) => self.keep(p),
      None => None,
    };
    // `open` and `pending` were just drained; re-arm the timeline so the next
    // epoch may restart span positions.
    self.last_start = None;
    SegmentTail::new(first, second)
  }

  /// Destructive discard: return to the freshly-constructed state, **dropping**
  /// any unemitted pending output.
  ///
  /// [`discontinuity`](Segmenter::discontinuity) is the non-lossy alternative
  /// when the timeline continues.
  pub fn reset(&mut self) {
    self.open = None;
    self.pending = None;
    self.last_start = None;
  }

  /// Merge a just-closed run into the `pending` accumulator (the streaming form
  /// of the batch merge-adjacent left fold), returning the accumulator it
  /// finalizes, if any.
  ///
  /// Runs close in ascending start order, so this is a left fold: a run within
  /// `merge_gap` of the accumulator folds into it (`end` grows), otherwise the
  /// accumulator is complete and the run starts a fresh one. `min_len` is
  /// applied only here, at finalization — never at close — so a short run a
  /// later run merges with survives, exactly as the batch merge-then-filter
  /// order.
  fn feed_raw_run(&mut self, run: Range) -> Option<Range> {
    let merge_gap = self.opts.merge_gap();
    // Comparing the gap with `saturating_sub` avoids the `pending.end +
    // merge_gap` overflow an unbounded `merge_gap` would cause, and folds an
    // overlapping run (start before the accumulator's end) to a zero gap
    // exactly as the batch `merge_adjacent` did.
    let completed: Option<Range> = match self.pending {
      None => {
        self.pending = Some(run);
        None
      }
      Some(ref mut p) if run.start.saturating_sub(p.end) <= merge_gap => {
        p.end = p.end.max(run.end);
        None
      }
      Some(ref mut p) => {
        let done = *p;
        *p = run;
        Some(done)
      }
    };
    completed.and_then(|r| self.keep(r))
  }

  /// Finalize `pending` when a span at `start` proves no future run can merge
  /// into it (`start` clears `pending.end + merge_gap`, and starts only grow).
  ///
  /// Sound only when nothing is `open`: an open run within `merge_gap` of
  /// `pending` will still fold into it, so this is invoked only where `open` is
  /// `None`.
  fn early_finalize(&mut self, start: usize) -> Option<Range> {
    let p = self.pending?;
    if start.saturating_sub(p.end) > self.opts.merge_gap() {
      self.pending = None;
      self.keep(p)
    } else {
      None
    }
  }

  /// Apply the `min_len` filter at finalization: emit `r` if it is long enough,
  /// otherwise drop it silently.
  fn keep(&self, r: Range) -> Option<Range> {
    if r.len() >= self.opts.min_len() {
      Some(r)
    } else {
      None
    }
  }
}

/// Bounded terminal emission from [`Segmenter::finish`] and
/// [`Segmenter::discontinuity`]: an iterator over at most two finalized ranges.
///
/// A concrete, fixed-size iterator that allocates nothing — it holds the ranges
/// inline. Implements [`Iterator<Item = Range>`](Iterator) and
/// [`ExactSizeIterator`], yielding the finalized ranges in ascending start
/// order.
#[derive(Clone, Debug)]
pub struct SegmentTail {
  /// The finalized ranges, compacted so the present ones lead and any absent
  /// slot trails.
  ranges: [Option<Range>; 2],
  /// The next slot to yield.
  idx: usize,
}

impl SegmentTail {
  /// Build a tail from the two candidate emissions, dropping the absent ones so
  /// iteration yields only present ranges (in order).
  fn new(first: Option<Range>, second: Option<Range>) -> Self {
    let mut ranges = [None, None];
    for (slot, r) in [first, second].into_iter().flatten().enumerate() {
      ranges[slot] = Some(r);
    }
    Self { ranges, idx: 0 }
  }
}

impl Iterator for SegmentTail {
  type Item = Range;

  fn next(&mut self) -> Option<Range> {
    let taken = self.ranges.get_mut(self.idx)?.take();
    if taken.is_some() {
      self.idx += 1;
    }
    taken
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.ranges[self.idx..]
      .iter()
      .filter(|r| r.is_some())
      .count();
    (remaining, Some(remaining))
  }
}

impl ExactSizeIterator for SegmentTail {}

#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

/// Push a finalized range onto the output, surfacing an allocation failure as
/// [`WinditError::AllocFailed`] rather than aborting.
///
/// `try_reserve(1)` is a no-op when spare capacity exists and grows (amortized)
/// otherwise, so the checked entry points stay checked without changing the
/// allocation pattern.
#[cfg(any(feature = "std", feature = "alloc"))]
fn push_checked(out: &mut Vec<Range>, r: Range) -> Result<(), WinditError> {
  out.try_reserve(1).map_err(|_| WinditError::AllocFailed {
    elements: out.len().saturating_add(1),
  })?;
  out.push(r);
  Ok(())
}

/// Group the windows `predicate` accepts into merged, length-filtered element
/// ranges, in input order.
///
/// This drives a fresh [`Segmenter`] over `seq` and collects everything it
/// emits, so the result is exactly the streaming core plus
/// [`finish`](Segmenter::finish).
///
/// The sequence must be in span order (ascending `span.start`), as planners
/// produce; a run's start is taken from its first window, not the minimum over
/// the run. A strictly backward start is a precondition violation reported as
/// [`WinditError::NonMonotonicSpan`], not silent nonsense — sort by `span.start`
/// first if the order was ever unknown.
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
/// largest [`Span::end`] among its windows (so
/// overlapping-window runs cover the union of their spans). The runs are then
/// merged when separated by at most [`SegmentOptions::merge_gap`] elements, and
/// any run shorter than [`SegmentOptions::min_len`] is dropped.
///
/// # Errors
///
/// - [`WinditError::NonMonotonicSpan`] if a span's `start` is strictly before
///   its predecessor's.
/// - [`WinditError::AllocFailed`] if the output ranges cannot be allocated.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub fn runs<V, F>(
  seq: &[Windowed<V>],
  predicate: F,
  opts: &SegmentOptions,
) -> Result<Vec<Range>, WinditError>
where
  F: Fn(&V) -> bool,
{
  let mut seg = Segmenter::new(*opts);
  let mut out: Vec<Range> = Vec::new();
  for w in seq {
    if let Some(r) = seg.push(predicate(w.value()), w.span())? {
      push_checked(&mut out, r)?;
    }
  }
  for r in seg.finish() {
    push_checked(&mut out, r)?;
  }
  Ok(out)
}

/// The longest range from [`runs`], breaking ties toward the earliest.
///
/// Returns `Ok(None)` when [`runs`] is empty.
///
/// # Errors
///
/// Propagates [`runs`]: [`WinditError::NonMonotonicSpan`] or
/// [`WinditError::AllocFailed`].
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub fn longest_run<V, F>(
  seq: &[Windowed<V>],
  predicate: F,
  opts: &SegmentOptions,
) -> Result<Option<Range>, WinditError>
where
  F: Fn(&V) -> bool,
{
  let mut best: Option<Range> = None;
  for r in runs(seq, predicate, opts)? {
    match best {
      // Keep the incumbent on a tie, so the earliest of equal-length runs wins.
      Some(b) if b.len() >= r.len() => {}
      _ => best = Some(r),
    }
  }
  Ok(best)
}

/// The ranges from [`runs`], sorted by length descending.
///
/// The sort is stable, so equal-length ranges keep their input order.
///
/// # Errors
///
/// Propagates [`runs`]: [`WinditError::NonMonotonicSpan`] or
/// [`WinditError::AllocFailed`].
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
pub fn runs_sorted<V, F>(
  seq: &[Windowed<V>],
  predicate: F,
  opts: &SegmentOptions,
) -> Result<Vec<Range>, WinditError>
where
  F: Fn(&V) -> bool,
{
  let mut all = runs(seq, predicate, opts)?;
  all.sort_by_key(|r| core::cmp::Reverse(r.len()));
  Ok(all)
}

/// Stateful binary decision over a windowed value sequence: one window in, one
/// `bool` out.
///
/// The decision plane feeding the [`Segmenter`] — `Ok(true)` means the window is
/// active (accepted), `Ok(false)` inactive. A `Gate` carries whatever state its
/// decision needs (a latch, a running count) in O(1) space and allocates nothing,
/// so it lives in the featureless core tier. The configuration that constructs one
/// is a [`GatePolicy`].
///
/// `Box<dyn Gate<f32>>` is a valid object, so a gate can be selected at run time.
pub trait Gate<V> {
  /// Advance by one window; `Ok(true)` means active/accepted.
  ///
  /// # Errors
  ///
  /// Returns a [`WinditError`] for a stage that reads spans out of order; the
  /// shipped gates ([`Threshold`], [`Hysteresis`]) are infallible and always
  /// return `Ok`, returning `Result` only for uniformity with the composable
  /// stages.
  fn push(&mut self, w: &Windowed<V>) -> Result<bool, WinditError>;

  /// Return to the freshly-constructed state.
  fn reset(&mut self);

  /// Declare a timeline break; for a gate that carries no pending output across
  /// epochs this is [`reset`](Gate::reset).
  fn discontinuity(&mut self) {
    self.reset();
  }
}

/// A gating configuration: names the strategy, constructs its streaming [`Gate`],
/// and drives it as a batch segmentation convenience.
///
/// Generic over the value type `V`; the shipped built-ins implement it for
/// `V = f32` (speech probabilities, energies, logits). Implement the factory
/// [`gate`](GatePolicy::gate) to add a strategy — the batch
#[cfg_attr(
  any(feature = "std", feature = "alloc"),
  doc = "[`segment`](GatePolicy::segment) method is provided over it."
)]
#[cfg_attr(
  not(any(feature = "std", feature = "alloc")),
  doc = "`segment` method is provided over it."
)]
pub trait GatePolicy<V> {
  /// The streaming state this configuration constructs.
  type Gate: Gate<V>;

  /// Fresh streaming state for this configuration.
  fn gate(&self) -> Self::Gate;

  /// Batch convenience: gate every window in `seq` through a fresh [`Gate`] and
  /// shape the accepted runs with `opts`, in input order.
  ///
  /// This drives the gate decisions through a fresh [`Segmenter`] and collects
  /// what it emits, so the result equals the streaming core plus
  /// [`finish`](Segmenter::finish) by construction. Fresh state per call, exactly
  /// as the 0.1.x policies documented.
  ///
  /// The sequence must be in span order (ascending `span.start`), as planners
  /// produce; a strictly backward start is a precondition violation reported as
  /// [`WinditError::NonMonotonicSpan`], not silent nonsense.
  ///
  /// # Errors
  ///
  /// - [`WinditError::NonMonotonicSpan`] if a span's `start` is strictly before
  ///   its predecessor's.
  /// - [`WinditError::AllocFailed`] if the output ranges cannot be allocated.
  /// - Any error the underlying [`Gate::push`] surfaces (none for the shipped
  ///   built-ins).
  #[cfg(any(feature = "std", feature = "alloc"))]
  #[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
  fn segment(&self, opts: &SegmentOptions, seq: &[Windowed<V>]) -> Result<Vec<Range>, WinditError> {
    let mut gate = self.gate();
    let mut seg = Segmenter::new(*opts);
    let mut out: Vec<Range> = Vec::new();
    for w in seq {
      let active = gate.push(w)?;
      if let Some(r) = seg.push(active, w.span())? {
        push_checked(&mut out, r)?;
      }
    }
    for r in seg.finish() {
      push_checked(&mut out, r)?;
    }
    Ok(out)
  }
}

/// Gate where the score is at or above a fixed threshold.
///
/// A window is active when `value >= thr`. The comparison is raw IEEE: a `NaN`
/// score is never active (even with `thr = -inf`); a `NaN` `thr` accepts nothing;
/// `thr = -inf` accepts every non-`NaN` score; `thr = +inf` accepts only `+inf`.
///
/// The morphology that shapes the accepted runs — `merge_gap`, `min_len` — is no
/// longer bundled here; it is passed to the segmentation call through
/// [`SegmentOptions`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Threshold {
  thr: f32,
}

impl Threshold {
  /// A gate that accepts values at or above `thr`.
  #[must_use]
  pub const fn new(thr: f32) -> Self {
    Self { thr }
  }

  /// The cutoff at or above which a value is active.
  #[must_use]
  pub const fn thr(&self) -> f32 {
    self.thr
  }
}

/// The streaming state of a [`Threshold`] gate: the immutable cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThresholdState {
  thr: f32,
}

impl Gate<f32> for ThresholdState {
  fn push(&mut self, w: &Windowed<f32>) -> Result<bool, WinditError> {
    Ok(*w.value() >= self.thr)
  }

  fn reset(&mut self) {}
}

impl GatePolicy<f32> for Threshold {
  type Gate = ThresholdState;

  fn gate(&self) -> ThresholdState {
    ThresholdState { thr: self.thr }
  }
}

/// Latching two-threshold gate.
///
/// The gate turns on when a value rises to `on` or above, turns off when a value
/// falls strictly below `off`, and otherwise holds its previous state — so a
/// value exactly at `off` holds rather than turning off. It starts off. Configure
/// `on >= off`; the half-open band `off <= value < on` is the hold region that
/// suppresses chatter. If misconfigured with `on < off`, the turn-on test is
/// evaluated first and wins, so the gate degrades to a single threshold at `on`.
/// This is the binary VAD smoothing generalized to any f32 score.
///
/// Comparisons are IEEE, and every comparison with `NaN` is false, so: a `NaN`
/// score holds the previous state (including the initial off state); `+inf`
/// activates whenever `on` is not `NaN`; `-inf` releases unless `on` is `-inf`
/// (activation wins) or `off` is `NaN` or `-inf` (then it holds — `off = -inf`
/// can never release, since no value is `< -inf`); a `NaN` `on` can never
/// activate, so the gate stays off; and a `NaN` `off` never releases once on.
///
/// This is a [`Gate`], not a smoother: it yields a typed `bool` decision. A caller
/// that genuinely needs a `0.0`/`1.0` float latch sequence maps the decision
/// itself — `if active { 1.0 } else { 0.0 }` — rather than the crate carrying that
/// role.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hysteresis {
  on: f32,
  off: f32,
}

impl Hysteresis {
  /// A gate that latches on at `>= on` and off at `< off`.
  ///
  /// Configure `on >= off`; the band `off <= value < on` is the hold region, so
  /// a value exactly at `off` holds rather than turning off. An `on < off` pair
  /// is deliberately not rejected: the type documents the single-threshold
  /// behaviour it degrades to, which is defined and deterministic rather than
  /// invalid.
  #[must_use]
  pub const fn new(on: f32, off: f32) -> Self {
    Self { on, off }
  }

  /// The turn-on threshold: a value `>= on` latches the gate on.
  #[must_use]
  pub const fn on(&self) -> f32 {
    self.on
  }

  /// The turn-off threshold: a value `< off` latches the gate off; a value
  /// exactly at `off` holds instead.
  #[must_use]
  pub const fn off(&self) -> f32 {
    self.off
  }

  /// Advance the latch by one value and return the resulting state: on at
  /// `>= on`, off strictly below `off`, hold otherwise (so a value exactly at
  /// `off`, and any NaN, holds the previous state). The turn-on test is
  /// evaluated first and wins.
  ///
  /// The single definition of the gate transition, shared by
  /// [`HysteresisState::push`](Gate::push) and any batch driver, so the streaming
  /// and batch paths cannot drift apart.
  pub(crate) const fn step(&self, active: bool, value: f32) -> bool {
    if value >= self.on {
      true
    } else if value < self.off {
      false
    } else {
      active
    }
  }
}

/// The streaming state of a [`Hysteresis`] gate: the thresholds plus the latch,
/// which starts off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HysteresisState {
  cfg: Hysteresis,
  active: bool,
}

impl Gate<f32> for HysteresisState {
  fn push(&mut self, w: &Windowed<f32>) -> Result<bool, WinditError> {
    self.active = self.cfg.step(self.active, *w.value());
    Ok(self.active)
  }

  fn reset(&mut self) {
    self.active = false;
  }
}

impl GatePolicy<f32> for Hysteresis {
  type Gate = HysteresisState;

  fn gate(&self) -> HysteresisState {
    HysteresisState {
      cfg: *self,
      active: false,
    }
  }
}
