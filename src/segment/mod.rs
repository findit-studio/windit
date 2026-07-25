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
//! The shipped gates are [`Threshold`] (active at or above a fixed cutoff),
//! [`Hysteresis`] (a latching two-threshold gate), and [`Vote`] (active once
//! at least `need` of the last `of` per-window comparisons were true).
//! [`Dwell`] and [`Hangover`] are gate combinators that wrap an inner gate and
//! reshape its flag sequence with an element-denominated on-delay or
//! off-delay; they shape the *causal flag plane*, distinct from the
//! *finalized-range plane* morphology
//! ([`SegmentOptions::merge_gap`], [`SegmentOptions::min_len`]) the
//! `Segmenter` applies afterward — the two are not interchangeable. A
//! [`GatePolicy`] is the
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
///
/// # Span contract
///
/// Spans arrive in ascending [`Span::start`] order, equal starts admitted — and
/// that is the only ordering guaranteed. **Ends are not monotone:** nested and
/// overlapping spans are legal, so a later span may end *before* one already
/// seen. A gate that keeps a temporal horizon must therefore fold it by maximum
/// (`horizon = max(horizon, span.end())`) and measure against that fold, as
/// [`Dwell`]'s confirmation and [`Hangover`]'s coverage horizon both do;
/// reading the current span's end alone would let the horizon move backward and
/// the decision retract. A strictly backward start is a contract violation,
/// reported as [`WinditError::NonMonotonicSpan`].
pub trait Gate<V> {
  /// Advance by one window; `Ok(true)` means active/accepted.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::NonMonotonicSpan`] from a span-reading stage fed a
  /// start before the previous one: [`Dwell`] and [`Hangover`] read spans to
  /// measure their confirm/hold distances and report out-of-order input here,
  /// each checking independently of any inner or nested combinator.
  /// [`Threshold`], [`Hysteresis`], and [`Vote`] read no spans and are
  /// infallible, always returning `Ok` and carrying the `Result` only for
  /// uniformity with the composable stages.
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

/// N-of-M voting gate: active once at least `need` of the last `of`
/// per-window comparisons `value >= thr` were true.
///
/// A window not yet pushed in the current epoch counts as an inactive vote —
/// the pre-fill is all-`false` — so the gate starts inactive and its decision
/// is causal and deterministic from the very first push. The comparison
/// itself is raw IEEE, exactly [`Threshold`]'s table: a `NaN` value is never a
/// true vote; a `NaN` `thr` accepts nothing; `thr = -inf` accepts every
/// non-`NaN` value; `thr = +inf` accepts only `+inf`. Only the vote *counts*
/// are checked at construction — `thr` is deliberately free to degrade the
/// same way `Threshold`'s does. (Lineage: FunASR window-count voting, cited
/// for lineage only — no recommendation.)
///
/// # Cadence coupling
///
/// `Vote` counts *windows*, not elements — like [`Ema`](crate::smooth::Ema)'s
/// per-step alpha, its physical meaning changes with the hop. That is
/// inherent to N-of-M voting, not a defect, and is stated plainly here: no
/// portability claim is made. [`Dwell`] is the element-denominated
/// confirmation alternative when portability across hops matters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vote {
  need: usize,
  of: usize,
  thr: f32,
}

impl Vote {
  /// A vote gate active once at least `need` of the last `of` comparisons
  /// `value >= thr` are true.
  ///
  /// # Panics
  ///
  /// Panics, in every build, unless `1 <= need <= of <= 64`. Use
  /// [`try_new`](Vote::try_new) to handle untrusted counts instead.
  #[must_use]
  pub const fn new(need: usize, of: usize, thr: f32) -> Self {
    match Self::try_new(need, of, thr) {
      Ok(v) => v,
      Err(_) => panic!("vote counts must satisfy 1 <= need <= of <= 64"),
    }
  }

  /// The checked counterpart of [`new`](Vote::new): validate the counts
  /// rather than panic on them.
  ///
  /// # Errors
  ///
  /// Returns [`WinditError::InvalidVote`] unless `1 <= need <= of <= 64`:
  /// `need = 0` would always activate, `need > of` would never activate,
  /// `of = 0` is vacuous, and `of > 64` exceeds the one-machine-word state
  /// bound documented on [`VoteState`].
  pub const fn try_new(need: usize, of: usize, thr: f32) -> Result<Self, WinditError> {
    if need >= 1 && need <= of && of <= 64 {
      Ok(Self { need, of, thr })
    } else {
      Err(WinditError::InvalidVote { need, of })
    }
  }

  /// The required-true count out of the last [`of`](Vote::of) votes.
  #[must_use]
  pub const fn need(&self) -> usize {
    self.need
  }

  /// The vote window length: how many recent comparisons are considered.
  #[must_use]
  pub const fn of(&self) -> usize {
    self.of
  }

  /// The cutoff each window's value is compared against: a vote is true when
  /// `value >= thr`.
  #[must_use]
  pub const fn thr(&self) -> f32 {
    self.thr
  }
}

/// The streaming state of a [`Vote`] gate: the cutoff plus a one-machine-word
/// ring buffer of the last up-to-64 votes and its popcount, all-`false` until
/// the first push.
///
/// `need` and `of` narrow from [`Vote`]'s `usize` (the crate's count
/// convention) down to `u8`, because a validated configuration's counts
/// always fit `1..=64`; [`gate`](GatePolicy::gate) is the single place that
/// defends the narrowing (see its doc). The ring holds the newest vote in bit
/// 0. The type is a fixed 16 bytes on every platform, independent of target
/// word size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoteState {
  thr: f32,
  need: u8,
  of: u8,
  ring: u64,
}

impl Gate<f32> for VoteState {
  fn push(&mut self, w: &Windowed<f32>) -> Result<bool, WinditError> {
    // Raw IEEE, Threshold's table: NaN is never a true vote.
    let vote = *w.value() >= self.thr;
    // The newest vote enters bit 0; bit 63 falls off the top, discarding any
    // history older than 64 votes ago. Vote reads no spans, so this is total
    // and branch-free.
    self.ring = (self.ring << 1) | (vote as u64);
    // `of` is defended into `1..=64` by `gate()`, so `64 - of` is always in
    // `0..=63` — a shift amount that can never panic.
    let window = self.ring & (u64::MAX >> (64 - self.of as u32));
    Ok(window.count_ones() >= self.need as u32)
  }

  fn reset(&mut self) {
    self.ring = 0;
  }
}

impl GatePolicy<f32> for Vote {
  type Gate = VoteState;

  fn gate(&self) -> VoteState {
    // `Vote::new`/`try_new` are the only public ways to set `need`/`of`, and
    // both validate `1 <= need <= of <= 64` there, so this re-clamp is
    // currently unreachable. It is kept as the last line of defence for the
    // shift-amount invariant `VoteState::push` relies on: a future
    // construction path that bypasses `new`/`try_new` — a serde derive being
    // the obvious one — degrades to the nearest valid configuration instead
    // of shifting a `u64` out of range, which would panic in debug and be
    // silently masked in release.
    let of = self.of.clamp(1, 64);
    let need = self.need.clamp(1, of);
    VoteState {
      thr: self.thr,
      need: need as u8,
      of: of as u8,
      ring: 0,
    }
  }
}

/// Onset-confirmation gate combinator: on-delay debounce.
///
/// Wraps an inner gate `P` and suppresses its `true` decisions until the inner
/// gate has been continuously active for `confirm` input elements. Config-level:
/// implements [`GatePolicy<V>`] for any inner `P: GatePolicy<V>`; the streaming
/// state, [`DwellState`], implements [`Gate<V>`] for *every* `V` given only
/// `G: Gate<V>` — the wrapper never reads the value, only the inner decision and
/// the span. (Lineage: fast-vad/LiveKit dwell time, cited for lineage only — no
/// recommendation.)
///
/// # Semantics
///
/// On an inner `true`, the wrapper records the *origin* — the `span.start()` of
/// the first `true` after a `false` (or after fresh/[`reset`](Gate::reset)/
/// [`discontinuity`](Gate::discontinuity)) — and folds the run's coverage
/// horizon forward: `horizon = max(horizon, span.end())`, a monotone maximum
/// rather than an overwrite, since ascending starts do not imply ascending
/// ends. It outputs `true` once that horizon reaches `confirm` elements past
/// the origin (`horizon - origin >= confirm`), otherwise `false` (a suppressed
/// head). Because the horizon only grows, a confirmed run stays confirmed: a
/// nested or overlapping span can never retract an activation. On an inner
/// `false`, the run is cleared and the output is `false`: releases are never
/// delayed, this is on-delay only.
///
/// `confirm = 0` is an exact pass-through: every span covers at least one
/// element, so `horizon - origin >= 0` always holds on an inner `true`. The
/// confirmation distance is measured to the run's coverage horizon — the
/// largest span end seen since the origin — so a window's own coverage counts
/// toward it: with unit spans and `confirm = 3`,
/// inner-true windows at positions 0, 1, 2 confirm on the third push
/// (`end 3 - origin 0 >= 3`). Consequently a `confirm` at or below the first
/// window's length never suppresses anything.
///
/// `confirm = usize::MAX` is the opposite pole and never confirms. It is
/// suppressed explicitly rather than left to the comparison: the widest run a
/// `Span` pair can describe — `[0, 1)` folded with
/// `[usize::MAX - 1, usize::MAX)` — reaches `horizon - origin == usize::MAX`,
/// which meets `>= confirm` exactly and would otherwise activate.
///
/// On a gapped plan (hop exceeding window), the distance is positional:
/// elements no span covers still count toward confirmation as long as the
/// inner flag run is continuous across the gap. That is the element-
/// denominated contract, not a coverage accounting — deliberate, not a bug.
///
/// A `true` is never retracted (emitted only once confirmation is real) and a
/// `false` is never upgraded after the fact: output `true` at some step implies
/// the inner gate was also `true` at that step (suppression only).
///
/// # The causal and finalized planes
///
/// `Dwell` reshapes the causal flag sequence that feeds the [`Segmenter`], so
/// it trims a finalized range's head: the range starts at the `span.start()`
/// of the first *confirmed* window, not at the origin. For example, with unit
/// spans and `confirm = 3`, an inner-true run at positions `0..6` confirms
/// starting at position 2 (the third true, `end 3 - origin 0 >= 3`), so the
/// finalized range is `[2, 6)`, not `[0, 6)`. A caller wanting the full extent
/// kept, with only whole short runs suppressed, uses
/// [`SegmentOptions::min_len`] (finalized-plane morphology) instead — the two
/// tools shape different planes and are not interchangeable.
///
/// # Errors
///
/// `Dwell` reads spans to measure `confirm`, so it checks
/// [`WinditError::NonMonotonicSpan`] itself, independently of the inner gate:
/// the check runs before the inner [`push`](Gate::push), and on the wrapper's
/// own violation neither the wrapper's nor the inner gate's state has advanced.
/// If the *inner* push errs instead (a nested span-reading combinator), the
/// wrapper's own fields are still unchanged — `last_start` is written only
/// after the inner call returns `Ok` — while the inner gate's state follows its
/// own error contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dwell<P> {
  inner: P,
  confirm: usize,
}

impl<P> Dwell<P> {
  /// A dwell combinator over `inner`, confirming activation only after
  /// `confirm` continuous input elements.
  ///
  /// Infallible: every `usize` is a meaningful confirmation distance,
  /// including `0` (exact pass-through) and [`usize::MAX`], which never
  /// confirms — a suppressed sentinel, not merely an unreachable comparison.
  #[must_use]
  pub const fn new(inner: P, confirm: usize) -> Self {
    Self { inner, confirm }
  }

  /// The wrapped gate configuration.
  #[must_use]
  pub const fn inner(&self) -> &P {
    &self.inner
  }

  /// The confirmation distance, in input elements.
  #[must_use]
  pub const fn confirm(&self) -> usize {
    self.confirm
  }
}

/// The streaming state of a [`Dwell`] combinator: the inner gate, the
/// confirmation distance, the current continuous run as
/// `(origin, coverage horizon)` (`None` between runs), and the last pushed
/// span's start (for the wrapper's own monotonicity check, independent of the
/// inner gate's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DwellState<G> {
  inner: G,
  confirm: usize,
  run: Option<(usize, usize)>,
  last_start: Option<usize>,
}

impl<V, G: Gate<V>> Gate<V> for DwellState<G> {
  fn push(&mut self, w: &Windowed<V>) -> Result<bool, WinditError> {
    let span = w.span();
    let start = span.start();
    // Checked before the inner push, so the wrapper's own violation leaves
    // both this wrapper and the inner gate unchanged.
    if let Some(prev) = self.last_start {
      if start < prev {
        return Err(WinditError::NonMonotonicSpan {
          prev_start: prev,
          start,
        });
      }
    }
    let inner_active = self.inner.push(w)?;
    self.last_start = Some(start);
    if inner_active {
      // The origin is the start of the first `true` since the last `false`
      // (or since fresh/reset/discontinuity); set here at most once per run.
      // The horizon is folded by MAX, exactly as `HangoverState`'s is: the
      // contract guarantees ascending starts, not ascending ends, so a nested
      // or overlapping span can end before one already seen. Comparing the
      // current span's end alone would let a confirmed gate deactivate while
      // the inner gate never released — an on-delay gate must not do that. The
      // max also makes confirmation permanent for the run: once the horizon
      // clears `confirm` it can never fall back below it.
      let (origin, horizon) = match self.run {
        Some((origin, horizon)) => (origin, horizon.max(span.end())),
        None => (start, span.end()),
      };
      self.run = Some((origin, horizon));
      // `usize::MAX` is the documented never-confirming configuration, and it
      // is the one value the `>=` test below cannot enforce on its own: the
      // widest run a `Span` pair can describe — `[0, 1)` then
      // `[usize::MAX - 1, usize::MAX)` — folds to `origin = 0` and
      // `horizon = usize::MAX`, so `horizon - origin` meets `confirm` exactly
      // and confirms. Suppressing the sentinel here keeps the promise total.
      // The run above is still folded, so the state machine stays uniform and
      // only the decision is withheld. `Hangover` needs no counterpart: its
      // test has the opposite sense (`gap < hold`), and `Span`'s own invariants
      // — `len >= 1`, and no `start + len` overflow — cap the gap at
      // `usize::MAX - 2`, two elements short of the sentinel.
      if self.confirm == usize::MAX {
        return Ok(false);
      }
      // `origin <= start <= horizon` under monotone starts, so this cannot
      // underflow; `saturating_sub` is the last line of defence, not
      // load-bearing.
      Ok(horizon.saturating_sub(origin) >= self.confirm)
    } else {
      self.run = None;
      Ok(false)
    }
  }

  fn reset(&mut self) {
    self.run = None;
    self.last_start = None;
    self.inner.reset();
  }

  // Forwarded explicitly, not left to the trait default: the default would
  // call this wrapper's own `reset`, which would in turn call the inner
  // gate's `reset` rather than its `discontinuity`, silently erasing an
  // override the inner gate carries (mirrors the `Box` impl below).
  fn discontinuity(&mut self) {
    self.run = None;
    self.last_start = None;
    self.inner.discontinuity();
  }
}

impl<V, P: GatePolicy<V>> GatePolicy<V> for Dwell<P> {
  type Gate = DwellState<P::Gate>;

  fn gate(&self) -> Self::Gate {
    DwellState {
      inner: self.inner.gate(),
      confirm: self.confirm,
      run: None,
      last_start: None,
    }
  }
}

/// Release-hold gate combinator: off-delay debounce.
///
/// Wraps an inner gate `P` and, once active, holds `true` while the gap since
/// the end of inner-active coverage stays below `hold` input elements — the
/// mirror of [`Dwell`]'s on-delay. Config-level: implements [`GatePolicy<V>`]
/// for any inner `P: GatePolicy<V>`; the streaming state, [`HangoverState`],
/// implements [`Gate<V>`] for *every* `V` given only `G: Gate<V>`. (Lineage:
/// WebRTC VAD hangover, cited for lineage only — no recommendation.)
///
/// # Semantics
///
/// On an inner `true`, the wrapper outputs `true` and folds the coverage
/// horizon forward: `last_yes_end = max(last_yes_end, span.end())` — a
/// monotone maximum, not an overwrite, so an overlapping or nested span can
/// never move the horizon backward and shorten the hold. On an inner `false`:
/// if no inner `true` has been seen this epoch, the output is `false`;
/// otherwise the output is `true` iff
/// `span.start().saturating_sub(last_yes_end) < hold`, else `false`. The
/// comparison is strict: a window starting exactly `hold` elements after
/// coverage ends is *not* held — with unit spans, `hold = N` extends the flag
/// run by exactly `N` elements past the last inner-active window's end. The
/// subtraction saturates, so an overlapping window
/// (`span.start() < last_yes_end`) has gap `0`, which is always held for any
/// positive `hold` — the same overlap-saturates-to-zero convention
/// [`Segmenter`]'s merge fold uses.
///
/// `hold = 0` is an exact pass-through: no gap is `< 0` (an adjacent window
/// has gap `0`, which is not `< 0`). `hold = usize::MAX` is the opposite pole
/// and never releases once activated. That one falls out of the comparison
/// rather than needing [`Dwell`]'s explicit guard, because the test has the
/// opposite sense: [`Span`] requires `len >= 1` and rejects a `start + len`
/// overflow, so a coverage horizon is at least `1` and a start at most
/// `usize::MAX - 1`, capping the gap at `usize::MAX - 2` — strictly below the
/// sentinel. An inner `true` seen again during the
/// hold re-folds the horizon, so `Hangover` causally bridges inner-flag gaps
/// shorter than `hold` into one flag run. `last_yes_end` is never cleared on
/// release — under monotone starts a later gap can only grow, so a stale
/// horizon can never re-activate anything — only [`reset`](Gate::reset) or
/// [`discontinuity`](Gate::discontinuity) clears it. A held `true` is never
/// retracted, and an inner `true` at some step always implies output `true`
/// at that step (extension only — the dual of [`Dwell`]'s suppression only).
///
/// # The causal and finalized planes
///
/// Held flags do not bridge *uncovered* elements in the finalized plane:
/// geometric continuity is span-based (the [`Segmenter`] only extends a run
/// across windows whose start falls at or before its current end), so on a
/// gapped plan (hop exceeding window) a held window separated from the last
/// inner-active one by elements no span covers still opens a new run.
/// [`SegmentOptions::merge_gap`] remains the only element-level bridge in that
/// plane: `Hangover` holds the *flag* causally, while `merge_gap` bridges the
/// finalized *ranges* acausally within its own horizon — the two tools shape
/// different planes and are not interchangeable.
///
/// # Errors
///
/// `Hangover` reads spans to measure `hold`, so it checks
/// [`WinditError::NonMonotonicSpan`] itself, independently of the inner gate:
/// the check runs before the inner [`push`](Gate::push), and on the wrapper's
/// own violation neither the wrapper's nor the inner gate's state has advanced.
/// If the *inner* push errs instead, the wrapper's own fields are still
/// unchanged — `last_start` is written only after the inner call returns `Ok`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hangover<P> {
  inner: P,
  hold: usize,
}

impl<P> Hangover<P> {
  /// A hangover combinator over `inner`, holding an activation for `hold`
  /// input elements after inner-active coverage ends.
  ///
  /// Infallible: every `usize` is a meaningful hold distance, including `0`
  /// (exact pass-through) and [`usize::MAX`] (never releases once activated).
  #[must_use]
  pub const fn new(inner: P, hold: usize) -> Self {
    Self { inner, hold }
  }

  /// The wrapped gate configuration.
  #[must_use]
  pub const fn inner(&self) -> &P {
    &self.inner
  }

  /// The hold distance, in input elements.
  #[must_use]
  pub const fn hold(&self) -> usize {
    self.hold
  }
}

/// The streaming state of a [`Hangover`] combinator: the inner gate, the hold
/// distance, the folded coverage horizon (`None` until the first inner `true`
/// this epoch), and the last pushed span's start (for the wrapper's own
/// monotonicity check, independent of the inner gate's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HangoverState<G> {
  inner: G,
  hold: usize,
  last_yes_end: Option<usize>,
  last_start: Option<usize>,
}

impl<V, G: Gate<V>> Gate<V> for HangoverState<G> {
  fn push(&mut self, w: &Windowed<V>) -> Result<bool, WinditError> {
    let span = w.span();
    let start = span.start();
    if let Some(prev) = self.last_start {
      if start < prev {
        return Err(WinditError::NonMonotonicSpan {
          prev_start: prev,
          start,
        });
      }
    }
    let inner_active = self.inner.push(w)?;
    self.last_start = Some(start);
    if inner_active {
      // F2 (plan conformance flag): fold by MAX, not last-write. A later span
      // can still end earlier than an already-covered one (an
      // overlapping/nested span), and a last-write horizon would move the
      // release boundary backward and shorten the hold.
      let end = span.end();
      self.last_yes_end = Some(self.last_yes_end.map_or(end, |prev| prev.max(end)));
      Ok(true)
    } else {
      match self.last_yes_end {
        Some(end) => Ok(start.saturating_sub(end) < self.hold),
        None => Ok(false),
      }
    }
  }

  fn reset(&mut self) {
    self.last_yes_end = None;
    self.last_start = None;
    self.inner.reset();
  }

  // Forwarded explicitly for the same reason as `DwellState`: the trait
  // default would silently route through the inner gate's `reset`.
  fn discontinuity(&mut self) {
    self.last_yes_end = None;
    self.last_start = None;
    self.inner.discontinuity();
  }
}

impl<V, P: GatePolicy<V>> GatePolicy<V> for Hangover<P> {
  type Gate = HangoverState<P::Gate>;

  fn gate(&self) -> Self::Gate {
    HangoverState {
      inner: self.inner.gate(),
      hold: self.hold,
      last_yes_end: None,
      last_start: None,
    }
  }
}

/// Forwarding so a boxed gate is itself a [`Gate`], letting a run-time-selected
/// `Box<dyn Gate<V>>` be *held* as a stage — not merely called through
/// auto-deref.
///
/// `?Sized` covers `Box<dyn Gate<V>>` and `Box<Concrete>` alike; the coverage
/// is conventional, mirroring std's `Box<impl Iterator>`. Policies stay
/// non-boxed — configs are `Copy`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
impl<V, T: Gate<V> + ?Sized> Gate<V> for std::boxed::Box<T> {
  fn push(&mut self, w: &Windowed<V>) -> Result<bool, WinditError> {
    (**self).push(w)
  }

  fn reset(&mut self) {
    (**self).reset();
  }

  // Forwarded explicitly, not left to the trait default: the default would
  // route this box's `discontinuity` to the box's own `reset`, silently
  // erasing any `discontinuity` override the concrete stage carries.
  fn discontinuity(&mut self) {
    (**self).discontinuity();
  }
}
