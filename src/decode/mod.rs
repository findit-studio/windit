//! Decoding: compose a smoother, a gate, and a segmenter into one streaming
//! pipeline.
//!
//! A [`Decoder`] threads each [`Windowed<V>`](crate::windowed::Windowed) through
//! three stages in order — a [`Smoother`] rewrites the
//! value, a [`Gate`] turns it into an active/inactive
//! decision, and a [`Segmenter`] groups those
//! decisions into finalized element [`Range`]s — and
//! returns a [`Step`] carrying both output planes for that window. It is the
//! common gate-and-segment pipeline as one object; the three stages stay
//! individually public for callers who prefer to wire them by hand.
//!
//! # The two output planes
//!
//! Every [`push`](Decoder::push) reports both a causal decision and any range
//! that push finalized:
//!
//! | Plane | Accessor | Latency | Retraction | Batch parity |
//! |---|---|---|---|---|
//! | Causal | [`Step::active`] | this window | never | not promised |
//! | Finalized | [`Step::finalized`] | deferred until no later input can change it | never | exact |
//!
//! `active` is the gate's immediate decision for the window just pushed. A
//! `finalized` range is one no future input can alter, so those ranges —
//! concatenated with the [`finish`](Decoder::finish) tail — reproduce the batch
//! composition exactly (see [`Decoder`] for the precise statement). The causal
//! flag stream carries no such batch-parity promise: a combinator gate shapes it
//! as a streaming decision.
//!
//! # Lifecycle
//!
//! - [`push`](Decoder::push) advances all three stages by one window.
//! - [`finish`](Decoder::finish) ends the stream, draining the segmenter's
//!   pending ranges (the stages hold none) and consuming the decoder.
//! - [`discontinuity`](Decoder::discontinuity) declares a timeline break: it
//!   drains the pending ranges exactly as `finish` would, then re-arms all three
//!   stages for the next epoch — the non-lossy way to continue past a break.
//! - [`reset`](Decoder::reset) discards all state, dropping any unemitted
//!   pending output.
//!
//! ```
//! use windit::decode::Decoder;
//! use windit::prelude::*;
//!
//! // Pass-through smoothing, a fixed cutoff, and no run morphology.
//! let mut dec = Decoder::new(
//!     Ema::new(1.0).smoother(),
//!     Threshold::new(0.5).gate(),
//!     SegmentOptions::new(),
//! );
//!
//! // A short active burst over unit spans; `active` is the causal decision.
//! for (i, &score) in [0.1_f32, 0.8, 0.9, 0.2].iter().enumerate() {
//!     let step = dec.push(Windowed::new(score, Span::new(i, 1, 1)))?;
//!     assert_eq!(step.active(), score >= 0.5);
//! }
//!
//! // Ending the stream finalizes the burst as the element range `[1, 3)`.
//! let mut tail = dec.finish();
//! assert_eq!(tail.next(), Some(Range::new(1, 3)));
//! assert_eq!(tail.next(), None);
//! # Ok::<(), windit::WinditError>(())
//! ```

use core::{fmt, marker::PhantomData};

use crate::{
  error::WinditError,
  segment::{Gate, Range, SegmentOptions, SegmentTail, Segmenter},
  smooth::Smoother,
  windowed::Windowed,
};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// A streaming [`Smoother`] →
/// [`Gate`] → [`Segmenter`]
/// pipeline: push windows in, read a [`Step`] per window, drain the tail at the
/// end.
///
/// Each [`push`](Decoder::push) runs the smoother, feeds the smoothed window to
/// the gate, and feeds the gate's decision with the smoothed span to the
/// segmenter, returning the causal flag and any finalized
/// [`Range`] as a [`Step`]. The stages are built from
/// their policies — `Decoder::new(cfg_s.smoother(), cfg_g.gate(), opts)` — and
/// are otherwise opaque: the decoder exposes no stage accessors, so the only way
/// to advance them is the parity-preserving [`push`](Decoder::push).
///
/// This is the fixed three-stage composition, not a configurable pipeline: for
/// anything else, drive the stages directly. There is no batch method either —
/// the batch story is the policy conveniences (`SmoothPolicy::smooth` then
/// `GatePolicy::segment`, behind the `alloc` tier), which the parity property
/// below ties to this streaming path.
///
/// # Parity
///
/// For a span-monotone sequence, driving a decoder to exhaustion equals the
/// batch composition of the same policies. Concatenating every
/// [`Step::finalized`] with the [`finish`](Decoder::finish) tail yields exactly
/// `cfg_g.segment(&opts, &cfg_s.smooth(seq)?)?`, and the [`Step::active`] stream
/// equals a fresh `cfg_g.gate()` driven over `cfg_s.smooth(seq)?` — because both
/// sides drive the same three state types, in the same order, from fresh state.
/// A [`discontinuity`](Decoder::discontinuity) partitions the run into
/// independent epochs: the output is the concatenation of the two whole-epoch
/// runs, on both planes.
///
/// # Type parameters
///
/// `S` and `G` are the smoother and gate *state* types; `V` is the window value
/// the pipeline carries. `V` is pinned on the struct through a
/// [`PhantomData<fn() -> V>`](core::marker::PhantomData) rather than left to the
/// stage methods: [`reset`](Decoder::reset) and
/// [`discontinuity`](Decoder::discontinuity) name no value yet must call the
/// stages' `V`-typed lifecycle methods, so `V` has to live on the type. The
/// `fn() -> V` form keeps the decoder covariant in `V` and leaves its `Send`/
/// `Sync` to `S` and `G` alone, with no drop-check entanglement.
///
/// On the common path `V` is inferred with no turbofish: the shipped gates
/// implement [`Gate<f32>`](crate::segment::Gate) only, so pairing any smoother
/// with one fixes `V = f32`.
pub struct Decoder<S, G, V> {
  smoother: S,
  gate: G,
  seg: Segmenter,
  _value: PhantomData<fn() -> V>,
}

impl<S, G, V> Decoder<S, G, V> {
  /// A decoder over the given smoother and gate states, segmenting with `opts`.
  ///
  /// Carries no trait bounds: construction only stores the stages alongside a
  /// fresh [`Segmenter`]. Build the stage states from
  /// their policies — `Decoder::new(cfg_s.smoother(), cfg_g.gate(), opts)`.
  #[must_use]
  pub const fn new(smoother: S, gate: G, opts: SegmentOptions) -> Self {
    Self {
      smoother,
      gate,
      seg: Segmenter::new(opts),
      _value: PhantomData,
    }
  }

  /// The morphology options the inner segmenter applies.
  #[must_use]
  pub const fn opts(&self) -> SegmentOptions {
    self.seg.opts()
  }

  /// End of stream: drain the segmenter's pending ranges (at most two) as a
  /// fixed-size iterator, consuming the decoder.
  ///
  /// Carries no trait bounds. A smoother and a gate hold no pending output —
  /// their trait contract — so finishing drains the segmenter alone; the two
  /// stages are simply dropped. Continuing past this point is
  /// [`discontinuity`](Decoder::discontinuity), which does not consume the
  /// decoder.
  #[must_use]
  pub fn finish(self) -> SegmentTail {
    self.seg.finish()
  }
}

impl<S, G, V> Decoder<S, G, V>
where
  S: Smoother<V>,
  G: Gate<V>,
{
  /// Advance every stage by one window and report the [`Step`].
  ///
  /// The window is smoothed, the smoothed window gated, and the gate's decision
  /// finalized through the segmenter against the smoothed span — the three
  /// stages in pipeline order.
  ///
  /// # Errors
  ///
  /// Surfaces the first error any stage reports. A span-reading stage — a
  /// smoother like [`CadenceEma`](crate::smooth::CadenceEma), a
  /// [`Dwell`](crate::segment::Dwell)/[`Hangover`](crate::segment::Hangover)
  /// combinator, or the segmenter — returns
  /// [`WinditError::NonMonotonicSpan`] on a start before the previous one.
  /// Because every span-reading stage checks the same contract against the same
  /// spans, at most one error surfaces per offending push, from the first such
  /// stage in the pipeline.
  ///
  /// The pipeline is **not** transactional: a stage earlier than the one that
  /// failed has already consumed the window — a value-only smoother advances
  /// before the segmenter rejects a backward span, say. After an error the
  /// decoder is still valid, but its stage states are unspecified relative to
  /// one another; recover with [`discontinuity`](Decoder::discontinuity) to keep
  /// the finalized output or [`reset`](Decoder::reset) to discard it, not by
  /// retrying the failed push. (Each span-reading stage still leaves *itself*
  /// unchanged on its own error, as its own contract states.)
  pub fn push(&mut self, w: Windowed<V>) -> Result<Step, WinditError> {
    let w = self.smoother.push(w)?;
    let active = self.gate.push(&w)?;
    let finalized = self.seg.push(active, w.span())?;
    Ok(Step { active, finalized })
  }

  /// Declare a timeline break: drain the segmenter's pending ranges as
  /// [`finish`](Decoder::finish) would, then re-arm all three stages for the
  /// next epoch.
  ///
  /// Each stage re-arms through its own
  /// [`discontinuity`](crate::segment::Gate::discontinuity), not
  /// [`reset`](crate::segment::Gate::reset), so a stage that distinguishes the
  /// two lifecycles keeps its semantics across the break. Unlike
  /// [`reset`](Decoder::reset) this *emits* the pending output rather than
  /// dropping it, so it is the non-lossy way to continue a stream past a
  /// declared discontinuity. Span positions may restart after the break.
  #[must_use]
  pub fn discontinuity(&mut self) -> SegmentTail {
    // Drain first, then re-arm the value/decision stages: the segmenter's own
    // `discontinuity` finalizes its pending ranges and re-arms it, and the
    // smoother and gate are re-armed through their `discontinuity` so an
    // epoch-aware stage never leaks state across the break.
    let tail = self.seg.discontinuity();
    self.smoother.discontinuity();
    self.gate.discontinuity();
    tail
  }

  /// Destructive discard: return every stage to its freshly-constructed state,
  /// **dropping** any unemitted pending range.
  ///
  /// [`discontinuity`](Decoder::discontinuity) is the non-lossy alternative when
  /// the timeline continues.
  pub fn reset(&mut self) {
    self.seg.reset();
    self.smoother.reset();
    self.gate.reset();
  }
}

// A derived `Clone` would bound `V: Clone` through the phantom, even though
// `PhantomData<fn() -> V>` is `Clone` for every `V` — the classic derive
// over-constraint. This hand-written impl bounds `S`/`G` alone.
impl<S: Clone, G: Clone, V> Clone for Decoder<S, G, V> {
  fn clone(&self) -> Self {
    Self {
      smoother: self.smoother.clone(),
      gate: self.gate.clone(),
      seg: self.seg.clone(),
      _value: PhantomData,
    }
  }
}

// Likewise `Debug`: a derive would demand `V: Debug` for a field that carries no
// `V` value. The phantom holds nothing to show, so it is omitted.
impl<S: fmt::Debug, G: fmt::Debug, V> fmt::Debug for Decoder<S, G, V> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Decoder")
      .field("smoother", &self.smoother)
      .field("gate", &self.gate)
      .field("seg", &self.seg)
      .finish()
  }
}

/// One [`push`](Decoder::push)'s output: the causal decision and any range that
/// push finalized.
///
/// The two fields are the two planes — [`active`](Step::active) is the gate's
/// immediate, causal, never-retracted decision for the window just pushed, and
/// [`finalized`](Step::finalized) is a [`Range`] no future
/// input can change (or `None`). Only [`Decoder::push`] constructs one.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[must_use = "a discarded Step may drop a finalized range"]
pub struct Step {
  active: bool,
  finalized: Option<Range>,
}

impl Step {
  /// The gate's decision for the window just pushed: `true` when active.
  ///
  /// Immediate and causal — never retracted by later input. Batch parity is
  /// deliberately not promised for this plane; the finalized plane
  /// ([`finalized`](Step::finalized)) is the one with exact batch parity.
  #[must_use]
  pub const fn active(&self) -> bool {
    self.active
  }

  /// The range this push finalized, if any — one no future input can change.
  ///
  /// `Some` on the push that proves a run complete, `None` otherwise.
  /// Concatenating these across a run with the [`Decoder::finish`] tail
  /// reproduces the batch segmentation exactly.
  #[must_use]
  pub const fn finalized(&self) -> Option<Range> {
    self.finalized
  }
}
