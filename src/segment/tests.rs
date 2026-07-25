use std::{boxed::Box, cell::Cell, rc::Rc, vec, vec::Vec};

use super::{
  longest_run, runs, runs_sorted, Dwell, DwellState, Gate, GatePolicy, Hangover, HangoverState,
  Hysteresis, Range, SegmentOptions, SegmentTail, Segmenter, Threshold, Vote, VoteState,
};
use crate::{
  error::WinditError,
  plan::{Span, WindowOptions, WindowPlan},
  windowed::Windowed,
};

/// One `Windowed<f32>` per value, each covering a single element (window 1), so
/// element units and frame indices coincide.
fn seq(values: &[f32]) -> Vec<Windowed<f32>> {
  values
    .iter()
    .enumerate()
    .map(|(i, &v)| Windowed::new(v, Span::new(i, 1, 1)))
    .collect()
}

/// A default `SegmentOptions`: no merging, no minimum length.
fn plain() -> SegmentOptions {
  SegmentOptions::new()
}

/// xorshift64 — deterministic and dependency-free; the seed must be nonzero.
fn xorshift(state: &mut u64) -> u64 {
  let mut x = *state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *state = x;
  x
}

/// A pseudo-random `f32` in `[0, 1)` from the generator's next 24 bits.
fn next_unit(state: &mut u64) -> f32 {
  (xorshift(state) >> 40) as f32 / (1u32 << 24) as f32
}

/// The retained 0.1.2 two-pass geometry core, kept verbatim as the independent
/// differential oracle for the [`Segmenter`]-driven batch drivers.
///
/// This is the behaviour that shipped before the state machine: a first pass
/// builds geometrically-continuous runs, a second merges adjacent runs and drops
/// the short ones. It proves the new path is *equivalent* to what shipped, not
/// that either is *correct*.
mod oracle {
  use super::{Hysteresis, Range, SegmentOptions, Span, Vec, Windowed};

  /// 0.1.2 `merge_adjacent`, verbatim.
  fn merge_adjacent(ranges: Vec<Range>, merge_gap: usize) -> Vec<Range> {
    let mut out: Vec<Range> = Vec::with_capacity(ranges.len());
    for r in ranges {
      match out.last_mut() {
        Some(last) if r.start.saturating_sub(last.end) <= merge_gap => {
          last.end = last.end.max(r.end);
        }
        _ => out.push(r),
      }
    }
    out
  }

  /// 0.1.2 `runs_from_flags`, verbatim.
  fn runs_from_flags<I>(flags: I, opts: &SegmentOptions) -> Vec<Range>
  where
    I: Iterator<Item = (bool, Span)>,
  {
    let mut raw: Vec<Range> = Vec::new();
    let mut current: Option<Range> = None;
    for (accepted, span) in flags {
      if accepted {
        let (start, end) = (span.start(), span.end());
        match current {
          Some(run) if start > run.end => {
            raw.push(run);
            current = Some(Range::new(start, end));
          }
          Some(ref mut run) => run.end = run.end.max(end),
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

  /// 0.1.2 `runs`, verbatim (predicate over values).
  pub fn runs<V, F>(seq: &[Windowed<V>], predicate: F, opts: &SegmentOptions) -> Vec<Range>
  where
    F: Fn(&V) -> bool,
  {
    runs_from_flags(seq.iter().map(|w| (predicate(w.value()), w.span())), opts)
  }

  /// 0.1.2 `HysteresisSegment::segment`, verbatim (the fused two-pass path).
  pub fn hysteresis_segment(
    on: f32,
    off: f32,
    opts: &SegmentOptions,
    seq: &[Windowed<f32>],
  ) -> Vec<Range> {
    let gate = Hysteresis::new(on, off);
    runs_from_flags(
      seq.iter().scan(false, |active, w| {
        *active = gate.step(*active, *w.value());
        Some((*active, w.span()))
      }),
      opts,
    )
  }
}

/// Drive a fresh [`Segmenter`] over explicit `(active, span)` decisions,
/// collecting every finalized range plus the [`finish`](Segmenter::finish) tail
/// — the streaming counterpart of the batch drivers, over arbitrary flags.
fn drive_flags(flags: &[(bool, Span)], opts: &SegmentOptions) -> Vec<Range> {
  let mut seg = Segmenter::new(*opts);
  let mut out = Vec::new();
  for &(active, span) in flags {
    if let Some(r) = seg.push(active, span).unwrap() {
      out.push(r);
    }
  }
  out.extend(seg.finish());
  out
}

/// Drive the hysteresis gate through a [`Segmenter`], the new-side counterpart of
/// [`oracle::hysteresis_segment`].
fn seg_hysteresis(on: f32, off: f32, opts: &SegmentOptions, s: &[Windowed<f32>]) -> Vec<Range> {
  let gate = Hysteresis::new(on, off);
  let mut seg = Segmenter::new(*opts);
  let mut active = false;
  let mut out = Vec::new();
  for w in s {
    active = gate.step(active, *w.value());
    if let Some(r) = seg.push(active, w.span()).unwrap() {
      out.push(r);
    }
  }
  out.extend(seg.finish());
  out
}

// ── batch geometry (the #3 acceptance floor, now `Ok`-wrapped) ───────────────

#[test]
fn runs_contiguous_maps_to_element_range() {
  // Frames 1 and 2 are above 0.5, so one run spanning elements [1, 3).
  let s = seq(&[0.1, 0.9, 0.8, 0.2, 0.1]);
  let out = runs(&s, |&v| v > 0.5, &plain()).unwrap();
  assert_eq!(out, vec![Range::new(1, 3)]);
}

#[test]
fn runs_maps_multi_element_spans() {
  // Three width-4 windows; the middle one is below threshold. Ranges come from
  // span.start .. span.start + span.len, i.e. element units, not frame indices.
  let s = [
    Windowed::new(0.9, Span::new(0, 4, 4)),
    Windowed::new(0.1, Span::new(4, 4, 4)),
    Windowed::new(0.9, Span::new(8, 4, 4)),
  ];
  let out = runs(&s, |&v| v > 0.5, &plain()).unwrap();
  assert_eq!(out, vec![Range::new(0, 4), Range::new(8, 12)]);
}

#[test]
fn merge_gap_bridges_and_zero_keeps_separate() {
  // Frames 0, 2, 3 above threshold: raw runs [0,1) and [2,4), a one-element gap.
  let s = seq(&[0.9, 0.1, 0.9, 0.9]);

  let separate = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(0)).unwrap();
  assert_eq!(separate, vec![Range::new(0, 1), Range::new(2, 4)]);

  let bridged = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(1)).unwrap();
  assert_eq!(bridged, vec![Range::new(0, 4)]);
}

/// The gapped spans a `hop > window` plan produces: elements `2..5` are covered
/// by no span, so the two accepted windows must stay two runs and only
/// `merge_gap` may bridge them.
fn gapped_plan() -> Vec<Windowed<f32>> {
  let spans = WindowPlan::spans(&WindowOptions::new(2).with_hop(5), 7).unwrap();
  assert_eq!(
    spans
      .iter()
      .map(|s| (s.start(), s.len()))
      .collect::<Vec<_>>(),
    vec![(0, 2), (5, 2)],
    "the planner must produce the gapped geometry this regression is about"
  );
  spans.into_iter().map(|s| Windowed::new(0.9, s)).collect()
}

#[test]
fn gapped_spans_are_not_fused_into_one_run() {
  let s = gapped_plan();

  // Both windows are accepted, but they are not geometrically continuous:
  // fusing them would silently select the uncovered elements 2..5.
  assert_eq!(
    runs(&s, |&v| v > 0.5, &plain()).unwrap(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );

  // Bridging the 3-element gap is `merge_gap`'s decision alone.
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(2)).unwrap(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(3)).unwrap(),
    vec![Range::new(0, 7)]
  );
}

#[test]
fn gapped_spans_feed_min_len_longest_and_sorted() {
  let s = gapped_plan();

  // Each run is 2 elements, so a 3-element minimum drops both; fusing them into
  // a single 7-element run would keep one.
  assert!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(3))
      .unwrap()
      .is_empty()
  );
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(2)).unwrap(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );

  // Equal lengths, so the earliest wins the tie and the sort keeps input order.
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()).unwrap(),
    Some(Range::new(0, 2))
  );
  assert_eq!(
    runs_sorted(&s, |&v| v > 0.5, &plain()).unwrap(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
}

#[test]
fn gapped_spans_split_under_the_threshold_policy() {
  let s = gapped_plan();

  // The policy drives the same `Segmenter`, so it inherits the split.
  assert_eq!(
    Threshold::new(0.5).segment(&plain(), &s).unwrap(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
}

#[test]
fn min_len_drops_short_runs() {
  // Runs [0,1) (len 1) and [2,5) (len 3); min_len 2 drops the first.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9]);

  let kept_all = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(0)).unwrap();
  assert_eq!(kept_all, vec![Range::new(0, 1), Range::new(2, 5)]);

  let filtered = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(2)).unwrap();
  assert_eq!(filtered, vec![Range::new(2, 5)]);
}

#[test]
fn longest_run_picks_longest_and_ties_earliest() {
  // Runs of length 1, 3, 2: the length-3 run wins.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()).unwrap(),
    Some(Range::new(2, 5))
  );

  // Two length-2 runs: the earliest wins the tie.
  let tie = seq(&[0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    longest_run(&tie, |&v| v > 0.5, &plain()).unwrap(),
    Some(Range::new(0, 2))
  );
}

#[test]
fn runs_sorted_descending_stable() {
  // Lengths 1, 3, 2 sort to 3, 2, 1.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    runs_sorted(&s, |&v| v > 0.5, &plain()).unwrap(),
    vec![Range::new(2, 5), Range::new(6, 8), Range::new(0, 1),]
  );

  // Equal lengths keep input order (stable): [0,2) before [3,5).
  let tie = seq(&[0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]);
  assert_eq!(
    runs_sorted(&tie, |&v| v > 0.5, &plain()).unwrap(),
    vec![Range::new(0, 2), Range::new(3, 5), Range::new(6, 7),]
  );
}

#[test]
fn vad_shaped_twenty_frames() {
  // 20 speech probabilities; threshold 0.5 yields three speech ranges, the
  // middle one longest.
  let probs = [
    0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.2, 0.1, 0.3, 0.9, 0.95, 0.99, 0.8, 0.85, 0.7, 0.9, 0.2, 0.1,
    0.7, 0.8,
  ];
  let s = seq(&probs);
  let speech = runs(&s, |&v| v > 0.5, &plain()).unwrap();
  assert_eq!(
    speech,
    vec![Range::new(2, 6), Range::new(9, 16), Range::new(18, 20),]
  );
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()).unwrap(),
    Some(Range::new(9, 16))
  );

  // merge_gap 2 bridges only the 2-element gap before the final range (the
  // 3-element gap between the first two stays); min_len 5 then drops the short
  // leading [2, 6) range, leaving the merged tail.
  let merged = runs(
    &s,
    |&v| v > 0.5,
    &SegmentOptions::new().with_merge_gap(2).with_min_len(5),
  )
  .unwrap();
  assert_eq!(merged, vec![Range::new(9, 20)]);
}

#[test]
fn threshold_policy_includes_boundary() {
  // thr 0.5 admits values >= 0.5, so the boundary value 0.5 is in-segment.
  let s = seq(&[0.5, 0.9, 0.2, 0.6]);
  let policy = Threshold::new(0.5);
  assert_eq!(
    policy.segment(&plain(), &s).unwrap(),
    vec![Range::new(0, 2), Range::new(3, 4)]
  );
}

#[test]
fn empty_and_all_false_yield_none() {
  let empty: Vec<Windowed<f32>> = Vec::new();
  assert!(runs(&empty, |&v| v > 0.5, &plain()).unwrap().is_empty());
  assert_eq!(longest_run(&empty, |&v| v > 0.5, &plain()).unwrap(), None);
  assert!(runs_sorted(&empty, |&v| v > 0.5, &plain())
    .unwrap()
    .is_empty());

  let all_low = seq(&[0.1, 0.2, 0.0]);
  assert!(runs(&all_low, |&v| v > 0.5, &plain()).unwrap().is_empty());
  assert_eq!(longest_run(&all_low, |&v| v > 0.5, &plain()).unwrap(), None);
}

#[test]
fn overlapping_spans_union_and_merge() {
  // (i) Two accepted overlapping windows inside one run cover the union of their
  // spans: [0,4) then [2,6) -> [0,6). The second start (2) is at or before the
  // open run's end (4), so it extends the run rather than starting a new one.
  let s = [
    Windowed::new(0.9, Span::new(0, 4, 4)),
    Windowed::new(0.9, Span::new(2, 4, 4)),
  ];
  assert_eq!(
    runs(&s, |&v| v > 0.5, &plain()).unwrap(),
    vec![Range::new(0, 6)]
  );

  // (ii) A rejected window between two accepted overlapping ranges splits them
  // into raw runs [0,6) and [4,10); with merge_gap 0, the merge fold folds the
  // overlap (start 4 before end 6) to a zero gap through `saturating_sub` and
  // merges them into [0,10).
  let s = [
    Windowed::new(0.9, Span::new(0, 6, 6)),
    Windowed::new(0.1, Span::new(2, 6, 6)),
    Windowed::new(0.9, Span::new(4, 6, 6)),
  ];
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(0)).unwrap(),
    vec![Range::new(0, 10)]
  );
}

#[test]
fn merge_gap_applies_before_min_len() {
  // Raw runs [0,2) and [3,5), a one-element gap. merge_gap 1 bridges them first,
  // so min_len then sees one length-5 run: min_len 4 keeps it, min_len 6 drops
  // it. If min_len ran before the merge it would drop both length-2 runs.
  let s = seq(&[0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    runs(
      &s,
      |&v| v > 0.5,
      &SegmentOptions::new().with_merge_gap(1).with_min_len(4)
    )
    .unwrap(),
    vec![Range::new(0, 5)]
  );
  assert!(runs(
    &s,
    |&v| v > 0.5,
    &SegmentOptions::new().with_merge_gap(1).with_min_len(6)
  )
  .unwrap()
  .is_empty());
}

// ── Range / options unit tests (unchanged) ───────────────────────────────────

#[test]
fn range_len_and_is_empty() {
  let r = Range::new(2, 5);
  assert_eq!(r.len(), 3);
  assert!(!r.is_empty());

  let empty = Range::new(4, 4);
  assert_eq!(empty.len(), 0);
  assert!(empty.is_empty());
}

#[test]
fn inverted_range_saturates_to_empty() {
  // Both constructors reject an inverted range in every build, so this one is
  // built through the struct literal — reachable here because `tests` is a child
  // of the defining module — to stand in for an in-crate write to `end` that
  // moved it downward. It pins `len` saturating to zero rather than underflowing
  // into a near-`usize::MAX` length.
  let inverted = Range { start: 10, end: 5 };
  assert_eq!(inverted.len(), 0);
  assert!(inverted.is_empty());
}

#[test]
#[should_panic(expected = "start <= end")]
fn range_new_rejects_an_inverted_range_in_every_build() {
  let _ = Range::new(10, 5);
}

#[test]
fn range_try_new_reports_an_inverted_range_as_a_typed_error() {
  assert_eq!(
    Range::try_new(10, 5),
    Err(WinditError::InvalidRange { start: 10, end: 5 })
  );
  assert_eq!(Range::try_new(4, 4).unwrap(), Range::new(4, 4));
}

#[test]
fn span_at_the_usize_boundary_never_produces_an_inverted_run() {
  // A span starting at `usize::MAX` used to wrap `start + len` to `0` here and
  // yield `Range { start: usize::MAX, end: 0 }`; it is now unconstructible.
  assert!(Span::try_new(usize::MAX, 1, 1).is_err());

  // The largest span that does exist segments to its exact element range.
  let span = Span::try_new(usize::MAX - 1, 1, 1).unwrap();
  let s = [Windowed::new(0.9f32, span)];
  let out = runs(&s, |&v| v > 0.5, &plain()).unwrap();
  assert_eq!(out, vec![Range::new(usize::MAX - 1, usize::MAX)]);
  assert_eq!(out[0].len(), 1);
}

#[test]
fn segment_options_builder_and_default() {
  let o = SegmentOptions::new().with_min_len(3).with_merge_gap(2);
  assert_eq!(o.min_len(), 3);
  assert_eq!(o.merge_gap(), 2);

  let d = SegmentOptions::default();
  assert_eq!(d.min_len(), 0);
  assert_eq!(d.merge_gap(), 0);
  assert_eq!(d, SegmentOptions::new());
}

#[test]
fn threshold_non_finite_scores_and_thresholds() {
  // `thr = -inf` admits every finite score and both infinities, but a NaN score
  // is still excluded because `NaN >= -inf` is false, so index 1 drops out.
  let s = seq(&[0.1, f32::NAN, 0.5]);
  assert_eq!(
    Threshold::new(f32::NEG_INFINITY)
      .segment(&plain(), &s)
      .unwrap(),
    vec![Range::new(0, 1), Range::new(2, 3)]
  );

  // `thr = NaN`: `value >= NaN` is never true, so nothing is in-segment.
  assert!(Threshold::new(f32::NAN)
    .segment(&plain(), &s)
    .unwrap()
    .is_empty());

  // `thr = +inf` admits only a `+inf` score.
  let s = seq(&[f32::INFINITY, 1.0]);
  assert_eq!(
    Threshold::new(f32::INFINITY).segment(&plain(), &s).unwrap(),
    vec![Range::new(0, 1)]
  );
}

// ── gate value semantics (moved from smooth, now a typed bool decision) ──────

/// Drive a fresh gate over `values` (each a unit span), collecting the decision
/// at every step.
fn gate_bools<P: GatePolicy<f32>>(policy: &P, values: &[f32]) -> Vec<bool> {
  let mut g = policy.gate();
  values
    .iter()
    .enumerate()
    .map(|(i, &v)| g.push(&Windowed::new(v, Span::new(i, 1, 1))).unwrap())
    .collect()
}

#[test]
fn hysteresis_gate_latches_and_holds() {
  // on=0.6, off=0.3: 0.1 off, 0.7 on, 0.5 hold(on), 0.2 off, 0.6 on.
  let out = gate_bools(&Hysteresis::new(0.6, 0.3), &[0.1, 0.7, 0.5, 0.2, 0.6]);
  assert_eq!(out, vec![false, true, true, false, true]);
}

#[test]
fn hysteresis_gate_holds_at_off_boundary_instead_of_turning_off() {
  // 0.7 latches on; a value exactly at `off` (0.3) holds that on state rather than
  // turning off, twice in a row; only the strictly-below 0.2 turns it off. The
  // strict-below boundary real VAD consumers rely on.
  let out = gate_bools(&Hysteresis::new(0.6, 0.3), &[0.7, 0.3, 0.3, 0.2]);
  assert_eq!(out, vec![true, true, true, false]);
}

#[test]
fn hysteresis_gate_exact_on_boundary_activates() {
  // The turn-on test is `value >= on`, so a value exactly at `on` activates.
  assert_eq!(gate_bools(&Hysteresis::new(0.6, 0.3), &[0.6]), vec![true]);
}

#[test]
fn hysteresis_gate_on_below_off_degrades_to_single_threshold() {
  // With `on < off` the turn-on test wins, so the gate degrades to the pointwise
  // single threshold `value >= on` (here 0.3): 0.4 on, 0.5 on, 0.2 off, 0.35 on.
  let out = gate_bools(&Hysteresis::new(0.3, 0.6), &[0.4, 0.5, 0.2, 0.35]);
  assert_eq!(out, vec![true, true, false, true]);
}

#[test]
fn hysteresis_gate_nan_score_holds_state() {
  // Both gate comparisons are false for NaN, so a NaN score holds the current
  // state — including the initial off state at index 0.
  let out = gate_bools(
    &Hysteresis::new(0.6, 0.3),
    &[f32::NAN, 0.7, f32::NAN, 0.2, f32::NAN],
  );
  assert_eq!(out, vec![false, true, true, false, false]);
}

#[test]
fn hysteresis_gate_infinite_scores_latch_and_release() {
  // `+inf >= on` activates; `-inf < off` releases; the finite holds between.
  let out = gate_bools(
    &Hysteresis::new(0.6, 0.3),
    &[f32::INFINITY, 0.5, f32::NEG_INFINITY, 0.5],
  );
  assert_eq!(out, vec![true, true, false, false]);
}

#[test]
fn hysteresis_gate_nan_thresholds_fail_closed_or_never_release() {
  // `on = NaN`: `value >= on` is never true, so the gate can never activate.
  let out = gate_bools(&Hysteresis::new(f32::NAN, 0.3), &[0.7, f32::INFINITY, 0.1]);
  assert_eq!(out, vec![false, false, false]);

  // `off = NaN`: `value < off` is never true, so once on the gate never releases.
  let out = gate_bools(
    &Hysteresis::new(0.6, f32::NAN),
    &[0.7, 0.1, f32::NEG_INFINITY],
  );
  assert_eq!(out, vec![true, true, true]);
}

#[test]
fn hysteresis_gate_infinite_thresholds() {
  // `on = -inf`: every non-NaN score activates (`-inf` via `-inf >= -inf`, and a
  // NaN in between holds the on state).
  let out = gate_bools(
    &Hysteresis::new(f32::NEG_INFINITY, 0.3),
    &[f32::NEG_INFINITY, f32::NAN, 0.1],
  );
  assert_eq!(out, vec![true, true, true]);

  // `on = +inf`: only a `+inf` score activates; the finite below `off` releases.
  let out = gate_bools(
    &Hysteresis::new(f32::INFINITY, 0.3),
    &[1.0, f32::INFINITY, 0.5, 0.2],
  );
  assert_eq!(out, vec![false, true, true, false]);

  // `off = -inf`: `value < off` is never true, so once on the gate never releases
  // — even a `-inf` score holds.
  let out = gate_bools(
    &Hysteresis::new(0.6, f32::NEG_INFINITY),
    &[0.7, f32::NEG_INFINITY, 0.1, -1e30],
  );
  assert_eq!(out, vec![true, true, true, true]);
}

#[test]
fn hysteresis_gate_new_exposes_thresholds() {
  let gate = Hysteresis::new(0.6, 0.3);
  assert_eq!((gate.on(), gate.off()), (0.6, 0.3));
}

#[test]
fn hysteresis_gate_reset_clears_the_latch() {
  let mut g = Hysteresis::new(0.6, 0.3).gate();
  assert!(g.push(&Windowed::new(0.7, Span::new(0, 1, 1))).unwrap()); // latched on
  g.reset();
  // A held value (0.5) now sees the initial off state rather than the latch.
  assert!(!g.push(&Windowed::new(0.5, Span::new(1, 1, 1))).unwrap());
}

#[test]
fn threshold_gate_membership_is_raw_ieee() {
  // `value >= thr`, raw IEEE: the boundary value is in; a NaN score never is.
  assert_eq!(
    gate_bools(&Threshold::new(0.5), &[0.5, 0.9, 0.2, 0.6]),
    vec![true, true, false, true]
  );
  // `thr = -inf` admits every finite score and both infinities but not NaN.
  assert_eq!(
    gate_bools(
      &Threshold::new(f32::NEG_INFINITY),
      &[0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
    ),
    vec![true, false, true, true]
  );
  // `thr = NaN` accepts nothing; a threshold state resets to nothing pending.
  assert_eq!(
    gate_bools(&Threshold::new(f32::NAN), &[0.9, f32::INFINITY]),
    vec![false, false]
  );
  assert_eq!(Threshold::new(0.5).thr(), 0.5);
}

// ── monotonic-span contract ──────────────────────────────────────────────────

#[test]
fn equal_starts_are_admitted() {
  // A degenerate but deterministic case: repeated equal starts are allowed (the
  // contract only rejects a *strictly* backward start).
  let mut seg = Segmenter::new(plain());
  assert_eq!(seg.push(true, Span::new(4, 2, 2)).unwrap(), None);
  assert_eq!(seg.push(true, Span::new(4, 2, 2)).unwrap(), None);
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(4, 6)]);
}

#[test]
fn backward_start_is_reported_and_leaves_state_unchanged() {
  let mut seg = Segmenter::new(plain());
  assert_eq!(seg.push(true, Span::new(5, 2, 2)).unwrap(), None);
  // A strictly earlier start with no discontinuity is a checked violation.
  assert_eq!(
    seg.push(true, Span::new(0, 2, 2)),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
  // The offending push was a no-op: the run opened at 5 is intact, and a valid
  // continuation still works.
  assert_eq!(seg.push(false, Span::new(7, 2, 2)).unwrap(), None);
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(5, 7)]);
}

#[test]
fn runs_reports_non_monotonic_input() {
  // 0.1.x returned "deterministic but unspecified" here; 0.2.0 upgrades that to
  // a checked contract.
  let s = [
    Windowed::new(0.9, Span::new(5, 2, 2)),
    Windowed::new(0.9, Span::new(0, 2, 2)),
  ];
  assert_eq!(
    runs(&s, |&v| v > 0.5, &plain()),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
  assert_eq!(
    runs_sorted(&s, |&v| v > 0.5, &plain()),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
}

#[test]
fn discontinuity_re_arms_the_monotonicity_check() {
  // A declared break lets span positions restart: the backward start after it is
  // accepted, and no run bridges the break.
  let mut seg = Segmenter::new(SegmentOptions::new().with_merge_gap(100));
  assert_eq!(seg.push(true, Span::new(50, 2, 2)).unwrap(), None);
  assert_eq!(
    seg.discontinuity().collect::<Vec<_>>(),
    vec![Range::new(50, 52)]
  );
  // Restarting at 0 is now fine, and merge_gap never bridged the epoch break.
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None);
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(0, 2)]);
}

// ── Segmenter transition rules ───────────────────────────────────────────────

#[test]
fn accepted_continuous_span_extends_and_emits_nothing() {
  let mut seg = Segmenter::new(plain());
  assert_eq!(seg.push(true, Span::new(0, 4, 4)).unwrap(), None); // open [0,4)
  assert_eq!(seg.push(true, Span::new(2, 4, 4)).unwrap(), None); // start 2 <= 4 → extend [0,6)
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(0, 6)]);
}

#[test]
fn geometric_gap_starts_a_new_run_and_does_not_bridge() {
  // start beyond the open run's end starts a new run; default opts do not bridge.
  let mut seg = Segmenter::new(plain());
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None);
  // [5,7) is beyond [0,2)'s end: the previous run closes, but is finalized only
  // once this run's start clears the (zero) gap horizon — here immediately.
  assert_eq!(
    seg.push(true, Span::new(5, 2, 2)).unwrap(),
    Some(Range::new(0, 2))
  );
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(5, 7)]);
}

#[test]
fn rejected_span_closes_the_open_run() {
  let mut seg = Segmenter::new(plain());
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None);
  // The rejected span closes [0,2) into pending; with a zero gap horizon the
  // rejected start (3) already clears it, so [0,2) finalizes now.
  assert_eq!(
    seg.push(false, Span::new(3, 2, 2)).unwrap(),
    Some(Range::new(0, 2))
  );
  assert_eq!(seg.finish().len(), 0);
}

#[test]
fn a_short_run_that_later_merges_survives_min_len() {
  // Rule 7: min_len is applied only at finalization, never at close. Raw runs
  // [0,2) and [3,5), a one-element gap; merge_gap 1 bridges them so min_len sees
  // one length-5 run. A close-time filter would have dropped [0,2) early.
  let opts = SegmentOptions::new().with_merge_gap(1).with_min_len(4);
  let flags = [
    (true, Span::new(0, 2, 2)),
    (false, Span::new(2, 2, 2)),
    (true, Span::new(3, 2, 2)),
  ];
  assert_eq!(drive_flags(&flags, &opts), vec![Range::new(0, 5)]);
}

#[test]
fn early_finalization_emits_at_the_first_span_beyond_the_gap_horizon() {
  // merge_gap 1: a span whose start clears pending.end + 1 finalizes it eagerly.
  let mut seg = Segmenter::new(SegmentOptions::new().with_merge_gap(1));
  // open [0,2)
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None);
  // rejected at start 3: gap(3,2) = 1 is within the horizon, so [0,2) waits.
  assert_eq!(seg.push(false, Span::new(3, 2, 2)).unwrap(), None);
  // start 5: gap(5,2) = 3 > 1 clears the horizon → [0,2) finalizes here.
  assert_eq!(
    seg.push(false, Span::new(5, 1, 1)).unwrap(),
    Some(Range::new(0, 2))
  );
}

#[test]
fn unbounded_merge_gap_keeps_o1_state_and_defers_to_finish() {
  // With a huge merge_gap nothing ever clears the horizon, so `pending` only
  // widens (never a growing collection) and emission defers to `finish`. A long
  // input with many gaps stays a single, ever-widening pending range.
  let opts = SegmentOptions::new().with_merge_gap(usize::MAX);
  let mut seg = Segmenter::new(opts);
  for i in 0..1_000usize {
    // Accepted unit spans every 10 elements: 1000 raw runs, all folded into one.
    assert_eq!(seg.push(true, Span::new(i * 10, 1, 1)).unwrap(), None);
  }
  let tail: Vec<_> = seg.finish().collect();
  assert_eq!(tail, vec![Range::new(0, 9991)]);
}

#[test]
fn extend_then_far_new_run_matches_batch_merge() {
  // A run that keeps extending past the gap horizon must still merge with the
  // pending accumulator when it finally closes — a case where finalizing the
  // pending on a raw span start (before closing the open run) would wrongly
  // split it. Spans: [0,10) then [13,14) (new run within gap 5) then [14,25)
  // (extends the new run past the horizon) then [30,31). merge_gap 5 merges the
  // lot into [0,31).
  let opts = SegmentOptions::new().with_merge_gap(5);
  let flags = [
    (true, Span::new(0, 10, 10)),
    (true, Span::new(13, 1, 1)),
    (true, Span::new(14, 11, 11)),
    (true, Span::new(30, 1, 1)),
  ];
  let out = drive_flags(&flags, &opts);
  assert_eq!(out, vec![Range::new(0, 31)]);
  // And it is exactly what the retained 0.1.2 oracle produces.
  let s: Vec<Windowed<f32>> = flags
    .iter()
    .map(|&(_, span)| Windowed::new(0.9f32, span))
    .collect();
  assert_eq!(out, oracle::runs(&s, |&v| v > 0.5, &opts));
}

#[test]
fn far_new_runs_finalize_the_predecessor_eagerly() {
  // With merge_gap 1, each accepted run starts far enough past the last to clear
  // the horizon, so the predecessor finalizes on the push that opens its
  // successor and finish is left with only the last run.
  let opts = SegmentOptions::new().with_merge_gap(1);
  let mut seg = Segmenter::new(opts);
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None); // open [0,2)
  assert_eq!(
    seg.push(true, Span::new(10, 2, 2)).unwrap(),
    Some(Range::new(0, 2)) // gap(10,2)=8 > 1 → [0,2) finalizes; open [10,12)
  );
  assert_eq!(
    seg.push(true, Span::new(20, 2, 2)).unwrap(),
    Some(Range::new(10, 12)) // [10,12) finalizes; open [20,22)
  );
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(20, 22)]);
}

#[test]
fn finish_emits_the_single_merged_pending() {
  // Eager early-finalization keeps the resting invariant "an open run within
  // merge_gap of pending folds into it", so `finish` never leaves both a
  // non-merging pending and an open run — it emits the one merged pending.
  let opts = SegmentOptions::new().with_merge_gap(2);
  let mut seg = Segmenter::new(opts);
  // Three runs each within the gap of the previous → one merged range at finish.
  seg.push(true, Span::new(0, 2, 2)).unwrap();
  seg.push(false, Span::new(2, 2, 2)).unwrap();
  seg.push(true, Span::new(4, 2, 2)).unwrap(); // gap(4,2)=2 <= 2 → folds
  seg.push(false, Span::new(6, 2, 2)).unwrap();
  seg.push(true, Span::new(7, 2, 2)).unwrap(); // gap(7,6)=1 <= 2 → folds
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(0, 9)]);
}

#[test]
fn finish_and_discontinuity_emit_at_most_one_range_over_all_small_inputs() {
  // Exhaustive check of the resting invariant: over every flag pattern and a
  // grid of geometries and options, a terminal drain yields at most one range.
  // The public `SegmentTail` still admits two (the design's upper bound); this
  // pins the tighter behaviour the eager finalization actually delivers.
  let geometries = [(1usize, 1usize), (2, 2), (2, 3), (2, 1)]; // (span_len, hop)
  let gaps = [0usize, 1, 3, usize::MAX];
  let min_lens = [0usize, 1, 3];
  for len in 0..=8usize {
    for bits in 0u32..(1 << len) {
      for &(span_len, hop) in &geometries {
        let window = span_len;
        for &merge_gap in &gaps {
          for &min_len in &min_lens {
            let opts = SegmentOptions::new()
              .with_merge_gap(merge_gap)
              .with_min_len(min_len);
            let flags: Vec<(bool, Span)> = (0..len)
              .map(|i| ((bits >> i) & 1 == 1, Span::new(i * hop, span_len, window)))
              .collect();
            let mut seg = Segmenter::new(opts);
            for &(active, span) in &flags {
              seg.push(active, span).unwrap();
            }
            assert!(
              seg.clone().finish().len() <= 1,
              "finish emitted >1: bits={bits:b} hop={hop} span_len={span_len} opts={opts:?}"
            );
            assert!(
              seg.discontinuity().len() <= 1,
              "discontinuity emitted >1: bits={bits:b} hop={hop} span_len={span_len} opts={opts:?}"
            );
          }
        }
      }
    }
  }
}

#[test]
fn reset_drops_pending_output() {
  let mut seg = Segmenter::new(plain());
  seg.push(true, Span::new(0, 2, 2)).unwrap();
  seg.reset();
  // Pending run [0,2) was dropped; the segmenter is fresh (a backward start is
  // now fine because the monotonicity check was re-armed too).
  assert_eq!(seg.push(true, Span::new(0, 2, 2)).unwrap(), None);
  assert_eq!(seg.finish().collect::<Vec<_>>(), vec![Range::new(0, 2)]);
}

// ── SegmentTail ──────────────────────────────────────────────────────────────

#[test]
fn segment_tail_is_exact_sized_and_bounded() {
  // Two ranges, in order.
  let opts = SegmentOptions::new().with_merge_gap(1);
  let mut seg = Segmenter::new(opts);
  seg.push(true, Span::new(0, 2, 2)).unwrap();
  seg.push(false, Span::new(1, 1, 1)).unwrap();
  seg.push(true, Span::new(9, 2, 2)).unwrap();
  let mut tail = seg.finish();
  assert_eq!(tail.len(), 1);
  assert_eq!(tail.size_hint(), (1, Some(1)));
  assert_eq!(tail.next(), Some(Range::new(9, 11)));
  assert_eq!(tail.len(), 0);
  assert_eq!(tail.next(), None);

  // Empty tail.
  let empty = Segmenter::new(plain()).finish();
  assert_eq!(empty.len(), 0);
  assert_eq!(empty.count(), 0);

  // min_len can drop a finish emission, so the tail is shorter than the pending
  // count.
  let mut seg = Segmenter::new(SegmentOptions::new().with_min_len(5));
  seg.push(true, Span::new(0, 2, 2)).unwrap();
  assert_eq!(seg.finish().len(), 0);
}

#[test]
fn segment_tail_holds_two_ranges_in_order() {
  // `finish` never fills both slots today, but the type admits two (the design's
  // upper bound). Exercise the two-slot path directly through the private
  // constructor, reachable because `tests` is a child of the defining module.
  let tail = SegmentTail::new(Some(Range::new(0, 2)), Some(Range::new(5, 7)));
  assert_eq!(tail.len(), 2);
  assert_eq!(
    tail.collect::<Vec<_>>(),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );

  // An absent leading slot compacts, so iteration still yields the present one.
  let tail = SegmentTail::new(None, Some(Range::new(5, 7)));
  assert_eq!(tail.len(), 1);
  assert_eq!(tail.collect::<Vec<_>>(), vec![Range::new(5, 7)]);
}

// ── differential vs the retained 0.1.2 oracle ────────────────────────────────

/// The fixed geometry grid: adjacent unit spans, the off-boundary hold, the
/// gapped plan under each option, overlapping spans, non-finite scores, and the
/// degenerate `on < off` / NaN-threshold configurations.
fn hysteresis_grid() -> Vec<(f32, f32, SegmentOptions, Vec<Windowed<f32>>)> {
  let overlap_union = vec![
    Windowed::new(0.9, Span::new(0, 4, 4)),
    Windowed::new(0.9, Span::new(2, 4, 4)),
  ];
  let overlap_split = vec![
    Windowed::new(0.9, Span::new(0, 6, 6)),
    Windowed::new(0.1, Span::new(2, 6, 6)),
    Windowed::new(0.9, Span::new(4, 6, 6)),
  ];
  vec![
    (0.6, 0.3, plain(), seq(&[0.1, 0.7, 0.5, 0.2, 0.6])),
    (0.6, 0.3, plain(), seq(&[0.7, 0.3, 0.3, 0.2])),
    (0.6, 0.3, plain(), gapped_plan()),
    (
      0.6,
      0.3,
      SegmentOptions::new().with_merge_gap(2),
      gapped_plan(),
    ),
    (
      0.6,
      0.3,
      SegmentOptions::new().with_merge_gap(3),
      gapped_plan(),
    ),
    (
      0.6,
      0.3,
      SegmentOptions::new().with_min_len(2),
      gapped_plan(),
    ),
    (
      0.6,
      0.3,
      SegmentOptions::new().with_min_len(3),
      gapped_plan(),
    ),
    (0.6, 0.3, plain(), overlap_union),
    (
      0.6,
      0.3,
      SegmentOptions::new().with_merge_gap(0),
      overlap_split,
    ),
    (0.6, 0.3, plain(), seq(&[f32::NAN, 0.7, f32::NAN, 0.2])),
    (
      0.6,
      0.3,
      plain(),
      seq(&[f32::INFINITY, 0.5, f32::NEG_INFINITY]),
    ),
    (0.3, 0.6, plain(), seq(&[0.4, 0.5, 0.2, 0.35])),
    (f32::NAN, 0.3, plain(), seq(&[0.7, 0.1, 0.1])),
    (0.6, f32::NAN, plain(), seq(&[0.7, 0.1, 0.1])),
  ]
}

#[test]
fn segmenter_matches_oracle_hysteresis_on_the_fixed_grid() {
  for (on, off, opts, s) in hysteresis_grid() {
    let new = seg_hysteresis(on, off, &opts, &s);
    let reference = oracle::hysteresis_segment(on, off, &opts, &s);
    assert_eq!(new, reference, "on={on} off={off} opts={opts:?}");
    // The reshaped provided method drives the same gate → Segmenter.
    assert_eq!(
      Hysteresis::new(on, off).segment(&opts, &s).unwrap(),
      reference,
      "Hysteresis::segment on={on} off={off} opts={opts:?}"
    );
  }
}

#[test]
fn segmenter_matches_oracle_on_randomized_finite_and_non_finite_inputs() {
  // ~200 deterministic pseudo-random cases exercising both the threshold and the
  // hysteresis paths against the retained 0.1.2 oracle, over the full geometry
  // grid (unit / adjacent / gapped / overlapping), non-finite scores and
  // thresholds included, in ascending span order by construction.
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 257) as usize;
    let window = (xorshift(&mut state) % 8 + 1) as usize;
    let geometry = xorshift(&mut state) % 4;
    let hop = match geometry {
      0 => 1,
      1 => window,
      2 => window + 1 + (xorshift(&mut state) % 4) as usize,
      _ => 1 + (xorshift(&mut state) % window as u64) as usize,
    };
    let span_len = if geometry == 0 { 1 } else { window };

    // Occasionally inject non-finite scores so the IEEE tables are exercised.
    let mut scores: Vec<f32> = Vec::with_capacity(n);
    let s: Vec<Windowed<f32>> = (0..n)
      .map(|i| {
        let v = match xorshift(&mut state) % 16 {
          0 => f32::NAN,
          1 => f32::INFINITY,
          2 => f32::NEG_INFINITY,
          _ => next_unit(&mut state),
        };
        scores.push(v);
        Windowed::new(v, Span::new(i * hop, span_len, window))
      })
      .collect();

    let threshold = |state: &mut u64| -> f32 {
      match xorshift(state) % 8 {
        0 => f32::NAN,
        1 if !scores.is_empty() => scores[(xorshift(state) % scores.len() as u64) as usize],
        _ => next_unit(state),
      }
    };
    let on = threshold(&mut state);
    let off = threshold(&mut state);
    let thr = threshold(&mut state);

    let merge_gap = [0usize, 1, 3, usize::MAX][(xorshift(&mut state) % 4) as usize];
    let min_len = [0usize, 2, 5][(xorshift(&mut state) % 3) as usize];
    let opts = SegmentOptions::new()
      .with_merge_gap(merge_gap)
      .with_min_len(min_len);

    // Threshold path: the new fallible driver against the 0.1.2 oracle.
    let new_thr = runs(&s, |&v| v >= thr, &opts).unwrap();
    let ref_thr = oracle::runs(&s, |&v| v >= thr, &opts);
    assert_eq!(
      new_thr, ref_thr,
      "threshold: n={n} window={window} hop={hop} thr={thr} opts={opts:?}"
    );

    // Hysteresis path: the Segmenter-driven gate against the 0.1.2 two-pass.
    let new_hy = seg_hysteresis(on, off, &opts, &s);
    let ref_hy = oracle::hysteresis_segment(on, off, &opts, &s);
    assert_eq!(
      new_hy, ref_hy,
      "hysteresis: n={n} window={window} hop={hop} on={on} off={off} opts={opts:?}"
    );

    // The reshaped `GatePolicy::segment` provided methods drive the same gate
    // through the same `Segmenter`, so they must match the oracle too — the P2
    // parity gate that batch `segment` still equals what 0.1.2 shipped.
    assert_eq!(
      Threshold::new(thr).segment(&opts, &s).unwrap(),
      ref_thr,
      "Threshold::segment: n={n} thr={thr} opts={opts:?}"
    );
    assert_eq!(
      Hysteresis::new(on, off).segment(&opts, &s).unwrap(),
      ref_hy,
      "Hysteresis::segment: n={n} on={on} off={off} opts={opts:?}"
    );
  }
}

// ── chunk-partition invariance ───────────────────────────────────────────────

#[test]
fn chunk_partition_invariance_over_random_splits() {
  // The core property: feeding the same decisions to one Segmenter as a single
  // batch versus split into arbitrary chunks (push… then a single finish) yields
  // identical finalized ranges. There is no chunk-level logic anywhere, so this
  // holds by construction; the property test guards it against regression.
  let mut state: u64 = 0x1357_9BDF_2468_ACE0;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 200) as usize;
    let window = (xorshift(&mut state) % 6 + 1) as usize;
    let geometry = xorshift(&mut state) % 4;
    let hop = match geometry {
      0 => 1,
      1 => window,
      2 => window + 1 + (xorshift(&mut state) % 3) as usize,
      _ => 1 + (xorshift(&mut state) % window as u64) as usize,
    };
    let span_len = if geometry == 0 { 1 } else { window };
    let merge_gap = [0usize, 1, 4, usize::MAX][(xorshift(&mut state) % 4) as usize];
    let min_len = [0usize, 3][(xorshift(&mut state) % 2) as usize];
    let opts = SegmentOptions::new()
      .with_merge_gap(merge_gap)
      .with_min_len(min_len);

    let flags: Vec<(bool, Span)> = (0..n)
      .map(|i| {
        let active = !xorshift(&mut state).is_multiple_of(3);
        (active, Span::new(i * hop, span_len, window))
      })
      .collect();

    // Whole-batch drive.
    let whole = drive_flags(&flags, &opts);

    // Chunked drive across the same single Segmenter: split at random points,
    // push within each chunk, and finish exactly once at the end.
    let mut seg = Segmenter::new(opts);
    let mut chunked = Vec::new();
    let mut i = 0;
    while i < flags.len() {
      let remaining = flags.len() - i;
      let chunk = 1 + (xorshift(&mut state) as usize % remaining);
      for &(active, span) in &flags[i..i + chunk] {
        if let Some(r) = seg.push(active, span).unwrap() {
          chunked.push(r);
        }
      }
      i += chunk;
    }
    chunked.extend(seg.finish());

    assert_eq!(whole, chunked, "n={n} hop={hop} opts={opts:?}");

    // And both equal the batch driver over the value-encoded flags.
    let values: Vec<Windowed<f32>> = flags
      .iter()
      .map(|&(active, span)| Windowed::new(if active { 1.0f32 } else { 0.0 }, span))
      .collect();
    let batch = runs(&values, |&v| v >= 0.5, &opts).unwrap();
    assert_eq!(whole, batch, "batch parity: n={n} hop={hop} opts={opts:?}");
  }
}

// ── Dwell / Hangover combinators ─────────────────────────────────────────────

/// A test-local [`Gate`] that replays a fixed flag script, one `bool` per push,
/// ignoring the pushed value and span entirely — both a scan-oracle driver for
/// the combinators and, via `Gate<()>`, the non-`f32` `V` witness that pins
/// `Dwell`/`Hangover`'s value-freeness (driven over `Windowed<()>`).
///
/// Deliberately concrete over `V = ()` rather than a blanket `impl<V>`: a
/// blanket impl would make `DwellState<ScriptGate>: Gate<V>` hold for every
/// `V` simultaneously, and calling `reset`/`discontinuity` — whose signatures
/// name no `V` — on such a value is genuinely ambiguous (multiple applicable
/// trait impls, none selectable from the call). Pinning `V = ()` here keeps
/// every call in this file unambiguous without a turbofish.
#[derive(Clone, Debug, PartialEq)]
struct ScriptGate {
  script: Vec<bool>,
  idx: usize,
}

impl ScriptGate {
  fn new(script: &[bool]) -> Self {
    Self {
      script: script.to_vec(),
      idx: 0,
    }
  }
}

impl Gate<()> for ScriptGate {
  fn push(&mut self, _w: &Windowed<()>) -> Result<bool, WinditError> {
    let v = self.script.get(self.idx).copied().unwrap_or(false);
    self.idx += 1;
    Ok(v)
  }

  fn reset(&mut self) {
    self.idx = 0;
  }
}

/// A test-local [`Gate<f32>`] recording whether [`reset`](Gate::reset) or
/// [`discontinuity`](Gate::discontinuity) fired — the regression probe for
/// conformance flag F3 (a wrapper or `Box` forwarding impl that fell back to
/// the trait default would call `reset` here instead of `discontinuity`).
///
/// The counters live behind a shared `Rc<Cell<_>>` so a test can keep a handle
/// after moving the probe into a wrapper or a `Box<dyn Gate<f32>>`, where the
/// concrete `ProbeGate` is no longer nameable to inspect directly. Concrete
/// over `f32`, for the same ambiguity reason as [`ScriptGate`].
#[derive(Clone, Debug)]
struct ProbeGate {
  active: bool,
  reset_calls: Rc<Cell<usize>>,
  discontinuity_calls: Rc<Cell<usize>>,
}

impl ProbeGate {
  /// A fresh probe plus the two counter handles the test retains.
  fn new(active: bool) -> (Self, Rc<Cell<usize>>, Rc<Cell<usize>>) {
    let reset_calls = Rc::new(Cell::new(0));
    let discontinuity_calls = Rc::new(Cell::new(0));
    (
      Self {
        active,
        reset_calls: reset_calls.clone(),
        discontinuity_calls: discontinuity_calls.clone(),
      },
      reset_calls,
      discontinuity_calls,
    )
  }
}

impl Gate<f32> for ProbeGate {
  fn push(&mut self, _w: &Windowed<f32>) -> Result<bool, WinditError> {
    Ok(self.active)
  }

  fn reset(&mut self) {
    self.reset_calls.set(self.reset_calls.get() + 1);
  }

  fn discontinuity(&mut self) {
    self
      .discontinuity_calls
      .set(self.discontinuity_calls.get() + 1);
  }
}

/// Reference implementation of [`Dwell`]'s semantics (the type's own doc),
/// computed directly over `(inner_flag, span)` pairs rather than through
/// [`DwellState`] — the independent oracle the exhaustive sweep checks the
/// real state machine against.
fn dwell_oracle(flags: &[(bool, Span)], confirm: usize) -> Vec<bool> {
  // `(origin, horizon)`: the run's first start and the largest end seen since.
  // The horizon is a max fold, exactly as `HangoverState`'s is — a later span
  // may end before an earlier one, and comparing the current span's end alone
  // would let a confirmed gate deactivate mid-activation.
  let mut run: Option<(usize, usize)> = None;
  let mut out = Vec::with_capacity(flags.len());
  for &(active, span) in flags {
    if active {
      let (origin, horizon) = match run {
        Some((origin, horizon)) => (origin, horizon.max(span.end())),
        None => (span.start(), span.end()),
      };
      run = Some((origin, horizon));
      out.push(horizon.saturating_sub(origin) >= confirm);
    } else {
      run = None;
      out.push(false);
    }
  }
  out
}

/// Reference implementation of [`Hangover`]'s semantics, the `Hangover`
/// counterpart of [`dwell_oracle`].
fn hangover_oracle(flags: &[(bool, Span)], hold: usize) -> Vec<bool> {
  let mut last_yes_end: Option<usize> = None;
  let mut out = Vec::with_capacity(flags.len());
  for &(active, span) in flags {
    if active {
      last_yes_end = Some(last_yes_end.map_or(span.end(), |end| end.max(span.end())));
      out.push(true);
    } else {
      out.push(match last_yes_end {
        Some(end) => span.start().saturating_sub(end) < hold,
        None => false,
      });
    }
  }
  out
}

/// Brute-force reference for [`Vote`]'s N-of-M semantics: keep the full
/// per-window vote history and recompute the last `of` votes from scratch on
/// every step, rather than the ring/popcount state machine — the independent
/// oracle the exhaustive and randomized suites check the real state machine
/// against. Equivalence with the all-`false`-prefill formulation is by
/// construction (only `min(of, i + 1)` votes exist before index `of - 1`);
/// this is what the tests actually prove: that the ring implements it.
fn vote_oracle(scores: &[f32], thr: f32, need: usize, of: usize) -> Vec<bool> {
  let mut history: Vec<bool> = Vec::with_capacity(scores.len());
  let mut out = Vec::with_capacity(scores.len());
  for &v in scores {
    history.push(v >= thr);
    let window = of.min(history.len());
    let trues = history[history.len() - window..]
      .iter()
      .filter(|&&b| b)
      .count();
    out.push(trues >= need);
  }
  out
}

/// Drive a fresh `DwellState<ScriptGate>` over `spans`, scripting the inner
/// gate's flags directly and pushing `Windowed<()>` — the streaming
/// counterpart of [`dwell_oracle`], and the value-freeness witness at once.
fn drive_dwell(inner_flags: &[bool], spans: &[Span], confirm: usize) -> Vec<bool> {
  let mut state = DwellState {
    inner: ScriptGate::new(inner_flags),
    confirm,
    run: None,
    last_start: None,
  };
  spans
    .iter()
    .map(|&span| state.push(&Windowed::new((), span)).unwrap())
    .collect()
}

/// The `Hangover` counterpart of [`drive_dwell`].
fn drive_hangover(inner_flags: &[bool], spans: &[Span], hold: usize) -> Vec<bool> {
  let mut state = HangoverState {
    inner: ScriptGate::new(inner_flags),
    hold,
    last_yes_end: None,
    last_start: None,
  };
  spans
    .iter()
    .map(|&span| state.push(&Windowed::new((), span)).unwrap())
    .collect()
}

/// Span geometries for the combinator sweeps, as `(hop, cycle of span lengths)`.
/// The first four are the fixed-length planner shapes (unit, adjacent, gapped,
/// overlapping); the last cycles its lengths, so a later span can end *before*
/// an earlier one. Ascending starts never imply ascending ends, and a
/// combinator that keeps a temporal horizon has to fold it by max to stay
/// correct on that geometry.
const COMBINATOR_GEOMETRIES: [(usize, &[usize]); 5] = [
  (1, &[1]),
  (2, &[2]),
  (3, &[2]),
  (1, &[2]),
  (1, &[6, 1, 3, 1, 5, 2]),
];

/// `len` spans of one [`COMBINATOR_GEOMETRIES`] entry: window `i` starts at
/// `i * hop` and covers the next length in the cycle.
fn geometry_spans(len: usize, hop: usize, lens: &[usize]) -> Vec<Span> {
  (0..len)
    .map(|i| {
      let span_len = lens[i % lens.len()];
      Span::new(i * hop, span_len, span_len)
    })
    .collect()
}

/// Feed `(flag, span)` pairs through a fresh [`Segmenter`] under `opts` and
/// collect every finalized range, including [`finish`](Segmenter::finish)'s
/// tail — shared by the Dwell/Hangover worked examples below.
fn finalize(flags: &[bool], spans: &[Span], opts: &SegmentOptions) -> Vec<Range> {
  let mut seg = Segmenter::new(*opts);
  let mut out = Vec::new();
  for (&flag, &span) in flags.iter().zip(spans) {
    if let Some(r) = seg.push(flag, span).unwrap() {
      out.push(r);
    }
  }
  out.extend(seg.finish());
  out
}

#[test]
fn dwell_new_exposes_inner_and_confirm() {
  let d = Dwell::new(Threshold::new(0.5), 7);
  assert_eq!(d.confirm(), 7);
  assert_eq!(d.inner().thr(), 0.5);
}

#[test]
fn hangover_new_exposes_inner_and_hold() {
  let h = Hangover::new(Threshold::new(0.5), 9);
  assert_eq!(h.hold(), 9);
  assert_eq!(h.inner().thr(), 0.5);
}

#[test]
fn dwell_confirm_zero_is_pass_through() {
  let inner_flags = [true, false, true, true, false, true];
  let spans: Vec<Span> = (0..inner_flags.len()).map(|i| Span::new(i, 1, 1)).collect();
  let got = drive_dwell(&inner_flags, &spans, 0);
  assert_eq!(got, inner_flags);
}

#[test]
fn hangover_hold_zero_is_pass_through() {
  // hold = 0: an adjacent window has gap 0, which is not < 0, so it releases
  // immediately and the output equals the inner flags exactly.
  let inner_flags = [true, false, true, true, false, false, true];
  let spans: Vec<Span> = (0..inner_flags.len()).map(|i| Span::new(i, 1, 1)).collect();
  let got = drive_hangover(&inner_flags, &spans, 0);
  assert_eq!(got, inner_flags);
}

#[test]
fn dwell_rising_edge_lag_example() {
  // Unit spans, confirm = 3: inner-true at positions 0, 1, 2 confirms on the
  // third push (end 3 - origin 0 >= 3) — the exact example from the type doc.
  let inner_flags = [true, true, true, true];
  let spans: Vec<Span> = (0..4).map(|i| Span::new(i, 1, 1)).collect();
  let got = drive_dwell(&inner_flags, &spans, 3);
  assert_eq!(got, vec![false, false, true, true]);
}

#[test]
fn dwell_confirm_at_or_below_first_window_len_never_suppresses() {
  // A single window of len 5 confirms immediately whenever confirm <= 5.
  let span = Span::new(0, 5, 5);
  for confirm in [0usize, 1, 5] {
    let got = drive_dwell(&[true], &[span], confirm);
    assert_eq!(got, vec![true], "confirm={confirm}");
  }
}

#[test]
fn dwell_counts_uncovered_elements_on_a_gapped_plan() {
  // hop 5 > window 2: elements 2..5 are covered by no span, but confirmation
  // distance is positional, so they still count toward `confirm` (F8).
  let spans = [Span::new(0, 2, 2), Span::new(5, 2, 2)];
  // First window: end 2, origin 0, 2 - 0 = 2 < confirm(3) -> false.
  // Second window: end 7, origin 0, 7 - 0 = 7 >= 3 -> true, despite the
  // 3-element gap no span covers.
  let got = drive_dwell(&[true, true], &spans, 3);
  assert_eq!(got, vec![false, true]);
}

#[test]
fn dwell_usize_max_never_activates() {
  let spans: Vec<Span> = (0..50).map(|i| Span::new(i, 1, 1)).collect();
  let inner_flags = vec![true; 50];
  let got = drive_dwell(&inner_flags, &spans, usize::MAX);
  assert!(got.iter().all(|&f| !f));
}

#[test]
fn dwell_head_trim_vs_min_len_worked_example() {
  // Inner-true run over unit spans [0, 6), confirm = 3: confirms at position 2
  // (end 3 - origin 0 >= 3), so the finalized range starts at 2, not 0 — the
  // causal/finalized-plane worked example from the type doc.
  let confirm = 3;
  let inner_flags = [true, true, true, true, true, true];
  let spans: Vec<Span> = (0..6).map(|i| Span::new(i, 1, 1)).collect();

  let flags = drive_dwell(&inner_flags, &spans, confirm);
  assert_eq!(finalize(&flags, &spans, &plain()), vec![Range::new(2, 6)]);

  // min_len instead keeps the full extent of any run it does not drop
  // entirely — it does not trim a kept run's head.
  let kept_whole = finalize(&inner_flags, &spans, &SegmentOptions::new().with_min_len(3));
  assert_eq!(kept_whole, vec![Range::new(0, 6)]);
}

#[test]
fn dwell_does_not_deactivate_when_a_later_span_ends_earlier() {
  // Ascending starts do not imply ascending ends: `[0, 10)` then the nested
  // `[1, 2)`, inner active throughout. Confirmation is measured against the
  // run's folded end horizon, so an on-delay gate that has confirmed cannot
  // deactivate while the inner gate never releases. Reading the current span's
  // end alone would report `2 - 0 = 2 < 10` and drop the flag.
  let spans = [Span::new(0, 10, 10), Span::new(1, 1, 1)];
  assert_eq!(drive_dwell(&[true, true], &spans, 10), vec![true, true]);

  // Equal ends and a further shrink keep it latched, and the finalized plane
  // stays the single run the flags describe.
  let spans = [
    Span::new(0, 10, 10),
    Span::new(1, 9, 9),
    Span::new(2, 3, 3),
    Span::new(4, 1, 1),
  ];
  let got = drive_dwell(&[true; 4], &spans, 10);
  assert_eq!(got, vec![true; 4]);
  assert_eq!(finalize(&got, &spans, &plain()), vec![Range::new(0, 10)]);
}

#[test]
fn dwell_never_deactivates_while_the_inner_gate_stays_active() {
  // The property behind the case above, over randomized geometries whose span
  // lengths vary, so ends rise, fall, and repeat under ascending starts.
  let mut state: u64 = 0x51A5_3C0D_7E11_9B42;
  for _ in 0..200 {
    let len = (xorshift(&mut state) % 40) as usize;
    let confirm = [0usize, 1, 2, 5, 10, 25][(xorshift(&mut state) % 6) as usize];
    let inner_flags: Vec<bool> = (0..len)
      .map(|_| !xorshift(&mut state).is_multiple_of(4))
      .collect();
    let mut start = 0usize;
    let spans: Vec<Span> = (0..len)
      .map(|_| {
        let span_len = 1 + (xorshift(&mut state) % 12) as usize;
        let span = Span::new(start, span_len, span_len);
        start += (xorshift(&mut state) % 4) as usize;
        span
      })
      .collect();

    let got = drive_dwell(&inner_flags, &spans, confirm);
    for i in 1..len {
      assert!(
        !(got[i - 1] && inner_flags[i] && !got[i]),
        "deactivated at {i} without an inner release: confirm={confirm} \
         flags={inner_flags:?} spans={spans:?} got={got:?}"
      );
    }
    // The finalized plane must agree with the oracle's flags too.
    let pairs: Vec<(bool, Span)> = inner_flags
      .iter()
      .copied()
      .zip(spans.iter().copied())
      .collect();
    let expected = dwell_oracle(&pairs, confirm);
    assert_eq!(got, expected, "confirm={confirm} spans={spans:?}");
    let opts = plain();
    assert_eq!(
      finalize(&got, &spans, &opts),
      finalize(&expected, &spans, &opts)
    );
  }
}

#[test]
fn dwell_deactivation_regression_survives_nesting_and_the_batch_driver() {
  // The same shrinking-end sequence through `Hangover(Dwell(Vote))`, the
  // canonical nesting, and through the batch driver. `hold = 0` is deliberate:
  // a positive hold masks this defect, because the shrinking span starts
  // *before* the hangover's coverage horizon and so is held at gap 0 — only a
  // pass-through hold lets the dwell decision reach the output unchanged.
  let policy = Hangover::new(Dwell::new(Vote::new(1, 1, 0.5), 10), 0);
  let spans = [Span::new(0, 10, 10), Span::new(1, 1, 1), Span::new(2, 2, 2)];
  let s: Vec<Windowed<f32>> = spans.iter().map(|&sp| Windowed::new(0.9, sp)).collect();

  let mut gate = policy.gate();
  let flags: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
  assert_eq!(flags, vec![true, true, true]);
  assert_eq!(
    policy.segment(&plain(), &s).unwrap(),
    finalize(&flags, &spans, &plain())
  );
}

#[test]
fn dwell_suppression_only_invariant_on_randomized_runs() {
  // Dual of Hangover's extension-only: Dwell only ever turns an inner `true`
  // into `false` (a suppressed head), never the reverse.
  let mut state: u64 = 0x0123_4567_89AB_CDEF;
  for _ in 0..200 {
    let len = (xorshift(&mut state) % 40) as usize;
    let confirm = [0usize, 1, 2, 5, 10][(xorshift(&mut state) % 5) as usize];
    let hop = 1 + (xorshift(&mut state) % 3) as usize;
    let inner_flags: Vec<bool> = (0..len)
      .map(|_| xorshift(&mut state).is_multiple_of(2))
      .collect();
    let spans: Vec<Span> = (0..len).map(|i| Span::new(i * hop, 1, 1)).collect();
    let got = drive_dwell(&inner_flags, &spans, confirm);
    for i in 0..len {
      assert!(
        !got[i] || inner_flags[i],
        "suppression-only violated at {i}: confirm={confirm} hop={hop}"
      );
    }
  }
}

#[test]
fn hangover_extension_only_invariant_on_randomized_runs() {
  let mut state: u64 = 0xABCD_EF01_2345_6789;
  for _ in 0..200 {
    let len = (xorshift(&mut state) % 40) as usize;
    let hold = [0usize, 1, 2, 5, 10][(xorshift(&mut state) % 5) as usize];
    let hop = 1 + (xorshift(&mut state) % 3) as usize;
    let inner_flags: Vec<bool> = (0..len)
      .map(|_| xorshift(&mut state).is_multiple_of(2))
      .collect();
    let spans: Vec<Span> = (0..len).map(|i| Span::new(i * hop, 1, 1)).collect();
    let got = drive_hangover(&inner_flags, &spans, hold);
    for i in 0..len {
      assert!(
        !inner_flags[i] || got[i],
        "extension-only violated at {i}: hold={hold} hop={hop}"
      );
    }
  }
}

#[test]
fn dwell_matches_oracle_over_exhaustive_flags_and_geometry() {
  let confirms = [0usize, 1, 2, 5, usize::MAX];
  for len in 0..=8usize {
    for bits in 0u32..(1 << len) {
      let inner_flags: Vec<bool> = (0..len).map(|i| (bits >> i) & 1 == 1).collect();
      for &(hop, lens) in &COMBINATOR_GEOMETRIES {
        let spans = geometry_spans(len, hop, lens);
        let pairs: Vec<(bool, Span)> = inner_flags
          .iter()
          .copied()
          .zip(spans.iter().copied())
          .collect();
        for &confirm in &confirms {
          let expected = dwell_oracle(&pairs, confirm);
          let got = drive_dwell(&inner_flags, &spans, confirm);
          assert_eq!(
            got, expected,
            "flags={inner_flags:?} hop={hop} lens={lens:?} confirm={confirm}"
          );

          // The finalized-range plane must agree too: driving the oracle
          // flags through a fresh Segmenter must equal driving DwellState's
          // own output through one.
          let opts = plain();
          assert_eq!(
            finalize(&got, &spans, &opts),
            finalize(&expected, &spans, &opts),
            "ranges: flags={inner_flags:?} hop={hop} lens={lens:?} confirm={confirm}"
          );
        }
      }
    }
  }
}

#[test]
fn hangover_matches_oracle_over_exhaustive_flags_and_geometry() {
  let holds = [0usize, 1, 2, 5, usize::MAX];
  for len in 0..=8usize {
    for bits in 0u32..(1 << len) {
      let inner_flags: Vec<bool> = (0..len).map(|i| (bits >> i) & 1 == 1).collect();
      for &(hop, lens) in &COMBINATOR_GEOMETRIES {
        let spans = geometry_spans(len, hop, lens);
        let pairs: Vec<(bool, Span)> = inner_flags
          .iter()
          .copied()
          .zip(spans.iter().copied())
          .collect();
        for &hold in &holds {
          let expected = hangover_oracle(&pairs, hold);
          let got = drive_hangover(&inner_flags, &spans, hold);
          assert_eq!(
            got, expected,
            "flags={inner_flags:?} hop={hop} lens={lens:?} hold={hold}"
          );

          let opts = plain();
          assert_eq!(
            finalize(&got, &spans, &opts),
            finalize(&expected, &spans, &opts),
            "ranges: flags={inner_flags:?} hop={hop} lens={lens:?} hold={hold}"
          );
        }
      }
    }
  }
}

#[test]
fn hangover_release_boundary_is_strict() {
  let hold = 5;
  // gap = hold - 1 (start 5, last_yes_end 1): still < hold, held.
  let mut held = HangoverState {
    inner: ScriptGate::new(&[true]),
    hold,
    last_yes_end: None,
    last_start: None,
  };
  assert!(held.push(&Windowed::new((), Span::new(0, 1, 1))).unwrap());
  assert!(held.push(&Windowed::new((), Span::new(5, 1, 1))).unwrap());

  // gap = hold exactly (start 6, last_yes_end 1): not < hold, released.
  let mut released = HangoverState {
    inner: ScriptGate::new(&[true]),
    hold,
    last_yes_end: None,
    last_start: None,
  };
  assert!(released
    .push(&Windowed::new((), Span::new(0, 1, 1)))
    .unwrap());
  assert!(!released
    .push(&Windowed::new((), Span::new(6, 1, 1)))
    .unwrap());
}

#[test]
fn hangover_overlapping_window_has_gap_zero_and_is_held() {
  // An inner-false window starting BEFORE the coverage horizon (an
  // overlapping window) saturates its gap to 0, which is always < any
  // positive hold.
  let mut state = HangoverState {
    inner: ScriptGate::new(&[true, false]),
    hold: 1,
    last_yes_end: None,
    last_start: None,
  };
  assert!(state.push(&Windowed::new((), Span::new(0, 5, 5))).unwrap()); // last_yes_end = 5
  assert!(state.push(&Windowed::new((), Span::new(2, 2, 2))).unwrap()); // start 2 < 5: gap 0
}

#[test]
fn hangover_relatch_during_hold_bridges_the_gap_into_one_run() {
  // hold = 3: an inner true, a short false gap within the hold, then another
  // inner true re-folds the horizon, causally bridging the two into one run.
  let hold = 3;
  let inner_flags = [true, false, false, true, false, false, false, false];
  let spans: Vec<Span> = (0..8).map(|i| Span::new(i, 1, 1)).collect();
  let got = drive_hangover(&inner_flags, &spans, hold);
  assert_eq!(got, vec![true, true, true, true, true, true, true, false]);
}

#[test]
fn hangover_usize_max_never_releases_after_activation() {
  let spans: Vec<Span> = (0..50).map(|i| Span::new(i * 3, 1, 1)).collect();
  let mut inner_flags = vec![false; 50];
  inner_flags[0] = true;
  let got = drive_hangover(&inner_flags, &spans, usize::MAX);
  assert!(
    got.iter().all(|&f| f),
    "usize::MAX hold must never release: {got:?}"
  );
}

#[test]
fn hangover_held_flags_do_not_bridge_uncovered_elements_without_merge_gap() {
  // hop 5 > window 2: an inner-true window at [0, 2), then an inner-false
  // window that stays held (within `hold`). The CAUSAL flag stays true, but
  // the FINALIZED plane still splits at the geometric gap unless `merge_gap`
  // bridges it — the two-plane worked example from the type doc.
  let hold = 10;
  let spans = [Span::new(0, 2, 2), Span::new(5, 2, 2)];
  let inner_flags = [true, false];
  let flags = drive_hangover(&inner_flags, &spans, hold);
  assert_eq!(flags, vec![true, true], "held across the gap causally");

  assert_eq!(
    finalize(&flags, &spans, &plain()),
    vec![Range::new(0, 2), Range::new(5, 7)],
    "still two ranges without merge_gap"
  );
  assert_eq!(
    finalize(&flags, &spans, &SegmentOptions::new().with_merge_gap(3)),
    vec![Range::new(0, 7)],
    "merge_gap bridges it, orthogonally"
  );
}

#[test]
fn hangover_folds_the_horizon_by_max_not_last_write() {
  // F2: span (0, len 6) then an overlapping (2, len 2) — ends 6 then 4. A
  // last-write-wins horizon would regress to 4 and release too early; the
  // max-fold keeps the horizon at 6.
  let mut state = HangoverState {
    inner: ScriptGate::new(&[true, true]),
    hold: 3,
    last_yes_end: None,
    last_start: None,
  };
  assert!(state.push(&Windowed::new((), Span::new(0, 6, 6))).unwrap());
  assert!(state.push(&Windowed::new((), Span::new(2, 2, 2))).unwrap());
  assert_eq!(state.last_yes_end, Some(6));

  // gap from end 6: start 8 -> gap 2 < hold(3): held.
  assert!(state.push(&Windowed::new((), Span::new(8, 1, 1))).unwrap());
  // start 9 -> gap 3, not < hold(3): released. (Measured from 6, not 4: a
  // last-write horizon would have released one step earlier, at start 7.)
  assert!(!state.push(&Windowed::new((), Span::new(9, 1, 1))).unwrap());
}

#[test]
fn dwell_reset_clears_origin_and_inner_state() {
  let mut g = Dwell::new(Threshold::new(0.5), 3).gate();
  assert!(!g.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(!g.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap());
  g.reset();
  // Post-reset: origin is cleared and a fresh run must reconfirm from
  // scratch, exactly like a freshly constructed gate; the monotonicity check
  // is re-armed too (span restarts at 0).
  assert!(!g.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(!g.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap());
  assert!(g.push(&Windowed::new(0.9, Span::new(2, 1, 1))).unwrap());
}

#[test]
fn hangover_reset_clears_horizon_and_inner_state() {
  let mut g = Hangover::new(Threshold::new(0.5), 3).gate();
  assert!(g.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  g.reset();
  // Post-reset: last_yes_end is cleared, so an inner-false push right after
  // is NOT held (no inner true has been seen yet this epoch); the span
  // restarts at 0 too.
  assert!(!g.push(&Windowed::new(0.1, Span::new(0, 1, 1))).unwrap());
}

#[test]
fn dwell_backward_start_errs_with_wrapper_and_inner_unchanged() {
  let mut state = DwellState {
    inner: ScriptGate::new(&[true, true, true]),
    confirm: 0,
    run: None,
    last_start: None,
  };
  assert!(state.push(&Windowed::new((), Span::new(5, 2, 2))).unwrap());
  assert_eq!(
    state.push(&Windowed::new((), Span::new(0, 2, 2))),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
  // The offending push was a no-op: last_start/origin are untouched, and the
  // inner gate's script cursor did not advance (the check runs before the
  // inner push).
  assert_eq!(state.last_start, Some(5));
  assert_eq!(state.run, Some((5, 7)));
  assert_eq!(state.inner.idx, 1);
  // A valid continuation behaves as if the bad push never happened.
  assert!(state.push(&Windowed::new((), Span::new(7, 2, 2))).unwrap());
  assert_eq!(state.inner.idx, 2);
}

#[test]
fn nested_combinators_surface_exactly_one_error() {
  // Hangover(Dwell(Threshold)): a backward start is caught by the OUTERMOST
  // span-reading stage before the inner combinator ever sees it, so exactly
  // one NonMonotonicSpan error surfaces.
  let policy = Hangover::new(Dwell::new(Threshold::new(0.5), 2), 3);
  let mut gate = policy.gate();
  assert!(!gate.push(&Windowed::new(0.9, Span::new(5, 1, 1))).unwrap());
  assert_eq!(
    gate.push(&Windowed::new(0.9, Span::new(0, 1, 1))),
    Err(WinditError::NonMonotonicSpan {
      prev_start: 5,
      start: 0,
    })
  );
}

#[test]
fn dwell_batch_segment_equals_streaming_drive() {
  let policy = Dwell::new(Threshold::new(0.5), 3);
  let s = seq(&[0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9]);
  let batch = policy.segment(&plain(), &s).unwrap();

  let mut gate = policy.gate();
  let flags: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
  let spans: Vec<Span> = s.iter().map(|w| w.span()).collect();
  assert_eq!(batch, finalize(&flags, &spans, &plain()));
}

#[test]
fn hangover_batch_segment_equals_streaming_drive() {
  let policy = Hangover::new(Threshold::new(0.5), 3);
  let s = seq(&[0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]);
  let batch = policy.segment(&plain(), &s).unwrap();

  let mut gate = policy.gate();
  let flags: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
  let spans: Vec<Span> = s.iter().map(|w| w.span()).collect();
  assert_eq!(batch, finalize(&flags, &spans, &plain()));
}

#[test]
fn composition_hangover_of_dwell_of_vote_streams_and_batch_drives_and_matches_oracle() {
  // The canonical nesting: Hangover(Dwell(Vote)) (T2 exercised this
  // composition against a placeholder Threshold inner before Vote existed;
  // T3 upgrades it to the real thing). Compiles as a config value, streams
  // via Gate::push, and batch-drives via GatePolicy::segment; both equal the
  // composed scan oracles (Vote -> dwell_oracle -> hangover_oracle).
  let need = 3;
  let of = 5;
  let thr = 0.5;
  let confirm = 2;
  let hold = 3;
  let policy = Hangover::new(
    Dwell::new(Vote::try_new(need, of, thr).unwrap(), confirm),
    hold,
  );

  let mut state: u64 = 0x2468_1357_9BDF_0246;
  for _ in 0..50 {
    let n = (xorshift(&mut state) % 30) as usize;
    let hop = 1 + (xorshift(&mut state) % 3) as usize;
    let scores: Vec<f32> = (0..n).map(|_| next_unit(&mut state)).collect();
    let spans: Vec<Span> = (0..n).map(|i| Span::new(i * hop, 1, 1)).collect();
    let s: Vec<Windowed<f32>> = scores
      .iter()
      .zip(&spans)
      .map(|(&v, &sp)| Windowed::new(v, sp))
      .collect();

    let mut gate = policy.gate();
    let streamed: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
    let batch_ranges = policy.segment(&plain(), &s).unwrap();

    let vote_flags = vote_oracle(&scores, thr, need, of);
    let vote_pairs: Vec<(bool, Span)> = vote_flags
      .iter()
      .zip(&spans)
      .map(|(&f, &sp)| (f, sp))
      .collect();
    let dwell_flags = dwell_oracle(&vote_pairs, confirm);
    let dwell_pairs: Vec<(bool, Span)> = dwell_flags
      .iter()
      .zip(&spans)
      .map(|(&f, &sp)| (f, sp))
      .collect();
    let hangover_flags = hangover_oracle(&dwell_pairs, hold);

    assert_eq!(streamed, hangover_flags, "n={n} hop={hop}");
    assert_eq!(
      batch_ranges,
      finalize(&hangover_flags, &spans, &plain()),
      "n={n} hop={hop}"
    );
  }
}

#[test]
fn canonical_nesting_example_from_the_spec_compiles_and_streams() {
  // The parent design's literal worked example
  // (`Hangover::new(Dwell::new(Vote::try_new(3, 5, 0.5)?, 160), 480)`): pins
  // that the exact expression type-checks as a config value and
  // streams/batch-drives without panicking. `confirm = 160` dwarfs this short
  // sequence, so every flag must be false and the batch empty — a concrete,
  // checkable consequence rather than a no-op smoke test. The property test
  // above uses smaller confirm/hold so its randomized sequences actually
  // exercise Dwell/Hangover transitions; this test exists only to pin the
  // spec's literal constants separately.
  let policy = Hangover::new(Dwell::new(Vote::try_new(3, 5, 0.5).unwrap(), 160), 480);
  let s = seq(&[0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9]);

  let mut gate = policy.gate();
  let flags: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
  assert!(
    flags.iter().all(|&f| !f),
    "confirm=160 must suppress every flag over an 8-element run: {flags:?}"
  );

  let batch = policy.segment(&plain(), &s).unwrap();
  assert!(batch.is_empty());
}

#[test]
fn nesting_order_changes_behavior() {
  // Dwell(Hangover(_)) and Hangover(Dwell(_)) are not interchangeable — a
  // brief blip that never reaches `confirm` on its own is invisible to
  // Dwell(Hangover(_)) (Hangover cannot make it longer than one blip's worth
  // by itself once released), but Hangover extends the SAME blip long enough
  // for Dwell to see it confirmed under Hangover(Dwell(_)).
  let confirm = 2;
  let hold = 2;
  let inner = Threshold::new(0.5);

  let a = Hangover::new(Dwell::new(inner, confirm), hold);
  let b = Dwell::new(Hangover::new(inner, hold), confirm);

  let scores = [0.9f32, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1];
  let spans: Vec<Span> = (0..scores.len()).map(|i| Span::new(i, 1, 1)).collect();
  let s: Vec<Windowed<f32>> = scores
    .iter()
    .zip(&spans)
    .map(|(&v, &sp)| Windowed::new(v, sp))
    .collect();

  let flags_a: Vec<bool> = {
    let mut g = a.gate();
    s.iter().map(|w| g.push(w).unwrap()).collect()
  };
  let flags_b: Vec<bool> = {
    let mut g = b.gate();
    s.iter().map(|w| g.push(w).unwrap()).collect()
  };

  assert_ne!(
    flags_a, flags_b,
    "nesting order must not be interchangeable"
  );
}

#[test]
fn dwell_and_hangover_are_object_safe_as_boxed_gates() {
  let mut boxed_dwell: Box<dyn Gate<f32>> = Box::new(Dwell::new(Threshold::new(0.5), 2).gate());
  assert!(!boxed_dwell
    .push(&Windowed::new(0.9, Span::new(0, 1, 1)))
    .unwrap());
  assert!(boxed_dwell
    .push(&Windowed::new(0.9, Span::new(1, 1, 1)))
    .unwrap());

  let mut boxed_hangover: Box<dyn Gate<f32>> =
    Box::new(Hangover::new(Threshold::new(0.5), 2).gate());
  assert!(boxed_hangover
    .push(&Windowed::new(0.9, Span::new(0, 1, 1)))
    .unwrap());
  assert!(boxed_hangover
    .push(&Windowed::new(0.1, Span::new(1, 1, 1)))
    .unwrap());
  assert!(!boxed_hangover
    .push(&Windowed::new(0.1, Span::new(3, 1, 1)))
    .unwrap());
}

#[test]
fn dwell_discontinuity_forwards_to_inner_discontinuity_not_reset() {
  let (probe, reset_calls, discontinuity_calls) = ProbeGate::new(false);
  let mut state = DwellState {
    inner: probe,
    confirm: 3,
    run: None,
    last_start: None,
  };
  let _ = state.push(&Windowed::new(0.0, Span::new(0, 1, 1))).unwrap();
  state.discontinuity();
  assert_eq!(discontinuity_calls.get(), 1);
  assert_eq!(reset_calls.get(), 0);
  state.reset();
  assert_eq!(reset_calls.get(), 1);
}

#[test]
fn hangover_discontinuity_forwards_to_inner_discontinuity_not_reset() {
  let (probe, reset_calls, discontinuity_calls) = ProbeGate::new(false);
  let mut state = HangoverState {
    inner: probe,
    hold: 3,
    last_yes_end: None,
    last_start: None,
  };
  let _ = state.push(&Windowed::new(0.0, Span::new(0, 1, 1))).unwrap();
  state.discontinuity();
  assert_eq!(discontinuity_calls.get(), 1);
  assert_eq!(reset_calls.get(), 0);
  state.reset();
  assert_eq!(reset_calls.get(), 1);
}

#[test]
fn box_dyn_gate_forwards_discontinuity_explicitly() {
  // F3 at the Box layer: a bare ProbeGate boxed as `Box<dyn Gate<f32>>` must
  // still route `discontinuity` to the concrete gate's `discontinuity`.
  let (probe, reset_calls, discontinuity_calls) = ProbeGate::new(false);
  let mut boxed: Box<dyn Gate<f32>> = Box::new(probe);
  boxed.discontinuity();
  assert_eq!(discontinuity_calls.get(), 1);
  assert_eq!(reset_calls.get(), 0);
  boxed.reset();
  assert_eq!(reset_calls.get(), 1);
}

#[test]
fn box_of_dwell_of_probe_forwards_discontinuity_through_both_layers() {
  // F3 through both layers at once: Box -> DwellState -> ProbeGate.
  let (probe, reset_calls, discontinuity_calls) = ProbeGate::new(false);
  let inner_state = DwellState {
    inner: probe,
    confirm: 2,
    run: None,
    last_start: None,
  };
  let mut boxed: Box<dyn Gate<f32>> = Box::new(inner_state);
  boxed.discontinuity();
  assert_eq!(
    discontinuity_calls.get(),
    1,
    "Box -> DwellState -> ProbeGate discontinuity chain broken"
  );
  assert_eq!(reset_calls.get(), 0);
}

// ── Vote gate ─────────────────────────────────────────────────────────────

#[test]
fn vote_new_exposes_need_of_thr() {
  let v = Vote::new(3, 5, 0.5);
  assert_eq!(v.need(), 3);
  assert_eq!(v.of(), 5);
  assert_eq!(v.thr(), 0.5);
}

#[test]
fn vote_accessors_report_nan_thr_verbatim() {
  // Only the counts are validated at construction; `thr` is carried through
  // exactly as given, NaN included (mirrors `Threshold`/`Hysteresis`).
  let v = Vote::new(2, 4, f32::NAN);
  assert_eq!(v.need(), 2);
  assert_eq!(v.of(), 4);
  assert!(v.thr().is_nan());
}

#[test]
fn vote_try_new_rejects_invalid_configurations() {
  // (0, n): need = 0 would always activate.
  assert_eq!(
    Vote::try_new(0, 5, 0.5),
    Err(WinditError::InvalidVote { need: 0, of: 5 })
  );
  // (n + 1, n): need > of would never activate.
  assert_eq!(
    Vote::try_new(6, 5, 0.5),
    Err(WinditError::InvalidVote { need: 6, of: 5 })
  );
  // (n, 0): of = 0 is vacuous.
  assert_eq!(
    Vote::try_new(5, 0, 0.5),
    Err(WinditError::InvalidVote { need: 5, of: 0 })
  );
  // (1, 65): of > 64 exceeds the one-word state bound.
  assert_eq!(
    Vote::try_new(1, 65, 0.5),
    Err(WinditError::InvalidVote { need: 1, of: 65 })
  );
  // (65, 65): both violations at once.
  assert_eq!(
    Vote::try_new(65, 65, 0.5),
    Err(WinditError::InvalidVote { need: 65, of: 65 })
  );
}

#[test]
fn vote_try_new_accepts_boundary_configurations() {
  assert_eq!(Vote::try_new(1, 1, 0.5).unwrap(), Vote::new(1, 1, 0.5));
  assert_eq!(Vote::try_new(64, 64, 0.5).unwrap(), Vote::new(64, 64, 0.5));
}

#[test]
#[should_panic(expected = "1 <= need <= of <= 64")]
fn vote_new_panics_on_invalid_pair() {
  let _ = Vote::new(6, 5, 0.5);
}

#[test]
fn vote_matches_brute_force_reference_on_exhaustive_patterns() {
  // Every vote pattern up to length 10, against the (need, of) grid from the
  // spec, checked at every step against the brute-force reference.
  let configs = [(1usize, 1usize), (1, 2), (2, 2), (2, 3), (3, 5), (5, 5)];
  let thr = 0.5;
  for len in 0..=10usize {
    for bits in 0u32..(1u32 << len) {
      let scores: Vec<f32> = (0..len)
        .map(|i| if (bits >> i) & 1 == 1 { 0.9 } else { 0.1 })
        .collect();
      for &(need, of) in &configs {
        let expected = vote_oracle(&scores, thr, need, of);
        let got = gate_bools(&Vote::new(need, of, thr), &scores);
        assert_eq!(got, expected, "len={len} bits={bits:b} need={need} of={of}");
      }
    }
  }
}

#[test]
fn vote_word_boundary_of_64_activation_saturation_and_decay() {
  // 64 true pushes then 64 false: of=64 means the vote window is exactly one
  // machine word, so this exercises activation (the window filling),
  // saturation (the window entirely true, popcount 64), and decay (the
  // window sliding off the true run one vote at a time). For i in
  // [63, 126], popcount(i) = 127 - i exactly: a clean linear decay across
  // the one-word boundary.
  let mut scores = vec![0.9f32; 64];
  scores.extend(vec![0.1f32; 64]);
  let thr = 0.5;

  for &(need, of) in &[(1usize, 64usize), (32, 64), (64, 64)] {
    let expected = vote_oracle(&scores, thr, need, of);
    let got = gate_bools(&Vote::new(need, of, thr), &scores);
    assert_eq!(got, expected, "need={need} of={of}");
  }

  // need=1: active as soon as any true vote is in the window (from index 0);
  // deactivates only once the last true vote ages out at index 127.
  let need1 = gate_bools(&Vote::new(1, 64, thr), &scores);
  assert!(need1[63], "need=1 active with a full window of true votes");
  assert!(
    need1[126],
    "need=1 still active: popcount(126) = 127-126 = 1"
  );
  assert!(
    !need1[127],
    "need=1 deactivates once the last true vote ages out"
  );

  // need=32: popcount(95) = 32 (active), popcount(96) = 31 (inactive) — the
  // exact release boundary.
  let need32 = gate_bools(&Vote::new(32, 64, thr), &scores);
  assert!(need32[95], "need=32: popcount(95) = 127-95 = 32, active");
  assert!(!need32[96], "need=32: popcount(96) = 127-96 = 31, inactive");

  // need=64: only a single-index pulse, exactly at the window's first full
  // saturation (index 63); cannot activate before, decays immediately after.
  let need64 = gate_bools(&Vote::new(64, 64, thr), &scores);
  assert!(
    need64[63],
    "need=64: window first fully saturates at index 63"
  );
  assert!(!need64[64], "need=64: decays the push after saturation");
  assert!(
    !need64[0],
    "need=64: cannot activate before the window fills"
  );
}

#[test]
fn vote_1_1_equals_threshold_on_randomized_sequences() {
  // Pins the comparison-table equivalence: with need = of = 1 there is no
  // history besides the current push, so Vote degrades exactly to
  // Threshold's raw-IEEE membership test.
  let mut state: u64 = 0x7F4A_7C15_9E37_79B9;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 40) as usize;
    let scores: Vec<f32> = (0..n)
      .map(|_| match xorshift(&mut state) % 16 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        _ => next_unit(&mut state),
      })
      .collect();
    let thr = match xorshift(&mut state) % 8 {
      0 => f32::NAN,
      1 => f32::INFINITY,
      2 => f32::NEG_INFINITY,
      3 if !scores.is_empty() => scores[(xorshift(&mut state) % scores.len() as u64) as usize],
      _ => next_unit(&mut state),
    };

    let vote_out = gate_bools(&Vote::new(1, 1, thr), &scores);
    let threshold_out = gate_bools(&Threshold::new(thr), &scores);
    assert_eq!(vote_out, threshold_out, "n={n} thr={thr}");
  }
}

#[test]
fn vote_matches_brute_force_reference_on_randomized_inputs() {
  // ~200 cases: random (need, of) with of <= 64, a threshold sampled with
  // NaN, a score collision, or a random unit value, and non-finite score
  // injections, checked against the brute-force reference.
  let mut state: u64 = 0x2545_F491_4F6C_DD1D;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 100) as usize;
    let of = 1 + (xorshift(&mut state) % 64) as usize;
    let need = 1 + (xorshift(&mut state) % of as u64) as usize;

    let scores: Vec<f32> = (0..n)
      .map(|_| match xorshift(&mut state) % 16 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        _ => next_unit(&mut state),
      })
      .collect();
    let thr = match xorshift(&mut state) % 8 {
      0 => f32::NAN,
      1 if !scores.is_empty() => scores[(xorshift(&mut state) % scores.len() as u64) as usize],
      _ => next_unit(&mut state),
    };

    let expected = vote_oracle(&scores, thr, need, of);
    let got = gate_bools(&Vote::new(need, of, thr), &scores);
    assert_eq!(got, expected, "n={n} need={need} of={of} thr={thr}");
  }
}

#[test]
fn vote_earliest_activation_is_at_index_need_minus_1() {
  // All-true pushes into a window exactly `need` wide: the need-th true vote
  // — and so the earliest possible activation — lands at index need - 1
  // (0-based), never earlier.
  for need in [1usize, 3, 10, 64] {
    let mut gate = Vote::new(need, need, 0.5).gate();
    for i in 0..need {
      let active = gate.push(&Windowed::new(0.9, Span::new(i, 1, 1))).unwrap();
      assert_eq!(
        active,
        i + 1 == need,
        "need={need} index={i}: activation must land exactly at index need-1"
      );
    }
  }
}

#[test]
fn vote_nan_value_is_a_false_vote_that_ages_out() {
  // NaN never satisfies `>= thr`, so it is an ordinary false vote — it
  // neither poisons the state nor is treated specially — and ages out of the
  // window exactly like any other false vote once `of` further pushes have
  // occurred.
  let mut gate = Vote::new(1, 3, 0.5).gate();
  assert!(gate.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap()); // true vote: need=1 met
  assert!(gate
    .push(&Windowed::new(f32::NAN, Span::new(1, 1, 1)))
    .unwrap()); // NaN false vote; the true vote is still in the window
  assert!(gate
    .push(&Windowed::new(f32::NAN, Span::new(2, 1, 1)))
    .unwrap()); // of=3: still in the window
                // The original true vote ages out of the 3-wide window on this push.
  assert!(!gate
    .push(&Windowed::new(f32::NAN, Span::new(3, 1, 1)))
    .unwrap());
}

#[test]
fn vote_deactivates_exactly_when_need_th_true_vote_leaves_the_window() {
  // need=2, of=4: two true votes at the front of the window, then false
  // pushes. Deactivation must land exactly when the older of the two true
  // votes ages out of the 4-wide window, not one push earlier or later.
  let mut gate = Vote::new(2, 4, 0.5).gate();
  assert!(!gate.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap()); // 1 true: below need
  assert!(gate.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap()); // 2 true: need met
  assert!(gate.push(&Windowed::new(0.1, Span::new(2, 1, 1))).unwrap()); // [t,t,f]: still 2 true
  assert!(gate.push(&Windowed::new(0.1, Span::new(3, 1, 1))).unwrap()); // [t,t,f,f]: window full, still 2 true
                                                                        // The window is now full; the next push evicts the oldest vote (the first
                                                                        // true, at index 0), leaving only 1 true — below need.
  assert!(!gate.push(&Windowed::new(0.1, Span::new(4, 1, 1))).unwrap());
}

#[test]
fn vote_reset_and_discontinuity_clear_history() {
  fn drive_four(g: &mut VoteState) -> Vec<bool> {
    [0.9f32, 0.9, 0.1, 0.9]
      .iter()
      .enumerate()
      .map(|(i, &v)| g.push(&Windowed::new(v, Span::new(i, 1, 1))).unwrap())
      .collect()
  }

  let policy = Vote::new(2, 3, 0.5);

  let mut fresh = policy.gate();
  let baseline = drive_four(&mut fresh);

  let mut via_reset = policy.gate();
  via_reset
    .push(&Windowed::new(0.9, Span::new(0, 1, 1)))
    .unwrap();
  via_reset
    .push(&Windowed::new(0.9, Span::new(1, 1, 1)))
    .unwrap();
  via_reset.reset();
  assert_eq!(drive_four(&mut via_reset), baseline);

  let mut via_discontinuity = policy.gate();
  via_discontinuity
    .push(&Windowed::new(0.9, Span::new(0, 1, 1)))
    .unwrap();
  via_discontinuity.discontinuity();
  assert_eq!(drive_four(&mut via_discontinuity), baseline);
}

#[test]
fn vote_batch_segment_equals_streaming_drive() {
  let policy = Vote::new(2, 3, 0.5);
  let s = seq(&[0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1]);
  let batch = policy.segment(&plain(), &s).unwrap();

  let mut gate = policy.gate();
  let flags: Vec<bool> = s.iter().map(|w| gate.push(w).unwrap()).collect();
  let spans: Vec<Span> = s.iter().map(|w| w.span()).collect();
  assert_eq!(batch, finalize(&flags, &spans, &plain()));
}

#[test]
fn vote_is_object_safe_as_boxed_gate() {
  let mut boxed: Box<dyn Gate<f32>> = Box::new(Vote::new(2, 3, 0.5).gate());
  assert!(!boxed.push(&Windowed::new(0.9, Span::new(0, 1, 1))).unwrap());
  assert!(boxed.push(&Windowed::new(0.9, Span::new(1, 1, 1))).unwrap());
  boxed.reset();
  assert!(!boxed.push(&Windowed::new(0.1, Span::new(2, 1, 1))).unwrap());
}
