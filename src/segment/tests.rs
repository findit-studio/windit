use std::{vec, vec::Vec};

use super::{
  longest_run, runs, runs_sorted, Gate, GatePolicy, Hysteresis, Range, SegmentOptions, SegmentTail,
  Segmenter, Threshold,
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
              "finish emitted >1: bits={bits:b} span_len={span_len} hop={hop} opts={opts:?}"
            );
            assert!(
              seg.discontinuity().len() <= 1,
              "discontinuity emitted >1: bits={bits:b} span_len={span_len} hop={hop} opts={opts:?}"
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
