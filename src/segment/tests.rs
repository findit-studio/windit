use std::{vec, vec::Vec};

use super::{
  longest_run, runs, runs_sorted, HysteresisSegment, Range, SegmentOptions, SegmentPolicy,
  Threshold,
};
use crate::{
  error::WinditError,
  plan::{Span, WindowOptions, WindowPlan},
  smooth::SmoothPolicy,
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

/// The pre-fusion two-pass composition, kept as the differential reference for
/// the fused [`HysteresisSegment::segment`]: latch the sequence with
/// `smooth::Hysteresis`, then group the latched-on windows with [`runs`]. This
/// is the behaviour that shipped before fusion, not an independent oracle — it
/// proves the fused path is *equivalent* to it, not that either is *correct*.
fn two_pass_reference(
  on: f32,
  off: f32,
  opts: &SegmentOptions,
  seq: &[Windowed<f32>],
) -> Vec<Range> {
  let gated = crate::smooth::Hysteresis::new(on, off).smooth(seq);
  runs(&gated, |&v| v >= 0.5, opts)
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

#[test]
fn runs_contiguous_maps_to_element_range() {
  // Frames 1 and 2 are above 0.5, so one run spanning elements [1, 3).
  let s = seq(&[0.1, 0.9, 0.8, 0.2, 0.1]);
  let out = runs(&s, |&v| v > 0.5, &plain());
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
  let out = runs(&s, |&v| v > 0.5, &plain());
  assert_eq!(out, vec![Range::new(0, 4), Range::new(8, 12)]);
}

#[test]
fn merge_gap_bridges_and_zero_keeps_separate() {
  // Frames 0, 2, 3 above threshold: raw runs [0,1) and [2,4), a one-element gap.
  let s = seq(&[0.9, 0.1, 0.9, 0.9]);

  let separate = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(0));
  assert_eq!(separate, vec![Range::new(0, 1), Range::new(2, 4)]);

  let bridged = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(1));
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
    runs(&s, |&v| v > 0.5, &plain()),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );

  // Bridging the 3-element gap is `merge_gap`'s decision alone.
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(2)),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(3)),
    vec![Range::new(0, 7)]
  );
}

#[test]
fn gapped_spans_feed_min_len_longest_and_sorted() {
  let s = gapped_plan();

  // Each run is 2 elements, so a 3-element minimum drops both; fusing them into
  // a single 7-element run would keep one.
  assert!(runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(3)).is_empty());
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(2)),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );

  // Equal lengths, so the earliest wins the tie and the sort keeps input order.
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()),
    Some(Range::new(0, 2))
  );
  assert_eq!(
    runs_sorted(&s, |&v| v > 0.5, &plain()),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
}

#[test]
fn gapped_spans_split_under_the_segment_policies() {
  let s = gapped_plan();

  // Both policies reach the geometry only through `runs`, so both inherit the
  // split. `HysteresisSegment` latches on at the first window and stays on
  // across the gap — a value decision that must not become a geometric one.
  assert_eq!(
    Threshold::new(0.5).segment(&s),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
  assert_eq!(
    HysteresisSegment::new(0.6, 0.3).segment(&s),
    vec![Range::new(0, 2), Range::new(5, 7)]
  );
}

#[test]
fn min_len_drops_short_runs() {
  // Runs [0,1) (len 1) and [2,5) (len 3); min_len 2 drops the first.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9]);

  let kept_all = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(0));
  assert_eq!(kept_all, vec![Range::new(0, 1), Range::new(2, 5)]);

  let filtered = runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_min_len(2));
  assert_eq!(filtered, vec![Range::new(2, 5)]);
}

#[test]
fn longest_run_picks_longest_and_ties_earliest() {
  // Runs of length 1, 3, 2: the length-3 run wins.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()),
    Some(Range::new(2, 5))
  );

  // Two length-2 runs: the earliest wins the tie.
  let tie = seq(&[0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    longest_run(&tie, |&v| v > 0.5, &plain()),
    Some(Range::new(0, 2))
  );
}

#[test]
fn runs_sorted_descending_stable() {
  // Lengths 1, 3, 2 sort to 3, 2, 1.
  let s = seq(&[0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9]);
  assert_eq!(
    runs_sorted(&s, |&v| v > 0.5, &plain()),
    vec![Range::new(2, 5), Range::new(6, 8), Range::new(0, 1),]
  );

  // Equal lengths keep input order (stable): [0,2) before [3,5).
  let tie = seq(&[0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]);
  assert_eq!(
    runs_sorted(&tie, |&v| v > 0.5, &plain()),
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
  let speech = runs(&s, |&v| v > 0.5, &plain());
  assert_eq!(
    speech,
    vec![Range::new(2, 6), Range::new(9, 16), Range::new(18, 20),]
  );
  assert_eq!(
    longest_run(&s, |&v| v > 0.5, &plain()),
    Some(Range::new(9, 16))
  );

  // merge_gap 2 bridges only the 2-element gap before the final range (the
  // 3-element gap between the first two stays); min_len 5 then drops the short
  // leading [2, 6) range, leaving the merged tail.
  let merged = runs(
    &s,
    |&v| v > 0.5,
    &SegmentOptions::new().with_merge_gap(2).with_min_len(5),
  );
  assert_eq!(merged, vec![Range::new(9, 20)]);
}

#[test]
fn threshold_policy_includes_boundary() {
  // thr 0.5 admits values >= 0.5, so the boundary value 0.5 is in-segment.
  let s = seq(&[0.5, 0.9, 0.2, 0.6]);
  let policy = Threshold::new(0.5);
  assert_eq!(policy.segment(&s), vec![Range::new(0, 2), Range::new(3, 4)]);
}

#[test]
fn hysteresis_segment_latches_then_segments() {
  // Same latch as smooth::Hysteresis (on 0.6, off 0.3): [0,1,1,0,1] over the
  // input, whose set frames form runs [1,3) and [4,5). The fused single pass must
  // match the two-pass composition here and everywhere else, which
  // `fused_matches_two_pass_reference_*` enforce exhaustively.
  let s = seq(&[0.1, 0.7, 0.5, 0.2, 0.6]);
  let policy = HysteresisSegment::new(0.6, 0.3);
  assert_eq!(policy.segment(&s), vec![Range::new(1, 3), Range::new(4, 5)]);
}

#[test]
fn hysteresis_segment_holds_at_off_boundary_instead_of_turning_off() {
  // Mirrors smooth::Hysteresis's own off-boundary regression (on 0.6, off
  // 0.3): the gate latches on at frame 0 and a value exactly at `off` (frame
  // 1) must hold it on rather than close the run early, so frames 0..3 stay
  // one run; only the strictly-below value at frame 3 (0.2) ends it.
  let s = seq(&[0.7, 0.3, 0.3, 0.2]);
  let policy = HysteresisSegment::new(0.6, 0.3);
  assert_eq!(policy.segment(&s), vec![Range::new(0, 3)]);
}

#[test]
fn empty_and_all_false_yield_none() {
  let empty: Vec<Windowed<f32>> = Vec::new();
  assert!(runs(&empty, |&v| v > 0.5, &plain()).is_empty());
  assert_eq!(longest_run(&empty, |&v| v > 0.5, &plain()), None);
  assert!(runs_sorted(&empty, |&v| v > 0.5, &plain()).is_empty());

  let all_low = seq(&[0.1, 0.2, 0.0]);
  assert!(runs(&all_low, |&v| v > 0.5, &plain()).is_empty());
  assert_eq!(longest_run(&all_low, |&v| v > 0.5, &plain()), None);
}

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
  let out = runs(&s, |&v| v > 0.5, &plain());
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
    Threshold::new(f32::NEG_INFINITY).segment(&s),
    vec![Range::new(0, 1), Range::new(2, 3)]
  );

  // `thr = NaN`: `value >= NaN` is never true, so nothing is in-segment.
  assert!(Threshold::new(f32::NAN).segment(&s).is_empty());

  // `thr = +inf` admits only a `+inf` score.
  let s = seq(&[f32::INFINITY, 1.0]);
  assert_eq!(
    Threshold::new(f32::INFINITY).segment(&s),
    vec![Range::new(0, 1)]
  );
}

#[test]
fn hysteresis_segment_non_finite_thresholds() {
  let s = seq(&[0.7, 0.1, 0.1]);
  // `on = NaN`: the gate can never activate, so nothing is segmented.
  assert!(HysteresisSegment::new(f32::NAN, 0.3).segment(&s).is_empty());
  // `off = NaN`: once latched on at index 0 the gate never releases, so the whole
  // sequence is a single run.
  assert_eq!(
    HysteresisSegment::new(0.6, f32::NAN).segment(&s),
    vec![Range::new(0, 3)]
  );
}

#[test]
fn hysteresis_segment_nan_score_holds_inside_run() {
  // on 0.6 off 0.3: index 0 latches on, the two NaN scores hold that on state,
  // and 0.2 releases — so the run is elements [0, 3), the NaNs do not split it.
  let s = seq(&[0.7, f32::NAN, f32::NAN, 0.2]);
  assert_eq!(
    HysteresisSegment::new(0.6, 0.3).segment(&s),
    vec![Range::new(0, 3)]
  );
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
  assert_eq!(runs(&s, |&v| v > 0.5, &plain()), vec![Range::new(0, 6)]);

  // (ii) A rejected window between two accepted overlapping ranges splits them
  // into raw runs [0,6) and [4,10); with merge_gap 0, `merge_adjacent` folds the
  // overlap (start 4 before end 6) to a zero gap through `saturating_sub` and
  // merges them into [0,10).
  let s = [
    Windowed::new(0.9, Span::new(0, 6, 6)),
    Windowed::new(0.1, Span::new(2, 6, 6)),
    Windowed::new(0.9, Span::new(4, 6, 6)),
  ];
  assert_eq!(
    runs(&s, |&v| v > 0.5, &SegmentOptions::new().with_merge_gap(0)),
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
    ),
    vec![Range::new(0, 5)]
  );
  assert!(runs(
    &s,
    |&v| v > 0.5,
    &SegmentOptions::new().with_merge_gap(1).with_min_len(6)
  )
  .is_empty());
}

#[test]
fn non_monotonic_spans_return_deterministically_without_panicking() {
  // `runs` documents ascending span order as a precondition. Non-monotonic input
  // is a documented precondition violation, not a supported case: the guarantee
  // is only that the call returns deterministically without panicking and every
  // returned range is well-formed. *Which* ranges is unspecified, so this pins
  // the guarantee and never the concrete geometry.
  let s = [
    Windowed::new(0.9, Span::new(5, 2, 2)),
    Windowed::new(0.9, Span::new(0, 2, 2)),
  ];
  let first = runs(&s, |&v| v > 0.5, &plain());
  let second = runs(&s, |&v| v > 0.5, &plain());
  assert_eq!(first, second, "runs must be deterministic");
  assert!(
    first.iter().all(|r| r.start() <= r.end()),
    "every returned range must be well-formed"
  );
}

#[test]
fn fused_matches_two_pass_reference_on_fixed_geometries() {
  // The fused `HysteresisSegment::segment` must equal the two-pass reference on
  // every input, not only finite ones. These fixed cases cover adjacent unit
  // spans, the off-boundary hold, the gapped plan under each option, overlapping
  // spans (union and merge), non-finite scores, and the degenerate `on < off`
  // and NaN-threshold configurations.
  let overlap_union = vec![
    Windowed::new(0.9, Span::new(0, 4, 4)),
    Windowed::new(0.9, Span::new(2, 4, 4)),
  ];
  let overlap_split = vec![
    Windowed::new(0.9, Span::new(0, 6, 6)),
    Windowed::new(0.1, Span::new(2, 6, 6)),
    Windowed::new(0.9, Span::new(4, 6, 6)),
  ];
  let cases: Vec<(f32, f32, SegmentOptions, Vec<Windowed<f32>>)> = vec![
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
  ];
  for (on, off, opts, s) in cases {
    let fused = HysteresisSegment::new(on, off).with_opts(opts).segment(&s);
    let reference = two_pass_reference(on, off, &opts, &s);
    assert_eq!(fused, reference, "on={on} off={off} opts={opts:?}");
  }
}

#[test]
fn fused_matches_two_pass_reference_on_randomized_finite_inputs() {
  // ~200 deterministic pseudo-random finite cases: the fused single pass must
  // equal the two-pass reference exactly for every one. Geometry is one of unit,
  // adjacent (hop == window), gapped (hop > window), or overlapping (hop <
  // window), all in ascending span order by construction; thresholds are either
  // uniform or sampled from the generated scores, so exact `v == on` / `v == off`
  // boundaries are exercised, and both `on >= off` and `on < off` orderings occur.
  let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
  for _ in 0..200 {
    let n = (xorshift(&mut state) % 257) as usize;
    let window = (xorshift(&mut state) % 8 + 1) as usize;
    let geometry = xorshift(&mut state) % 4;
    let hop = match geometry {
      0 => 1,                                                   // unit spans (len 1)
      1 => window,                                              // adjacent
      2 => window + 1 + (xorshift(&mut state) % 4) as usize,    // gapped (hop > window)
      _ => 1 + (xorshift(&mut state) % window as u64) as usize, // overlapping (hop <= window)
    };
    let span_len = if geometry == 0 { 1 } else { window };

    let mut scores: Vec<f32> = Vec::with_capacity(n);
    let seq: Vec<Windowed<f32>> = (0..n)
      .map(|i| {
        let v = next_unit(&mut state);
        scores.push(v);
        Windowed::new(v, Span::new(i * hop, span_len, window))
      })
      .collect();

    // Draw each threshold either uniformly in [0,1) or from an actual score, so
    // exact-boundary equality is hit; independence makes both orderings appear.
    let threshold = |state: &mut u64| -> f32 {
      if xorshift(state).is_multiple_of(2) || scores.is_empty() {
        next_unit(state)
      } else {
        scores[(xorshift(state) % scores.len() as u64) as usize]
      }
    };
    let on = threshold(&mut state);
    let off = threshold(&mut state);

    let merge_gap = [0usize, 1, 3][(xorshift(&mut state) % 3) as usize];
    let min_len = [0usize, 2, 5][(xorshift(&mut state) % 3) as usize];
    let opts = SegmentOptions::new()
      .with_merge_gap(merge_gap)
      .with_min_len(min_len);

    let fused = HysteresisSegment::new(on, off)
      .with_opts(opts)
      .segment(&seq);
    let reference = two_pass_reference(on, off, &opts, &seq);
    assert_eq!(
      fused, reference,
      "n={n} window={window} hop={hop} on={on} off={off} opts={opts:?}"
    );
  }
}
