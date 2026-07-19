use std::{vec, vec::Vec};

use super::{
  longest_run, runs, runs_sorted, HysteresisSegment, Range, SegmentOptions, SegmentPolicy,
  Threshold,
};
use crate::{plan::Span, windowed::Windowed};

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
fn hysteresis_segment_reuses_smooth_then_runs() {
  // Same latch as smooth::Hysteresis (on 0.6, off 0.3): [0,1,1,0,1] over the
  // input, whose set frames form runs [1,3) and [4,5).
  let s = seq(&[0.1, 0.7, 0.5, 0.2, 0.6]);
  let policy = HysteresisSegment::new(0.6, 0.3);
  assert_eq!(policy.segment(&s), vec![Range::new(1, 3), Range::new(4, 5)]);
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
  // `Range::new` rejects an inverted range only through a debug assertion, so a
  // release build can still hold one. This test builds it through the struct
  // literal — reachable here because `tests` is a child of the defining module —
  // to stand in for that release-build range, and pins `len` saturating to zero
  // rather than underflowing.
  let inverted = Range { start: 10, end: 5 };
  assert_eq!(inverted.len(), 0);
  assert!(inverted.is_empty());
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
