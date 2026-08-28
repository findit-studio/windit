use super::*;
use std::{vec, vec::Vec};

#[test]
fn exact_windows_no_overlap() {
  let o = WindowOptions::new(4);
  let s = WindowPlan::spans(&o, 12).unwrap();
  assert_eq!(s.len(), 3);
  assert_eq!((s[0].start(), s[0].len()), (0, 4));
  assert_eq!((s[2].start(), s[2].len()), (8, 4));
  assert_eq!(s[0].coverage(), 1.0);
}

#[test]
fn ragged_tail_keeps_partial_with_coverage() {
  let o = WindowOptions::new(4); // 10 elems -> [0..4],[4..8],[8..10 len2 cov .5]
  let s = WindowPlan::spans(&o, 10).unwrap();
  assert_eq!(s.len(), 3);
  assert_eq!((s[2].start(), s[2].len()), (8, 2));
  assert!((s[2].coverage() - 0.5).abs() < 1e-6);
}

#[test]
fn overlap_hops_correctly() {
  let o = WindowOptions::new(4).with_overlap(2); // hop 2
  let s = WindowPlan::spans(&o, 8).unwrap(); // [0..4],[2..6],[4..8]
  assert_eq!(
    s.iter().map(|x| x.start()).collect::<Vec<_>>(),
    vec![0, 2, 4]
  );
}

#[test]
fn drop_below_min_drops_short_tail() {
  let o = WindowOptions::new(4).with_tail(TailPolicy::DropBelowMin(2));
  let s = WindowPlan::spans(&o, 9).unwrap(); // tail len 1 < 2 -> dropped
  assert_eq!(s.last().map(|x| (x.start(), x.len())), Some((4, 4)));
}

#[test]
fn fits_in_one_window_is_single_span() {
  let o = WindowOptions::new(512);
  let s = WindowPlan::spans(&o, 40).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].start(), s[0].len()), (0, 40));
}

#[test]
fn max_windows_errors() {
  let o = WindowOptions::new(4).with_max_windows(2);
  assert!(matches!(
    WindowPlan::spans(&o, 100),
    Err(WinditError::TooManyWindows { .. })
  ));
}

#[test]
fn spans_no_overflow_near_usize_max() {
  // `input_len` within `window` of usize::MAX must not overflow the tail check
  // (a debug panic) nor wrap into an unbounded loop (release). The tail is
  // detected on the second window and packing stops.
  let opts = WindowOptions::new(usize::MAX - 10).with_hop(100);
  let s = WindowPlan::spans(&opts, usize::MAX - 5).unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!((s[0].start(), s[1].start()), (0, 100));

  // A hop whose advance would overflow `start + hop` breaks cleanly instead of
  // panicking: two windows are placed, then the advance saturates and stops.
  let hop = usize::MAX / 2 + 1;
  let s = WindowPlan::spans(&WindowOptions::new(10).with_hop(hop), usize::MAX).unwrap();
  assert_eq!(s.len(), 2);
  assert_eq!((s[0].start(), s[1].start()), (0, hop));
}

#[test]
fn zero_window_and_bad_overlap_error() {
  assert!(matches!(
    WindowOptions::new(0).validate(),
    Err(WinditError::ZeroWindow)
  ));
}

#[test]
fn zero_window_span_is_unconstructible_so_coverage_stays_finite() {
  // A zero window is ruled out by `0 < len <= window` rather than by a separate
  // check, which is what lets `coverage` divide unguarded.
  assert!(matches!(
    Span::try_new(0, 5, 0),
    Err(WinditError::InvalidSpan {
      start: 0,
      len: 5,
      window: 0
    })
  ));
  assert!(matches!(
    Span::try_new(0, 0, 0),
    Err(WinditError::InvalidSpan { .. })
  ));

  let cov = Span::new(0, 1, 4).coverage();
  assert!(cov.is_finite() && cov > 0.0 && cov <= 1.0);
}

#[test]
fn span_new_exposes_geometry_through_accessors() {
  let span = Span::new(8, 3, 4);
  assert_eq!((span.start(), span.len(), span.window()), (8, 3, 4));
  assert!((span.coverage() - 0.75).abs() < 1e-6);
}

/// A geometry past `f32`'s integer-exact range must report the exactly-correct
/// ratio, not one whose operands rounded before the division.
///
/// A window of `16_777_217` (`2^24 + 1`, the first integer `f32` cannot hold)
/// over `16_777_216` elements plans a single ragged tail, one element short of
/// full. Narrowing the operands first rounds that window down to `2^24` and the
/// tail reports as a full window at exactly `1.0` — the ragged tail and the full
/// window become indistinguishable. Both operands are exact in `f64`, so the
/// quotient there is the correctly-rounded ratio.
#[test]
fn coverage_past_f32_integer_range_is_the_exact_ratio() {
  let s = WindowPlan::spans(&WindowOptions::new(16_777_217), 16_777_216).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].len(), s[0].window()), (16_777_216, 16_777_217));

  let cov = s[0].coverage();
  assert!(
    cov < 1.0,
    "a tail one element short of the window must not report full coverage, got {cov:?}"
  );
  assert_eq!(
    cov,
    16_777_216.0 / 16_777_217.0,
    "coverage must be the exact ratio, not a ratio of rounded operands"
  );
}

#[test]
#[should_panic(expected = "0 < len <= window")]
fn span_new_rejects_len_above_window_in_every_build() {
  let _ = Span::new(0, 2, 1);
}

#[test]
#[should_panic(expected = "0 < len <= window")]
fn span_new_rejects_zero_len_in_every_build() {
  let _ = Span::new(0, 0, 4);
}

#[test]
#[should_panic(expected = "representable start + len")]
fn span_new_rejects_an_unrepresentable_end_in_every_build() {
  let _ = Span::new(usize::MAX, 1, 1);
}

#[test]
fn span_try_new_reports_the_same_invariant_as_a_typed_error() {
  for (start, len, window) in [(0, 2, 1), (0, 0, 4), (usize::MAX, 1, 1)] {
    assert_eq!(
      Span::try_new(start, len, window),
      Err(WinditError::InvalidSpan { start, len, window })
    );
  }

  let span = Span::try_new(8, 3, 4).unwrap();
  assert_eq!((span.start(), span.len(), span.window()), (8, 3, 4));
}

#[test]
fn span_end_is_exact_at_the_usize_boundary() {
  // The largest constructible span: its end is exactly `usize::MAX`, so `end`
  // neither wraps nor saturates away a real element.
  let span = Span::try_new(usize::MAX - 1, 1, 1).unwrap();
  assert_eq!(span.end(), usize::MAX);
  assert_eq!(Span::new(8, 3, 4).end(), 11);
}

#[cfg(feature = "serde")]
#[test]
fn window_options_serde_round_trip() {
  // Every field set to a non-default value, including the tuple tail variant and
  // the window cap, so the round trip covers the whole geometry.
  let opts = WindowOptions::new(512)
    .with_overlap(64)
    .with_tail(TailPolicy::DropBelowMin(10))
    .with_max_windows(8);
  let json = serde_json::to_string(&opts).unwrap();
  let back: WindowOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(opts, back);

  // The default tail and absent cap (a `None`) round-trip as well.
  let simple = WindowOptions::new(4);
  let back: WindowOptions = serde_json::from_str(&serde_json::to_string(&simple).unwrap()).unwrap();
  assert_eq!(simple, back);
}

#[cfg(feature = "serde")]
#[test]
fn window_options_serde_optional_cap_and_rejects_unknown_keys() {
  // `max_windows` may be omitted; it then defaults to no cap.
  let opts: WindowOptions =
    serde_json::from_str(r#"{"window":4,"hop":4,"tail":"KeepWithCoverage"}"#).unwrap();
  assert_eq!(opts.max_windows(), None);

  // A required (non-optional) field is still enforced.
  assert!(serde_json::from_str::<WindowOptions>(r#"{"window":4,"hop":4}"#).is_err());

  // An unknown key (e.g. a typo'd `max_window`) is rejected by
  // `deny_unknown_fields` rather than silently ignored.
  assert!(serde_json::from_str::<WindowOptions>(
    r#"{"window":4,"hop":4,"tail":"KeepWithCoverage","max_window":2}"#
  )
  .is_err());
}
