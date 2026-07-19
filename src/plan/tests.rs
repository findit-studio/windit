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
fn zero_window_span_coverage_is_zero() {
  // `Span::new` rejects a zero window only through a debug assertion, so a
  // release build can still hold one. This test builds it through the struct
  // literal — reachable here because `tests` is a child of the defining module —
  // to stand in for that release-build span, and pins `coverage` reporting `0.0`
  // rather than dividing by zero (which would yield `inf`, or `NaN` for a zero
  // `len` too). The planner itself never produces one.
  let span = Span {
    start: 0,
    len: 5,
    window: 0,
  };
  assert_eq!(span.coverage(), 0.0);
  assert!(span.coverage().is_finite());
}

#[test]
fn span_new_exposes_geometry_through_accessors() {
  let span = Span::new(8, 3, 4);
  assert_eq!((span.start(), span.len(), span.window()), (8, 3, 4));
  assert!((span.coverage() - 0.75).abs() < 1e-6);
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
