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

/// The same class one domain wider, and this is the one that survived widening:
/// `f64` holds every integer only to `2^53`, so a window of `2^53 + 1` casts to
/// the very same `f64` as the `2^53`-element tail inside it and the quotient is
/// exactly `1.0` again.
///
/// The defect was never "`f32` rounds too early". It is that **both operands are
/// rounded independently before the division**, and no wider domain removes
/// that — it only moves the first geometry that shows it. Reached through the
/// public planner in a single allocation (`input_len` is a count, not a slice
/// length, and this plan is one span), so nothing about it needs a hand-built
/// `Span`. 64-bit only: where `usize` is 32 bits every count is exact in `f64`
/// and the regime does not exist.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_past_f64_integer_range_is_below_one_for_a_ragged_tail() {
  let len = 1_usize << 53;
  let window = len + 1;
  assert_eq!(
    len as f64, window as f64,
    "the premise: both operands cast to one f64"
  );

  let s = WindowPlan::spans(&WindowOptions::new(window), len).unwrap();
  assert_eq!(s.len(), 1);
  assert_eq!((s[0].len(), s[0].window()), (len, window));

  let cov = s[0].coverage();
  assert!(
    cov < 1.0,
    "a tail one element short of a 2^53 + 1 window must not report full coverage, got {cov:?}"
  );
  assert_eq!(
    cov,
    f64::from_bits(0x3fef_ffff_ffff_ffff),
    "coverage must be the correctly rounded 2^53 / (2^53 + 1)"
  );
}

/// The integer path against an oracle: `Fraction(len, window)` rounded to the
/// nearest `f64` by an implementation with unbounded rationals (CPython), for ten
/// geometries whose window is past `2^53` and whose ratios span `2^-64` to within
/// two ulps of one.
///
/// The table is the whole point: nothing in this crate produced these bits, so
/// agreeing with them is evidence rather than a restatement.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_past_f64_integer_range_is_correctly_rounded() {
  const ORACLE: [(usize, usize, u64); 10] = [
    (1, usize::MAX, 0x3bf0_0000_0000_0000),
    (1, (1 << 63) + 1, 0x3c00_0000_0000_0000),
    (3, (1 << 54) + 1, 0x3ca8_0000_0000_0000),
    (7, 1 << 60, 0x3c5c_0000_0000_0000),
    (usize::MAX / 3, usize::MAX, 0x3fd5_5555_5555_5555),
    ((1 << 53) - 1, (1 << 54) + 1, 0x3fdf_ffff_ffff_ffff),
    ((1 << 62) + 12_345, (1 << 63) - 7, 0x3fe0_0000_0000_000c),
    (1 << 55, (1 << 55) + 1024, 0x3fef_ffff_ffff_ff00),
    ((1 << 53) + 1, (1 << 53) + 3, 0x3fef_ffff_ffff_fffe),
    (1 << 53, (1 << 53) + 1, 0x3fef_ffff_ffff_ffff),
  ];
  for (len, window, bits) in ORACLE {
    let got = Span::new(0, len, window).coverage();
    assert_eq!(
      got.to_bits(),
      bits,
      "{len}/{window}: got {got:?} ({:#018x}), oracle {:?}",
      got.to_bits(),
      f64::from_bits(bits)
    );
  }
}

/// The integer path against the `f64` division it replaces, over a sweep.
///
/// A ratio is unchanged by scaling both counts by a power of two, so
/// `ratio_to_f64(l << k, w << k)` must equal `ratio_to_f64(l, w)` — and for `w`
/// small the second is the exact-operand `f64` division, which IEEE already
/// requires to be correctly rounded. That makes this a cross-check of the two
/// paths against each other over 2485 geometries rather than a table of pinned
/// constants.
///
/// The shift decides which path the scaled pair takes, and `56` is the one that
/// matters: `window << 56` is past `2^53` for every window here, so every
/// geometry is checked on the integer path at least once (the test counts them
/// and asserts the count). `10` keeps the pair on the fast path, checking that a
/// ratio is scale-free there too, and `50` straddles the two.
///
/// The saturation regime is reached at no shift. Its test,
/// `(w - l) * 2^54 <= w`, has both sides scaled by `2^k` and so is invariant
/// under the shift — it reduces to the unshifted geometry, where a deficit of at
/// least `1` against a window of at most `70` cannot satisfy it.
#[test]
#[cfg(target_pointer_width = "64")]
fn the_integer_path_agrees_with_exact_operand_division() {
  let (mut checked, mut on_the_integer_path) = (0_u32, 0_u32);
  for window in 1_usize..=70 {
    for len in 1..=window {
      let direct = ratio_to_f64(len, window);
      assert_eq!(
        direct,
        len as f64 / window as f64,
        "{len}/{window} must take the exact-operand fast path"
      );
      for k in [10, 50, 56] {
        let (l, w) = (len << k, window << k);
        let scaled = ratio_to_f64(l, w);
        assert_eq!(
          scaled, direct,
          "{len}/{window} scaled by 2^{k} changed the ratio: {scaled:?} vs {direct:?}"
        );
        if w > 1 << 53 {
          on_the_integer_path += 1;
        }
      }
      checked += 1;
    }
  }
  assert_eq!(
    checked, 2485,
    "the sweep must cover every len <= window <= 70"
  );
  assert!(
    on_the_integer_path >= checked,
    "every geometry must reach the integer path at least once, got \
     {on_the_integer_path} integer-path checks over {checked} geometries"
  );
}

/// `coverage() == 1.0` if and only if `len == window`, including where the true
/// ratio is inside half an ulp of one and would round there.
///
/// `2^54 - 1` real elements in a window of `2^54` is the exact midpoint
/// `1 - 2^-54`, which ties to `1.0` — and `2^64 - 2` in `2^64 - 1` is far past
/// it. Both saturate down to the largest `f64` below one instead, which is what
/// keeps a ragged tail distinguishable from a full window at every geometry
/// rather than only at the ones `f64` resolves. The under-report is at most one
/// ulp and never in the direction of claiming coverage the span does not have.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_ragged_span_never_reports_full_coverage() {
  for (len, window) in [
    ((1_usize << 54) - 1, 1_usize << 54),
    ((1 << 63) - 1, 1 << 63),
    (1 << 63, (1 << 63) + 1),
    (usize::MAX - 1, usize::MAX),
  ] {
    let cov = Span::new(0, len, window).coverage();
    assert_eq!(
      cov, NEAREST_BELOW_ONE,
      "{len}/{window} must saturate below one, got {cov:?}"
    );
    assert!(cov < 1.0);
  }

  // And the equivalence in the other direction, at the same windows: only a full
  // span reports `1.0`.
  for window in [1_usize, 4, (1 << 54) - 1, 1 << 54, usize::MAX] {
    assert_eq!(Span::new(0, window, window).coverage(), 1.0);
  }
}

/// The tie-breaking rule, which only an exactly-halfway ratio can observe.
///
/// A tie needs the window to divide the scaled real length exactly *and* the
/// quotient's dropped bits to be `100…0`; both happen, and the rule is round to
/// nearest with ties to **even**, the same rule IEEE division follows on the fast
/// path. `2^54 - 3` and `2^54 - 5` real elements in a window of `2^54` are the
/// pair that pins it from both sides: their exact ratios are the two midpoints
/// either side of `1 - 2^-52`, so one must round *down* to it and the other *up*,
/// and any rule that always breaks a tie the same way gets one of them wrong.
#[test]
#[cfg(target_pointer_width = "64")]
fn coverage_breaks_an_exact_tie_to_even() {
  for (len, window, bits) in [
    ((1_usize << 54) - 3, 1_usize << 54, 0x3fef_ffff_ffff_fffe),
    ((1 << 54) - 5, 1 << 54, 0x3fef_ffff_ffff_fffe),
    (14_001_415_880_023_897, 1 << 54, 0x3fe8_df1b_55f0_cfac),
  ] {
    let got = Span::new(0, len, window).coverage();
    assert_eq!(
      got.to_bits(),
      bits,
      "{len}/{window} is an exact tie and must round to even: got {got:?} ({:#018x})",
      got.to_bits()
    );
  }
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
