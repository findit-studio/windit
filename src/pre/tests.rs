use super::*;
use std::vec;

use crate::plan::{WindowOptions, WindowPlan};

#[test]
fn planner_window_beyond_the_allocator_is_a_typed_error() {
  // `WindowPlan::spans` with window `usize::MAX` over a one-element input is a
  // legal plan: the sole span is (start 0, len 1, window usize::MAX), which
  // satisfies every `Span` invariant and is in bounds for the slice it is then
  // applied to. So the checked entry point is reached with nothing left to
  // reject -- and buffering `usize::MAX` elements is what it must not do by
  // panicking, since returning `Result` instead of panicking is the whole point
  // of the `try_` variant.
  //
  // Debug and release both run this: the failure was `Vec::with_capacity`'s
  // capacity overflow, which is not a debug assertion.
  let spans = WindowPlan::spans(&WindowOptions::new(usize::MAX), 1).unwrap();
  assert_eq!(spans, vec![Span::new(0, 1, usize::MAX)]);

  assert_eq!(
    try_slice_pad_mask(&[42u8], &spans[0], 0),
    Err(WinditError::AllocFailed {
      elements: usize::MAX
    })
  );
}

#[test]
fn plan_longer_than_the_allocator_is_a_typed_error() {
  // The same class one step earlier: `input_len` is a caller-supplied count, not
  // a slice length, so an untrusted one drives the plan's own `Vec`. Uncapped,
  // this walked toward `usize::MAX` spans one `push` at a time; the plan length
  // is now computed and reserved up front, so an unservable one is reported
  // rather than approached.
  assert_eq!(
    WindowPlan::spans(&WindowOptions::new(1), usize::MAX),
    Err(WinditError::AllocFailed {
      elements: usize::MAX
    })
  );

  // A cap keeps its own error: planning aborts at the cap, so it never asks the
  // allocator for the full (unservable) plan in the first place.
  assert_eq!(
    WindowPlan::spans(&WindowOptions::new(1).with_max_windows(10), usize::MAX),
    Err(WinditError::TooManyWindows { got: 11, max: 10 })
  );
}

#[test]
fn full_window_all_real() {
  let s = Span::new(0, 4, 4);
  let (w, m) = slice_pad_mask(&[1, 2, 3, 4, 5], &s, 0);
  assert_eq!(w, vec![1, 2, 3, 4]);
  assert_eq!(m, vec![1, 1, 1, 1]);
}

#[test]
fn partial_window_right_padded_masked() {
  let s = Span::new(8, 2, 4);
  let (w, m) = slice_pad_mask(&[0; 10], &s, 7);
  assert_eq!(w.len(), 4);
  assert_eq!(m, vec![1, 1, 0, 0]);
  assert_eq!(&w[2..], &[7, 7]);
}

#[test]
fn try_variant_errors_on_oob() {
  let s = Span::new(9, 4, 4);
  assert!(try_slice_pad_mask(&[0; 10], &s, 0i32).is_err());
}

#[test]
fn unrepresentable_span_end_never_reaches_the_slice() {
  // Before the span invariant was enforced in release, this geometry reached
  // `try_slice_pad_mask`, wrapped `start + len` to `0`, passed the length check
  // and panicked on the slice index. It is now rejected where it is built.
  assert!(matches!(
    Span::try_new(usize::MAX, 1, 1),
    Err(WinditError::InvalidSpan { .. })
  ));

  // The largest span that does exist runs past any real slice, so the checked
  // entry point reports the mismatch with the exact required length rather than
  // wrapping into a length that would appear to fit.
  let s = Span::try_new(usize::MAX - 1, 1, 1).unwrap();
  assert_eq!(
    try_slice_pad_mask(&[0u8; 4], &s, 0),
    Err(WinditError::DimMismatch {
      got: 4,
      expected: usize::MAX
    })
  );
}

#[test]
fn try_variant_ok_matches_unchecked() {
  let s = Span::new(8, 2, 4);
  let checked = try_slice_pad_mask(&[0; 10], &s, 7).unwrap();
  let unchecked = slice_pad_mask(&[0; 10], &s, 7);
  assert_eq!(checked, unchecked);
}
