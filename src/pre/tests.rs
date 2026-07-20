use super::*;
use std::vec;

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
