use super::*;
use alloc::vec;

#[test]
fn full_window_all_real() {
  let s = Span {
    start: 0,
    len: 4,
    window: 4,
  };
  let (w, m) = slice_pad_mask(&[1, 2, 3, 4, 5], &s, 0);
  assert_eq!(w, vec![1, 2, 3, 4]);
  assert_eq!(m, vec![1, 1, 1, 1]);
}

#[test]
fn partial_window_right_padded_masked() {
  let s = Span {
    start: 8,
    len: 2,
    window: 4,
  };
  let (w, m) = slice_pad_mask(&[0; 10], &s, 7);
  assert_eq!(w.len(), 4);
  assert_eq!(m, vec![1, 1, 0, 0]);
  assert_eq!(&w[2..], &[7, 7]);
}

#[test]
fn try_variant_errors_on_oob() {
  let s = Span {
    start: 9,
    len: 4,
    window: 4,
  };
  assert!(try_slice_pad_mask(&[0; 10], &s, 0i32).is_err());
}

#[test]
fn try_variant_ok_matches_unchecked() {
  let s = Span {
    start: 8,
    len: 2,
    window: 4,
  };
  let checked = try_slice_pad_mask(&[0; 10], &s, 7).unwrap();
  let unchecked = slice_pad_mask(&[0; 10], &s, 7);
  assert_eq!(checked, unchecked);
}
