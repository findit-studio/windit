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
fn try_variant_ok_matches_unchecked() {
  let s = Span::new(8, 2, 4);
  let checked = try_slice_pad_mask(&[0; 10], &s, 7).unwrap();
  let unchecked = slice_pad_mask(&[0; 10], &s, 7);
  assert_eq!(checked, unchecked);
}
