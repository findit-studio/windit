use super::*;
use crate::{
  scalar::Scalar,
  test_support::{assert_close, RawF64Emb, TestVec},
};
use std::{borrow::Cow, vec, vec::Vec};

#[test]
fn from_unnormalized_l2_normalizes() {
  // [3, 4] has norm 5, so the unit vector is [0.6, 0.8].
  let v = TestVec::from_unnormalized(&[3.0, 4.0]).unwrap();
  assert_close(v.as_slice(), &[0.6, 0.8]);
  assert_eq!(v.dim(), 2);
}

#[test]
fn windowed_carries_value_and_span() {
  let span = Span::new(0, 2, 4);
  let e: WindowEmbedding<TestVec> =
    Windowed::new(TestVec::from_unnormalized(&[3.0, 4.0]).unwrap(), span);
  assert_eq!(e.span(), span);
  assert_eq!(e.span().start(), 0);
  assert_close(e.value().as_slice(), &[0.6, 0.8]);
  assert_eq!(e.value.dim(), 2);
}

#[test]
fn windowed_moves_value_out() {
  let span = Span::new(0, 2, 4);
  let e: WindowEmbedding<TestVec> =
    Windowed::new(TestVec::from_unnormalized(&[3.0, 4.0]).unwrap(), span);
  assert_close(e.into_value().as_slice(), &[0.6, 0.8]);

  let e: WindowEmbedding<TestVec> =
    Windowed::new(TestVec::from_unnormalized(&[3.0, 4.0]).unwrap(), span);
  let (value, got_span) = e.into_parts();
  assert_eq!(got_span, span);
  assert_close(value.as_slice(), &[0.6, 0.8]);
}

#[test]
fn zero_norm_is_nonfinite_error() {
  assert!(matches!(
    TestVec::from_unnormalized(&[0.0, 0.0]),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn non_finite_input_errors() {
  assert!(matches!(
    TestVec::from_unnormalized(&[f64::INFINITY, 1.0]),
    Err(WinditError::NonFinite)
  ));
  assert!(matches!(
    TestVec::from_unnormalized(&[f64::NAN, 1.0]),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn empty_input_errors() {
  assert!(matches!(
    TestVec::from_unnormalized(&[]),
    Err(WinditError::Empty)
  ));
}

#[test]
fn default_projection_widens_f32_like_to_compute() {
  // A TestVec stores f32, so the default projection cannot borrow: it widens
  // elementwise through `Scalar::to_compute`, bitwise-identical to the widening
  // loop `aggregate` used before the projection method existed.
  let v = TestVec::from_unnormalized(&[3.0, 4.0]).unwrap(); // [0.6, 0.8] in f32
  let cow = v.compute_components().unwrap();
  assert!(
    matches!(cow, Cow::Owned(_)),
    "f32 storage must own its widening"
  );
  let want: Vec<f64> = v.as_slice().iter().map(|s| s.to_compute()).collect();
  assert_eq!(cow.len(), want.len());
  for (g, w) in cow.iter().zip(&want) {
    assert_eq!(
      g.to_bits(),
      w.to_bits(),
      "widening must match to_compute bitwise"
    );
  }
}

#[test]
fn default_projection_borrows_f64_zero_copy() {
  // f64 storage IS the compute scalar, so the default projection borrows it with
  // no copy: the returned slice points at the very storage `as_slice` returns.
  let e = RawF64Emb {
    data: vec![1.0, -2.0, 3.5],
    captured: Vec::new(),
  };
  let cow = e.compute_components().unwrap();
  assert!(matches!(cow, Cow::Borrowed(_)), "f64 storage must borrow");
  assert!(core::ptr::eq(cow.as_ref().as_ptr(), e.as_slice().as_ptr()));
  assert_eq!(cow.as_ref(), e.as_slice());
}
