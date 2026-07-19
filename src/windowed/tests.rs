use super::*;
use crate::test_support::{assert_close, TestVec};

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
fn zero_norm_is_nonfinite_error() {
  assert!(matches!(
    TestVec::from_unnormalized(&[0.0, 0.0]),
    Err(WinditError::NonFinite)
  ));
}

#[test]
fn non_finite_input_errors() {
  assert!(matches!(
    TestVec::from_unnormalized(&[f32::INFINITY, 1.0]),
    Err(WinditError::NonFinite)
  ));
  assert!(matches!(
    TestVec::from_unnormalized(&[f32::NAN, 1.0]),
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
