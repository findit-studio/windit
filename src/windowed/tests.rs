use super::*;

/// A minimal embedding double that L2-normalizes on construction. Uses `sqrt`,
/// so the whole test module is `std`-gated; the [`Vector`] trait itself is
/// no_std.
struct TestVec(Vec<f32>);

impl Vector for TestVec {
  fn as_slice(&self) -> &[f32] {
    &self.0
  }

  fn from_unnormalized(v: &[f32]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(v.iter().map(|x| x / norm).collect()))
  }
}

fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(got.len(), want.len(), "len mismatch: {got:?} vs {want:?}");
  for (g, w) in got.iter().zip(want) {
    assert!((g - w).abs() < 1e-6, "value mismatch: {got:?} vs {want:?}");
  }
}

#[test]
fn from_unnormalized_l2_normalizes() {
  // [3, 4] has norm 5, so the unit vector is [0.6, 0.8].
  let v = TestVec::from_unnormalized(&[3.0, 4.0]).unwrap();
  assert_close(v.as_slice(), &[0.6, 0.8]);
  assert_eq!(v.dim(), 2);
}

#[test]
fn windowed_carries_value_and_span() {
  let span = Span {
    start: 0,
    len: 2,
    window: 4,
  };
  let e: WindowEmbedding<TestVec> =
    Windowed::new(TestVec::from_unnormalized(&[3.0, 4.0]).unwrap(), span);
  assert_eq!(e.span(), span);
  assert_eq!(e.span.start, 0);
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
