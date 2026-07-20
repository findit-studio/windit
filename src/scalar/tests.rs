use super::{Real, Scalar, TestQuant};

#[test]
fn identities_hold_at_both_scalars() {
  assert_eq!(<f32 as Real>::ZERO + 1.5, 1.5);
  assert_eq!(<f32 as Real>::ONE * 1.5, 1.5);
  assert_eq!(<f64 as Real>::ZERO + 1.5, 1.5);
  assert_eq!(<f64 as Real>::ONE * 1.5, 1.5);
}

#[test]
fn from_f32_is_exact() {
  // Every f32 is exactly representable in f64, so widening a coverage or an EMA
  // alpha must not perturb it: this is what lets policy configuration stay f32
  // while the math runs at the embedding's precision.
  for x in [0.0f32, 1.0, 0.5, 0.25, 0.1, 0.7, -3.25, f32::MIN_POSITIVE] {
    assert_eq!(<f32 as Real>::from_f32(x), x);
    assert_eq!(<f64 as Real>::from_f32(x), f64::from(x));
    assert_eq!(<f64 as Real>::from_f32(x) as f32, x);
  }
}

#[test]
fn sqrt_matches_known_values() {
  assert_eq!(<f32 as Real>::sqrt(0.0), 0.0);
  assert_eq!(<f32 as Real>::sqrt(1.0), 1.0);
  assert_eq!(<f32 as Real>::sqrt(4.0), 2.0);
  assert_eq!(<f32 as Real>::sqrt(6.25), 2.5);
  assert_eq!(<f64 as Real>::sqrt(0.0), 0.0);
  assert_eq!(<f64 as Real>::sqrt(1.0), 1.0);
  assert_eq!(<f64 as Real>::sqrt(4.0), 2.0);
  assert_eq!(<f64 as Real>::sqrt(6.25), 2.5);

  // sqrt is correctly rounded per IEEE-754, so the f64 root of 2 carries digits
  // the f32 root cannot: the two traits are not the same computation.
  assert!(<f64 as Real>::sqrt(2.0) != f64::from(<f32 as Real>::sqrt(2.0)));
}

#[test]
fn is_finite_rejects_infinities_and_nan() {
  // Also the guard against `Real::is_finite` resolving to itself: a recursive
  // implementation would overflow the stack here rather than return.
  assert!(<f32 as Real>::is_finite(1.0));
  assert!(!<f32 as Real>::is_finite(f32::INFINITY));
  assert!(!<f32 as Real>::is_finite(f32::NEG_INFINITY));
  assert!(!<f32 as Real>::is_finite(f32::NAN));
  assert!(<f64 as Real>::is_finite(1.0));
  assert!(!<f64 as Real>::is_finite(f64::INFINITY));
  assert!(!<f64 as Real>::is_finite(f64::NEG_INFINITY));
  assert!(!<f64 as Real>::is_finite(f64::NAN));
}

#[test]
fn to_compute_is_the_identity_for_the_shipped_scalars() {
  assert_eq!(<f32 as Scalar>::to_compute(1.5), 1.5f32);
  assert_eq!(<f64 as Scalar>::to_compute(1.5), 1.5f64);
}

#[test]
fn as_compute_slice_borrows_only_when_the_types_coincide() {
  let f32s = [1.0f32, 2.0];
  let borrowed = <f32 as Scalar>::as_compute_slice(&f32s).expect("f32 computes in itself");
  assert_eq!(borrowed, &f32s);
  assert!(core::ptr::eq(borrowed.as_ptr(), f32s.as_ptr()));

  let f64s = [1.0f64, 2.0];
  let borrowed = <f64 as Scalar>::as_compute_slice(&f64s).expect("f64 computes in itself");
  assert_eq!(borrowed, &f64s);
  assert!(core::ptr::eq(borrowed.as_ptr(), f64s.as_ptr()));

  // The widening double reports no borrow, which is what routes `aggregate`
  // down its elementwise path.
  assert!(<TestQuant as Scalar>::as_compute_slice(&[TestQuant(1)]).is_none());
}

#[test]
fn test_quant_widens_to_its_scale() {
  assert_eq!(TestQuant(127).to_compute(), 1.0);
  assert_eq!(TestQuant(0).to_compute(), 0.0);
  assert_eq!(TestQuant(-127).to_compute(), -1.0);
}
