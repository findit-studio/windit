use super::{Real, Scalar, TestQuant};

#[test]
fn identities_hold_at_f64() {
  // `f64` is the only shipped `Real`; `f32` is storage only and has no `Real`
  // identities to assert.
  assert_eq!(<f64 as Real>::ZERO + 1.5, 1.5);
  assert_eq!(<f64 as Real>::ONE * 1.5, 1.5);
}

#[test]
fn from_f32_is_exact() {
  // Every f32 is exactly representable in f64, so widening a coverage or an EMA
  // alpha must not perturb it: this is what lets policy configuration stay f32
  // while the math runs at the embedding's precision.
  for x in [0.0f32, 1.0, 0.5, 0.25, 0.1, 0.7, -3.25, f32::MIN_POSITIVE] {
    assert_eq!(<f64 as Real>::from_f32(x), f64::from(x));
    assert_eq!(<f64 as Real>::from_f32(x) as f32, x);
  }
}

#[test]
fn sqrt_matches_known_values() {
  assert_eq!(<f64 as Real>::sqrt(0.0), 0.0);
  assert_eq!(<f64 as Real>::sqrt(1.0), 1.0);
  assert_eq!(<f64 as Real>::sqrt(4.0), 2.0);
  assert_eq!(<f64 as Real>::sqrt(6.25), 2.5);

  // sqrt is correctly rounded per IEEE-754, so the f64 root of 2 carries digits
  // an f32 root cannot: the compute domain is genuinely f64, not a widened f32.
  assert!(<f64 as Real>::sqrt(2.0) != f64::from(core::f32::consts::SQRT_2));
  assert!((<f64 as Real>::sqrt(2.0) - core::f64::consts::SQRT_2).abs() < 1e-15);
}

#[test]
fn abs_drops_the_sign_and_keeps_the_magnitude() {
  // Also the guard against `Real::abs` resolving to itself: a recursive
  // implementation would not return `libm::fabs`'s exact magnitude here.
  assert_eq!(<f64 as Real>::abs(2.5), 2.5);
  assert_eq!(<f64 as Real>::abs(-2.5), 2.5);
  assert_eq!(<f64 as Real>::abs(0.0), 0.0);
  assert_eq!(<f64 as Real>::abs(-0.0), 0.0);

  // Exact at the extremes the scale-aware norm divides by, where an approximate
  // magnitude would reintroduce the overflow it exists to avoid.
  assert_eq!(<f64 as Real>::abs(-f64::MAX), f64::MAX);
  assert_eq!(<f64 as Real>::abs(-f64::MIN_POSITIVE), f64::MIN_POSITIVE);
  assert!(!<f64 as Real>::abs(f64::NAN).is_finite());
  assert_eq!(<f64 as Real>::abs(f64::NEG_INFINITY), f64::INFINITY);
}

#[test]
fn exponent_brackets_the_magnitude() {
  // The contract the scale-aware reductions divide by: `2^e <= |x| < 2^(e+1)`,
  // so `x / 2^e` always lands in [1, 2) and a sum of such ratios can neither
  // overflow nor vanish.
  for x in [
    1.0f64,
    1.5,
    2.0,
    3.0,
    0.5,
    0.75,
    1e200,
    1e-300,
    f64::MAX,
    f64::MIN_POSITIVE,
  ] {
    let e = <f64 as Real>::exponent(x);
    assert!(
      <f64 as Real>::ldexp(1.0, e) <= x && x < <f64 as Real>::ldexp(1.0, e + 1),
      "f64 {x:e} is not bracketed by 2^{e}"
    );
    assert_eq!(
      <f64 as Real>::exponent(-x),
      e,
      "the sign must not move the exponent of {x:e}"
    );
  }

  // Subnormals report their true exponent rather than a flushed one, which is
  // what lets an embedding of `f64::from_bits(1)` be normalized rather than
  // divided by a zero scale and rejected.
  assert_eq!(<f64 as Real>::exponent(f64::from_bits(1)), -1074);
  assert_eq!(<f64 as Real>::exponent(f64::MIN_POSITIVE), -1022);
}

#[test]
fn ldexp_is_an_exact_power_of_two_rescale() {
  // The property the scale-aware norm rests on: a power of two moves the
  // exponent and leaves the significand, so shifting and shifting back recovers
  // the value bit for bit.
  for x in [
    1.0f64,
    0.6,
    3.0,
    -2.5,
    1e200,
    1e-300,
    f64::MAX,
    f64::from_bits(1),
  ] {
    for n in [-300i32, -1, 1, 300] {
      let there = <f64 as Real>::ldexp(x, n);
      // Only a normal intermediate round-trips: a shift into the subnormals
      // discards significand bits, which is why the renormalization sizes its
      // scale to keep the magnitudes it cares about normal.
      if there.is_finite() && <f64 as Real>::abs(there) >= f64::MIN_POSITIVE {
        assert_eq!(
          <f64 as Real>::ldexp(there, -n),
          x,
          "f64 {x:e} shifted by {n} must round-trip"
        );
      }
    }
  }

  // `n == 0` is the identity, which is what lets the renormalization apply its
  // scale unconditionally and still leave the ordinary case unchanged.
  for x in [0.0f64, 1.0, -2.5, f64::MAX, f64::MIN_POSITIVE] {
    assert_eq!(<f64 as Real>::ldexp(x, 0).to_bits(), x.to_bits());
  }
}

#[test]
fn is_finite_rejects_infinities_and_nan() {
  // Also the guard against `Real::is_finite` resolving to itself: a recursive
  // implementation would overflow the stack here rather than return.
  assert!(<f64 as Real>::is_finite(1.0));
  assert!(!<f64 as Real>::is_finite(f64::INFINITY));
  assert!(!<f64 as Real>::is_finite(f64::NEG_INFINITY));
  assert!(!<f64 as Real>::is_finite(f64::NAN));
}

#[test]
fn to_compute_widens_each_storage_scalar_to_f64() {
  // `f64` computes in itself; `f32` widens to `f64` exactly (subnormals
  // included), which is what routes an `f32` embedding through the compute
  // domain that ends the `f32` fold's numerical defect class.
  assert_eq!(<f64 as Scalar>::to_compute(1.5), 1.5f64);
  assert_eq!(<f32 as Scalar>::to_compute(1.5f32), 1.5f64);
  assert_eq!(
    <f32 as Scalar>::to_compute(f32::from_bits(1)),
    f64::from(f32::from_bits(1))
  );
}

#[test]
fn as_compute_slice_borrows_only_when_the_types_coincide() {
  // `f64` computes in itself, so its storage is reborrowed with no copy.
  let f64s = [1.0f64, 2.0];
  let borrowed = <f64 as Scalar>::as_compute_slice(&f64s).expect("f64 computes in itself");
  assert_eq!(borrowed, &f64s);
  assert!(core::ptr::eq(borrowed.as_ptr(), f64s.as_ptr()));

  // `f32` computes in `f64`, so it cannot be reborrowed and reports no borrow —
  // which is what routes an `f32` embedding down `aggregate`'s widening path.
  assert!(<f32 as Scalar>::as_compute_slice(&[1.0f32, 2.0]).is_none());

  // The integer widening double does the same.
  assert!(<TestQuant as Scalar>::as_compute_slice(&[TestQuant(1)]).is_none());
}

#[test]
fn test_quant_widens_to_its_scale() {
  assert_eq!(TestQuant(127).to_compute(), 1.0f64);
  assert_eq!(TestQuant(0).to_compute(), 0.0f64);
  assert_eq!(TestQuant(-127).to_compute(), -1.0f64);
}
