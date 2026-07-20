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
fn abs_drops_the_sign_and_keeps_the_magnitude() {
  // Also the guard against `Real::abs` resolving to itself, as for `is_finite`.
  assert_eq!(<f32 as Real>::abs(2.5), 2.5);
  assert_eq!(<f32 as Real>::abs(-2.5), 2.5);
  assert_eq!(<f32 as Real>::abs(0.0), 0.0);
  assert_eq!(<f32 as Real>::abs(-0.0), 0.0);
  assert_eq!(<f64 as Real>::abs(-2.5), 2.5);
  assert_eq!(<f64 as Real>::abs(0.0), 0.0);

  // Exact at the extremes the scale-aware norm divides by, where an approximate
  // magnitude would reintroduce the overflow it exists to avoid.
  assert_eq!(<f32 as Real>::abs(-f32::MAX), f32::MAX);
  assert_eq!(<f32 as Real>::abs(-f32::MIN_POSITIVE), f32::MIN_POSITIVE);
  assert_eq!(<f64 as Real>::abs(-f64::MAX), f64::MAX);
  assert!(!<f32 as Real>::abs(f32::NAN).is_finite());
  assert_eq!(<f32 as Real>::abs(f32::NEG_INFINITY), f32::INFINITY);
}

#[test]
fn exponent_brackets_the_magnitude() {
  // The contract the scale-aware reductions divide by: `2^e <= |x| < 2^(e+1)`,
  // so `x / 2^e` always lands in [1, 2) and a sum of such ratios can neither
  // overflow nor vanish.
  for x in [
    1.0f32,
    1.5,
    2.0,
    3.0,
    0.5,
    0.75,
    1e20,
    1e-30,
    f32::MAX,
    f32::MIN_POSITIVE,
  ] {
    let e = <f32 as Real>::exponent(x);
    assert!(
      <f32 as Real>::ldexp(1.0, e) <= x && x < <f32 as Real>::ldexp(1.0, e + 1),
      "f32 {x:e} is not bracketed by 2^{e}"
    );
    assert_eq!(
      <f32 as Real>::exponent(-x),
      e,
      "the sign must not move the exponent of {x:e}"
    );
  }
  for x in [1.0f64, 3.0, 1e200, 1e-300, f64::MAX, f64::MIN_POSITIVE] {
    let e = <f64 as Real>::exponent(x);
    assert!(
      <f64 as Real>::ldexp(1.0, e) <= x && x < <f64 as Real>::ldexp(1.0, e + 1),
      "f64 {x:e} is not bracketed by 2^{e}"
    );
  }

  // Subnormals report their true exponent rather than a flushed one, which is
  // what lets an embedding of `f32::from_bits(1)` be normalized rather than
  // divided by a zero scale and rejected.
  assert_eq!(<f32 as Real>::exponent(f32::from_bits(1)), -149);
  assert_eq!(<f32 as Real>::exponent(f32::MIN_POSITIVE), -126);
  assert_eq!(<f64 as Real>::exponent(f64::from_bits(1)), -1074);
  assert_eq!(<f64 as Real>::exponent(f64::MIN_POSITIVE), -1022);
}

#[test]
fn ldexp_is_an_exact_power_of_two_rescale() {
  // The property the whole scale-aware design rests on: a power of two moves the
  // exponent and leaves the significand, so shifting and shifting back recovers
  // the value bit for bit. An approximate rescale is exactly what turns a
  // cancelling sum into a small non-zero residue with an arbitrary direction.
  for x in [
    1.0f32,
    0.6,
    3.0,
    -2.5,
    1e20,
    1e-30,
    f32::MAX,
    f32::from_bits(1),
  ] {
    for n in [-40i32, -1, 1, 40] {
      let there = <f32 as Real>::ldexp(x, n);
      // Only a normal intermediate round-trips: a shift into the subnormals
      // discards significand bits, which is why the aggregation sizes its shifts
      // to keep the magnitudes it cares about normal.
      if there.is_finite() && <f32 as Real>::abs(there) >= f32::MIN_POSITIVE {
        assert_eq!(
          <f32 as Real>::ldexp(there, -n),
          x,
          "f32 {x:e} shifted by {n} must round-trip"
        );
      }
    }
  }

  // `n == 0` is the identity, which is what lets the accumulation apply its
  // shift unconditionally and still leave the ordinary case the fold it was.
  for x in [0.0f32, 1.0, -2.5, f32::MAX, f32::MIN_POSITIVE] {
    assert_eq!(<f32 as Real>::ldexp(x, 0).to_bits(), x.to_bits());
    let wide = f64::from(x);
    assert_eq!(<f64 as Real>::ldexp(wide, 0).to_bits(), wide.to_bits());
  }
}

#[test]
fn exponent_bounds_delimit_the_representable_powers_of_two() {
  // The accumulation sizes its shifts against these two constants, so what they
  // mean is asserted rather than restated: `MAX_EXP` is one past the largest
  // representable power of two, and `MIN_EXP - 1` is the smallest normal one.
  assert!(<f32 as Real>::ldexp(1.0, <f32 as Real>::MAX_EXP - 1).is_finite());
  assert!(!<f32 as Real>::ldexp(1.0, <f32 as Real>::MAX_EXP).is_finite());
  assert_eq!(
    <f32 as Real>::ldexp(1.0, <f32 as Real>::MIN_EXP - 1),
    f32::MIN_POSITIVE
  );
  assert!(<f64 as Real>::ldexp(1.0, <f64 as Real>::MAX_EXP - 1).is_finite());
  assert!(!<f64 as Real>::ldexp(1.0, <f64 as Real>::MAX_EXP).is_finite());
  assert_eq!(
    <f64 as Real>::ldexp(1.0, <f64 as Real>::MIN_EXP - 1),
    f64::MIN_POSITIVE
  );
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
