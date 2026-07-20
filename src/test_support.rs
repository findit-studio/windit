//! Shared test doubles and helpers, compiled only under `cfg(test)`.
//!
//! Two embedding doubles and one float-closeness assertion ([`assert_close`])
//! serve every module's `tests.rs`, so the crate carries its test [`Vector`]
//! implementations in one place instead of redefining them per module.
//! [`TestVec`] is the ordinary f32 embedding; [`TestQuantVec`] stores a narrower
//! scalar than it computes in, which is the only way to reach `aggregate`'s
//! widening path. No helper needs `std`: they normalize through `libm::sqrtf`
//! and avoid the std-only float methods.

use std::vec::Vec;

use crate::{
  error::WinditError,
  scalar::TestQuant,
  windowed::{ComputeOf, Vector},
};

/// A minimal embedding double that L2-normalizes on construction.
///
/// `from_unnormalized` divides by the L2 norm (via `libm::sqrtf`), rejecting an
/// empty slice with [`WinditError::Empty`] and a zero or non-finite norm with
/// [`WinditError::NonFinite`].
pub(crate) struct TestVec(pub(crate) Vec<f32>);

impl Vector for TestVec {
  type Scalar = f32;

  fn as_slice(&self) -> &[f32] {
    &self.0
  }

  fn from_unnormalized(v: &[f32]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>());
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(v.iter().map(|x| x / norm).collect()))
  }
}

/// An embedding double whose storage is narrower than its compute domain: it
/// keeps [`TestQuant`] bytes but aggregates in `f32`.
///
/// This is the in-crate stand-in for a quantized embedding, and the only
/// implementor that drives `aggregate`'s widening branch. It also pins the
/// requantization contract [`Vector::from_unnormalized`] describes: normalize in
/// the compute domain, then round half away from zero and clamp symmetrically
/// (excluding `-128`, which has no positive counterpart) on the way into
/// storage.
pub(crate) struct TestQuantVec(pub(crate) Vec<TestQuant>);

impl Vector for TestQuantVec {
  type Scalar = TestQuant;

  fn as_slice(&self) -> &[TestQuant] {
    &self.0
  }

  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>());
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(
      v.iter()
        .map(|x| {
          let scaled = libm::roundf(x / norm * 127.0);
          TestQuant(scaled.clamp(-127.0, 127.0) as i8)
        })
        .collect(),
    ))
  }
}

/// Assert two f32 slices are elementwise within `1e-6`.
///
/// The difference is taken without `f32::abs` (std-only), so the helper stays
/// usable in the crate's `no_std` test configurations.
pub(crate) fn assert_close(got: &[f32], want: &[f32]) {
  assert_eq!(got.len(), want.len(), "len mismatch: {got:?} vs {want:?}");
  for (g, w) in got.iter().zip(want) {
    let diff = if g > w { g - w } else { w - g };
    assert!(diff < 1e-6, "value mismatch: {got:?} vs {want:?}");
  }
}

/// Assert two f64 slices are elementwise within `1e-12`.
///
/// The tolerance is six orders tighter than [`assert_close`]'s deliberately: f32
/// arithmetic cannot reach it, so a golden asserted through this helper is
/// evidence the math genuinely ran at f64 precision rather than being widened
/// after the fact.
pub(crate) fn assert_close_f64(got: &[f64], want: &[f64]) {
  assert_eq!(got.len(), want.len(), "len mismatch: {got:?} vs {want:?}");
  for (g, w) in got.iter().zip(want) {
    let diff = if g > w { g - w } else { w - g };
    assert!(diff < 1e-12, "value mismatch: {got:?} vs {want:?}");
  }
}
