//! Shared test doubles and helpers, compiled only under `cfg(test)`.
//!
//! One embedding double ([`TestVec`]) and one float-closeness assertion
//! ([`assert_close`]) serve every module's `tests.rs`, so the crate carries a
//! single [`Vector`] implementation for its tests instead of redefining one per
//! module. Neither helper needs `std`: [`TestVec`] normalizes through
//! `libm::sqrtf` and [`assert_close`] avoids the std-only float methods.

use std::vec::Vec;

use crate::{error::WinditError, windowed::Vector};

/// A minimal embedding double that L2-normalizes on construction.
///
/// `from_unnormalized` divides by the L2 norm (via `libm::sqrtf`), rejecting an
/// empty slice with [`WinditError::Empty`] and a zero or non-finite norm with
/// [`WinditError::NonFinite`].
pub(crate) struct TestVec(pub(crate) Vec<f32>);

impl Vector for TestVec {
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
