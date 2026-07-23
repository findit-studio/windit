//! Shared test doubles and helpers, compiled only under `cfg(test)`.
//!
//! The embedding doubles and two float-closeness assertions ([`assert_close`],
//! [`assert_close_f64`]) serve every module's `tests.rs`, so the crate carries
//! its test [`Vector`] implementations in one place instead of redefining them
//! per module. [`TestVec`] is the ordinary f32 embedding — which now computes in
//! `f64`, so its `from_unnormalized` takes `&[f64]` and narrows on the way into
//! storage; [`TestQuantVec`] stores a fixed-scale integer scalar. [`QuantEmb`],
//! [`RawF64Emb`], and [`BareI8Emb`] are the quantization doubles: a scale-bearing
//! `i8` embedding that overrides `compute_components`, its hand-dequantized `f64`
//! reference, and a bare `i8` embedding with no override (for the fail-closed
//! guard). No helper needs `std`: they normalize through `libm::sqrt` and avoid
//! the std-only float methods.

use std::{borrow::Cow, vec::Vec};

use crate::{
  error::WinditError,
  scalar::TestQuant,
  windowed::{ComputeOf, Vector},
};

/// A minimal embedding double that L2-normalizes on construction.
///
/// It stores `f32` but computes in `f64` (`f32`'s compute type), so
/// `from_unnormalized` takes `&[f64]`, normalizes there, and narrows each
/// component to `f32` on the way into storage. It rejects an empty slice with
/// [`WinditError::Empty`] and a zero or non-finite norm with
/// [`WinditError::NonFinite`].
pub(crate) struct TestVec(pub(crate) Vec<f32>);

impl Vector for TestVec {
  type Scalar = f32;

  fn as_slice(&self) -> &[f32] {
    &self.0
  }

  fn from_unnormalized(v: &[f64]) -> Result<Self, WinditError> {
    if v.is_empty() {
      return Err(WinditError::Empty);
    }
    let norm = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(v.iter().map(|x| (x / norm) as f32).collect()))
  }
}

/// An embedding double whose storage is narrower than its compute domain: it
/// keeps [`TestQuant`] bytes but aggregates in `f64`.
///
/// A fixed-scale (1/127) stand-in for a quantized embedding: its `TestQuant`
/// widening is value-preserving, so it needs no `compute_components` override and
/// drives the default projection's elementwise-widening path. It also pins the
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
    let norm = libm::sqrt(v.iter().map(|x| x * x).sum::<f64>());
    if !norm.is_finite() || norm == 0.0 {
      return Err(WinditError::NonFinite);
    }
    Ok(Self(
      v.iter()
        .map(|x| {
          let scaled = libm::round(x / norm * 127.0);
          TestQuant(scaled.clamp(-127.0, 127.0) as i8)
        })
        .collect(),
    ))
  }
}

/// A quantized-embedding double: `i8` codes plus per-embedding affine parameters,
/// overriding [`Vector::compute_components`] with the dequantization only it
/// knows. It is the crate's reference implementation of the pattern the
/// [`WinditError::MissingDequantization`] guard steers quantized storage toward.
///
/// `from_unnormalized` captures the incoming compute-domain slice verbatim into
/// `captured` (leaving `codes` empty), so a test can read an aggregation's result
/// at full `f64` precision without the requantization loss a real quantized
/// `from_unnormalized` would impose.
pub(crate) struct QuantEmb {
  pub(crate) codes: Vec<i8>,
  pub(crate) scale: f64,
  pub(crate) zero_point: i8,
  pub(crate) captured: Vec<f64>,
}

impl Vector for QuantEmb {
  type Scalar = i8;

  fn as_slice(&self) -> &[i8] {
    &self.codes
  }

  fn compute_components(&self) -> Result<Cow<'_, [f64]>, WinditError> {
    // Affine dequant `scale * (q - zero_point)`: the subtraction is exact in i16
    // (127 - (-128) does not overflow i8), a code equal to zero_point yields
    // exactly 0.0, and each component costs one rounding.
    let mut out = crate::aggregate::try_vec_with_capacity(self.codes.len())?;
    for &q in &self.codes {
      out.push(self.scale * f64::from(i16::from(q) - i16::from(self.zero_point)));
    }
    Ok(Cow::Owned(out))
  }

  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    Ok(Self {
      codes: Vec::new(),
      scale: 1.0,
      zero_point: 0,
      captured: v.to_vec(),
    })
  }
}

/// An `f64`-storage double that uses the default `compute_components` projection
/// (the zero-copy `Cow::Borrowed` path), capturing the aggregation result
/// verbatim like [`QuantEmb`]. The reference side of the quantized differential:
/// the same aggregation over hand-dequantized `f64` values.
pub(crate) struct RawF64Emb {
  pub(crate) data: Vec<f64>,
  pub(crate) captured: Vec<f64>,
}

impl Vector for RawF64Emb {
  type Scalar = f64;

  fn as_slice(&self) -> &[f64] {
    &self.data
  }

  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    Ok(Self {
      data: Vec::new(),
      captured: v.to_vec(),
    })
  }
}

/// An `i8`-storage double with **no** `compute_components` override, so the
/// default projection refuses it with [`WinditError::MissingDequantization`].
/// Exists only to pin that fail-closed guard.
pub(crate) struct BareI8Emb(pub(crate) Vec<i8>);

impl Vector for BareI8Emb {
  type Scalar = i8;

  fn as_slice(&self) -> &[i8] {
    &self.0
  }

  fn from_unnormalized(_v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    // Never reached: the guard fires before reconstruction. Total because the
    // trait method is.
    Ok(Self(Vec::new()))
  }
}

/// An `f16`-storage double using the default projection (widens elementwise to
/// `f64`), capturing the aggregation result verbatim like [`QuantEmb`].
#[cfg(feature = "half")]
pub(crate) struct HalfEmb {
  pub(crate) data: Vec<crate::scalar::f16>,
  pub(crate) captured: Vec<f64>,
}

#[cfg(feature = "half")]
impl Vector for HalfEmb {
  type Scalar = crate::scalar::f16;

  fn as_slice(&self) -> &[crate::scalar::f16] {
    &self.data
  }

  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    Ok(Self {
      data: Vec::new(),
      captured: v.to_vec(),
    })
  }
}

/// A `bf16`-storage double, the `bf16` counterpart to [`HalfEmb`].
#[cfg(feature = "half")]
pub(crate) struct Bf16Emb {
  pub(crate) data: Vec<crate::scalar::bf16>,
  pub(crate) captured: Vec<f64>,
}

#[cfg(feature = "half")]
impl Vector for Bf16Emb {
  type Scalar = crate::scalar::bf16;

  fn as_slice(&self) -> &[crate::scalar::bf16] {
    &self.data
  }

  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError> {
    Ok(Self {
      data: Vec::new(),
      captured: v.to_vec(),
    })
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
