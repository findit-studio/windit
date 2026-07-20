//! Scalar types an embedding can be stored and aggregated in.
//!
//! [`Scalar`] is the storage type — [`Vector::as_slice`](crate::windowed::Vector::as_slice)
//! yields `&[Self::Scalar]` — and [`Real`] is the floating-point domain the
//! aggregation math runs in. The two are deliberately *not* the same for `f32`:
//! an `f32` embedding stores `f32` but computes in `f64`
//! ([`f32::Compute`](Scalar::Compute) is `f64`). That is the whole reason the
//! storage and compute types are split, and it is what a future narrow storage
//! type (an `f16`, a quantized byte) reuses to compute in a wider float without
//! a breaking change to
//! [`Vector::from_unnormalized`](crate::windowed::Vector::from_unnormalized).
//!
//! Widening `f32` to `f64` before accumulating is what ends a whole class of
//! numerical defect. Every `f32` is exact in `f64`, every `f32` subnormal is a
//! normal `f64`, and `f32::MAX` squared (~1.2e77) sits far inside `f64`'s range
//! (~1.8e308), so a sum that overflowed, flushed a subnormal, or lost a
//! cancellation in `f32` does none of those in `f64`. `f32` is therefore a
//! storage scalar only: it is **not** a [`Real`], and no embedding accumulates
//! in it.
//!
//! Two scalars are implemented: `f32` (storage, computes in `f64`) and `f64`
//! (both storage and its own compute domain — the sole [`Real`]). Neither is
//! feature-gated: both are `core` types, and monomorphization already makes an
//! unused one free.
//!
//! # Sealed
//!
//! Both traits are sealed: downstream code can name them and bound on them, but
//! only this crate can implement them. The aggregation math relies on invariants
//! these implementations uphold — a finite, ordered field with an exact widening
//! from the `f32` configuration values — which a downstream implementation could
//! silently violate. Sealing also means adding a scalar later is not a breaking
//! change. To request one, open an issue.
//!
//! Deliberately absent, and why:
//!
//! - **Bare integers (`i8`, `i16`, …).** Mechanically they would fit, since the
//!   square roots happen in [`Real`]. Semantically they do not: a bare `i8` has
//!   no value without a per-tensor quantization scale, and this crate cannot
//!   know it. Implementing `to_compute` would mean picking a scale of `1.0` and
//!   silently returning a wrong answer to everyone whose scale differs. The
//!   right shape is a storage wrapper that carries its own scale.
//! - **`f16` / `bf16`.** The intended next addition, and the case the
//!   storage/compute split exists for — `f16` has no arithmetic in `core`, so
//!   computing in `f64` (as `f32` already does) is mandatory rather than
//!   optional. Left out only to avoid taking a dependency for a use that has not
//!   arrived yet.

#[cfg(test)]
mod tests;

mod private {
  /// The seal. Implemented only for the scalars this crate ships, so
  /// [`Scalar`](super::Scalar) and [`Real`](super::Real) cannot be implemented
  /// downstream.
  pub trait Sealed {}

  impl Sealed for f32 {}
  impl Sealed for f64 {}
}

/// A scalar type an embedding can be stored in.
///
/// This trait is sealed; see the [module documentation](self) for the reason and
/// for the list of types that are deliberately not implemented.
pub trait Scalar: private::Sealed + Copy {
  /// The floating-point type this scalar's aggregation math runs in.
  ///
  /// `f64` for both shipped scalars: `f64` computes in itself, and `f32` widens
  /// to `f64` so no embedding ever accumulates in `f32`. A storage type equal to
  /// its own `Compute` (only `f64`, here) is a [`Real`]; one narrower than its
  /// `Compute` (`f32`) is storage only.
  type Compute: Real;

  /// Widen one stored value into the compute domain.
  fn to_compute(self) -> Self::Compute;

  /// Borrow a stored slice as compute values when the two types coincide.
  ///
  /// Returns `Some` — a zero-copy reborrow — when `Compute` is `Self`, which
  /// lets aggregation skip a widening pass, and `None` when they differ, in
  /// which case the caller widens elementwise through
  /// [`to_compute`](Scalar::to_compute).
  fn as_compute_slice(v: &[Self]) -> Option<&[Self::Compute]>;
}

/// A floating-point scalar the aggregation math can run in.
///
/// This trait is sealed; see the [module documentation](self).
pub trait Real:
  Scalar<Compute = Self>
  + core::ops::Add<Output = Self>
  + core::ops::Sub<Output = Self>
  + core::ops::Mul<Output = Self>
  + core::ops::Div<Output = Self>
  + PartialOrd
{
  /// The additive identity.
  const ZERO: Self;

  /// The multiplicative identity.
  const ONE: Self;

  /// Widen an `f32` configuration value — a [`Span::coverage`] or an EMA
  /// smoothing factor — into this domain. Exact for every implementor.
  ///
  /// [`Span::coverage`]: crate::plan::Span::coverage
  fn from_f32(x: f32) -> Self;

  /// The square root, for L2 norms.
  fn sqrt(self) -> Self;

  /// The magnitude, ignoring sign.
  ///
  /// Aggregation uses this to find a vector's largest component and normalize
  /// against it, which is what lets an embedding whose *squares* leave the
  /// compute scalar's range — `f64::MAX` squares to infinity, `f64::MIN_POSITIVE`
  /// to zero — still be normalized rather than rejected.
  fn abs(self) -> Self;

  /// The binary exponent of `self`'s magnitude: the `e` for which
  /// `2^e <= |self| < 2^(e + 1)`.
  ///
  /// Defined for a finite, non-zero `self`; aggregation only ever asks for the
  /// exponent of a magnitude it has already checked for both. Subnormals report
  /// their true exponent (`f64::from_bits(1)` is `-1074`), not a flushed one.
  fn exponent(self) -> i32;

  /// `self * 2^n`, exact whenever the result is representable.
  ///
  /// A power of two leaves the significand untouched and moves only the
  /// exponent, so dividing a component by its own `2^exponent` and multiplying
  /// the root back afterwards is exact: renormalization computes a unit vector
  /// whose norm was never representable (`[f64::MAX, f64::MAX]`) without ever
  /// forming that norm, and the quotient is the direct `v_i / norm` to the bit
  /// wherever the direct computation was valid. `n == 0` returns `self`
  /// unchanged.
  fn ldexp(self, n: i32) -> Self;

  /// Whether the value is finite: neither infinite nor NaN.
  fn is_finite(self) -> bool;
}

impl Scalar for f32 {
  // Storage only: an `f32` embedding computes in `f64`. `f32` is not a `Real`,
  // so nothing ever accumulates in it — the exact property that ends the
  // overflow/underflow/cancellation class the `f32` fold kept reintroducing.
  type Compute = f64;

  fn to_compute(self) -> Self::Compute {
    // Widening `f32` to `f64` is exact for every value, subnormals included.
    f64::from(self)
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    // `Compute` (`f64`) differs from `Self` (`f32`), so there is no zero-copy
    // reborrow; `aggregate` widens elementwise through `to_compute` instead.
    None
  }
}

impl Scalar for f64 {
  type Compute = Self;

  fn to_compute(self) -> Self::Compute {
    self
  }

  fn as_compute_slice(v: &[Self]) -> Option<&[Self::Compute]> {
    Some(v)
  }
}

impl Real for f64 {
  const ZERO: Self = 0.0;
  const ONE: Self = 1.0;

  fn from_f32(x: f32) -> Self {
    f64::from(x)
  }

  fn sqrt(self) -> Self {
    libm::sqrt(self)
  }

  fn abs(self) -> Self {
    libm::fabs(self)
  }

  fn exponent(self) -> i32 {
    libm::frexp(self).1 - 1
  }

  fn ldexp(self, n: i32) -> Self {
    libm::ldexp(self, n)
  }

  fn is_finite(self) -> bool {
    f64::is_finite(self)
  }
}

/// A test-only storage scalar narrower than `f64`, its compute type.
///
/// `f32` also widens (to `f64`), so the widening half of
/// [`Scalar::as_compute_slice`] is no longer reachable only through this double.
/// It survives because it is the one *integer* storage scalar — a symmetric int8
/// quantization at a fixed scale of `1/127`, the arrangement a real quantized
/// embedding would use — so it keeps the requantization round-trip
/// (`f64` compute back to `i8` storage) under test.
///
/// Sealing is what confines this to the crate: an integration test cannot
/// implement [`Scalar`], so this widening path can only be exercised from here.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TestQuant(pub(crate) i8);

#[cfg(test)]
impl private::Sealed for TestQuant {}

#[cfg(test)]
impl Scalar for TestQuant {
  type Compute = f64;

  fn to_compute(self) -> Self::Compute {
    f64::from(self.0) / 127.0
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    None
  }
}
