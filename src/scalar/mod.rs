//! Scalar types an embedding can be stored and aggregated in.
//!
//! [`Scalar`] is the storage type — [`Vector::as_slice`](crate::windowed::Vector::as_slice)
//! yields `&[Self::Scalar]` — and [`Real`] is the floating-point domain the
//! aggregation math runs in. For every scalar this crate ships the two coincide,
//! but they are kept separate so a future narrow storage type (an `f16`, a
//! quantized byte) can compute in a wider float without a breaking change to
//! [`Vector::from_unnormalized`](crate::windowed::Vector::from_unnormalized).
//!
//! Two scalars are implemented: `f32` and `f64`. Neither is feature-gated —
//! both are `core` types, and monomorphization already makes an unused one free.
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
//!   computing in `f32` is mandatory rather than optional. Left out only to
//!   avoid taking a dependency for a use that has not arrived yet.

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
  /// Equal to `Self` for every scalar this crate currently ships.
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

  /// The exponent one past the largest representable power of two, matching
  /// [`f32::MAX_EXP`]: `2^(MAX_EXP - 1)` is finite and `2^MAX_EXP` is not.
  const MAX_EXP: i32;

  /// The exponent one past the smallest *normal* power of two, matching
  /// [`f32::MIN_EXP`]: `2^(MIN_EXP - 1)` is the smallest positive normal value.
  const MIN_EXP: i32;

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
  /// scalar's range — `1e20_f32` squares to infinity, `1e-30_f32` to zero — still
  /// be normalized rather than rejected.
  fn abs(self) -> Self;

  /// The binary exponent of `self`'s magnitude: the `e` for which
  /// `2^e <= |self| < 2^(e + 1)`.
  ///
  /// Defined for a finite, non-zero `self`; aggregation only ever asks for the
  /// exponent of a magnitude it has already checked for both. Subnormals report
  /// their true exponent (`f32::from_bits(1)` is `-149`), not a flushed one.
  fn exponent(self) -> i32;

  /// `self * 2^n`, exact whenever the result is representable.
  ///
  /// This exactness is the whole basis of the aggregation math's scale
  /// handling. A power of two leaves the significand untouched and moves only
  /// the exponent, so a sum rescaled this way is the *same* sum with its
  /// exponent shifted: it keeps the direction, keeps the rounding, and — the
  /// property a rescale by an arbitrary divisor destroys — keeps an exactly
  /// cancelling sum at exactly zero. `n == 0` returns `self` unchanged.
  fn ldexp(self, n: i32) -> Self;

  /// Whether the value is finite: neither infinite nor NaN.
  fn is_finite(self) -> bool;
}

impl Scalar for f32 {
  type Compute = Self;

  fn to_compute(self) -> Self::Compute {
    self
  }

  fn as_compute_slice(v: &[Self]) -> Option<&[Self::Compute]> {
    Some(v)
  }
}

impl Real for f32 {
  const ZERO: Self = 0.0;
  const ONE: Self = 1.0;
  const MAX_EXP: i32 = f32::MAX_EXP;
  const MIN_EXP: i32 = f32::MIN_EXP;

  fn from_f32(x: f32) -> Self {
    x
  }

  fn sqrt(self) -> Self {
    libm::sqrtf(self)
  }

  // Through libm for the same reason as `sqrt`: `f32::abs` is std-only, and
  // libm's `arch` feature lowers this to the hardware instruction anyway.
  fn abs(self) -> Self {
    libm::fabsf(self)
  }

  // `frexpf` splits into a significand in [0.5, 1) and an exponent, so its
  // exponent is one above the `2^e <= |x|` convention this trait states.
  fn exponent(self) -> i32 {
    libm::frexpf(self).1 - 1
  }

  fn ldexp(self, n: i32) -> Self {
    libm::ldexpf(self, n)
  }

  // Inherent-first path resolution picks `core`'s `f32::is_finite`, not this
  // trait method; the deny-by-default `unconditional_recursion` lint and the
  // infinity/NaN cases in `tests.rs` are what keep that true.
  fn is_finite(self) -> bool {
    f32::is_finite(self)
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
  const MAX_EXP: i32 = f64::MAX_EXP;
  const MIN_EXP: i32 = f64::MIN_EXP;

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

/// A test-only storage scalar whose compute type is *not* `Self`.
///
/// Both shipped scalars widen to themselves, so without this double the
/// widening half of [`Scalar::as_compute_slice`] — and the branch in
/// `aggregate` that consumes it — would ship untested. It models a symmetric
/// int8 quantization at a fixed scale of `1/127`, the arrangement a real
/// quantized embedding would use.
///
/// Sealing is what confines this to the crate: an integration test cannot
/// implement [`Scalar`], so the widening path can only be exercised from here.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TestQuant(pub(crate) i8);

#[cfg(test)]
impl private::Sealed for TestQuant {}

#[cfg(test)]
impl Scalar for TestQuant {
  type Compute = f32;

  fn to_compute(self) -> Self::Compute {
    f32::from(self.0) / 127.0
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    None
  }
}
