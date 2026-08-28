//! Scalar types an embedding can be stored and aggregated in.
//!
//! [`Scalar`] is the storage type — [`Vector::as_slice`](crate::windowed::Vector::as_slice)
//! yields `&[Self::Scalar]` — and [`Real`] is the floating-point domain the
//! aggregation math runs in. The two are deliberately *not* the same for `f32`:
//! an `f32` embedding stores `f32` but computes in `f64`
//! ([`f32::Compute`](Scalar::Compute) is `f64`). That is the whole reason the
//! storage and compute types are split, and it is what the narrow storage
//! scalars (`f16`, `bf16`, and the quantized `i8` byte) reuse to compute in a
//! wider float without a breaking change to
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
//! The storage scalars are the two core floats `f32` (storage only — it computes
//! in `f64`) and `f64` (both storage and its own compute domain, the sole
//! [`Real`]); the code scalar `i8`; and, behind the `half` feature, `f16` and
//! `bf16` (two more storage-only floats that compute in `f64`, exactly as `f32`
//! does). The core scalars are not feature-gated: monomorphization already makes
//! an unused one free.
//!
//! `i8` is a *code* scalar, not a value scalar: its
//! [`to_compute`](Scalar::to_compute) widens the raw stored quantization code,
//! which becomes a value only once an embedding applies the dequantization scale
//! this crate cannot know. [`TO_COMPUTE_IS_VALUE`](Scalar::TO_COMPUTE_IS_VALUE)
//! records that distinction, and the `Vector::compute_components` projection
//! enforces it.
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
//! Two notes on the narrow scalars:
//!
//! - **`i8` is a code, not a value.** A bare `i8` has no value without a
//!   per-tensor (or per-row, per-block) quantization scale this crate cannot
//!   know, so folding raw codes as if they were values is right only by
//!   coincidence — symmetric, uniform-scale, renormalizing — and silently wrong
//!   otherwise. `i8` is therefore admitted as a code scalar whose value lives
//!   with the [`Vector`](crate::windowed::Vector) that knows its scale, projected
//!   by its `compute_components` method; the default projection refuses raw codes
//!   with [`WinditError::MissingDequantization`](crate::WinditError::MissingDequantization)
//!   rather than fold a wrong answer. This is the insight the old "use a storage
//!   wrapper that carries its own scale" note carried, now enforced.
//! - **Wider integers (`i16`, `i32`, …) and unsigned types** stay out: `i8` is
//!   the width a quantized embedding stores, and the code-scalar mechanism does
//!   not become more general by widening it. To request one, open an issue.

#[cfg(test)]
mod tests;

mod private {
  /// The seal. Implemented only for the scalars this crate ships, so
  /// [`Scalar`](super::Scalar) and [`Real`](super::Real) cannot be implemented
  /// downstream.
  pub trait Sealed {}

  impl Sealed for f32 {}
  impl Sealed for f64 {}
  impl Sealed for i8 {}

  #[cfg(feature = "half")]
  impl Sealed for half::f16 {}
  #[cfg(feature = "half")]
  impl Sealed for half::bf16 {}
}

/// A scalar type an embedding can be stored in.
///
/// This trait is sealed; see the [module documentation](self) for the reason and
/// for the list of types that are deliberately not implemented.
pub trait Scalar: private::Sealed + Copy {
  /// The floating-point type this scalar's aggregation math runs in.
  ///
  /// `f64` for every shipped scalar: `f64` computes in itself, and `f32`, `i8`,
  /// and the `half` scalars all widen to `f64` so no embedding ever accumulates
  /// in a narrower type. A storage type equal to its own `Compute` (only `f64`,
  /// here) is a [`Real`]; one narrower than its `Compute` (`f32`, `i8`, `f16`,
  /// `bf16`) is storage only.
  type Compute: Real;

  /// Whether [`to_compute`](Scalar::to_compute) yields the value this scalar
  /// *represents*, rather than its raw stored code.
  ///
  /// `true` for every float scalar (`f32`, `f64`, and the `half` scalars):
  /// widening is exact and value-preserving. `false` for `i8`, whose stored code
  /// has no value without a quantization scale this crate cannot know; the value
  /// projection for such a scalar lives at the embedding level, in the
  /// [`Vector`](crate::windowed::Vector) `compute_components` method, whose
  /// default refuses to fold raw codes.
  const TO_COMPUTE_IS_VALUE: bool;

  /// Widen one stored scalar into the compute domain.
  ///
  /// For a code scalar (`i8`) this widens the raw stored code, which is a value
  /// only after an embedding applies its dequantization scale — see
  /// [`TO_COMPUTE_IS_VALUE`](Scalar::TO_COMPUTE_IS_VALUE).
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
  // Every implementor is a core float, so this costs implementors nothing and
  // buys the generic code its only way to *show* a compute value: without it a
  // `C: Real` is unformattable, and `VectorEmaState` had to report its shape
  // while withholding the coefficient it was configured with.
  + core::fmt::Debug
  + core::ops::Add<Output = Self>
  + core::ops::Sub<Output = Self>
  + core::ops::Mul<Output = Self>
  + core::ops::Div<Output = Self>
  + PartialOrd
  // Every implementor is an owned, borrow-free arithmetic type, so this costs
  // implementors nothing and spares every caller a bound: a policy configured
  // in `C` is boxed as `Box<dyn AggregatePolicy<C>>`, whose implicit `'static`
  // would otherwise have to be re-spelled at each such site (and by every
  // downstream wrapper around one).
  + 'static
{
  /// The additive identity.
  const ZERO: Self;

  /// The multiplicative identity.
  const ONE: Self;

  /// The unit roundoff: the gap between `1` and the next larger representable
  /// value (`2^-52` for `f64`).
  ///
  /// Aggregation scales this by the accumulated term magnitude to decide when a
  /// folded result lies at or below its own rounding floor, and so determines no
  /// direction at working precision.
  const EPSILON: Self;

  /// The smallest magnitude a nonzero input component may carry into aggregation
  /// (`2^-400` for `f64`).
  ///
  /// With [`MAX_AGG_MAGNITUDE`](Real::MAX_AGG_MAGNITUDE) it bounds the input
  /// domain within which every intermediate of every built-in aggregation policy
  /// stays finite, and — for the policies whose weights the domain itself bounds
  /// below (`MeanRenormalized` at the constant `1`, `SaliencyWeighted` at a norm
  /// this bound itself puts at `2^-400`) — every nonzero intermediate stays a
  /// normal value, with no overflow and no flush to a subnormal.
  /// `EmaRenormalized` and `CoverageWeightedMean` are the two that reach below
  /// it, and in neither is the cause a small weight *scale*: EMA's ideal weights
  /// sum to exactly `1`, and a normalized coverage's largest is exactly a power
  /// of two whatever scale the slice arrived in. It is the *ratio* — EMA's
  /// recency factors decaying without limit, a coverage far below
  /// the fullest window's — that drives a product toward a subnormal. The
  /// determinacy gate's [`MIN_GATE_THRESHOLD`](Real::MIN_GATE_THRESHOLD) floor
  /// keeps that regime sound (see the `aggregate` module's Input domain note). A
  /// nonzero component below this bound is rejected before any arithmetic. Every
  /// magnitude an `f32`-storage embedding can produce sits far above it.
  const MIN_AGG_MAGNITUDE: Self;

  /// The largest magnitude a nonzero input component may carry into aggregation
  /// (`2^400` for `f64`).
  ///
  /// The upper companion to [`MIN_AGG_MAGNITUDE`](Real::MIN_AGG_MAGNITUDE),
  /// sized so that even the norm-weighted saliency policy — which squares a
  /// magnitude — keeps every intermediate a normal value. A component above it
  /// is rejected before any arithmetic.
  const MAX_AGG_MAGNITUDE: Self;

  /// The absolute floor of the aggregation determinacy threshold (`2^-1000` for
  /// `f64`).
  ///
  /// The determinacy gate rejects a folded result whose norm is at or below
  /// `16 * `[`EPSILON`](Real::EPSILON)` * ||M|| + MIN_GATE_THRESHOLD`, where `M`
  /// is the accumulated term-magnitude vector. The relative `16 * EPSILON * ||M||`
  /// part underflows to zero once an unbounded weight *ratio* drives the fold's
  /// products into the subnormal range, where per-term rounding is absolute (at
  /// most `2^-1075`) rather than relative. This floor dominates the largest
  /// residue such an exactly-cancelling fold can leave
  /// (`sqrt(dim) * n * 2^-1075 <= 2^-1018` for any `n <= 2^40`, `dim <= 2^32`), so
  /// the gate cannot degenerate into an exact-zero check that a nonzero subnormal
  /// residue slips past. It sits far below `16 * EPSILON * ||M||` for any mass a
  /// fold whose heaviest window carries its own accumulates, so those verdicts
  /// are bit-for-bit unchanged. It engages whenever the accumulated mass falls below about
  /// `2^-948`, including folds whose products are still normal — where it
  /// monotonically turns a sub-floor direction into `NonFinite` (an engagement
  /// boundary a regression test pins).
  ///
  /// **This bound is absolute, so it is sound only against a quantity carried in
  /// the embedding's own units** — the accumulator's norm and `M`, which the
  /// input domain bounds below, and not a dimensionless *weight*, whose scale a
  /// caller sets and which every policy divides back out. Reaching the regime
  /// therefore takes an unbounded weight ratio: `EmaRenormalized`'s decaying
  /// recency factors, or a `CoverageWeightedMean` fold whose fullest windows are
  /// all zero. No realizable `f32` workload gets there. See the `aggregate`
  /// module's Input domain note.
  const MIN_GATE_THRESHOLD: Self;

  /// Widen one of the determinacy gate's own dimensionless constants — the `16`
  /// its threshold carries — into this domain. Exact for every implementor.
  ///
  /// Nothing the fold *multiplies an embedding by* arrives this way any more.
  /// Both such weights — an EMA smoothing factor and a [`Span::coverage`] — are
  /// resolved at the accumulator's own width and reach a `Real` through
  /// [`from_f64`](Real::from_f64); a coverage stopped coming through here in
  /// `0.3.0`, when it stopped being typed by its provenance. What is left is a
  /// small exact integer written into a threshold, which is not a weight and
  /// never was a tuning knob, so the narrow constructor names it without
  /// implying either.
  ///
  /// [`Span::coverage`]: crate::plan::Span::coverage
  fn from_f32(x: f32) -> Self;

  /// Widen an `f64` value the fold will multiply an embedding by — an EMA
  /// smoothing factor, or a [`Span::coverage`] — into this domain. Exact for
  /// every implementor, and the identity for `f64`.
  ///
  /// Every implementor of this trait *is* the compute domain (`Real` is
  /// `Scalar<Compute = Self>`), and that domain is `f64` for every shipped
  /// scalar. A coefficient narrower than the domain it multiplies cannot
  /// express the filters that domain can carry: no `f32` holds `1 - 2^-30` —
  /// the nearest one is exactly `1.0` — and the `f32` grid is `2^-24` where the
  /// arithmetic rounds at `2^-53`, so a caller would be tuning on a grid coarser
  /// than the recurrence it is tuning. So the two smoothing-factor
  /// constructors — `aggregate::EmaRenormalized::new` and
  /// `smooth::VectorEma::new`, named rather than linked because both sit behind
  /// `alloc` and this tier is featureless — take the compute domain itself
  /// rather than widening into it. Two kinds of value cannot be asked for in the
  /// compute domain because they exist before one does, and both reach it
  /// through here instead: the serde selector `AggregatePolicyKind`, a wire value
  /// read before any embedding, and a `Span::coverage`, which the planner derives
  /// from two `usize`s in a tier that has no compute scalar at all. Neither is
  /// configured at the accumulator's width; both are nonetheless resolved at it,
  /// because what a value multiplies is what decides how finely it must be
  /// carried.
  ///
  /// [`Span::coverage`]: crate::plan::Span::coverage
  fn from_f64(x: f64) -> Self;

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

  // Widening f32 to f64 is exact and value-preserving.
  const TO_COMPUTE_IS_VALUE: bool = true;

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

  // f64 is its own compute domain, so `to_compute` is the identity — trivially
  // value-preserving.
  const TO_COMPUTE_IS_VALUE: bool = true;

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
  const EPSILON: Self = f64::EPSILON;
  // 2^-400 and 2^400, the aggregation magnitude domain (sized in the `aggregate`
  // module docs). `from_bits` is const well below the 1.95 MSRV.
  const MIN_AGG_MAGNITUDE: Self = f64::from_bits(0x26F0_0000_0000_0000);
  const MAX_AGG_MAGNITUDE: Self = f64::from_bits(0x58F0_0000_0000_0000);
  // 2^-1000, the absolute determinacy-threshold floor (sized in the `aggregate`
  // module docs).
  const MIN_GATE_THRESHOLD: Self = f64::from_bits(0x0170_0000_0000_0000);

  fn from_f32(x: f32) -> Self {
    f64::from(x)
  }

  fn from_f64(x: f64) -> Self {
    x
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

impl Scalar for i8 {
  // A quantized code stored as one byte. It widens to `f64` like the other
  // narrow scalars, but the widened code is not the value it represents.
  type Compute = f64;

  // A stored code, not a value: folding raw codes is right only by coincidence
  // (symmetric, uniform-scale, renormalizing), so nothing in this crate ever
  // does. A quantized `Vector` overrides `compute_components` to dequantize; its
  // default projection refuses a code scalar rather than fold raw codes.
  const TO_COMPUTE_IS_VALUE: bool = false;

  fn to_compute(self) -> Self::Compute {
    // The raw code, widened exactly (every `i8` is exact in `f64`). NOT a
    // represented value: `TO_COMPUTE_IS_VALUE` being `false` is what keeps every
    // crate path from treating it as one.
    f64::from(self)
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    // `Compute` (`f64`) differs from `Self` (`i8`): no zero-copy reborrow.
    None
  }
}

// f16: an 11-bit effective significand and value exponents in [-24, 15], both
// strictly inside f64's (53-bit, [-1074, 1023]), so every finite f16 is exact in
// f64 and `to_compute` rounds nothing — the same property f32 already has.
#[cfg(feature = "half")]
impl Scalar for half::f16 {
  type Compute = f64;

  // Widening f16 to f64 is exact and value-preserving.
  const TO_COMPUTE_IS_VALUE: bool = true;

  fn to_compute(self) -> Self::Compute {
    f64::from(self)
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    None
  }
}

// bf16: an 8-bit effective significand and value exponents in [-133, 127], both
// strictly inside f64's, so every finite bf16 is exact in f64 as well.
#[cfg(feature = "half")]
impl Scalar for half::bf16 {
  type Compute = f64;

  // Widening bf16 to f64 is exact and value-preserving.
  const TO_COMPUTE_IS_VALUE: bool = true;

  fn to_compute(self) -> Self::Compute {
    f64::from(self)
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    None
  }
}

/// The half-precision storage scalars, re-exported so a consumer's
/// `type Scalar = f16;` names the same type this crate implemented [`Scalar`]
/// for, whatever `half` version the consumer resolves.
#[cfg(feature = "half")]
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
pub use half::{bf16, f16};

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

  // Its fixed 1/127 scale is baked into `to_compute`, so the widened value is a
  // genuine represented value, not a raw code.
  const TO_COMPUTE_IS_VALUE: bool = true;

  fn to_compute(self) -> Self::Compute {
    f64::from(self.0) / 127.0
  }

  fn as_compute_slice(_: &[Self]) -> Option<&[Self::Compute]> {
    None
  }
}
