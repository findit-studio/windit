//! Aggregation policies: combine a window sequence into a single embedding.
//!
//! `AggregatePolicy` is the object-safe seam: its one method works on plain
//! slices of a [`Real`] compute scalar (`aggregate_values`), so
//! `&dyn AggregatePolicy` is usable. The scalar is a trait type parameter
//! defaulting to `f32`, which is what keeps that bare `dyn` spelling valid; a
//! non-`f32` embedding names it (`dyn AggregatePolicy<f64>`). The generic free
//! function `aggregate` extracts the compute slices and per-window coverages
//! from a `&[WindowEmbedding<E>]`, runs the policy, and reconstructs the
//! embedding type `E` through `Vector::from_unnormalized`. Keeping
//! reconstruction out of the trait is what lets the trait stay object-safe while
//! embedding reconstruction stays generic.
//!
//! Policy *configuration* stays `f32` — an EMA smoothing factor is a
//! dimensionless constant, not an embedding value — and is widened where it is
//! used, through [`Real::from_f32`]. That is what leaves `AggregatePolicyKind`
//! and its serde representation untouched by the scalar generalization.
//!
//! Built-in strategies weight the windows by different signals:
//!
//! - `CoverageWeightedMean` (the default) weights by [`Span::coverage`], so
//!   fuller windows count more.
//! - `MeanRenormalized` weights uniformly (a renormalized arithmetic mean).
//! - `EmaRenormalized` weights by recency (an exponential moving average).
//! - `SaliencyWeighted` weights by each input's L2 norm, so higher-magnitude
//!   (more salient) inputs dominate. Because `aggregate` passes already-unit
//!   embeddings, saliency is meaningful only when a caller invokes
//!   `aggregate_values` directly with vectors that still carry magnitude.
//!
//! `keep_separate` is the multi-vector path: it returns every window unchanged
//! for callers that want per-window embeddings rather than one summary.
//!
//! # Scale
//!
//! Aggregation runs in [`ComputeOf<E>`](crate::windowed::ComputeOf), which is
//! `f64` for both shipped scalars: an `f32` embedding widens to `f64` before a
//! single value is folded. That is what disarms the whole class of magnitude
//! hazard for `f32` inputs — every `f32` is exact in `f64`, every `f32`
//! subnormal is a normal `f64`, and `f32::MAX` squared is ~1.2e77 against
//! `f64`'s ~1.8e308 ceiling — so a sum that would have overflowed, flushed a
//! subnormal, or lost a cancellation in `f32` does none of those here. No power
//! of-two prescaling of the fold is applied, and none is needed.
//!
//! Two hazards survive into `f64` itself, because `f64` is the widest domain
//! there is, and both are handled where they arise rather than by a blanket
//! shift:
//!
//! - **Cancellation across a wide exponent spread.** A weighted sum whose exact
//!   value is zero can fold to an order-dependent non-zero residue once a small
//!   term is absorbed into a large partial sum and the large term is later
//!   subtracted away. The accumulation is therefore a compensated
//!   (Neumaier's variant of Kahan-Babuška) sum, and a determinacy gate then
//!   rejects any result at or below the fold's own provable rounding floor — so
//!   [`WinditError::NonFinite`] keeps meaning "no direction determined at
//!   working precision" rather than "the fold happened to round to a residue".
//!   The [Input domain](self#input-domain) note states the bound this rests on.
//! - **A norm that is not representable although the vector is.**
//!   `[f64::MAX, f64::MAX]` is an ordinary diagonal whose norm, `sqrt(2) *
//!   f64::MAX`, overflows. The renormalization divides each component by its own
//!   `2^exponent` power-of-two scale and by the scaled norm separately, so it
//!   never forms that norm; dividing by a power of two is exact, so the quotient
//!   is the direct `v_i / norm` to the bit wherever the direct computation was
//!   valid. A vector's squares leaving the range (`[f64::MAX, 0.0]` squares to
//!   infinity, `[f64::MIN_POSITIVE, 0.0]` to zero) is the same mechanism and the
//!   same fix.
//!
//! There is no second attempt anywhere, which is what keeps
//! [`WinditError::NonFinite`] meaning what it says: an all-zero (or exactly
//! cancelling) vector, or a component that is itself not finite. A retry cannot
//! tell those from a norm that merely overflowed, and a vector whose components
//! cancel exactly is not a vector whose norm was unrepresentable — it is a
//! vector with no direction.
//!
//! Rather than let any policy reach for the edge of `f64`, aggregation enforces
//! an input domain that keeps every fold clear of it — see below.
//!
//! # Input domain
//!
//! Every input component must be finite and either zero or of a magnitude
//! between [`Real::MIN_AGG_MAGNITUDE`] and [`Real::MAX_AGG_MAGNITUDE`]
//! (`[2^-400, 2^400]` for `f64`, about `[3.9e-121, 2.6e120]`); every coverage
//! must be finite and in `[0, 1]`. Inputs outside this domain are rejected with
//! [`WinditError::MagnitudeOutOfRange`] or [`WinditError::CoverageOutOfRange`]
//! before any arithmetic. The bounds are sized so that within them every
//! intermediate of every built-in policy is finite and every nonzero
//! intermediate is a normal `f64` — no overflow, no subnormal flush — including
//! the squared term [`SaliencyWeighted`] forms. Every value an `f32`-storage
//! embedding can produce lies more than 250 binary orders inside this window on
//! both sides, so no realizable `f32` input ever reaches a boundary.
//!
//! Within the domain, an aggregated result is the direction of a vector within
//! `4 * `[`Real::EPSILON`]` * ||M||` of the exact weighted sum, where `M` is the
//! componentwise sum of the folded term magnitudes; any result whose norm is at
//! or below `16 * `[`Real::EPSILON`]` * ||M||` is reported as
//! [`WinditError::NonFinite`] — no direction is determined at working precision.
//! This is the crate's one accuracy claim, and it is a theorem rather than an
//! observation: per dimension the fold's error is at most the product rounding
//! (`<= u * M_j`, with `u = EPSILON / 2`) plus the Neumaier bound (`<= 2u * M_j`,
//! plus an `O(n * u^2) * M_j` tail negligible for any window count that fits in
//! memory), together at most `4 * EPSILON * M_j`; summing over dimensions gives
//! `||R - exact|| <= 4 * EPSILON * ||M||`. An exactly cancelling sum therefore
//! has `||R|| <= 4 * EPSILON * ||M|| < 16 * EPSILON * ||M||` and is always gated,
//! whatever the ordering or tier structure — so no fold can fabricate a
//! direction from cancellation without violating the bound.
//!
//! [`Real`]: crate::scalar::Real
//! [`Real::from_f32`]: crate::scalar::Real::from_f32
//! [`Span`]: crate::plan::Span
//! [`Span::coverage`]: crate::plan::Span::coverage

use std::vec::Vec;

use crate::{
  error::WinditError,
  scalar::{Real, Scalar},
  windowed::{ComputeOf, Vector, WindowEmbedding},
};

#[cfg(test)]
mod tests;

/// A policy that combines a sequence of window embeddings into one embedding.
///
/// The single required method operates on plain slices so the trait is
/// object-safe (`&dyn AggregatePolicy` works). Embedding reconstruction lives in
/// the generic free function [`aggregate`], not here.
///
/// `C` is the compute scalar — the [`Real`] domain the math runs in — and
/// defaults to `f64`. Because both shipped scalars compute in `f64` (an `f32`
/// embedding widens to it), that default is the domain every built-in
/// aggregation actually uses, and it keeps `dyn AggregatePolicy` and
/// `Box<dyn AggregatePolicy>` spelling the object every ordinary embedding
/// needs. A custom compute scalar names it, as in `Box<dyn AggregatePolicy<C>>`.
/// Note that trait objects are per-scalar: two `AggregatePolicy` objects over
/// different `C` are unrelated types and cannot share one collection.
///
/// # Custom policies
///
/// Implement [`aggregate_values`](AggregatePolicy::aggregate_values) to add a
/// strategy. This one keeps the first window unchanged, and serves the default
/// `f64` compute domain by leaving the type parameter off:
///
/// ```
/// use windit::aggregate::AggregatePolicy;
/// use windit::WinditError;
///
/// struct FirstWindow;
///
/// impl AggregatePolicy for FirstWindow {
///   fn aggregate_values(
///     &self,
///     embeddings: &[&[f64]],
///     _coverages: &[f32],
///     dim: usize,
///   ) -> Result<Vec<f64>, WinditError> {
///     let first = embeddings.first().ok_or(WinditError::Empty)?;
///     if first.len() != dim {
///       return Err(WinditError::DimMismatch { got: first.len(), expected: dim });
///     }
///     Ok(first.to_vec())
///   }
/// }
/// ```
///
/// Writing `impl<C: Real> AggregatePolicy<C> for FirstWindow` instead — with
/// `&[&[C]]` and `Vec<C>` — makes the same policy serve every compute scalar.
pub trait AggregatePolicy<C: Real = f64> {
  /// Combine `embeddings` (each a `dim`-length slice of the compute scalar)
  /// with their matching `coverages` into a single `dim`-length vector.
  ///
  /// The built-in policies return an L2-normalized vector, and a custom policy
  /// should do the same; either way [`aggregate`] re-normalizes the result
  /// through [`Vector::from_unnormalized`]. `coverages` must have the same length
  /// as `embeddings` even for policies that do not weight by coverage. They stay
  /// `f32` at every scalar: a coverage is a geometric fraction from
  /// [`Span::coverage`](crate::plan::Span::coverage), not an embedding value.
  ///
  /// Every component must be finite and either zero or of magnitude within
  /// `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`, and every coverage finite and in
  /// `[0, 1]`; see the module [Input domain](self#input-domain) note.
  ///
  /// # Errors
  ///
  /// - [`WinditError::Empty`] if `embeddings` is empty.
  /// - [`WinditError::DimMismatch`] if `coverages.len() != embeddings.len()` or
  ///   any embedding's length differs from `dim`.
  /// - [`WinditError::MagnitudeOutOfRange`] if a nonzero component's magnitude is
  ///   outside `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`.
  /// - [`WinditError::CoverageOutOfRange`] if a coverage is not a finite fraction
  ///   in `[0, 1]`.
  /// - [`WinditError::NonFinite`] if the combined vector cannot be normalized to
  ///   a finite unit vector (zero norm or a non-finite component).
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<C>, WinditError>;
}

/// Aggregate a sequence of window embeddings into one embedding of type `E`.
///
/// Extracts each window's values and [`Span::coverage`](crate::plan::Span::coverage),
/// runs `policy` in `E`'s compute domain, and reconstructs `E` via
/// [`Vector::from_unnormalized`]. Works with any policy, including
/// `&dyn AggregatePolicy`.
///
/// # Errors
///
/// [`WinditError::Empty`] if `windows` is empty; otherwise any error from the
/// policy or from [`Vector::from_unnormalized`].
pub fn aggregate<E, P>(policy: &P, windows: &[WindowEmbedding<E>]) -> Result<E, WinditError>
where
  E: Vector,
  P: AggregatePolicy<ComputeOf<E>> + ?Sized,
{
  if windows.is_empty() {
    return Err(WinditError::Empty);
  }
  let dim = windows[0].value.dim();
  let mut coverages = try_vec_with_capacity(windows.len())?;
  for w in windows {
    coverages.push(w.span.coverage());
  }

  // Fast path: borrow the storage when it already is the compute scalar (true of
  // `f64` storage) rather than widen it into fresh buffers. `f32` storage
  // computes in `f64`, so its `as_compute_slice` returns `None` and a single
  // one sends the whole aggregation down the widening branch below.
  let mut borrowed: Vec<&[ComputeOf<E>]> = try_vec_with_capacity(windows.len())?;
  let mut all_borrowable = true;
  for w in windows {
    match <E::Scalar as Scalar>::as_compute_slice(w.value.as_slice()) {
      Some(s) => borrowed.push(s),
      None => {
        all_borrowable = false;
        break;
      }
    }
  }

  let raw = if all_borrowable {
    policy.aggregate_values(&borrowed, &coverages, dim)?
  } else {
    let mut widened: Vec<Vec<ComputeOf<E>>> = try_vec_with_capacity(windows.len())?;
    for w in windows {
      let stored = w.value.as_slice();
      let mut col = try_vec_with_capacity(stored.len())?;
      for s in stored {
        col.push(s.to_compute());
      }
      widened.push(col);
    }
    let mut embeddings: Vec<&[ComputeOf<E>]> = try_vec_with_capacity(widened.len())?;
    for col in &widened {
      embeddings.push(col.as_slice());
    }
    policy.aggregate_values(&embeddings, &coverages, dim)?
  };
  E::from_unnormalized(&raw)
}

/// A `Vec` that can hold `n` elements, or [`WinditError::AllocFailed`] when the
/// allocator cannot (or refuses to) provide the space.
///
/// The fallible counterpart to `Vec::with_capacity` for the growing buffers on
/// these `Result`-returning paths. Every buffer an aggregation grows is sized by
/// the caller's window count or embedding dimension — counts that need not
/// correspond to memory that exists — so a refused allocation must surface as a
/// typed error rather than abort the process. `try_reserve_exact` because each
/// buffer is then filled to exactly `n` and never grown again.
fn try_vec_with_capacity<T>(n: usize) -> Result<Vec<T>, WinditError> {
  let mut v = Vec::new();
  v.try_reserve_exact(n)
    .map_err(|_| WinditError::AllocFailed { elements: n })?;
  Ok(v)
}

/// A `dim`-length vector of [`Real::ZERO`], or [`WinditError::AllocFailed`].
///
/// The accumulator every weighted sum folds into; [`try_vec_with_capacity`]
/// reserves it and `resize` fills the reserved space without growing again.
fn try_zeroed<C: Real>(dim: usize) -> Result<Vec<C>, WinditError> {
  let mut v = try_vec_with_capacity(dim)?;
  v.resize(dim, C::ZERO);
  Ok(v)
}

/// The multi-vector path: return every window unchanged.
///
/// The counterpart to [`aggregate`], for callers that keep per-window embeddings
/// (for example, one speaker centroid per window) instead of collapsing them.
#[must_use]
pub fn keep_separate<E>(windows: Vec<WindowEmbedding<E>>) -> Vec<WindowEmbedding<E>> {
  windows
}

/// Coverage-weighted mean, then L2 renormalization (the default strategy).
///
/// Each window contributes in proportion to its [`Span::coverage`](crate::plan::Span::coverage),
/// so a padded ragged tail counts less than a full window.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoverageWeightedMean;

/// Uniform (unweighted) mean, then L2 renormalization.
///
/// Every window contributes equally regardless of coverage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeanRenormalized;

/// Exponential moving average across the window sequence, then L2 renormalization.
///
/// State advances `s_i = alpha * emb_i + (1 - alpha) * s_{i-1}` from `s_0 = emb_0`,
/// so later windows weigh more (recency). `coverages` are ignored beyond the
/// length check. `alpha` must be in `[0, 1]`; an out-of-range or non-finite
/// `alpha` is rejected with [`WinditError::AlphaOutOfRange`] rather than
/// silently producing a non-convex (sign-flipping) combination.
///
/// Like [`WindowOptions`](crate::plan::WindowOptions), construction is
/// infallible and the range is checked where the value is used — here in
/// [`aggregate_values`](AggregatePolicy::aggregate_values), which already returns a
/// `Result`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaRenormalized {
  alpha: f32,
}

impl EmaRenormalized {
  /// An EMA aggregation with the given smoothing factor.
  ///
  /// `alpha` is not validated here: a value outside `[0, 1]` (or a NaN) is
  /// reported as [`AlphaOutOfRange`](WinditError::AlphaOutOfRange) by
  /// [`aggregate_values`](AggregatePolicy::aggregate_values). Deferring the check is
  /// what keeps this constructor usable from `AggregatePolicyKind::into_policy`,
  /// which builds a policy from deserialized configuration and has no error
  /// channel of its own.
  #[must_use]
  pub const fn new(alpha: f32) -> Self {
    Self { alpha }
  }

  /// The smoothing factor: larger values track recent windows more.
  #[must_use]
  pub const fn alpha(&self) -> f32 {
    self.alpha
  }
}

/// L2-norm-weighted mean, then renormalization: higher-magnitude inputs dominate.
///
/// Each window is weighted by the L2 norm of its input slice, so more salient
/// (larger-magnitude) vectors pull the result toward them. `coverages` are
/// ignored beyond the length check. This differs from the other strategies only
/// when the inputs carry magnitude; [`aggregate`] feeds unit vectors, so use
/// [`aggregate_values`](AggregatePolicy::aggregate_values) directly to exploit it.
///
/// Because it squares magnitudes (weight times component, and the weight is
/// itself a norm), its intermediates reach higher than any linear policy's. The
/// crate-level [input domain](self#input-domain) is sized so even that square
/// stays a finite, normal `f64`, which is why this policy needs no window of its
/// own: a component outside the domain is rejected by the shared input check
/// before the square is ever formed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SaliencyWeighted;

impl<C: Real> AggregatePolicy<C> for CoverageWeightedMean {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| C::from_f32(coverages[i]))
  }
}

impl<C: Real> AggregatePolicy<C> for MeanRenormalized {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |_, _| C::ONE)
  }
}

impl<C: Real> AggregatePolicy<C> for SaliencyWeighted {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    check_inputs(embeddings, coverages, dim)?;
    // Materialized rather than recomputed inside the fold: a norm is a full pass
    // over a window, and `weighted_sum_renorm` reads each weight once per
    // dimension. Each norm is taken against the window's own power-of-two scale
    // ([`l2_norm`]), so a window whose norm is not representable still weighs in;
    // the shared magnitude divides back out in the renormalization that ends the
    // policy, leaving only the ratios a weight means here.
    let mut weights = try_vec_with_capacity(embeddings.len())?;
    for emb in embeddings {
      weights.push(l2_norm(emb));
    }
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| weights[i])
  }
}

impl<C: Real> AggregatePolicy<C> for EmaRenormalized {
  fn aggregate_values(
    &self,
    embeddings: &[&[C]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<C>, WinditError> {
    // A convex EMA needs alpha in [0, 1]; anything else (including NaN, which
    // fails the range test) is a configuration error, checked first and on the
    // f32 configuration field so the same range is enforced at every compute
    // scalar.
    if !(0.0..=1.0).contains(&self.alpha) {
      return Err(WinditError::AlphaOutOfRange);
    }
    // The recurrence `s_i = alpha*e_i + (1-alpha)*s_{i-1}` from `s_0 = e_0` is
    // the convex combination `sum_i w_i * e_i` with `w_0 = (1-alpha)^(n-1)` and
    // `w_i = alpha*(1-alpha)^(n-1-i)` for `i >= 1`. Building those weights and
    // folding through `weighted_sum_renorm` gives EMA the same input domain,
    // determinacy gate, and error bound as every other policy from one proof —
    // there is no separate recurrence fold left to fabricate a direction. The
    // dyadic case (`alpha = 0.5` over basis vectors) reproduces the old
    // recurrence bit for bit. `1 - alpha` is formed in `C`, not folded in `f32`
    // first, so the `f64` path keeps the precision its type promises.
    let alpha = C::from_f32(self.alpha);
    let complement = C::ONE - alpha;
    let n = embeddings.len();
    let mut weights = try_zeroed::<C>(n)?;
    // One backward pass carrying `power = complement^(n-1-i)`; the oldest window
    // gets the bare `complement^(n-1)` with no `alpha` factor.
    let mut power = C::ONE;
    for i in (1..n).rev() {
      weights[i] = alpha * power;
      power = power * complement;
    }
    if n > 0 {
      weights[0] = power;
    }
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| weights[i])
  }
}

/// Serde-serializable selector over the built-in aggregation policies.
///
/// Deserialize a configured choice, then [`into_policy`](AggregatePolicyKind::into_policy)
/// to obtain a boxed [`AggregatePolicy`]. Requires `alloc` (for the boxed policy)
/// in addition to `serde`.
#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum AggregatePolicyKind {
  /// Selects [`CoverageWeightedMean`].
  CoverageWeightedMean,
  /// Selects [`MeanRenormalized`].
  MeanRenormalized,
  /// Selects [`EmaRenormalized`] with the given smoothing factor.
  Ema {
    /// The EMA smoothing factor, forwarded to [`EmaRenormalized::new`].
    alpha: f32,
  },
  /// Selects [`SaliencyWeighted`].
  SaliencyWeighted,
}

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
impl AggregatePolicyKind {
  /// Build the boxed built-in policy this kind selects, at the compute scalar
  /// `C`.
  ///
  /// `C` is normally inferred from the embeddings the policy is about to run
  /// over — `aggregate(kind.into_policy().as_ref(), &windows)` needs no
  /// annotation. A turbofish is required only when the boxed policy is bound to
  /// a `let` that nothing downstream pins, as in `into_policy::<f32>()`.
  #[must_use]
  pub fn into_policy<C: Real>(self) -> std::boxed::Box<dyn AggregatePolicy<C>> {
    use std::boxed::Box;
    match self {
      Self::CoverageWeightedMean => Box::new(CoverageWeightedMean),
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::Ema { alpha } => Box::new(EmaRenormalized::new(alpha)),
      Self::SaliencyWeighted => Box::new(SaliencyWeighted),
    }
  }
}

/// Validate `embeddings`, `coverages`, and every component against the
/// aggregation input domain, before any arithmetic runs.
///
/// Beyond the structural checks — non-empty, `coverages` length matching
/// `embeddings`, every embedding of length `dim` — this rejects a non-finite
/// component ([`NonFinite`](WinditError::NonFinite)), a nonzero component whose
/// magnitude is outside `[MIN_AGG_MAGNITUDE, MAX_AGG_MAGNITUDE]`
/// ([`MagnitudeOutOfRange`](WinditError::MagnitudeOutOfRange)), and a coverage
/// that is not a finite fraction in `[0, 1]`
/// ([`CoverageOutOfRange`](WinditError::CoverageOutOfRange)). Enforcing the
/// domain here — the one choke point every built-in policy passes through — is
/// what lets each fold run without overflow or subnormal flush; see the module
/// [Scale](self#scale) note.
fn check_inputs<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f32],
  dim: usize,
) -> Result<(), WinditError> {
  if embeddings.is_empty() {
    return Err(WinditError::Empty);
  }
  if coverages.len() != embeddings.len() {
    return Err(WinditError::DimMismatch {
      got: coverages.len(),
      expected: embeddings.len(),
    });
  }
  for (window, (emb, &coverage)) in embeddings.iter().zip(coverages).enumerate() {
    if emb.len() != dim {
      return Err(WinditError::DimMismatch {
        got: emb.len(),
        expected: dim,
      });
    }
    if !(coverage.is_finite() && (0.0..=1.0).contains(&coverage)) {
      return Err(WinditError::CoverageOutOfRange { window });
    }
    for (component, &x) in emb.iter().enumerate() {
      if !x.is_finite() {
        return Err(WinditError::NonFinite);
      }
      if x != C::ZERO && (x.abs() < C::MIN_AGG_MAGNITUDE || x.abs() > C::MAX_AGG_MAGNITUDE) {
        return Err(WinditError::MagnitudeOutOfRange { window, component });
      }
    }
  }
  Ok(())
}

/// Accumulate `sum_i weight(i, emb_i) * emb_i`, gate it against its own rounding
/// floor, and L2-renormalize it.
///
/// One pass, no retry, no prescaling: the compute scalar is `f64` (an `f32`
/// embedding widened before this ran), and [`check_inputs`] has confined every
/// component to the input domain, so every product and partial sum is finite and
/// normal. The sum is *compensated* (Neumaier), and alongside it the routine
/// accumulates `M`, the componentwise sum of the term magnitudes. Before
/// normalizing, a determinacy gate rejects any result whose norm is at or below
/// `16 * EPSILON * ||M||`: within the fold's provable `4 * EPSILON * ||M||` error
/// bound the exact weighted sum is indistinguishable from zero there, so a
/// smaller residue is rounding noise with no direction — not a vector for
/// [`l2_renorm`] to amplify into a fabricated unit direction. The bound is the
/// crate's one accuracy claim; see the module [Input domain](self#input-domain)
/// note for the proof.
fn weighted_sum_renorm<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f32],
  dim: usize,
  weight: impl Fn(usize, &[C]) -> C,
) -> Result<Vec<C>, WinditError> {
  // `dim` is caller-supplied, but this has just proved it is the length of an
  // embedding that exists, so the accumulators below are bounded by real data.
  check_inputs(embeddings, coverages, dim)?;
  let mut acc = try_zeroed(dim)?;
  // The running Neumaier compensation, one term per dimension: the sum of the
  // low-order bits `acc` could not hold as it grew.
  let mut comp = try_zeroed::<C>(dim)?;
  // `M`: the running sum of term magnitudes per dimension. Plain monotone adds
  // (no cancellation), so it is a faithful measure of the mass the fold summed,
  // and its own accumulation error only tightens the gate.
  let mut mag = try_zeroed::<C>(dim)?;
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb);
    for (((a, c), m), &e) in acc
      .iter_mut()
      .zip(comp.iter_mut())
      .zip(mag.iter_mut())
      .zip(emb.iter())
    {
      let term = w * e;
      neumaier_add(a, c, term);
      *m = *m + term.abs();
    }
  }
  for (a, &c) in acc.iter_mut().zip(comp.iter()) {
    *a = *a + c;
  }
  // Determinacy gate: reject a result at or below the fold's own rounding floor
  // rather than let `l2_renorm` amplify rounding noise into a direction. `K = 16`
  // against the proven `<= 4 * EPSILON * ||M||` bound (module Input domain note),
  // so exact cancellation (`||exact|| = 0`) is always caught, at every ordering.
  let tau = C::from_f32(16.0) * C::EPSILON * l2_norm(&mag);
  if l2_norm(&acc) <= tau {
    return Err(WinditError::NonFinite);
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// Add `term` into the running sum `acc` with Neumaier compensation `comp`.
///
/// The correction is `(larger - new_sum) + smaller`: the part of the smaller
/// magnitude that `new_sum` could not represent, which is exactly what a naive
/// `acc + term` discards. Accumulated into `comp` and folded back once at the
/// end, it holds the fold's error to a small multiple of the accumulated term
/// magnitude, which is what makes the determinacy gate in
/// [`weighted_sum_renorm`] sound.
fn neumaier_add<C: Real>(acc: &mut C, comp: &mut C, term: C) {
  let sum = *acc + term;
  *comp = *comp
    + if acc.abs() >= term.abs() {
      (*acc - sum) + term
    } else {
      (term - sum) + *acc
    };
  *acc = sum;
}

/// The largest absolute component of `v`, or `ZERO` for an empty one.
///
/// NaN compares false against everything, so it never becomes the maximum; it
/// reaches the caller's own sum instead and carries through to the non-finite
/// result that rejects the vector.
fn max_magnitude<C: Real>(v: &[C]) -> C {
  let mut max = C::ZERO;
  for &x in v {
    let m = x.abs();
    if m > max {
      max = m;
    }
  }
  max
}

/// The exponent of the power of two a reduction over `v` divides by: that of
/// `v`'s largest component.
///
/// `None` when `v` has no scale to speak of — it is empty or all zero — or when
/// a component is infinite. Both are conditions to reject rather than reduce.
fn scale_exponent<C: Real>(v: &[C]) -> Option<i32> {
  let m = max_magnitude(v);
  if m == C::ZERO || !m.is_finite() {
    return None;
  }
  Some(m.exponent())
}

/// `sum_i (v_i / 2^exp)^2`, as an explicit left fold from `ZERO`.
///
/// With `exp` from [`scale_exponent`] every ratio is under two, so the sum is
/// under `4 * v.len()` and cannot overflow for any slice that fits in memory;
/// nor can it be zero, since the largest component divides to at least one.
/// Both bounds hold however far the unscaled squares would have left the scalar.
/// Dividing by a power of two is exact, so this is not an approximation of the
/// unscaled sum of squares but that same sum with its exponent moved by
/// `-2 * exp`.
///
/// Spelled out rather than through `Iterator::sum`, which would cost a `Sum`
/// supertrait on [`Real`] for a syntax preference.
fn scaled_sum_of_squares<C: Real>(v: &[C], exp: i32) -> C {
  let scale = C::ONE.ldexp(exp);
  let mut sum = C::ZERO;
  for &x in v {
    let r = x / scale;
    sum = sum + r * r;
  }
  sum
}

/// The L2 norm of `v`, via [`Real::sqrt`] (core has no `f32::sqrt`).
///
/// Taken against `v`'s own power-of-two scale and shifted back afterwards, so
/// the sum of squares never leaves the compute scalar even when the norm itself
/// would: `[f64::MAX, f64::MAX]` has norm `sqrt(2) * f64::MAX`, which overflows,
/// yet this returns it (as an overflow to infinity) rather than by squaring into
/// one. The result is `sqrt(sum(v_i^2))` to the bit wherever that direct
/// computation was valid, and the vector's actual norm wherever it was not.
///
/// [`SaliencyWeighted`] weights each window by this, so a window whose norm is
/// unrepresentable still contributes its direction: the shared magnitude divides
/// back out in the final renormalization, and only the ratios between norms —
/// what the weighting means — survive.
///
/// A vector with no scale returns its own largest magnitude: zero when it is all
/// zero, and the non-finite component itself when one is infinite. Each is
/// already the weight — and then the accumulator — that the caller must reject.
fn l2_norm<C: Real>(v: &[C]) -> C {
  let Some(exp) = scale_exponent(v) else {
    return max_magnitude(v);
  };
  scaled_sum_of_squares(v, exp).sqrt().ldexp(exp)
}

/// Normalize `v` to unit L2 length in place.
///
/// # Errors
///
/// [`WinditError::NonFinite`] if `v` cannot be normalized to a finite unit
/// vector: it is all zero, or some component is not finite.
fn l2_renorm<C: Real>(v: &mut [C]) -> Result<(), WinditError> {
  // The one rejection, and a property of the input rather than of some
  // intermediate leaving range: an all-zero vector has no direction to normalize
  // to, and neither has one with an infinite component.
  let Some(exp) = scale_exponent(v) else {
    return Err(WinditError::NonFinite);
  };
  let scale = C::ONE.ldexp(exp);
  // `unit` is the norm divided by `scale`, which puts it in [1, 2*sqrt(len)]:
  // always representable, and always at least one, even for a vector whose norm
  // is not representable at all (`[f64::MAX, f64::MAX]`). Dividing by `scale` and
  // by `unit` separately is what avoids ever forming that norm. Both divisors are
  // exact power-of-two relatives of the direct computation's, so the quotient is
  // `v_i / norm` to the bit wherever the direct computation was valid.
  let unit = scaled_sum_of_squares(v, exp).sqrt();
  // Only a NaN component reaches this: it never becomes the maximum, so it
  // passes the scale check above and surfaces in the sum instead. Nothing is
  // written until every rejection is past, so a rejected vector is left as it
  // was.
  if !unit.is_finite() {
    return Err(WinditError::NonFinite);
  }
  for x in v.iter_mut() {
    *x = (*x / scale) / unit;
  }
  Ok(())
}
