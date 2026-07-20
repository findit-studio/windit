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
//!   (Neumaier's variant of Kahan-Babuška) sum: it carries the low-order bits every naive
//!   fold discards, so an exactly cancelling sum lands at exactly zero
//!   regardless of association order, and [`WinditError::NonFinite`] keeps
//!   meaning "no direction" rather than "the fold happened to round to a
//!   residue".
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
//! One magnitude window is irreducible, because `f64` is the widest domain there
//! is. [`SaliencyWeighted`] weights each window by its L2 norm, so it forms the
//! square of a magnitude where the other policies stay linear in it. Past about
//! `1.3e154` (roughly `sqrt(f64::MAX)`) that square overflows, and below about
//! `1.5e-162` it underflows to zero — and no power-of-two prescaling can pull
//! either back without the subnormal-flushing fabrication a scaled fold
//! reintroduces, so that one policy returns [`WinditError::NonFinite`] outside
//! the window rather than invent a direction. The three linear policies have no
//! such window, and every realistic embedding magnitude (unit-ish, and never
//! past `~1e38`) sits more than a hundred orders of magnitude inside it.
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
  /// # Errors
  ///
  /// - [`WinditError::Empty`] if `embeddings` is empty.
  /// - [`WinditError::DimMismatch`] if `coverages.len() != embeddings.len()` or
  ///   any embedding's length differs from `dim`.
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
/// itself a norm), it is the one built-in with a bounded `f64` magnitude window:
/// a component past about `sqrt(f64::MAX)` (`~1.3e154`) overflows and one below
/// about `1.5e-162` underflows to zero, and either is rejected with
/// [`WinditError::NonFinite`] rather than rescaled by the fabrication-prone shift
/// that widening to `f64` exists to retire. Realistic magnitudes sit deep inside
/// the window; see the module [Scale](self#scale) note.
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
    check_inputs(embeddings, coverages, dim)?;
    // A convex EMA needs alpha in [0, 1]; anything else (including NaN, which
    // fails the range test) is a configuration error, not a normalizable vector.
    // The test runs on the f32 configuration field, before widening, so the
    // same range is enforced at every compute scalar.
    if !(0.0..=1.0).contains(&self.alpha) {
      return Err(WinditError::AlphaOutOfRange);
    }
    // `1 - alpha` is formed in C rather than folded in f32 first: at C = f64
    // the f32 fold would round the complement to f32 precision and make the
    // f64 path gratuitously less accurate than its type promises.
    let alpha = C::from_f32(self.alpha);
    let complement = C::ONE - alpha;
    let mut state = try_vec_with_capacity(embeddings[0].len())?;
    state.extend_from_slice(embeddings[0]);
    // No prescaling of the state: computing in f64 keeps `alpha * e` exact for
    // every f32-derived subnormal (`0.5 * f32::from_bits(1)` is `2^-150`, a
    // normal f64), so the convex step never flushes the small component a
    // power-of-two shift used to discard by pushing it below the subnormal floor.
    for emb in &embeddings[1..] {
      for (s, &e) in state.iter_mut().zip(emb.iter()) {
        *s = alpha * e + complement * *s;
      }
    }
    l2_renorm(&mut state)?;
    Ok(state)
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

/// Validate that `embeddings` is non-empty, `coverages` matches its length, and
/// every embedding has length `dim`.
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
  for emb in embeddings {
    if emb.len() != dim {
      return Err(WinditError::DimMismatch {
        got: emb.len(),
        expected: dim,
      });
    }
  }
  Ok(())
}

/// Accumulate `sum_i weight(i, emb_i) * emb_i` and L2-renormalize it.
///
/// One pass, no retry, no prescaling: the compute scalar is `f64` (an `f32`
/// embedding widened before this ran), which has the range to fold every `f32`
/// derived product without overflow or underflow. The sum is *compensated*
/// (Neumaier), so a wide exponent spread cannot fabricate a direction out of a
/// sum that exactly cancels: the low-order bits a naive fold would drop when a
/// small term meets a large partial sum are carried and added back, leaving an
/// exactly cancelling accumulation at exactly zero for [`l2_renorm`] to reject
/// as the directionless vector it is, whatever order the windows fold in.
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
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb);
    for ((a, c), &e) in acc.iter_mut().zip(comp.iter_mut()).zip(emb.iter()) {
      neumaier_add(a, c, w * e);
    }
  }
  for (a, &c) in acc.iter_mut().zip(comp.iter()) {
    *a = *a + c;
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// Add `term` into the running sum `acc` with Neumaier compensation `comp`.
///
/// The correction is `(larger - new_sum) + smaller`: the part of the smaller
/// magnitude that `new_sum` could not represent, which is exactly what a naive
/// `acc + term` discards. Accumulated into `comp` and folded back once at the
/// end, it makes the total independent of association order — so an exactly
/// cancelling set of terms totals exactly zero however the windows are ordered.
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
