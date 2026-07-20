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
//! Every policy ends in an L2 renormalization, and the arithmetic can leave the
//! scalar's range long before the vector does. A vector's squares go first:
//! `[1e20_f32, 0.0]` squares to infinity and `[1e-30_f32, 0.0]` to zero, though
//! both normalize to exactly `[1, 0]`. A vector's norm can follow while the unit
//! vector stays ordinary: `[3e38_f32, 3e38]` has a norm of `sqrt(2) * 3e38`.
//! None of that is a property of the vector, and none of it is grounds to reject
//! one.
//!
//! Every reduction here is therefore taken in a domain shifted by a power of
//! two, chosen from the inputs' own exponents before any of them is folded.
//! Scaling by a power of two moves the exponent and leaves the significand
//! alone, so the shifted reduction is the unshifted one to the bit — same
//! direction, same rounding — wherever the unshifted one was valid, and the
//! shift divides back out of a renormalized result. What it buys is range: each
//! shift is the one nearest zero that keeps every product, partial sum, and sum
//! of squares clear of both ends of the scalar.
//!
//! There is no second attempt anywhere, which is what keeps
//! [`WinditError::NonFinite`] meaning what it says: an all-zero vector, or a
//! component that is itself not finite. A retry cannot tell those from a norm
//! that merely overflowed, and a vector whose components cancel exactly is not a
//! vector whose norm was unrepresentable — it is a vector with no direction.
//! Because every scale is a power of two, a cancelling sum stays exactly zero
//! and is rejected on the only path there is, rather than rounding to some
//! arbitrary residue that normalizes to an arbitrary answer.
//!
//! [`Real`]: crate::scalar::Real
//! [`Real::from_f32`]: crate::scalar::Real::from_f32
//! [`Span`]: crate::plan::Span
//! [`Span::coverage`]: crate::plan::Span::coverage

use std::{vec, vec::Vec};

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
/// `C` is the compute scalar — the [`Real`] domain the math
/// runs in — and defaults to `f32`. The default is what keeps `dyn
/// AggregatePolicy` and `Box<dyn AggregatePolicy>` spelling the `f32` policy
/// object; a policy used at another scalar names it, as in
/// `Box<dyn AggregatePolicy<f64>>`. Note that trait objects are per-scalar: an
/// `AggregatePolicy<f32>` object and an `AggregatePolicy<f64>` object are
/// unrelated types and cannot share one collection.
///
/// # Custom policies
///
/// Implement [`aggregate_values`](AggregatePolicy::aggregate_values) to add a
/// strategy. This one keeps the first window unchanged, and stays `f32`-only by
/// leaving the type parameter at its default:
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
///     embeddings: &[&[f32]],
///     _coverages: &[f32],
///     dim: usize,
///   ) -> Result<Vec<f32>, WinditError> {
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
/// `&[&[C]]` and `Vec<C>` — makes the same policy serve every scalar.
pub trait AggregatePolicy<C: Real = f32> {
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
  let mut coverages = Vec::with_capacity(windows.len());
  for w in windows {
    coverages.push(w.span.coverage());
  }

  // Fast path: the stored scalar already is the compute scalar (true of every
  // scalar this crate ships), so borrow the storage rather than widen it into
  // fresh buffers. The collect short-circuits on the first `None`, making this
  // one `Option` check per window — without it, every f32 aggregation would
  // gain an allocation and a full copy.
  let borrowed: Option<Vec<&[ComputeOf<E>]>> = windows
    .iter()
    .map(|w| <E::Scalar as Scalar>::as_compute_slice(w.value.as_slice()))
    .collect();

  let raw = if let Some(embeddings) = borrowed {
    policy.aggregate_values(&embeddings, &coverages, dim)?
  } else {
    let widened: Vec<Vec<ComputeOf<E>>> = windows
      .iter()
      .map(|w| w.value.as_slice().iter().map(|s| s.to_compute()).collect())
      .collect();
    let embeddings: Vec<&[ComputeOf<E>]> = widened.iter().map(Vec::as_slice).collect();
    policy.aggregate_values(&embeddings, &coverages, dim)?
  };
  E::from_unnormalized(&raw)
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
    // Materialized rather than recomputed inside the fold: `weighted_sum_renorm`
    // reads each weight twice — once to size its shift, once to apply it — and a
    // norm is a full pass over the window.
    let shift = saliency_shift(embeddings, dim);
    let weights: Vec<C> = embeddings
      .iter()
      .map(|emb| scaled_l2_norm(emb, shift))
      .collect();
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
    let shift = ema_shift(embeddings);
    let mut state = embeddings[0].to_vec();
    if shift != 0 {
      for s in &mut state {
        *s = s.ldexp(shift);
      }
    }
    // The shift rides on `alpha` rather than on each component: the state is
    // already in the shifted domain, so `(alpha * 2^s) * e` is the incoming term
    // scaled to match. At the ordinary `shift == 0` this is `alpha` itself and
    // the fold is unchanged.
    let scaled_alpha = alpha.ldexp(shift);
    for emb in &embeddings[1..] {
      for (s, &e) in state.iter_mut().zip(emb.iter()) {
        *s = scaled_alpha * e + complement * *s;
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
/// One pass, no retry. The power-of-two shift that keeps the accumulation in
/// range is chosen from the inputs before any of them is folded, so there is no
/// failure to recover from and nothing to recompute a second way. An
/// accumulator that lands on the zero vector landed there because the inputs
/// cancel, and [`l2_renorm`] says so.
fn weighted_sum_renorm<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f32],
  dim: usize,
  weight: impl Fn(usize, &[C]) -> C,
) -> Result<Vec<C>, WinditError> {
  // `dim` is caller-supplied, but this has just proved it is the length of an
  // embedding that exists, so the accumulator below is bounded by real data.
  check_inputs(embeddings, coverages, dim)?;
  let shift = accumulation_shift(embeddings, &weight);
  let mut acc = vec![C::ZERO; dim];
  for (i, emb) in embeddings.iter().enumerate() {
    // The whole shift rides on the weight: `(w * 2^s) * e` and `w * (e * 2^s)`
    // are the same number, and applying it once per window rather than once per
    // component leaves the inner loop the plain multiply-accumulate it was.
    let w = weight(i, emb).ldexp(shift);
    for (a, &e) in acc.iter_mut().zip(emb.iter()) {
      *a = *a + w * e;
    }
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// The power-of-two exponent to fold into every weight so that
/// `sum_i w_i * emb_ij` neither overflows nor is flushed to zero.
///
/// Zero — the ordinary case, and every case whose direct fold was already in
/// range — leaves the accumulation exactly the fold it has always been. Both
/// bounds are deliberately tight rather than normalizing: `[[1e30], [-1e30],
/// [1e-30]]` spans the f32 range internally and is shifted by nothing, so the
/// `1e-30` that survives the cancellation survives here too. A shift that always
/// brought the largest product to one would flush it away and report a zero
/// vector for a sequence that has a direction.
fn accumulation_shift<C: Real>(embeddings: &[&[C]], weight: &impl Fn(usize, &[C]) -> C) -> i32 {
  let mut w_max = C::ZERO;
  let mut e_max = C::ZERO;
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb).abs();
    if w > w_max {
      w_max = w;
    }
    let m = max_magnitude(emb);
    if m > e_max {
      e_max = m;
    }
  }
  // Nothing a shift can do: a zero on either side makes the accumulator the zero
  // vector and a non-finite one makes it non-finite, and `l2_renorm` rejects
  // both. A NaN never becomes a maximum, so it leaves these bounds finite and
  // reaches that same rejection through the fold instead.
  if w_max == C::ZERO || e_max == C::ZERO || !w_max.is_finite() || !e_max.is_finite() {
    return 0;
  }
  let product = w_max.exponent() + e_max.exponent();
  // `|acc_j| <= count * w_max * e_max`, and each factor is under twice its own
  // power of two, so the accumulation stays under `2^(product + count_bits + 2)`.
  // Holding that a binade below `2^MAX_EXP` — the first power of two the scalar
  // cannot represent — keeps every product and every partial sum finite.
  let count_bits = (usize::BITS - embeddings.len().leading_zeros()) as i32;
  let highest = C::MAX_EXP - 3 - count_bits - product;
  // At the other end the largest product must stay normal, or an accumulation of
  // uniformly tiny magnitudes — `1e-30_f32` weighted by its own norm — underflows
  // to the zero vector that means "no direction".
  let lowest = (C::MIN_EXP - 1) - product;
  // `highest - lowest` is `MAX_EXP - MIN_EXP - 2 - count_bits`, positive at both
  // scalars for every count a slice can hold, so this is the value nearest zero
  // within a range that is never empty.
  lowest.max(0).min(highest)
}

/// The power-of-two exponent an EMA folds its state in.
///
/// Non-zero only when every component is subnormal, and then only by enough to
/// make the largest of them normal. `alpha * e` halves a subnormal toward zero —
/// `0.5 * f32::from_bits(1)` rounds to nothing — so a state assembled from such
/// components flushes to the zero vector and is reported as a direction that
/// does not exist. A convex combination never grows past its largest input, so
/// there is nothing to shift down and no overflow to guard; ordinary magnitudes
/// shift by nothing and fold exactly as they did.
fn ema_shift<C: Real>(embeddings: &[&[C]]) -> i32 {
  let mut e_max = C::ZERO;
  for emb in embeddings {
    let m = max_magnitude(emb);
    if m > e_max {
      e_max = m;
    }
  }
  if e_max == C::ZERO || !e_max.is_finite() {
    return 0;
  }
  ((C::MIN_EXP - 1) - e_max.exponent()).max(0)
}

/// The power-of-two exponent every saliency weight is scaled by, so that no
/// window's norm has to be representable on its own.
///
/// `[3e38_f32, 3e38]` is a finite vector with an infinite norm. Weighting by
/// that infinity and then dividing it back out gives NaN, which is how the one
/// policy that reads magnitudes came to reject vectors every other policy
/// normalized. Scaling every norm by one shared exact factor keeps them in range
/// and leaves their ratios — the only thing a weight means here — untouched; the
/// common factor divides out again in the renormalization that ends the policy.
/// Zero, the ordinary case, leaves each weight the window's own L2 norm.
fn saliency_shift<C: Real>(embeddings: &[&[C]], dim: usize) -> i32 {
  let mut e_max = C::ZERO;
  for emb in embeddings {
    let m = max_magnitude(emb);
    if m > e_max {
      e_max = m;
    }
  }
  if e_max == C::ZERO || !e_max.is_finite() {
    return 0;
  }
  // A norm is under `2 * sqrt(dim)` times its vector's largest component, so it
  // lands under `2^(exponent + 2 + dim_bits / 2)`. Halved as an unsigned count
  // because `i32::div_ceil` is not stable at this crate's MSRV.
  let dim_bits = usize::BITS - dim.leading_zeros();
  let excess = e_max.exponent() + 2 + dim_bits.div_ceil(2) as i32 - C::MAX_EXP;
  if excess > 0 {
    -excess
  } else {
    0
  }
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

/// The L2 norm of `v`, scaled by `2^shift`, via [`Real::sqrt`] (core has no
/// `f32::sqrt`).
///
/// Taken against `v`'s own power-of-two scale and shifted back afterwards, so
/// the sum of squares never leaves the scalar. At `shift == 0` the result is
/// `sqrt(sum(v_i^2))` to the bit wherever that direct computation was valid, and
/// the vector's actual norm wherever it was not.
///
/// `shift` exists because a norm can be unrepresentable although every component
/// is ordinary; see [`saliency_shift`], its only caller.
///
/// A vector with no scale returns its own largest magnitude: zero when it is all
/// zero, and the non-finite component itself when one is infinite. Each is
/// already the weight — and then the accumulator — that the caller must reject.
fn scaled_l2_norm<C: Real>(v: &[C], shift: i32) -> C {
  let Some(exp) = scale_exponent(v) else {
    return max_magnitude(v);
  };
  scaled_sum_of_squares(v, exp).sqrt().ldexp(exp + shift)
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
  // is not representable at all (`[3e38_f32, 3e38]`). Dividing by `scale` and by
  // `unit` separately is what avoids ever forming that norm. Both divisors are
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
