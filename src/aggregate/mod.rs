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
    weighted_sum_renorm(embeddings, coverages, dim, |_, emb| l2_norm(emb))
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
    let mut state = embeddings[0].to_vec();
    for emb in &embeddings[1..] {
      for (s, &e) in state.iter_mut().zip(emb.iter()) {
        *s = alpha * e + (C::ONE - alpha) * *s;
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
fn weighted_sum_renorm<C: Real>(
  embeddings: &[&[C]],
  coverages: &[f32],
  dim: usize,
  weight: impl Fn(usize, &[C]) -> C,
) -> Result<Vec<C>, WinditError> {
  check_inputs(embeddings, coverages, dim)?;
  let mut acc = vec![C::ZERO; dim];
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb);
    for (a, &e) in acc.iter_mut().zip(emb.iter()) {
      *a = *a + w * e;
    }
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// The L2 norm of `v`, via [`Real::sqrt`] (core has no `f32::sqrt`).
///
/// The sum is an explicit left fold from `ZERO` rather than `Iterator::sum`,
/// which would cost a `Sum` supertrait on [`Real`] for a syntax preference. The
/// association order is identical, so the f32 result is unchanged.
fn l2_norm<C: Real>(v: &[C]) -> C {
  let mut sum = C::ZERO;
  for &x in v {
    sum = sum + x * x;
  }
  sum.sqrt()
}

/// Normalize `v` to unit L2 length in place.
///
/// # Errors
///
/// [`WinditError::NonFinite`] if the norm is zero or not finite (which also
/// catches a non-finite component, since it propagates into the norm).
fn l2_renorm<C: Real>(v: &mut [C]) -> Result<(), WinditError> {
  let norm = l2_norm(v);
  if !norm.is_finite() || norm == C::ZERO {
    return Err(WinditError::NonFinite);
  }
  for x in v.iter_mut() {
    *x = *x / norm;
  }
  Ok(())
}
