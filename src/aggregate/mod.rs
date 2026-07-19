//! Aggregation policies: combine a window sequence into a single embedding.
//!
//! `AggregatePolicy` is the object-safe seam: it works entirely in f32 space
//! (`aggregate_f32`), so `&dyn AggregatePolicy` is usable. The generic free
//! function `aggregate` extracts the f32 slices and per-window coverages from
//! a `&[WindowEmbedding<E>]`, runs the policy, and reconstructs the embedding
//! type `E` through `Vector::from_unnormalized`. Keeping reconstruction out
//! of the trait is what lets the trait stay object-safe while embedding
//! reconstruction stays generic.
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
//!   `aggregate_f32` directly with vectors that still carry magnitude.
//!
//! `keep_separate` is the multi-vector path: it returns every window unchanged
//! for callers that want per-window embeddings rather than one summary.
//!
//! [`Span`]: crate::plan::Span
//! [`Span::coverage`]: crate::plan::Span::coverage

use std::{vec, vec::Vec};

use crate::{
  error::WinditError,
  windowed::{Vector, WindowEmbedding},
};

#[cfg(test)]
mod tests;

/// A policy that combines a sequence of window embeddings into one embedding.
///
/// The single required method operates in f32 space so the trait is object-safe
/// (`&dyn AggregatePolicy` works). Embedding reconstruction lives in the generic
/// free function [`aggregate`], not here.
///
/// # Custom policies
///
/// Implement [`aggregate_f32`](AggregatePolicy::aggregate_f32) to add a strategy.
/// This one keeps the first window unchanged:
///
/// ```
/// use windit::aggregate::AggregatePolicy;
/// use windit::WinditError;
///
/// struct FirstWindow;
///
/// impl AggregatePolicy for FirstWindow {
///   fn aggregate_f32(
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
pub trait AggregatePolicy {
  /// Combine `embeddings` (each a `dim`-length f32 slice) with their matching
  /// `coverages` into a single `dim`-length f32 vector.
  ///
  /// The built-in policies return an L2-normalized vector, and a custom policy
  /// should do the same; either way [`aggregate`] re-normalizes the result
  /// through [`Vector::from_unnormalized`]. `coverages` must have the same length
  /// as `embeddings` even for policies that do not weight by coverage.
  ///
  /// # Errors
  ///
  /// - [`WinditError::Empty`] if `embeddings` is empty.
  /// - [`WinditError::DimMismatch`] if `coverages.len() != embeddings.len()` or
  ///   any embedding's length differs from `dim`.
  /// - [`WinditError::NonFinite`] if the combined vector cannot be normalized to
  ///   a finite unit vector (zero norm or a non-finite component).
  fn aggregate_f32(
    &self,
    embeddings: &[&[f32]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<f32>, WinditError>;
}

/// Aggregate a sequence of window embeddings into one embedding of type `E`.
///
/// Extracts each window's f32 slice and [`Span::coverage`](crate::plan::Span::coverage),
/// runs `policy`, and reconstructs `E` via [`Vector::from_unnormalized`]. Works
/// with any policy, including `&dyn AggregatePolicy`.
///
/// # Errors
///
/// [`WinditError::Empty`] if `windows` is empty; otherwise any error from the
/// policy or from [`Vector::from_unnormalized`].
pub fn aggregate<E, P>(policy: &P, windows: &[WindowEmbedding<E>]) -> Result<E, WinditError>
where
  E: Vector,
  P: AggregatePolicy + ?Sized,
{
  if windows.is_empty() {
    return Err(WinditError::Empty);
  }
  let dim = windows[0].value.dim();
  let mut embeddings = Vec::with_capacity(windows.len());
  let mut coverages = Vec::with_capacity(windows.len());
  for w in windows {
    embeddings.push(w.value.as_slice());
    coverages.push(w.span.coverage());
  }
  let raw = policy.aggregate_f32(&embeddings, &coverages, dim)?;
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmaRenormalized {
  /// The smoothing factor, which must be in `[0, 1]`: larger values track recent
  /// windows more. Outside that range (or NaN) is an
  /// [`AlphaOutOfRange`](WinditError::AlphaOutOfRange) error.
  pub alpha: f32,
}

/// L2-norm-weighted mean, then renormalization: higher-magnitude inputs dominate.
///
/// Each window is weighted by the L2 norm of its input slice, so more salient
/// (larger-magnitude) vectors pull the result toward them. `coverages` are
/// ignored beyond the length check. This differs from the other strategies only
/// when the inputs carry magnitude; [`aggregate`] feeds unit vectors, so use
/// [`aggregate_f32`](AggregatePolicy::aggregate_f32) directly to exploit it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SaliencyWeighted;

impl AggregatePolicy for CoverageWeightedMean {
  fn aggregate_f32(
    &self,
    embeddings: &[&[f32]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<f32>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |i, _| coverages[i])
  }
}

impl AggregatePolicy for MeanRenormalized {
  fn aggregate_f32(
    &self,
    embeddings: &[&[f32]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<f32>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |_, _| 1.0)
  }
}

impl AggregatePolicy for SaliencyWeighted {
  fn aggregate_f32(
    &self,
    embeddings: &[&[f32]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<f32>, WinditError> {
    weighted_sum_renorm(embeddings, coverages, dim, |_, emb| l2_norm(emb))
  }
}

impl AggregatePolicy for EmaRenormalized {
  fn aggregate_f32(
    &self,
    embeddings: &[&[f32]],
    coverages: &[f32],
    dim: usize,
  ) -> Result<Vec<f32>, WinditError> {
    check_inputs(embeddings, coverages, dim)?;
    // A convex EMA needs alpha in [0, 1]; anything else (including NaN, which
    // fails the range test) is a configuration error, not a normalizable vector.
    if !(0.0..=1.0).contains(&self.alpha) {
      return Err(WinditError::AlphaOutOfRange);
    }
    let mut state = embeddings[0].to_vec();
    for emb in &embeddings[1..] {
      for (s, &e) in state.iter_mut().zip(emb.iter()) {
        *s = self.alpha * e + (1.0 - self.alpha) * *s;
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
    /// The EMA smoothing factor, forwarded to [`EmaRenormalized::alpha`].
    alpha: f32,
  },
  /// Selects [`SaliencyWeighted`].
  SaliencyWeighted,
}

#[cfg(all(feature = "serde", any(feature = "std", feature = "alloc")))]
impl AggregatePolicyKind {
  /// Build the boxed built-in policy this kind selects.
  #[must_use]
  pub fn into_policy(self) -> std::boxed::Box<dyn AggregatePolicy> {
    use std::boxed::Box;
    match self {
      Self::CoverageWeightedMean => Box::new(CoverageWeightedMean),
      Self::MeanRenormalized => Box::new(MeanRenormalized),
      Self::Ema { alpha } => Box::new(EmaRenormalized { alpha }),
      Self::SaliencyWeighted => Box::new(SaliencyWeighted),
    }
  }
}

/// Validate that `embeddings` is non-empty, `coverages` matches its length, and
/// every embedding has length `dim`.
fn check_inputs(embeddings: &[&[f32]], coverages: &[f32], dim: usize) -> Result<(), WinditError> {
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
fn weighted_sum_renorm(
  embeddings: &[&[f32]],
  coverages: &[f32],
  dim: usize,
  weight: impl Fn(usize, &[f32]) -> f32,
) -> Result<Vec<f32>, WinditError> {
  check_inputs(embeddings, coverages, dim)?;
  let mut acc = vec![0.0f32; dim];
  for (i, emb) in embeddings.iter().enumerate() {
    let w = weight(i, emb);
    for (a, &e) in acc.iter_mut().zip(emb.iter()) {
      *a += w * e;
    }
  }
  l2_renorm(&mut acc)?;
  Ok(acc)
}

/// The L2 norm of `v`, via `libm::sqrtf` (core has no `f32::sqrt`).
fn l2_norm(v: &[f32]) -> f32 {
  libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>())
}

/// Normalize `v` to unit L2 length in place.
///
/// # Errors
///
/// [`WinditError::NonFinite`] if the norm is zero or not finite (which also
/// catches a non-finite component, since it propagates into the norm).
fn l2_renorm(v: &mut [f32]) -> Result<(), WinditError> {
  let norm = l2_norm(v);
  if !norm.is_finite() || norm == 0.0 {
    return Err(WinditError::NonFinite);
  }
  for x in v.iter_mut() {
    *x /= norm;
  }
  Ok(())
}
