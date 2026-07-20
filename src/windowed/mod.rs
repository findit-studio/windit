//! Windowed values: the minimal vector trait and the value-plus-span carrier.
//!
//! [`Windowed<V>`] pairs a per-window value with the [`Span`] it covers, generic
//! over `V` (an embedding, a probability, a logit). Embeddings additionally
//! implement [`Vector`], the minimal slice trait through which the crate
//! aggregates and reconstructs them, generic over the [`Scalar`] they store.

use crate::{error::WinditError, plan::Span, scalar::Scalar};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// The compute type behind an embedding: the floating-point domain its
/// aggregation runs in.
///
/// `f64` for both shipped scalars: it is `V::Scalar` again for an `f64`
/// embedding, and the wider `f64` for an `f32` one, which stores `f32` but
/// computes in `f64`. See the [`scalar`](crate::scalar) module.
pub type ComputeOf<V> = <<V as Vector>::Scalar as Scalar>::Compute;

/// A fixed-dimension embedding stored as a slice of some [`Scalar`].
///
/// The crate aggregates, compares, and reconstructs embeddings only through this
/// trait, so any 384-, 512-, or 768-dimension type fits unchanged. Implementors
/// decide what "normalized" means — L2 unit length is typical — inside
/// [`from_unnormalized`](Vector::from_unnormalized).
///
/// An `f32` embedding — which is every embedding in practice — writes
/// `type Scalar = f32;`; [`ComputeOf<Self>`](ComputeOf) is then `f64`, so
/// [`from_unnormalized`](Vector::from_unnormalized) receives `&[f64]` and
/// narrows to `f32` storage.
pub trait Vector: Sized {
  /// The scalar type this embedding stores.
  type Scalar: Scalar;

  /// The embedding as a contiguous slice of its stored scalar.
  fn as_slice(&self) -> &[Self::Scalar];

  /// Build a normalized embedding from raw (unnormalized) values in the
  /// embedding's compute domain.
  ///
  /// Aggregation runs in [`ComputeOf<Self>`](ComputeOf) and hands its result
  /// here, so an implementor whose storage is narrower than its compute type
  /// does its own conversion — including any rounding or saturation — in this
  /// method. That keeps the "normalize to a finite unit vector" contract stated
  /// where it is true, in the float domain, and leaves the quantization scale
  /// with the only code that knows it.
  ///
  /// # Errors
  ///
  /// Returns a [`WinditError`] when the input cannot form a valid embedding —
  /// for example [`WinditError::Empty`] for an empty slice, or
  /// [`WinditError::NonFinite`] when it cannot be normalized to a finite unit
  /// vector.
  fn from_unnormalized(v: &[ComputeOf<Self>]) -> Result<Self, WinditError>;

  /// The embedding dimension: the length of [`as_slice`](Vector::as_slice).
  fn dim(&self) -> usize {
    self.as_slice().len()
  }
}

/// A value paired with the [`Span`] of input it covers.
///
/// Generic over the value type `V` — an embedding, a probability, a logit, an
/// energy — so one post-processing surface serves embedding aggregation, VAD
/// smoothing and segmentation, and ASR stitching. The span is the thread that
/// links each value back to the input region it summarizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Windowed<V> {
  /// The per-window value produced from the span.
  pub(crate) value: V,
  /// The span of input this value covers.
  pub(crate) span: Span,
}

impl<V> Windowed<V> {
  /// Pair a value with the span it covers.
  pub const fn new(value: V, span: Span) -> Self {
    Self { value, span }
  }

  /// A reference to the per-window value.
  pub const fn value(&self) -> &V {
    &self.value
  }

  /// The span this value covers.
  pub const fn span(&self) -> Span {
    self.span
  }

  /// The per-window value, moved out of the pairing.
  ///
  /// The owning counterpart to [`value`](Windowed::value), for callers that need
  /// the `V` itself — the embeddings behind a `keep_separate` result, say —
  /// rather than a borrow of it. The fields are not public, so this is the only
  /// way to obtain an owned `V` without cloning.
  #[must_use]
  pub fn into_value(self) -> V {
    self.value
  }

  /// The value and its span, moved out of the pairing.
  ///
  /// [`into_value`](Windowed::into_value) when the span is not needed.
  #[must_use]
  pub fn into_parts(self) -> (V, Span) {
    (self.value, self.span)
  }
}

/// A [`Windowed`] embedding: a value that implements [`Vector`] paired with the
/// [`Span`] it covers. The multi-vector aggregation path is `Vec<WindowEmbedding<E>>`.
pub type WindowEmbedding<E> = Windowed<E>;
