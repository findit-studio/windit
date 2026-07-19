//! Windowed values: the minimal vector trait and the value-plus-span carrier.
//!
//! [`Windowed<V>`] pairs a per-window value with the [`Span`] it covers, generic
//! over `V` (an embedding, a probability, a logit). Embeddings additionally
//! implement [`Vector`], the minimal f32-slice trait through which the crate
//! aggregates and reconstructs them.

use crate::{error::WinditError, plan::Span};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// A fixed-dimension embedding living in f32 space.
///
/// The crate aggregates, compares, and reconstructs embeddings only through this
/// trait, so any 384-, 512-, or 768-dimension type fits unchanged. Implementors
/// decide what "normalized" means — L2 unit length is typical — inside
/// [`from_unnormalized`](Vector::from_unnormalized).
pub trait Vector: Sized {
  /// The embedding as a contiguous f32 slice.
  fn as_slice(&self) -> &[f32];

  /// Build a normalized embedding from a raw (unnormalized) f32 slice.
  ///
  /// # Errors
  ///
  /// Returns a [`WinditError`] when the input cannot form a valid embedding —
  /// for example [`WinditError::Empty`] for an empty slice, or
  /// [`WinditError::NonFinite`] when it cannot be normalized to a finite unit
  /// vector.
  fn from_unnormalized(v: &[f32]) -> Result<Self, WinditError>;

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
  pub value: V,
  /// The span of input this value covers.
  pub span: Span,
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
}

/// A [`Windowed`] embedding: a value that implements [`Vector`] paired with the
/// [`Span`] it covers. The multi-vector aggregation path is `Vec<WindowEmbedding<E>>`.
pub type WindowEmbedding<E> = Windowed<E>;
