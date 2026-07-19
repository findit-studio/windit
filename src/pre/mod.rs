//! Pre-processing: turn a span into a fixed-width, padded window and its mask.
//!
//! A [`Span`] from a [`WindowPlan`](crate::plan::WindowPlan) selects a run of
//! real input elements; [`slice_pad_mask`] copies them into a `window`-length
//! buffer, right-pads the remainder, and produces the matching attention mask
//! (`1` for real elements, `0` for padding).

#[cfg(any(feature = "std", feature = "alloc"))]
use std::vec::Vec;

#[cfg(any(feature = "std", feature = "alloc"))]
use crate::{error::WinditError, plan::Span};

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// Slice `elements` to `span`, right-pad to the span's window size with `pad`,
/// and build the matching mask.
///
/// Both returned vectors have length `span.window`. The first `span.len`
/// positions are the real elements `elements[span.start..span.start + span.len]`
/// (mask `1`); the remaining `span.window - span.len` positions are `pad`
/// (mask `0`).
///
/// The span must be valid for `elements`, i.e. produced by a
/// [`WindowPlan`](crate::plan::WindowPlan) over the same sequence. A span that
/// runs past `elements` trips a debug assertion; use [`try_slice_pad_mask`] for
/// a checked variant.
///
/// # Panics
///
/// Panics via a debug assertion (debug builds only) if
/// `span.start + span.len > elements.len()`.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn slice_pad_mask<T: Copy>(elements: &[T], span: &Span, pad: T) -> (Vec<T>, Vec<u8>) {
  debug_assert!(
    span.start + span.len <= elements.len(),
    "span (start {}, len {}) runs past element length {}",
    span.start,
    span.len,
    elements.len()
  );
  let window = span.window;

  let mut values = Vec::with_capacity(window);
  values.extend_from_slice(&elements[span.start..span.start + span.len]);
  values.resize(window, pad);

  let mut mask = Vec::with_capacity(window);
  mask.resize(span.len, 1u8);
  mask.resize(window, 0u8);

  (values, mask)
}

/// The checked counterpart of [`slice_pad_mask`]: validate the span against
/// `elements` before slicing.
///
/// # Errors
///
/// Returns [`WinditError::DimMismatch`] when the span runs past `elements`
/// (`span.start + span.len > elements.len()`), reporting `got = elements.len()`
/// and `expected = span.start + span.len`.
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn try_slice_pad_mask<T: Copy>(
  elements: &[T],
  span: &Span,
  pad: T,
) -> Result<(Vec<T>, Vec<u8>), WinditError> {
  let required = span.start + span.len;
  if required > elements.len() {
    return Err(WinditError::DimMismatch {
      got: elements.len(),
      expected: required,
    });
  }
  Ok(slice_pad_mask(elements, span, pad))
}
