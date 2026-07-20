//! Pre-processing: turn a span into a fixed-width, padded window and its mask.
//!
//! A `Span` from a `WindowPlan` selects a run of
//! real input elements; `slice_pad_mask` copies them into a `window`-length
//! buffer, right-pads the remainder, and produces the matching attention mask
//! (`1` for real elements, `0` for padding).

use std::vec::Vec;

use crate::{error::WinditError, plan::Span};

#[cfg(test)]
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
/// Whether the span is well formed in itself is not this function's concern:
/// [`Span`] enforces `0 < len <= window` and a representable
/// [`end`](Span::end) at construction, in every build.
///
/// # Panics
///
/// Panics via a debug assertion (debug builds only) if `span.end() >
/// elements.len()`. Release builds panic on the slice index instead, which is
/// bounds-checked regardless.
///
/// Panics, in every build, if two `span.window()`-element buffers cannot be
/// allocated. A window is a caller-supplied count that need not correspond to
/// memory that exists — `WindowOptions::new(usize::MAX)` plans a perfectly valid
/// span over a one-element input — so this is reachable from a well-formed
/// geometry rather than only from a bug. [`try_slice_pad_mask`] reports it as
/// [`WinditError::AllocFailed`] instead.
pub fn slice_pad_mask<T: Copy>(elements: &[T], span: &Span, pad: T) -> (Vec<T>, Vec<u8>) {
  debug_assert!(
    span.end() <= elements.len(),
    "span (start {}, len {}) runs past element length {}",
    span.start(),
    span.len(),
    elements.len()
  );
  let window = span.window();

  let mut values = Vec::with_capacity(window);
  values.extend_from_slice(&elements[span.start()..span.end()]);
  values.resize(window, pad);

  let mut mask = Vec::with_capacity(window);
  mask.resize(span.len(), 1u8);
  mask.resize(window, 0u8);

  (values, mask)
}

/// The checked counterpart of [`slice_pad_mask`]: validate the span against
/// `elements` before slicing.
///
/// # Errors
///
/// Returns [`WinditError::DimMismatch`] when the span runs past `elements`
/// (`span.end() > elements.len()`), reporting `got = elements.len()` and
/// `expected = span.end()`. Returns [`WinditError::InvalidSpan`] if the span's
/// end is not representable — which [`Span`]'s constructors already rule out,
/// so this arm exists only to keep the checked entry point total on its own
/// terms rather than by appeal to a distant invariant. Returns
/// [`WinditError::AllocFailed`] if the two `span.window()`-element buffers
/// cannot be allocated.
///
/// The allocation arm is the one that makes this variant worth calling. A span
/// is bounded by the elements it is applied to, but its *window* is not: a plan
/// built from `WindowOptions::new(usize::MAX)` over a one-element input yields
/// `(start 0, len 1, window usize::MAX)`, which satisfies every span invariant
/// and passes every check above. [`slice_pad_mask`] would then panic on a
/// capacity overflow; here the window is asked for rather than assumed.
pub fn try_slice_pad_mask<T: Copy>(
  elements: &[T],
  span: &Span,
  pad: T,
) -> Result<(Vec<T>, Vec<u8>), WinditError> {
  let Some(required) = span.start().checked_add(span.len()) else {
    return Err(WinditError::InvalidSpan {
      start: span.start(),
      len: span.len(),
      window: span.window(),
    });
  };
  if required > elements.len() {
    return Err(WinditError::DimMismatch {
      got: elements.len(),
      expected: required,
    });
  }
  let window = span.window();
  let too_large = || WinditError::AllocFailed { elements: window };

  // Reserved rather than delegated to `slice_pad_mask`: the infallible variant
  // reaches the window through `Vec::with_capacity`, which aborts the process on
  // a capacity overflow instead of returning. Both vectors are grown to exactly
  // `window` afterwards, and `span.len() <= window` holds by the span invariant,
  // so neither `extend_from_slice` nor `resize` can reallocate past what was
  // reserved here.
  let mut values = Vec::new();
  values.try_reserve_exact(window).map_err(|_| too_large())?;
  values.extend_from_slice(&elements[span.start()..span.end()]);
  values.resize(window, pad);

  let mut mask = Vec::new();
  mask.try_reserve_exact(window).map_err(|_| too_large())?;
  mask.resize(span.len(), 1u8);
  mask.resize(window, 0u8);

  Ok((values, mask))
}
