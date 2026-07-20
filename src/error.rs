//! The crate-wide error type.

/// An error produced by window planning, pre-processing, embedding
/// normalization, aggregation, or segmentation.
///
/// Every fallible operation in the crate returns this type. Variants carry the
/// offending values so callers can report or recover without re-deriving them.
///
/// This enum is `#[non_exhaustive]`: new variants may be added without that
/// being a breaking change. The taxonomy has already grown more than once — for
/// example adding [`WinditError::AlphaOutOfRange`] rather than overloading the
/// ill-fitting [`WinditError::NonFinite`] for an out-of-range EMA `alpha` — so
/// callers outside this crate must match with a wildcard arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WinditError {
  /// The configured window size was zero, so no geometry can be produced.
  ZeroWindow,
  /// The requested overlap was greater than or equal to the window size, which
  /// would leave the hop at zero and never advance the plan.
  OverlapGeWindow {
    /// The requested overlap, in elements.
    overlap: usize,
    /// The window size the overlap must stay strictly below, in elements.
    window: usize,
  },
  /// Planning produced more windows than the configured `max_windows` cap.
  TooManyWindows {
    /// The window count at which planning aborted: always the configured
    /// maximum plus one. Planning stops as soon as the cap is first exceeded,
    /// so this is the abort count, not the size of the full (unbounded) plan.
    got: usize,
    /// The configured maximum.
    max: usize,
  },
  /// A span was not well formed: it covered no real element, covered more than
  /// its window, or its end (`start + len`) was not representable as a `usize`.
  InvalidSpan {
    /// The rejected start, in input elements.
    start: usize,
    /// The rejected count of real elements.
    len: usize,
    /// The window the length must stay within, in elements.
    window: usize,
  },
  /// A range was inverted: its start lay past its end.
  InvalidRange {
    /// The rejected start, in input elements.
    start: usize,
    /// The rejected end, in input elements.
    end: usize,
  },
  /// Two quantities that had to share a dimension did not — for example an
  /// embedding whose length differed from its peers, or a span that ran past
  /// the elements it was applied to.
  DimMismatch {
    /// The dimension (or length) that was found.
    got: usize,
    /// The dimension (or length) that was expected.
    expected: usize,
  },
  /// An embedding could not be normalized to a finite unit vector: a component
  /// was not finite, or the vector had zero norm.
  NonFinite,
  /// An exponential-moving-average smoothing factor (`alpha`) was outside the
  /// valid `[0, 1]` range, or was not a number.
  ///
  /// Returned by `EmaRenormalized`, whose
  /// out-of-range `alpha` would otherwise silently produce a sign-flipping
  /// "average" rather than a moving average. The infallible
  /// `Ema` smoother has no error channel and instead
  /// clamps `alpha` into range.
  AlphaOutOfRange,
  /// The input sequence was empty where at least one element was required.
  Empty,
  /// A buffer whose size is set by the caller's geometry could not be allocated:
  /// the requested element count exceeded what the allocator can address, or the
  /// allocation was refused.
  ///
  /// This is what keeps the checked entry points checked. A window size and a
  /// planned input length are both caller-controlled counts that need not
  /// correspond to memory that exists, so a geometry can be entirely well formed
  /// — every [`InvalidSpan`](WinditError::InvalidSpan) condition satisfied, every
  /// span in bounds — and still name more elements than can be buffered. The
  /// `try_` entry points report that here rather than panicking on a capacity
  /// overflow; the infallible ones document the panic instead.
  AllocFailed {
    /// The number of elements the buffer would have held.
    elements: usize,
  },
}

impl core::fmt::Display for WinditError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    match self {
      Self::ZeroWindow => f.write_str("window size must be non-zero"),
      Self::OverlapGeWindow { overlap, window } => {
        write!(
          f,
          "overlap {overlap} must be less than the window size {window}"
        )
      }
      Self::TooManyWindows { got, max } => {
        write!(f, "produced {got} windows, exceeding the maximum of {max}")
      }
      Self::InvalidSpan { start, len, window } => {
        write!(
          f,
          "span (start {start}, len {len}, window {window}) must satisfy 0 < len <= window with a representable start + len"
        )
      }
      Self::InvalidRange { start, end } => {
        write!(f, "range start {start} must not exceed its end {end}")
      }
      Self::DimMismatch { got, expected } => {
        write!(f, "dimension {got} does not match the expected {expected}")
      }
      Self::NonFinite => f.write_str("embedding could not be normalized to a finite unit vector"),
      Self::AlphaOutOfRange => f.write_str("EMA smoothing factor alpha must be in [0, 1]"),
      Self::Empty => f.write_str("input sequence was empty"),
      Self::AllocFailed { elements } => {
        write!(f, "could not allocate a buffer of {elements} elements")
      }
    }
  }
}

impl core::error::Error for WinditError {}
