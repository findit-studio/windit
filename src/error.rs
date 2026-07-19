//! The crate-wide error type.

/// An error produced by window planning, pre-processing, embedding
/// normalization, or aggregation.
///
/// Every fallible operation in the crate returns this type. Variants carry the
/// offending values so callers can report or recover without re-deriving them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
  /// Returned by [`EmaRenormalized`](crate::aggregate::EmaRenormalized), whose
  /// out-of-range `alpha` would otherwise silently produce a sign-flipping
  /// "average" rather than a moving average. The infallible
  /// [`Ema`](crate::smooth::Ema) smoother has no error channel and instead
  /// clamps `alpha` into range.
  AlphaOutOfRange,
  /// The input sequence was empty where at least one element was required.
  Empty,
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
      Self::DimMismatch { got, expected } => {
        write!(f, "dimension {got} does not match the expected {expected}")
      }
      Self::NonFinite => f.write_str("embedding could not be normalized to a finite unit vector"),
      Self::AlphaOutOfRange => f.write_str("EMA smoothing factor alpha must be in [0, 1]"),
      Self::Empty => f.write_str("input sequence was empty"),
    }
  }
}

#[cfg(feature = "std")]
impl std::error::Error for WinditError {}
