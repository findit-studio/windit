//! Window geometry: configurable options, tail handling, spans, and the planner.
//!
//! A `WindowPlan` turns an input length plus a [`WindowOptions`] into a list of
//! [`Span`]s — plain `usize` element counts, unit-agnostic across samples,
//! tokens, patches, and frames. The same spans drive pre-processing (slice /
//! pad / mask) and post-processing (aggregate / smooth / segment).

use crate::error::WinditError;

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests;

/// A single window's placement over the input: where it starts, how many real
/// elements it covers, and the fixed window size it pads to.
///
/// `len` is the number of *real* input elements (`0 < len <= window`); the
/// remaining `window - len` positions are padding. [`coverage`](Span::coverage)
/// reports the real fraction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Span {
  /// Index of the first real element, in input elements.
  pub start: usize,
  /// Number of real elements in the window (`0 < len <= window`).
  pub len: usize,
  /// The fixed window size this span pads to, in elements.
  pub window: usize,
}

impl Span {
  /// The fraction of the window filled by real elements, in `(0, 1]` for
  /// planner-produced spans.
  ///
  /// Reports `0.0` for a zero-window span (`window == 0`), which a caller can
  /// construct through the public fields instead of dividing by zero; the
  /// planner itself never produces one.
  #[must_use]
  pub fn coverage(&self) -> f32 {
    if self.window == 0 {
      0.0
    } else {
      self.len as f32 / self.window as f32
    }
  }
}

/// How the planner treats a final window that does not fill a whole [`Span`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum TailPolicy {
  /// Keep the ragged tail as a partial span; its [`Span::coverage`] is below
  /// `1.0`. This is the default.
  #[default]
  KeepWithCoverage,
  /// Drop a ragged tail whose real length is below the given minimum. A full
  /// window (`len == window`) is always kept.
  DropBelowMin(usize),
  /// Keep the ragged tail; pre-processing pads it to a full window. The produced
  /// span is identical to [`KeepWithCoverage`](TailPolicy::KeepWithCoverage);
  /// the two differ only in downstream padding and weighting intent.
  PadFull,
}

/// Fully configurable window geometry: window size, hop/overlap, tail handling,
/// and an optional cap on the number of windows.
///
/// Construct with [`WindowOptions::new`] (non-overlapping windows that keep a
/// ragged tail) and refine with the `with_*` builders. Nothing is validated
/// until `WindowPlan::spans` runs — or you call
/// [`validate`](WindowOptions::validate) directly — so construction is
/// infallible.
///
/// With the `serde` feature every field is serialized, so a self-persisted
/// geometry always round-trips exactly. On deserialization the `window`, `hop`,
/// and `tail` fields are required; `max_windows` may be omitted and then
/// defaults to no cap (`None`). Unknown keys are rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct WindowOptions {
  window: usize,
  hop: usize,
  tail: TailPolicy,
  max_windows: Option<usize>,
}

impl WindowOptions {
  /// A geometry with the given window size, no overlap (`hop == window`),
  /// [`TailPolicy::KeepWithCoverage`], and no window cap.
  ///
  /// The window size is not validated here; a zero window is rejected by
  /// [`validate`](WindowOptions::validate) and `WindowPlan::spans`.
  #[must_use]
  pub const fn new(window: usize) -> Self {
    Self {
      window,
      hop: window,
      tail: TailPolicy::KeepWithCoverage,
      max_windows: None,
    }
  }

  /// Set the hop (stride between consecutive window starts), in elements.
  ///
  /// A hop below the window overlaps consecutive windows; a hop equal to the
  /// window tiles them. A zero hop is rejected by
  /// [`validate`](WindowOptions::validate). For full input coverage the hop
  /// should not exceed the window size.
  #[must_use]
  pub const fn with_hop(mut self, hop: usize) -> Self {
    self.hop = hop;
    self
  }

  /// Set the overlap between consecutive windows, in elements
  /// (`hop == window - overlap`).
  ///
  /// An overlap of `0` tiles the windows. The subtraction saturates, so an
  /// overlap of at least the window size yields a zero hop that
  /// [`validate`](WindowOptions::validate) reports as
  /// [`WinditError::OverlapGeWindow`].
  #[must_use]
  pub const fn with_overlap(mut self, overlap: usize) -> Self {
    self.hop = self.window.saturating_sub(overlap);
    self
  }

  /// Set the tail policy for a final ragged window.
  #[must_use]
  pub const fn with_tail(mut self, tail: TailPolicy) -> Self {
    self.tail = tail;
    self
  }

  /// Cap the number of windows the plan may produce; exceeding it is an error.
  #[must_use]
  pub const fn with_max_windows(mut self, max_windows: usize) -> Self {
    self.max_windows = Some(max_windows);
    self
  }

  /// The window size, in elements.
  #[must_use]
  pub const fn window(&self) -> usize {
    self.window
  }

  /// The hop (stride between window starts), in elements.
  #[must_use]
  pub const fn hop(&self) -> usize {
    self.hop
  }

  /// The overlap between consecutive windows, in elements (`window - hop`,
  /// saturating at `0`).
  #[must_use]
  pub const fn overlap(&self) -> usize {
    self.window.saturating_sub(self.hop)
  }

  /// The tail policy.
  #[must_use]
  pub const fn tail(&self) -> &TailPolicy {
    &self.tail
  }

  /// The configured window cap, if any.
  #[must_use]
  pub const fn max_windows(&self) -> Option<usize> {
    self.max_windows
  }

  /// Check the geometry: the window must be non-zero and the hop must advance
  /// (equivalently, the overlap must be below the window size).
  ///
  /// # Errors
  ///
  /// - [`WinditError::ZeroWindow`] if the window size is zero.
  /// - [`WinditError::OverlapGeWindow`] if the hop is zero — the overlap is at
  ///   least the window size.
  pub const fn validate(&self) -> Result<(), WinditError> {
    if self.window == 0 {
      return Err(WinditError::ZeroWindow);
    }
    if self.hop == 0 {
      return Err(WinditError::OverlapGeWindow {
        overlap: self.overlap(),
        window: self.window,
      });
    }
    Ok(())
  }
}

/// The window planner: turns an input length and [`WindowOptions`] into the list
/// of [`Span`]s that drives pre- and post-processing.
#[derive(Clone, Copy, Debug)]
pub struct WindowPlan;

#[cfg(any(feature = "std", feature = "alloc"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "std", feature = "alloc"))))]
const _: () = {
  use std::vec::Vec;

  impl WindowPlan {
    /// Plan the windows covering `input_len` elements under `opts`, in input
    /// order.
    ///
    /// An empty input yields an empty plan; a non-empty input can also yield an
    /// empty plan when [`TailPolicy::DropBelowMin`] drops the sole, too-short
    /// window. The final window may be ragged according to the [`TailPolicy`].
    /// When `opts.hop() <= opts.window()` every input element is covered by at
    /// least one span; a larger hop strides over the input, leaving gaps
    /// uncovered.
    ///
    /// # Errors
    ///
    /// Propagates [`WindowOptions::validate`], and returns
    /// [`WinditError::TooManyWindows`] if the plan would exceed
    /// [`WindowOptions::max_windows`].
    pub fn spans(opts: &WindowOptions, input_len: usize) -> Result<Vec<Span>, WinditError> {
      opts.validate()?;
      let (w, hop) = (opts.window(), opts.hop());
      let mut out = Vec::new();
      if input_len == 0 {
        return Ok(out);
      }
      let mut start = 0usize;
      loop {
        // Reachable only when `hop > w` (a stride wider than the window); keeps
        // the function total instead of underflowing `input_len - start`.
        if start >= input_len {
          break;
        }
        let len = core::cmp::min(w, input_len - start);
        // `input_len - start` is safe under the loop guard (`start < input_len`);
        // computing the tail this way avoids overflowing `start + w` when
        // `input_len` is within a window of `usize::MAX`.
        let is_tail = input_len - start <= w;
        let keep = match opts.tail() {
          TailPolicy::DropBelowMin(m) => len >= *m || len == w,
          _ => true,
        };
        if keep {
          out.push(Span {
            start,
            len,
            window: w,
          });
        }
        if let Some(max) = opts.max_windows() {
          if out.len() > max {
            return Err(WinditError::TooManyWindows {
              got: out.len(),
              max,
            });
          }
        }
        if is_tail {
          break;
        }
        // A hop that would carry `start` past `usize::MAX` cannot reach any further
        // in-bounds element, so stop rather than wrap.
        match start.checked_add(hop) {
          Some(s) => start = s,
          None => break,
        }
      }
      Ok(out)
    }
  }
};
