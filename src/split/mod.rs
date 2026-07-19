//! Split policies: decide how an input is divided before windowing.
//!
//! [`SplitPolicy`] maps an input length and [`WindowOptions`] to the [`Span`]s
//! that cover it. [`FixedWindow`] is the mechanical element-count split — it
//! delegates to [`WindowPlan::spans`], so any unit (samples, tokens, patches)
//! is divided the same way.
//!
//! [`ContentAware`] (feature `text`) is the tokenizer-free string chunker: it
//! packs text into chunks that respect paragraph, sentence, and word boundaries,
//! measuring length through a caller-supplied `len_fn` so the caller's own
//! tokenizer defines "how long". It is a separate surface from [`SplitPolicy`]
//! because it returns byte-offset ranges into the text rather than element spans.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "text")]
use unicode_segmentation::UnicodeSegmentation;

#[cfg(feature = "alloc")]
use crate::{
  error::WinditError,
  plan::{Span, WindowOptions, WindowPlan},
};

#[cfg(all(test, feature = "alloc"))]
mod tests;

/// A policy that divides an input of `input_len` elements into [`Span`]s.
///
/// The trait is object-safe (`&dyn SplitPolicy` works): the only method takes an
/// element count and [`WindowOptions`] and returns the plan.
#[cfg(feature = "alloc")]
pub trait SplitPolicy {
  /// Plan the spans covering `input_len` elements under `opts`.
  ///
  /// # Errors
  ///
  /// Returns whatever the underlying geometry rejects — see
  /// [`WindowPlan::spans`] (for example [`WinditError::ZeroWindow`] or
  /// [`WinditError::TooManyWindows`]).
  fn split(&self, input_len: usize, opts: &WindowOptions) -> Result<Vec<Span>, WinditError>;
}

/// The mechanical fixed-window split: a direct delegation to [`WindowPlan::spans`].
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedWindow;

#[cfg(feature = "alloc")]
impl SplitPolicy for FixedWindow {
  fn split(&self, input_len: usize, opts: &WindowOptions) -> Result<Vec<Span>, WinditError> {
    WindowPlan::spans(opts, input_len)
  }
}

/// A tokenizer-free, content-aware string chunker (feature `text`).
///
/// [`chunk`](ContentAware::chunk) splits text on recursive boundaries —
/// paragraphs (`\n\n`), then sentences, then words — and greedily packs the
/// pieces into chunks no longer than [`WindowOptions::window`], as measured by
/// the caller's `len_fn`. Because length is whatever `len_fn` returns, the caller
/// supplies its own notion of a token (word count, a tokenizer's id count, code
/// points) without this crate depending on a tokenizer.
#[cfg(feature = "text")]
#[derive(Clone, Copy)]
pub struct ContentAware<'a> {
  /// Measures the length of a text slice in the caller's units. `chunk`
  /// guarantees every returned range measures at most the window under this
  /// function, save for a single `char` that alone exceeds the window (it
  /// cannot be split further).
  pub len_fn: &'a dyn Fn(&str) -> usize,
}

#[cfg(feature = "text")]
impl ContentAware<'_> {
  /// Chunk `text` into byte-offset ranges, each measuring at most
  /// [`WindowOptions::window`] under `len_fn`.
  ///
  /// Each returned `(start, end)` is a byte range on `char` boundaries, usable
  /// directly as `&text[start..end]`. Boundaries are preferred coarse-to-fine:
  /// a paragraph, sentence, or word that fits is kept whole, and only a unit
  /// that overflows the window on its own is split further — a long sentence
  /// into words, a long word into `len_fn`-measured character slices. Chunks are
  /// packed greedily; when [`WindowOptions::overlap`] is non-zero, consecutive
  /// chunks repeat at most that many trailing tokens' worth of whole boundary
  /// units (as measured by `len_fn`).
  ///
  /// Ranges cover the tokenized content in order; inter-token whitespace falls
  /// inside a chunk's span but is never a chunk of its own. An empty or
  /// whitespace-only input yields no chunks.
  ///
  /// Every returned range measures at most the window, with a single exception:
  /// a lone `char` whose own measure exceeds the window is emitted as-is,
  /// because it cannot be split further. In the oversized-sentence word
  /// fallback, punctuation lying outside word boundaries (a trailing period,
  /// say) is not covered by any chunk. If `opts` is not valid — a zero window,
  /// or an overlap at least the window — no chunks are produced.
  #[must_use]
  pub fn chunk(&self, text: &str, opts: &WindowOptions) -> Vec<(usize, usize)> {
    if opts.validate().is_err() {
      return Vec::new();
    }
    let atoms = split_range(
      text,
      0,
      text.len(),
      opts.window(),
      self.len_fn,
      Level::Paragraph,
    );
    pack(text, &atoms, opts.window(), opts.overlap(), self.len_fn)
  }
}

/// The recursion level: which boundary to split on when a range overflows.
#[cfg(feature = "text")]
#[derive(Clone, Copy)]
enum Level {
  Paragraph,
  Sentence,
  Word,
}

/// Split `text[start..end]` into atoms that each fit the window, descending
/// through boundary levels only for the parts that overflow.
#[cfg(feature = "text")]
fn split_range(
  text: &str,
  start: usize,
  end: usize,
  window: usize,
  len_fn: &dyn Fn(&str) -> usize,
  level: Level,
) -> Vec<(usize, usize)> {
  if text[start..end].trim().is_empty() {
    return Vec::new();
  }
  if len_fn(&text[start..end]) <= window {
    return alloc::vec![(start, end)];
  }
  let mut out = Vec::new();
  match level {
    Level::Paragraph => {
      for (s, e) in paragraph_ranges(text, start, end) {
        out.extend(split_range(text, s, e, window, len_fn, Level::Sentence));
      }
    }
    Level::Sentence => {
      for (s, e) in sentence_ranges(text, start, end) {
        out.extend(split_range(text, s, e, window, len_fn, Level::Word));
      }
    }
    Level::Word => {
      let words = word_ranges(text, start, end);
      if words.is_empty() {
        out.extend(split_chars(text, start, end, window, len_fn));
      } else {
        for (s, e) in words {
          if len_fn(&text[s..e]) <= window {
            out.push((s, e));
          } else {
            out.extend(split_chars(text, s, e, window, len_fn));
          }
        }
      }
    }
  }
  out
}

/// Byte ranges of the `\n\n`-separated paragraphs within `text[start..end]`.
#[cfg(feature = "text")]
fn paragraph_ranges(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
  let slice = &text[start..end];
  let mut out = Vec::new();
  let mut last = 0usize;
  for (pos, sep) in slice.match_indices("\n\n") {
    out.push((start + last, start + pos));
    last = pos + sep.len();
  }
  out.push((start + last, end));
  out
}

/// Byte ranges of the Unicode sentences within `text[start..end]`.
#[cfg(feature = "text")]
fn sentence_ranges(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
  text[start..end]
    .split_sentence_bound_indices()
    .map(|(off, sent)| (start + off, start + off + sent.len()))
    .collect()
}

/// Byte ranges of the Unicode words within `text[start..end]` (whitespace and
/// punctuation between words are excluded).
#[cfg(feature = "text")]
fn word_ranges(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
  text[start..end]
    .unicode_word_indices()
    .map(|(off, word)| (start + off, start + off + word.len()))
    .collect()
}

/// Split `text[start..end]` into maximal `char`-aligned slices that each fit the
/// window: the hard fallback for a single unit longer than the window.
#[cfg(feature = "text")]
fn split_chars(
  text: &str,
  start: usize,
  end: usize,
  window: usize,
  len_fn: &dyn Fn(&str) -> usize,
) -> Vec<(usize, usize)> {
  let mut out = Vec::new();
  let mut seg_start = start;
  for (i, ch) in text[start..end].char_indices() {
    let abs = start + i;
    let next = abs + ch.len_utf8();
    // Close the current slice before the char that would overflow it, but only
    // once it holds at least one char, so a single oversized char still emits.
    if abs > seg_start && len_fn(&text[seg_start..next]) > window {
      out.push((seg_start, abs));
      seg_start = abs;
    }
  }
  if seg_start < end {
    out.push((seg_start, end));
  }
  out
}

/// Greedily pack `atoms` into chunks no longer than the window, repeating at
/// most `overlap` tokens' worth of trailing whole atoms between consecutive
/// chunks (`overlap` is a token budget measured by `len_fn`, not an atom count).
///
/// `atoms` must be in order; each is assumed to fit the window on its own.
#[cfg(feature = "text")]
fn pack(
  text: &str,
  atoms: &[(usize, usize)],
  window: usize,
  overlap: usize,
  len_fn: &dyn Fn(&str) -> usize,
) -> Vec<(usize, usize)> {
  let mut chunks = Vec::new();
  let mut i = 0usize;
  while i < atoms.len() {
    let chunk_start = atoms[i].0;
    let mut chunk_end = atoms[i].1;
    let mut j = i + 1;
    while j < atoms.len() {
      let candidate_end = atoms[j].1;
      if len_fn(&text[chunk_start..candidate_end]) <= window {
        chunk_end = candidate_end;
        j += 1;
      } else {
        break;
      }
    }
    chunks.push((chunk_start, chunk_end));
    if j >= atoms.len() {
      break;
    }
    // Back up for the next chunk by TOKEN budget, not atom count: repeat as many
    // trailing whole atoms as fit in `overlap` tokens (measured over the real
    // contiguous text, so it is robust to non-additive tokenizers), but always
    // advance at least one atom so packing terminates.
    let mut next = j;
    while next > i + 1 && len_fn(&text[atoms[next - 1].0..atoms[j - 1].1]) <= overlap {
      next -= 1;
    }
    i = next;
  }
  chunks
}
